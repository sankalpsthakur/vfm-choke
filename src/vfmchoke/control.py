"""Choke control on the calibrated VFM. Advisory setpoints only: an operator and the control system decide.

The twin of a well is its design deliverability (reservoir pressure, productivity, tubing; corrected by labels
when there are any) with the calibrated VFM as its choke: the physics rate at nominal discharge coefficient and
vendor trim, times exp(f + b0) * opening**b1. Twin and truth share one solver, so a setpoint found on the twin can be
applied to the truth field as is.
"""
from dataclasses import fields, replace

import numpy as np
from scipy.optimize import minimize

from . import network, pvt, well


def replicate(w, n):
    """Wells of shape (W,) as n identical groups, shape (n, W)."""
    km, ke = w._vfm_terms()
    kw = {f.name: np.broadcast_to(np.asarray(getattr(w, f.name)), (n,) + np.shape(w.p_res)).copy()
          for f in fields(w) if f.name not in ('fluid', 'k_mult', 'k_exp')}
    fl = pvt.Fluid(*(np.broadcast_to(getattr(w.fluid, f.name), (n,) + np.shape(w.p_res)).copy()
                     for f in fields(w.fluid)))
    return well.Wells(fluid=fl, k_mult=np.broadcast_to(km, (n,) + np.shape(w.p_res)).copy(),
                      k_exp=np.broadcast_to(ke, (n,) + np.shape(w.p_res)).copy(), **kw)


def replicate_line(line, g, n):
    return network.Line(*(np.full(n, float(np.asarray(getattr(line, f.name))[g])) for f in fields(line)))


def replicate_table(t, n):
    return {k: np.broadcast_to(v, (n,) + v.shape).copy() for k, v in t.items()}


def rates(w, table, openings, line, iters=28):
    """Network solve for stacked candidate opening vectors (n, W). Returns per-candidate totals and well rates."""
    n = openings.shape[0]
    out = network.solve(replicate(w, n), replicate_table(table, n), openings, replicate_line(line, 0, n),
                        iters=iters)
    return {'q_o': out['q_o'].sum(1), 'q_g': out['q_g'].sum(1), 'q_w': out['q_w'].sum(1), 'q_l': out['q_l'],
            'p_man': out['p_man'], 'p_wh': out['p_wh']}


def single_well_setpoint(w1, table1, p2, q_target, u_lo=0.05, u_hi=1.0, iters=24):
    """Opening at which the twin well delivers q_target (Sm3/s liquid) against downstream pressure p2.
    Returns (opening, achievable) where achievable is False when even full opening falls short."""
    q_full = float(np.ravel(well.solve_rate(w1, table1, np.asarray(p2), np.asarray(u_hi))['q_l'])[0])
    if q_full <= q_target:
        return u_hi, False
    lo, hi = u_lo, u_hi
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        q = float(np.ravel(well.solve_rate(w1, table1, np.asarray(p2), np.asarray(mid))['q_l'])[0])
        lo, hi = (mid, hi) if q < q_target else (lo, mid)
    return 0.5 * (lo + hi), True


def scale_heuristic(w_mean, w_hi, table, line, u0, gas_cap, water_cap, iters=18):
    """Operator baseline: scale every open choke by one factor until the cautious twin meets both caps."""
    lo, hi = 0.0, 1.0 / max(float(u0.max()), 1e-6)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        r = rates(w_hi, table, np.clip(u0 * mid, 0, 1)[None, :], line)
        ok = (r['q_g'][0] <= gas_cap) and (r['q_w'][0] <= water_cap)
        lo, hi = (mid, hi) if ok else (lo, mid)
    return np.clip(u0 * lo, 0, 1)


def optimize_group(w_mean, w_hi, table, line, u_start, open_mask, gas_cap, water_cap, h=0.02, maxiter=30):
    """Maximise twin oil over the openings of the open wells, subject to the cautious twin meeting the caps.
    Gradients by batched finite differences: all perturbed candidates go through one network solve."""
    idx = np.nonzero(open_mask)[0]
    cache = {}

    def evaluate(x):
        key = tuple(np.round(x, 6))
        if key in cache:
            return cache[key]
        base = u_start.copy()
        base[idx] = x
        cands = [base]
        for j in idx:
            c = base.copy()
            c[j] = min(c[j] + h, 1.0) if c[j] + h <= 1.0 else c[j] - h
            cands.append(c)
        cands = np.array(cands)
        rm = rates(w_mean, table, cands, line)
        rh = rates(w_hi, table, cands, line)
        steps = (cands[1:, idx] - base[idx]).diagonal()
        f0 = (-rm['q_o'][0], gas_cap - rh['q_g'][0], water_cap - rh['q_w'][0])
        grads = (-(rm['q_o'][1:] - rm['q_o'][0]) / steps, -(rh['q_g'][1:] - rh['q_g'][0]) / steps,
                 -(rh['q_w'][1:] - rh['q_w'][0]) / steps)
        cache[key] = (f0, grads)
        return cache[key]

    scale = max(abs(float(rates(w_mean, table, u_start[None, :], line)['q_o'][0])), 1e-9)
    res = minimize(lambda x: evaluate(x)[0][0] / scale, u_start[idx], jac=lambda x: evaluate(x)[1][0] / scale,
                   method='SLSQP', bounds=[(0.05, 1.0)] * len(idx),
                   constraints=[{'type': 'ineq', 'fun': lambda x: evaluate(x)[0][1] / gas_cap,
                                 'jac': lambda x: evaluate(x)[1][1] / gas_cap},
                                {'type': 'ineq', 'fun': lambda x: evaluate(x)[0][2] / max(water_cap, 1e-9),
                                 'jac': lambda x: evaluate(x)[1][2] / max(water_cap, 1e-9)}],
                   options={'maxiter': maxiter, 'ftol': 1e-6})
    u = u_start.copy()
    u[idx] = np.clip(res.x, 0.05, 1.0)
    return u, {'success': bool(res.success), 'nit': int(res.nit), 'message': str(res.message)}


def cautious(w, z_sd):
    """The same twin with its choke multiplier raised by z standard deviations (upper bound for constraints)."""
    km, ke = w._vfm_terms()
    return replace(w, k_mult=km * np.exp(z_sd), k_exp=ke)
