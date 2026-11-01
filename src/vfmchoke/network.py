"""Subsea production network: wells -> chokes -> manifold -> flowline -> riser -> separator, with optional leaks.

Arrays are shaped (G, W): G manifold groups, each with up to W wells (pad unused slots with pi = 0, they never
flow). The manifold pressure of every group is found by bisection: a higher manifold pressure lets less flow in,
while the flowline needs a higher inlet pressure for more flow, so the balance is unique.

Leak kinds: 0 none, 1 jumper of one well (that well's fluid, at manifold pressure), 2 manifold (commingled fluid),
3 flowline at fraction `pos` of its length from the manifold, 4 riser at fraction `pos` of its height from the
base. A leak is a sharp-edged hole (two-phase orifice, choke model at full opening) discharging to seawater
hydrostatic pressure at its depth; nothing leaks where the line pressure is below ambient.
"""
from dataclasses import dataclass, fields

import numpy as np

from . import choke, pvt, units as u, well

NONE, JUMPER, MANIFOLD, FLOWLINE, RISER = 0, 1, 2, 3, 4
CD_HOLE = 0.62


@dataclass
class Line:
    """Flowline and riser of each group, arrays of shape (G,)."""
    length: np.ndarray      # horizontal flowline length, m
    d: np.ndarray           # inner diameter (flowline and riser), m
    depth: np.ndarray       # water depth = riser height, m
    p_sep: np.ndarray       # separator pressure, Pa

    def __post_init__(self):
        for f in fields(self):
            setattr(self, f.name, np.asarray(getattr(self, f.name), dtype=float))


@dataclass
class Leak:
    """One optional leak per group, arrays of shape (G,)."""
    kind: np.ndarray        # NONE, JUMPER, MANIFOLD, FLOWLINE or RISER
    well: np.ndarray        # well slot for JUMPER leaks
    pos: np.ndarray         # fraction along the flowline (FLOWLINE) or up the riser (RISER)
    d_hole: np.ndarray      # equivalent hole diameter, m

    @staticmethod
    def none(g):
        return Leak(kind=np.zeros(g, int), well=np.zeros(g, int), pos=np.zeros(g), d_hole=np.zeros(g))


def section_up(p_down, m, fluid, t, d, length, vertical, n_seg=8, rough=4.5e-5):
    """Upstream pressure of a pipe section from its downstream pressure, marching against the flow."""
    q_l = np.maximum(m, 0.0) / pvt.mass_per_liquid_sc(fluid)
    p = np.array(p_down, dtype=float)
    dz = length / n_seg
    for _ in range(n_seg):
        p = p + well.gradient(p, t, q_l, fluid, d, rough, vertical) * dz
    return p


def ambient(depth):
    return u.P_SC + u.RHO_SEAWATER * u.G * np.asarray(depth, dtype=float)


def hole_rate(p_node, depth_node, t, fluid, d_hole):
    """Two-phase mass rate out of a hole to seawater at its depth."""
    return choke.mass_rate(p_node, ambient(depth_node), t, fluid, 1.0, d_hole, CD_HOLE)


def std_mass(q_o, q_g, q_w, fluid):
    return q_o * pvt.rho_oil_sc(fluid.api) + q_g * pvt.rho_gas_sc(fluid.gas_sg) + q_w * pvt.rho_water_sc(fluid.salinity)


def _evaluate(p_man, wells, table, opening, line, leak, inner_iters=40):
    g_idx = np.arange(p_man.shape[0])
    op = well.solve_rate(wells, table, p_man[:, None] * np.ones_like(opening), opening)
    q_o, q_g, q_w = choke.std_rates(op['m'], wells.fluid)

    # jumper or manifold leak: removed before the flowline, at manifold pressure
    k = np.clip(leak.well, 0, opening.shape[1] - 1)
    jf = wells.fluid.take((g_idx, k))
    t_mix = (op['m'] * op['t_wh']).sum(1) / np.maximum(op['m'].sum(1), 1e-12)
    t_mix = np.where(op['m'].sum(1) > 0, t_mix, wells.t_sea.max(1))
    mf = pvt.mix(q_o, q_g, q_w, wells.fluid, axis=1)
    m_in_all = op['m'].sum(1)
    m_jumper = np.minimum(hole_rate(p_man, line.depth, op['t_wh'][g_idx, k], jf, leak.d_hole), op['m'][g_idx, k])
    m_manifold = np.minimum(hole_rate(p_man, line.depth, t_mix, mf, leak.d_hole), m_in_all)
    m_up_leak = np.where(leak.kind == JUMPER, m_jumper, np.where(leak.kind == MANIFOLD, m_manifold, 0.0))
    leak_fluid_up = pvt.Fluid(*(np.where(leak.kind == JUMPER, getattr(jf, n), getattr(mf, n))
                                for n in ('api', 'gas_sg', 'gor', 'wc', 'salinity')))
    lq_o, lq_g, lq_w = choke.std_rates(m_up_leak, leak_fluid_up)

    # stream entering the flowline
    fq_o = q_o.sum(1) - lq_o
    fq_g = q_g.sum(1) - lq_g
    fq_w = q_w.sum(1) - lq_w
    ff = pvt.mix(np.stack([q_o.sum(1), -lq_o], 1), np.stack([q_g.sum(1), -lq_g], 1),
                 np.stack([q_w.sum(1), -lq_w], 1),
                 pvt.Fluid(*(np.stack([getattr(mf, n), getattr(leak_fluid_up, n)], 1)
                             for n in ('api', 'gas_sg', 'gor', 'wc', 'salinity'))), axis=1)
    m_in = std_mass(fq_o, fq_g, fq_w, ff)

    # flowline and riser carry the inflow to the separator
    p_rb = section_up(line.p_sep, m_in, ff, t_mix, line.d, line.depth, 1.0)
    p_man_implied = section_up(p_rb, m_in, ff, t_mix, line.d, line.length, 0.0)
    p_node = p_rb.copy()
    m_down_leak = np.zeros_like(m_in)
    is_fl, is_rs = leak.kind == FLOWLINE, leak.kind == RISER
    if np.any(is_fl | is_rs):
        # leak node pressure: the hydraulics downstream of the node must carry what the hole does not take
        node_depth = np.where(is_rs, line.depth * (1.0 - leak.pos), line.depth)
        riser_above = line.depth * (1.0 - np.where(is_rs, leak.pos, 0.0))
        fl_after = line.length * (1.0 - np.where(is_fl, leak.pos, 0.0))

        def node_needed(m_down):
            p_top_part = section_up(line.p_sep, m_down, ff, t_mix, line.d, riser_above, 1.0)
            return np.where(is_fl, section_up(p_top_part, m_down, ff, t_mix, line.d, fl_after, 0.0), p_top_part)

        lo, hi = line.p_sep.copy(), line.p_sep + 400e5
        for _ in range(inner_iters):
            mid = 0.5 * (lo + hi)
            m_l = np.minimum(hole_rate(mid, node_depth, t_mix, ff, leak.d_hole), m_in)
            f = mid - node_needed(m_in - m_l)
            lo = np.where(f < 0, mid, lo)
            hi = np.where(f < 0, hi, mid)
        node = 0.5 * (lo + hi)
        leaking = is_fl | is_rs
        m_down_leak = np.where(leaking, np.minimum(hole_rate(node, node_depth, t_mix, ff, leak.d_hole), m_in), 0.0)
        p_node = np.where(leaking, node, p_node)
        # upstream of the node the full inflow flows
        rb_fl = section_up(line.p_sep, m_in - m_down_leak, ff, t_mix, line.d, line.depth, 1.0)
        rb_rs = section_up(node, m_in, ff, t_mix, line.d, line.depth * leak.pos, 1.0)
        man_fl = section_up(node, m_in, ff, t_mix, line.d, line.length * leak.pos, 0.0)
        man_rs = section_up(rb_rs, m_in, ff, t_mix, line.d, line.length, 0.0)
        p_rb = np.where(is_fl, rb_fl, np.where(is_rs, rb_rs, p_rb))
        p_man_implied = np.where(is_fl, man_fl, np.where(is_rs, man_rs, p_man_implied))

    dq_o, dq_g, dq_w = choke.std_rates(m_down_leak, ff)
    leak_m = m_up_leak + m_down_leak
    out = {
        'p_man': p_man, 'p_man_implied': p_man_implied, 'p_riser_base': p_rb, 'p_node': p_node, 't_mix': t_mix,
        'q_l': op['q_l'], 'p_wh': op['p_wh'], 't_wh': op['t_wh'], 'm': op['m'],
        'q_o': q_o, 'q_g': q_g, 'q_w': q_w,
        'leak_m': leak_m,
        'leak_q_o': lq_o + dq_o, 'leak_q_g': lq_g + dq_g, 'leak_q_w': lq_w + dq_w,
        'sep_q_o': fq_o - dq_o, 'sep_q_g': fq_g - dq_g, 'sep_q_w': fq_w - dq_w,
        'sep_fluid': ff, 'm_in': m_in,
    }
    out['sep_m'] = std_mass(out['sep_q_o'], out['sep_q_g'], out['sep_q_w'], ff)
    return out


def solve(wells, table, opening, line, leak=None, iters=40):
    """Operating point of every group. wells/opening shaped (G, W); table from well.vlp_table(wells)."""
    opening = np.asarray(opening, dtype=float)
    leak = leak if leak is not None else Leak.none(opening.shape[0])
    lo = line.p_sep.copy()
    hi = np.maximum(table['p_wh'][..., 0].max(axis=1), line.p_sep + 1e5)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        res = _evaluate(mid, wells, table, opening, line, leak)
        f = mid - res['p_man_implied']
        lo = np.where(f < 0, mid, lo)
        hi = np.where(f < 0, hi, mid)
    return _evaluate(0.5 * (lo + hi), wells, table, opening, line, leak)
