"""Reservoir inflow and vertical tubing lift for a set of wells (all arrays, vectorised over wells).

Inflow: linear productivity above the bubble point, Vogel below (composite IPR), on stock-tank liquid.
Lift: a vertical tubing traverse marched upward with drift-flux gas holdup (C0 = 1.2, Taylor-bubble drift
velocity) and homogeneous friction. Each well gets a lookup table of wellhead pressure and temperature against
liquid rate, and its operating point is where the choke passes exactly what the well delivers.
"""
from dataclasses import dataclass, fields

import numpy as np

from . import choke, pvt, units as u

PI_FIELD = u.BBL / u.DAY / u.PSI        # one stb/d/psi in Sm3/s/Pa


@dataclass
class Wells:
    """Per-well parameters, all arrays of the same shape."""
    p_res: np.ndarray       # reservoir pressure, Pa
    t_res: np.ndarray       # reservoir temperature, K
    pi: np.ndarray          # productivity index on stock-tank liquid, Sm3/s/Pa
    depth: np.ndarray       # true vertical depth of the tubing, m
    d_tub: np.ndarray       # tubing inner diameter, m
    d_choke: np.ndarray     # choke bean diameter at full opening, m
    cd: np.ndarray          # choke discharge coefficient
    trim_gamma: np.ndarray  # choke trim exponent (effective area = opening**gamma)
    t_sea: np.ndarray       # seabed / ambient temperature, K
    q_t: np.ndarray         # rate scale of wellhead heating, Sm3/s
    fluid: pvt.Fluid
    k_mult: np.ndarray = None   # choke-rate multiplier (a calibrated VFM: exp(f + b0)); truth wells use 1
    k_exp: np.ndarray = None    # extra opening exponent (the VFM latent b1); truth wells use 0

    def __post_init__(self):
        if self.k_mult is None:
            self.k_mult = np.ones_like(np.asarray(self.p_res, dtype=float))
        if self.k_exp is None:
            self.k_exp = np.zeros_like(np.asarray(self.p_res, dtype=float))
        for f in fields(self):
            if f.name != 'fluid':
                setattr(self, f.name, np.asarray(getattr(self, f.name), dtype=float))

    def _vfm_terms(self):
        """(k_mult, k_exp), defaulting for objects pickled before these fields existed."""
        km, ke = getattr(self, 'k_mult', None), getattr(self, 'k_exp', None)
        shape = np.shape(self.p_res)
        return (np.ones(shape) if km is None else km), (np.zeros(shape) if ke is None else ke)

    def choke_rate(self, p1, p2, t1, opening, expand=False):
        """Mass rate through each well's choke (with the VFM multiplier and exponent when set)."""
        ex = (lambda a: np.asarray(a)[..., None]) if expand else (lambda a: a)
        fl = self.fluid.expand() if expand else self.fluid
        km, ke = self._vfm_terms()
        m = choke.mass_rate(p1, p2, t1, fl, opening, ex(self.d_choke), ex(self.cd), ex(self.trim_gamma))
        return m * ex(km) * np.maximum(np.asarray(opening, dtype=float), 1e-3) ** ex(ke)

    @property
    def p_b(self):
        return pvt.bubble_point(self.t_res, self.fluid)

    def take(self, idx):
        km, ke = self._vfm_terms()
        kw = {f.name: np.asarray(getattr(self, f.name))[idx] for f in fields(self)
              if f.name not in ('fluid', 'k_mult', 'k_exp')}
        return Wells(fluid=self.fluid.take(idx), k_mult=np.asarray(km)[idx], k_exp=np.asarray(ke)[idx], **kw)


def ipr_rate(p_wf, p_res, p_b, pi):
    """Stock-tank liquid rate (Sm3/s) at bottomhole flowing pressure p_wf."""
    p_wf = np.minimum(p_wf, p_res)
    pb = np.minimum(p_b, p_res)
    q_b = pi * (p_res - pb)
    r = np.clip(p_wf / np.maximum(pb, 1.0), 0.0, 1.0)
    below = q_b + pi * pb / 1.8 * (1.0 - 0.2 * r - 0.8 * r ** 2)
    return np.where(p_wf >= pb, pi * (p_res - p_wf), below)


def aof(p_res, p_b, pi):
    """Absolute open flow: the rate at zero bottomhole pressure."""
    pb = np.minimum(p_b, p_res)
    return pi * (p_res - pb) + pi * pb / 1.8


def ipr_pwf(q, p_res, p_b, pi):
    """Inverse of ipr_rate: bottomhole pressure that delivers liquid rate q."""
    pb = np.minimum(p_b, p_res)
    q_b = pi * (p_res - pb)
    above = p_res - q / pi
    s = np.clip(1.0 - 1.8 * (q - q_b) / np.maximum(pi * pb, 1e-30), 0.0, 1.0)
    r = (-0.2 + np.sqrt(0.04 + 3.2 * s)) / 1.6
    return np.where(q <= q_b, above, np.maximum(r, 0.0) * pb)


def wellhead_temperature(q, t_res, t_sea, q_t):
    """Flowing wellhead temperature: hotter with rate, approaching reservoir temperature at high rates."""
    return t_sea + (t_res - t_sea) * q / (q + q_t)


def friction_factor(re, rel_rough):
    """Darcy friction factor: laminar below Re 2300, Swamee-Jain above."""
    re = np.maximum(re, 1.0)
    turb = 0.25 / np.log10(rel_rough / 3.7 + 5.74 / re ** 0.9) ** 2
    return np.where(re < 2300.0, 64.0 / re, turb)


def gas_holdup(v_sg, v_sl, rho_l, rho_g, d, c0=1.2):
    """Drift-flux gas holdup in vertical upflow (Taylor-bubble drift velocity)."""
    v_m = v_sg + v_sl
    v_d = 0.35 * np.sqrt(u.G * d * np.maximum(rho_l - rho_g, 0.0) / rho_l)
    alpha = np.where(v_sg > 0, v_sg / np.maximum(c0 * v_m + v_d, 1e-12), 0.0)
    return np.clip(alpha, 0.0, 0.999)


def gradient(p, t, q_l, fluid, d, rough, vertical=1.0):
    """Pressure gradient (Pa/m) of upward flow: hydrostatic (times the vertical fraction) plus friction."""
    st = pvt.state(p, t, fluid)
    area = np.pi * d ** 2 / 4.0
    v_sl = q_l * st.v_l / area
    v_sg = q_l * st.v_g / area
    alpha = gas_holdup(v_sg, v_sl, st.rho_l, st.rho_g, d)
    rho_mix = alpha * st.rho_g + (1.0 - alpha) * st.rho_l
    v_m = v_sl + v_sg
    lam = np.where(v_m > 0, v_sl / np.maximum(v_m, 1e-12), 1.0)
    rho_ns = lam * st.rho_l + (1.0 - lam) * st.rho_g
    mu_ns = lam * st.mu_l + (1.0 - lam) * st.mu_g
    f = friction_factor(rho_ns * v_m * d / mu_ns, rough / d)
    return rho_mix * u.G * vertical + f * rho_ns * v_m ** 2 / (2.0 * d)


def traverse_up(p_bottom, q_l, fluid, t_bottom, t_top, depth, d_tub, n_seg=24, rough=4.5e-5):
    """Wellhead pressure for liquid rate q_l flowing up a vertical tubing from bottomhole pressure p_bottom."""
    p = np.array(np.broadcast_to(p_bottom, np.broadcast(p_bottom, q_l).shape), dtype=float)
    dz = depth / n_seg
    for i in range(n_seg):
        t = t_bottom + (t_top - t_bottom) * (i + 0.5) / n_seg
        p = np.maximum(p - gradient(p, t, q_l, fluid, d_tub, rough) * dz, u.P_SC)
    return p


def vlp_table(w, n_q=48):
    """Per-well table on a uniform liquid-rate grid from 0 to 99.9% of the open-flow potential.
    Returns arrays of shape (..., n_q): q (Sm3/s), p_wh (Pa), t_wh (K)."""
    q_max = 0.999 * aof(w.p_res, w.p_b, w.pi)
    q = q_max[..., None] * np.linspace(0.0, 1.0, n_q)
    ex = lambda a: np.asarray(a)[..., None]
    p_wf = ipr_pwf(q, ex(w.p_res), ex(w.p_b), ex(w.pi))
    t_wh = wellhead_temperature(q, ex(w.t_res), ex(w.t_sea), ex(w.q_t))
    p_wh = traverse_up(p_wf, q, w.fluid.expand(), ex(w.t_res), t_wh, ex(w.depth), ex(w.d_tub))
    return {'q': q, 'p_wh': p_wh, 't_wh': t_wh}


def interp(table, q):
    """Wellhead pressure and temperature at liquid rate q, interpolated linearly on each well's uniform grid."""
    grid = table['q']
    n = grid.shape[-1]
    q_max = grid[..., -1]
    s = np.clip(np.asarray(q, dtype=float) / np.maximum(q_max, 1e-30), 0.0, 1.0) * (n - 1)
    i0 = np.clip(np.floor(s).astype(int), 0, n - 2)
    w = (s - i0)[..., None]
    idx = i0[..., None]

    def at(tab):
        a = np.take_along_axis(tab, idx, axis=-1)
        b = np.take_along_axis(tab, idx + 1, axis=-1)
        return (a * (1.0 - w) + b * w)[..., 0]

    return at(table['p_wh']), at(table['t_wh'])


def solve_rate(w, table, p2, opening, iters=30):
    """Operating point of each well for downstream (manifold) pressure p2 and choke opening.
    F(q) = choke rate at the wellhead pressure the well delivers at q, minus q. A gassy well's wellhead pressure can
    rise before it falls, giving an unstable low-rate root as well; the stable operating point is the highest-rate
    sign change of F from + to -, which is the one taken here.
    Returns q_l (Sm3/s stock-tank liquid), p_wh, t_wh, m (kg/s); a well that cannot flow returns zero."""
    q = table['q']
    n = q.shape[-1]
    ex = lambda a: np.asarray(a)[..., None]
    m_grid = w.choke_rate(table['p_wh'], ex(p2), table['t_wh'], ex(opening), expand=True)
    f_grid = m_grid / ex(pvt.mass_per_liquid_sc(w.fluid)) - q
    nonneg = f_grid >= 0
    nonneg[..., 0] = f_grid[..., 0] > 0           # F(0) = 0 means the well cannot open against p2
    any_pos = nonneg.any(axis=-1)
    i_last = n - 1 - np.argmax(nonneg[..., ::-1], axis=-1)
    at_max = any_pos & (i_last == n - 1)
    i0 = np.clip(i_last, 0, n - 2)
    lo = np.take_along_axis(q, i0[..., None], axis=-1)[..., 0]
    hi = np.take_along_axis(q, (i0 + 1)[..., None], axis=-1)[..., 0]
    mls = pvt.mass_per_liquid_sc(w.fluid)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        p_wh, t_wh = interp(table, mid)
        f_mid = w.choke_rate(p_wh, p2, t_wh, opening) / mls - mid
        lo = np.where(f_mid >= 0, mid, lo)
        hi = np.where(f_mid >= 0, hi, mid)
    q_star = np.where(at_max, q[..., -1], 0.5 * (lo + hi))
    q_star = np.where(any_pos, q_star, 0.0)
    p_wh, t_wh = interp(table, q_star)
    return {'q_l': q_star, 'p_wh': p_wh, 't_wh': t_wh, 'm': q_star * mls}
