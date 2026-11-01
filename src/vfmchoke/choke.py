"""Wellhead choke models.

Mechanistic (after Sachdeva et al. 1986): homogeneous no-slip two-phase flow through a restriction, liquid
incompressible, free gas expanding isentropically from the upstream pressure P1 to y*P1. The mass flux is

    G(y) = sqrt(2 P1 rho_m2^2 [(1-x)(1-y) v_L + x k/(k-1) (v_G1 - y v_G2)]),   v_G2 = v_G1 y^(-1/k),

with x the free-gas mass fraction at upstream conditions. Flow is critical (choked) below the ratio y_c where G
peaks: there the rate no longer depends on the downstream pressure. Sachdeva's closed-form y_c uses a polytropic
exponent; here y_c is the exact maximiser of G, which keeps the rate continuous at the critical boundary.

Empirical (Gilbert family): P_wh = A q GLR^B / S^C in field units (psig, stb/d, scf/stb, 64ths of an inch),
valid for critical flow only.
"""
import numpy as np

from . import pvt

Y_MIN = 1e-3
CD_NOMINAL = 0.80
GOLDEN_ITERS = 48              # interval shrinks to ~1e-10 of [Y_MIN, 1]

GILBERT_FAMILY = {             # name: (A, B, C) in P_wh[psig] = A * q * GLR**B / S**C
    'gilbert': (10.00, 0.546, 1.89),
    'ros': (17.40, 0.500, 2.00),
    'baxendell': (9.56, 0.546, 1.93),
    'achong': (3.82, 0.650, 1.88),
    'pilehvari': (46.67, 0.313, 2.11),
}


def trim(opening, gamma=1.0):
    """Effective-area fraction of a choke trim at an opening in [0, 1]: opening**gamma.
    gamma = 1 is a linear trim; gamma > 1 approximates equal-percentage trims."""
    return np.clip(np.asarray(opening, dtype=float), 0.0, 1.0) ** gamma


def flux_shape(y, x, v_l, v_g1, k=pvt.K_GAS):
    """G**2 / (2 P1): depends only on the pressure ratio y and the upstream specific volumes and quality."""
    y = np.clip(np.asarray(y, dtype=float), 1e-12, 1.0)
    v_g2 = v_g1 * y ** (-1.0 / k)
    rho_m2 = 1.0 / (x * v_g2 + (1.0 - x) * v_l)
    energy = (1.0 - x) * (1.0 - y) * v_l + x * k / (k - 1.0) * (v_g1 - y * v_g2)
    return rho_m2 ** 2 * np.maximum(energy, 0.0)


def critical_ratio_golden(x, v_l, v_g1, k=pvt.K_GAS, iters=GOLDEN_ITERS):
    """Reference: the flux maximiser by golden-section search on [Y_MIN, 1] (kept for tests)."""
    shape = np.broadcast(np.asarray(x), np.asarray(v_l), np.asarray(v_g1)).shape
    lo = np.full(shape, Y_MIN)
    hi = np.ones(shape)
    g = (np.sqrt(5.0) - 1.0) / 2.0
    for _ in range(iters):
        c = hi - g * (hi - lo)
        d = lo + g * (hi - lo)
        left = flux_shape(c, x, v_l, v_g1, k) >= flux_shape(d, x, v_l, v_g1, k)
        hi = np.where(left, d, hi)
        lo = np.where(left, lo, c)
    return 0.5 * (lo + hi)


def critical_ratio(x, v_l, v_g1, k=pvt.K_GAS, iters=32):          # 2^-32 of [Y_MIN, 1]: ~2e-10
    """Pressure ratio P2/P1 at which the mass flux peaks; below it the choke is critical.

    With D(y) = x v_G1 y^(-1/k) + (1-x) v_L (= 1/rho_m2) and E(y) the energy term of flux_shape, dE/dy = -D, so
    d(E/D^2)/dy = -g(y)/D^3 with g = D^2 - (2/k) E x v_G1 y^(-1/k-1). The flux rises while g < 0 and falls once g > 0;
    the maximiser is the root of g, found by bisection (one evaluation per step). If g >= 0 already at Y_MIN (no
    free gas) the flux only falls, and the ratio is Y_MIN."""
    x, v_l, v_g1 = np.broadcast_arrays(np.asarray(x, float), np.asarray(v_l, float), np.asarray(v_g1, float))

    def g(y):
        a = y ** (-1.0 / k)
        d = x * v_g1 * a + (1.0 - x) * v_l
        e = (1.0 - x) * (1.0 - y) * v_l + x * k / (k - 1.0) * v_g1 * (1.0 - y * a)
        return d * d - (2.0 / k) * e * x * v_g1 * a / y

    lo = np.full(x.shape, Y_MIN)
    hi = np.ones(x.shape)
    none = g(lo) >= 0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        rising = g(mid) < 0
        lo = np.where(rising, mid, lo)
        hi = np.where(rising, hi, mid)
    return np.where(none, Y_MIN, 0.5 * (lo + hi))


def mass_rate(p1, p2, t1, fluid, opening, d_choke, cd=CD_NOMINAL, trim_gamma=1.0, k=pvt.K_GAS, info=False, st=None):
    """Total mass rate (kg/s) through a choke of bean diameter d_choke (m) at an opening in [0, 1].
    st: the upstream pvt.State when the caller already has it."""
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    st = pvt.state(p1, t1, fluid) if st is None else st
    v_l = 1.0 / st.rho_l
    v_g1 = 1.0 / st.rho_g
    yc = critical_ratio(st.x, v_l, v_g1, k)
    y = np.clip(p2 / np.maximum(p1, 1.0), 0.0, 1.0)
    y_eff = np.maximum(y, yc)
    area = np.pi / 4.0 * np.asarray(d_choke, dtype=float) ** 2 * trim(opening, trim_gamma)
    m = cd * area * np.sqrt(2.0 * p1 * flux_shape(y_eff, st.x, v_l, v_g1, k))
    m = np.where(p2 >= p1, 0.0, m)
    if info:
        return m, {'y': y, 'yc': yc, 'critical': y <= yc, 'x': st.x}
    return m


def std_rates(m, fluid):
    """Stock-tank oil, gas and water rates (Sm3/s) carried by a total mass rate m (kg/s) of a stream."""
    q_l = np.asarray(m, dtype=float) / pvt.mass_per_liquid_sc(fluid)
    q_o = (1.0 - fluid.wc) * q_l
    return q_o, fluid.gor * q_o, fluid.wc * q_l


def gilbert_rate(p_wh_psig, s64, glr_scf_stb, name='gilbert'):
    """Liquid rate (stb/d) from a Gilbert-family correlation; field units, critical flow."""
    a, b, c = GILBERT_FAMILY[name]
    return (np.asarray(p_wh_psig, dtype=float) * np.asarray(s64, dtype=float) ** c
            / (a * np.maximum(np.asarray(glr_scf_stb, dtype=float), 1e-9) ** b))
