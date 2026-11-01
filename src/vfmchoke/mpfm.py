"""Gamma-ray multiphase flow meter (MPFM) emulator: dual-energy densitometer plus venturi.

Densitometer (Beer-Lambert): ln(I0/I)_E = d * sum_i alpha_i * mu_i(E) * rho_i over oil, water and gas, at a low
energy (32 keV, photoelectric-sensitive) and a high energy (356 keV, Compton-dominated). Two attenuations plus
the closure sum(alpha) = 1 give the three phase holdups by a 3x3 linear solve.

Venturi: total mass rate from the differential pressure and the gamma mixture density, so a density error moves
the mass rate by sqrt(rho_meter / rho_true).

Holdups become flow fractions through the meter's drift-flux slip model. Errors emulated: Poisson counting noise,
a stale water salinity in the meter configuration (shifts the low-energy water attenuation and the water cut),
a slip-model mismatch, a venturi discharge-coefficient bias, and multiplicative flow-regime noise.

Mass attenuation coefficients are approximate values (cm2/g) interpolated from NIST XCOM tables; the emulator needs
the contrast between phases and energies, not metrology-grade constants.
"""
from dataclasses import dataclass

import numpy as np

from . import pvt

MU_MASS = {            # cm2/g at (32 keV, 356 keV)
    'oil': (0.245, 0.114),
    'water': (0.348, 0.111),
    'gas': (0.262, 0.125),
    'nacl': (2.00, 0.100),
}


def mu_linear(rho_o, rho_w, rho_g, salinity):
    """Linear attenuation coefficients (1/m), shape (..., 2 energies, 3 phases: oil, water, gas)."""
    sal = np.asarray(salinity, dtype=float)
    out = []
    for e in (0, 1):
        mw = (1.0 - sal) * MU_MASS['water'][e] + sal * MU_MASS['nacl'][e]
        out.append(np.stack(np.broadcast_arrays(MU_MASS['oil'][e] * 0.1 * rho_o, mw * 0.1 * rho_w,
                                                MU_MASS['gas'][e] * 0.1 * rho_g), axis=-1))
    return np.stack(out, axis=-2)


def attenuation(alpha, mu, d):
    """ln(I0/I) at both energies for holdups alpha (..., 3) through a path length d (m)."""
    return np.asarray(d, dtype=float)[..., None] * np.einsum('...ep,...p->...e', mu, alpha)


def invert(a, mu, d):
    """Holdups (..., 3) from attenuations (..., 2): two energies plus closure."""
    m = np.concatenate([mu, np.ones(mu.shape[:-2] + (1, 3))], axis=-2)
    rhs = np.concatenate([a / np.asarray(d, dtype=float)[..., None], np.ones(a.shape[:-1] + (1,))], axis=-1)
    return np.linalg.solve(m, rhs[..., None])[..., 0]


@dataclass
class Config:
    """What the meter believes, and its counting setup. Arrays broadcast against the wells."""
    salinity: np.ndarray            # configured water salinity (NaCl mass fraction)
    c0: float = 1.2                 # slip model distribution parameter
    vd_scale: float = 1.0           # slip model drift-velocity multiplier
    cd_ratio: float = 1.0           # venturi discharge coefficient, meter / true (scalar or array)
    counts: tuple = (2e5, 1e5)      # empty-pipe count rates (1/s) at (32 keV, 356 keV)
    integration_s: float = 86400.0  # averaging time of one reading
    noise_l: float = 0.0            # flow-regime noise on liquid (relative std)
    noise_g: float = 0.0            # flow-regime noise on gas (relative std)


def _flow_from_holdup(alpha_g, m, area, rho_l, rho_g, d, c0, vd_scale):
    """Superficial velocities from gas holdup and total mass flux, inverting the drift-flux slip model."""
    v_d = vd_scale * 0.35 * np.sqrt(9.80665 * d * np.maximum(rho_l - rho_g, 0.0) / rho_l)
    v_m = (m / area - alpha_g * v_d * (rho_g - rho_l)) / (rho_g * alpha_g * c0 + rho_l * (1.0 - alpha_g * c0))
    v_sg = alpha_g * (c0 * v_m + v_d)
    return v_m - v_sg, v_sg


def read(p, t, fluid, q_l, d, cfg, rng=None, true_c0=1.2, true_vd_scale=1.0):
    """One MPFM reading of a well producing q_l (Sm3/s stock-tank liquid) at line conditions (p, t).
    Returns reported standard oil, gas and water rates (Sm3/s) and the meter's holdups."""
    st = pvt.state(p, t, fluid)
    area = np.pi * np.asarray(d, dtype=float) ** 2 / 4.0
    v_sl = q_l * st.v_l / area
    v_sg = q_l * st.v_g / area
    v_d = true_vd_scale * 0.35 * np.sqrt(9.80665 * d * np.maximum(st.rho_l - st.rho_g, 0.0) / st.rho_l)
    alpha_g = np.where(v_sg > 0, v_sg / np.maximum(true_c0 * (v_sl + v_sg) + v_d, 1e-12), 0.0)
    wl = st.v_w / st.v_l
    alpha = np.stack([(1 - alpha_g) * (1 - wl), (1 - alpha_g) * wl, alpha_g], axis=-1)
    mu_true = mu_linear(st.rho_o, st.rho_w, st.rho_g, fluid.salinity)
    a = attenuation(alpha, mu_true, d)
    if rng is not None:
        n0 = np.asarray(cfg.counts) * cfg.integration_s
        n = rng.poisson(n0 * np.exp(-a))
        a = np.log(n0 / np.maximum(n, 1))
    rho_w_meter = pvt.rho_water_sc(cfg.salinity) * np.ones_like(st.rho_w)
    mu_meter = mu_linear(st.rho_o, rho_w_meter, st.rho_g, cfg.salinity)
    al = invert(a, mu_meter, d)
    rho_true = (alpha * np.stack([st.rho_o, st.rho_w, st.rho_g], -1)).sum(-1)
    rho_meter = (al * np.stack([st.rho_o, rho_w_meter, st.rho_g], -1)).sum(-1)
    m_true = q_l * pvt.mass_per_liquid_sc(fluid)
    m_meter = m_true * cfg.cd_ratio * np.sqrt(np.maximum(rho_meter, 1e-6) / rho_true)
    liq_al = np.maximum(al[..., 0] + al[..., 1], 1e-12)
    wl_m = np.clip(al[..., 1] / liq_al, 0.0, 1.0)
    rho_l_m = (1 - wl_m) * st.rho_o + wl_m * rho_w_meter
    v_sl_m, v_sg_m = _flow_from_holdup(np.clip(al[..., 2], 0.0, 0.999), m_meter, area, rho_l_m, st.rho_g, d,
                                       cfg.c0, cfg.vd_scale)
    q_o_line = v_sl_m * area * (1 - wl_m)
    q_w_line = v_sl_m * area * wl_m
    q_g_line = np.maximum(v_sg_m, 0.0) * area
    q_o = q_o_line / st.bo
    q_w = q_w_line
    q_g = q_g_line / st.bg + st.rs * q_o
    if rng is not None and (cfg.noise_l > 0 or cfg.noise_g > 0):
        el = np.exp(rng.normal(0.0, cfg.noise_l, np.shape(q_o)))
        q_o, q_w = q_o * el, q_w * el
        q_g = q_g * np.exp(rng.normal(0.0, cfg.noise_g, np.shape(q_g)))
    return {'q_o': q_o, 'q_g': q_g, 'q_w': q_w, 'alpha': al, 'alpha_true': alpha}
