"""Black-oil PVT from standard correlations. SI in and out; the correlations run in field units inside.

Standing (1947) solution gas, bubble point and oil formation volume factor; Beggs & Robinson (1975) oil viscosity;
Papay (1968) gas Z factor with Sutton (1985) pseudo-criticals; Lee, Gonzalez & Eakin (1966) gas viscosity;
Brill & Beggs water viscosity. Water formation volume factor is taken as 1 and brine density follows salinity.

A stream is described per standard cubic metre of stock-tank liquid (oil + water), so wells with any water cut in
[0, 1) work, including water-dominated ones.
"""
from dataclasses import dataclass, fields

import numpy as np

from . import units as u

RHO_W_SC_FRESH = 999.012          # kg/m3, fresh water at 60 F
K_GAS = 1.28                      # heat capacity ratio of natural gas


@dataclass
class Fluid:
    """Ratios and phase properties of a produced stream (scalars or broadcastable arrays)."""
    api: np.ndarray        # stock-tank oil gravity, degrees API
    gas_sg: np.ndarray     # gas specific gravity (air = 1)
    gor: np.ndarray        # producing gas-oil ratio, Sm3 gas per Sm3 stock-tank oil (free + dissolved)
    wc: np.ndarray         # water cut: water volume fraction of stock-tank liquid, in [0, 1)
    salinity: np.ndarray   # NaCl mass fraction of the produced water

    def __post_init__(self):
        for f in fields(self):
            setattr(self, f.name, np.asarray(getattr(self, f.name), dtype=float))

    def expand(self, axis=-1):
        """The same fluid with a new axis, so it broadcasts against per-well grids."""
        return Fluid(*(np.expand_dims(getattr(self, f.name), axis) for f in fields(self)))

    def take(self, idx):
        return Fluid(*(np.asarray(getattr(self, f.name))[idx] for f in fields(self)))

    def broadcast_to(self, shape):
        return Fluid(*(np.broadcast_to(getattr(self, f.name), shape).copy() for f in fields(self)))


def rho_oil_sc(api):
    return RHO_W_SC_FRESH * u.api_to_sg(api)


def rho_gas_sc(gas_sg):
    return np.asarray(gas_sg, dtype=float) * u.RHO_AIR_SC


def rho_water_sc(salinity):
    """Brine density at standard conditions from the NaCl mass fraction (linear fit, within ~1% up to 20 wt%)."""
    return RHO_W_SC_FRESH * (1.0 + 0.695 * np.asarray(salinity, dtype=float))


def mass_per_liquid_sc(f):
    """Total mass (kg) carried with one Sm3 of stock-tank liquid: oil, all its gas, and water."""
    return (1.0 - f.wc) * (rho_oil_sc(f.api) + f.gor * rho_gas_sc(f.gas_sg)) + f.wc * rho_water_sc(f.salinity)


def bubble_point(t, f):
    """Standing (1947) bubble-point pressure (Pa) for a solution GOR equal to the producing GOR."""
    rsb = np.maximum(u.si_to_scf_stb(f.gor), 1e-6)
    t_f = u.k_to_f(t)
    pb = 18.2 * ((rsb / f.gas_sg) ** 0.83 * 10.0 ** (0.00091 * t_f - 0.0125 * f.api) - 1.4)
    return u.psia_to_pa(np.maximum(pb, u.P_SC / u.PSI))


def solution_gor(p, t, f):
    """Standing (1947) solution GOR (Sm3/Sm3), capped at the producing GOR."""
    p_psia = np.maximum(u.pa_to_psia(p), u.P_SC / u.PSI)
    t_f = u.k_to_f(t)
    rs = f.gas_sg * ((p_psia / 18.2 + 1.4) * 10.0 ** (0.0125 * f.api - 0.00091 * t_f)) ** (1.0 / 0.83)
    return np.minimum(u.scf_stb_to_si(rs), f.gor)


def oil_fvf(rs, t, f):
    """Standing (1947) oil formation volume factor (rm3/Sm3)."""
    rs_scf = u.si_to_scf_stb(rs)
    t_f = u.k_to_f(t)
    return 0.9759 + 0.00012 * (rs_scf * np.sqrt(f.gas_sg / u.api_to_sg(f.api)) + 1.25 * t_f) ** 1.2


def z_factor(p, t, gas_sg):
    """Papay (1968) Z factor with Sutton (1985) pseudo-critical properties."""
    gas_sg = np.asarray(gas_sg, dtype=float)
    tpc = 169.2 + 349.5 * gas_sg - 74.0 * gas_sg ** 2          # R
    ppc = 756.8 - 131.0 * gas_sg - 3.6 * gas_sg ** 2           # psia
    tpr = u.k_to_r(t) / tpc
    ppr = u.pa_to_psia(p) / ppc
    z = 1.0 - 3.52 * ppr / 10.0 ** (0.9813 * tpr) + 0.274 * ppr ** 2 / 10.0 ** (0.8157 * tpr)
    return np.clip(z, 0.3, 1.2)


def gas_fvf(p, t, gas_sg):
    """Gas formation volume factor (rm3/Sm3)."""
    return u.P_SC / np.asarray(p, dtype=float) * np.asarray(t, dtype=float) / u.T_SC * z_factor(p, t, gas_sg)


def gas_density(p, t, gas_sg):
    return rho_gas_sc(gas_sg) / gas_fvf(p, t, gas_sg)


def dead_oil_viscosity(t, api):
    """Beggs & Robinson (1975) dead-oil viscosity, Pa s."""
    t_f = np.maximum(u.k_to_f(t), 60.0)
    x = t_f ** -1.163 * np.exp(6.9824 - 0.04658 * np.asarray(api, dtype=float))
    return (10.0 ** x - 1.0) * 1e-3


def live_oil_viscosity(rs, t, api):
    """Beggs & Robinson (1975) live-oil viscosity, Pa s."""
    rs_scf = u.si_to_scf_stb(rs)
    a = 10.715 * (rs_scf + 100.0) ** -0.515
    b = 5.44 * (rs_scf + 150.0) ** -0.338
    return a * (dead_oil_viscosity(t, api) * 1e3) ** b * 1e-3


def gas_viscosity(p, t, gas_sg):
    """Lee, Gonzalez & Eakin (1966) gas viscosity, Pa s."""
    m = np.asarray(gas_sg, dtype=float) * u.M_AIR
    t_r = u.k_to_r(t)
    rho = gas_density(p, t, gas_sg) / 1000.0                  # g/cm3
    k = (9.4 + 0.02 * m) * t_r ** 1.5 / (209.0 + 19.0 * m + t_r)
    x = 3.5 + 986.0 / t_r + 0.01 * m
    y = 2.4 - 0.2 * x
    return 1e-4 * k * np.exp(x * rho ** y) * 1e-3


def water_viscosity(t):
    """Brill & Beggs water viscosity, Pa s."""
    t_f = u.k_to_f(t)
    return np.exp(1.003 - 1.479e-2 * t_f + 1.982e-5 * t_f ** 2) * 1e-3


@dataclass
class State:
    """In-situ properties at (p, t), per Sm3 of stock-tank liquid where noted."""
    rs: np.ndarray       # solution GOR, Sm3/Sm3
    bo: np.ndarray       # oil FVF, rm3/Sm3
    bg: np.ndarray       # gas FVF, rm3/Sm3
    rho_o: np.ndarray    # live-oil density, kg/m3
    rho_g: np.ndarray    # gas density, kg/m3
    rho_w: np.ndarray    # brine density, kg/m3
    rho_l: np.ndarray    # oil + water liquid density, kg/m3
    mu_l: np.ndarray     # liquid viscosity (volume-weighted), Pa s
    mu_g: np.ndarray     # gas viscosity, Pa s
    v_o: np.ndarray      # in-situ oil volume per Sm3 liquid, m3
    v_w: np.ndarray      # in-situ water volume per Sm3 liquid, m3
    v_g: np.ndarray      # in-situ free-gas volume per Sm3 liquid, m3
    m_l: np.ndarray      # liquid mass per Sm3 liquid, kg
    m_g: np.ndarray      # free-gas mass per Sm3 liquid, kg
    x: np.ndarray        # free-gas mass fraction (quality)

    @property
    def v_l(self):
        return self.v_o + self.v_w


def state(p, t, f):
    """Phase split and properties of the stream at pressure p (Pa) and temperature t (K)."""
    p = np.maximum(np.asarray(p, dtype=float), u.P_SC)
    t = np.asarray(t, dtype=float)
    rs = solution_gor(p, t, f)
    bo = oil_fvf(rs, t, f)
    bg = gas_fvf(p, t, f.gas_sg)
    rho_osc, rho_gsc, rho_wsc = rho_oil_sc(f.api), rho_gas_sc(f.gas_sg), rho_water_sc(f.salinity)
    oil = 1.0 - f.wc
    free = np.maximum(f.gor - rs, 0.0)
    m_o = oil * (rho_osc + rs * rho_gsc)
    m_w = f.wc * rho_wsc
    m_g = oil * free * rho_gsc
    v_o = oil * bo
    v_w = f.wc * np.ones_like(bo)
    v_g = oil * free * bg
    rho_o = (rho_osc + rs * rho_gsc) / bo
    rho_g = rho_gsc / bg
    m_l = m_o + m_w
    v_l = v_o + v_w
    rho_l = m_l / v_l
    mu_o = live_oil_viscosity(rs, t, f.api)
    mu_w = water_viscosity(t)
    mu_l = (v_o * mu_o + v_w * mu_w) / v_l
    return State(rs=rs, bo=bo, bg=bg, rho_o=rho_o, rho_g=rho_g, rho_w=rho_wsc * np.ones_like(bo), rho_l=rho_l,
                 mu_l=mu_l, mu_g=gas_viscosity(p, t, f.gas_sg), v_o=v_o, v_w=v_w, v_g=v_g, m_l=m_l, m_g=m_g,
                 x=m_g / (m_g + m_l))


def mix(q_o, q_g, q_w, fluids, axis=-1):
    """Commingled stream of several wells: rate-weighted ratios and phase properties (mass-exact)."""
    so, sg_, sw = q_o.sum(axis), q_g.sum(axis), q_w.sum(axis)
    oil_sg = (q_o * u.api_to_sg(fluids.api)).sum(axis) / np.maximum(so, 1e-30)
    gas_sg = (q_g * fluids.gas_sg).sum(axis) / np.maximum(sg_, 1e-30)
    sal = (q_w * fluids.salinity).sum(axis) / np.maximum(sw, 1e-30)
    return Fluid(api=u.sg_to_api(np.where(so > 0, oil_sg, 0.85)),
                 gas_sg=np.where(sg_ > 0, gas_sg, 0.7),
                 gor=np.where(so > 0, sg_ / np.maximum(so, 1e-30), 0.0),
                 wc=np.clip(sw / np.maximum(so + sw, 1e-30), 0.0, 0.999),
                 salinity=np.where(sw > 0, sal, 0.0))
