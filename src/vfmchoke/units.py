"""Unit conversions. Everything inside the package is SI (Pa, K, kg, m, s); field units appear only at the edges."""
import numpy as np

G = 9.80665                      # m/s2
PSI = 6894.757293168             # Pa per psi
BAR = 1.0e5
P_SC = 101325.0                  # Pa, standard pressure (14.696 psia)
T_SC = 288.705556                # K, standard temperature (60 F)
BBL = 0.158987294928             # m3 per barrel
SCF = 0.028316846592             # m3 per cubic foot
DAY = 86400.0
INCH = 0.0254
R_GAS = 8314.462618              # J/(kmol K)
M_AIR = 28.9647                  # kg/kmol
RHO_AIR_SC = P_SC * M_AIR / (R_GAS * T_SC)     # ~1.2228 kg/m3
SCF_PER_STB = SCF / BBL          # Sm3/Sm3 in one scf/stb (~0.1781)
RHO_SEAWATER = 1025.0            # kg/m3


def f_to_k(t_f):
    return (np.asarray(t_f, dtype=float) - 32.0) * 5.0 / 9.0 + 273.15


def k_to_f(t_k):
    return (np.asarray(t_k, dtype=float) - 273.15) * 9.0 / 5.0 + 32.0


def c_to_k(t_c):
    return np.asarray(t_c, dtype=float) + 273.15


def k_to_r(t_k):
    return np.asarray(t_k, dtype=float) * 1.8


def pa_to_psia(p):
    return np.asarray(p, dtype=float) / PSI


def psia_to_pa(p):
    return np.asarray(p, dtype=float) * PSI


def pa_to_psig(p):
    return pa_to_psia(p) - P_SC / PSI


def psig_to_pa(p):
    return psia_to_pa(np.asarray(p, dtype=float) + P_SC / PSI)


def api_to_sg(api):
    return 141.5 / (np.asarray(api, dtype=float) + 131.5)


def sg_to_api(sg):
    return 141.5 / np.asarray(sg, dtype=float) - 131.5


def scf_stb_to_si(r):
    """Gas-oil or gas-liquid ratio, scf/stb -> Sm3/Sm3."""
    return np.asarray(r, dtype=float) * SCF_PER_STB


def si_to_scf_stb(r):
    return np.asarray(r, dtype=float) / SCF_PER_STB


def stb_d_to_si(q):
    """Liquid rate, stb/d -> Sm3/s."""
    return np.asarray(q, dtype=float) * BBL / DAY


def si_to_stb_d(q):
    return np.asarray(q, dtype=float) * DAY / BBL


def choke64_to_m(s64):
    """Choke bean size in 64ths of an inch -> diameter in metres."""
    return np.asarray(s64, dtype=float) / 64.0 * INCH


def m_to_choke64(d):
    return np.asarray(d, dtype=float) / INCH * 64.0
