"""M1 invariants: PVT bounds and bookkeeping, choke limits and monotonicity, IPR inverse, network conservation,
leak signatures, and exact MPFM inversion."""
import numpy as np

from vfmchoke import choke, mpfm, network, pvt, units as u, well


def envelope(n=4000, seed=1):
    rng = np.random.default_rng(seed)
    f = pvt.Fluid(api=rng.uniform(20, 50, n), gas_sg=rng.uniform(0.6, 1.0, n), gor=rng.uniform(20, 400, n),
                  wc=rng.uniform(0, 0.95, n), salinity=rng.uniform(0, 0.15, n))
    return rng.uniform(5e5, 400e5, n), u.c_to_k(rng.uniform(20, 150, n)), f


# ---------- PVT ----------
def test_pvt_bounds():
    p, t, f = envelope()
    st = pvt.state(p, t, f)
    assert np.all(st.rs >= 0) and np.all(st.rs <= f.gor + 1e-9)
    assert np.all(st.bo >= 1.0) and np.all(st.bo < 3.0)
    assert np.all((st.x >= 0) & (st.x <= 1))
    assert np.all(st.mu_l > 0) and np.all(st.mu_g > 0)
    assert np.all((st.rho_l > 500) & (st.rho_l < 1300))
    z = pvt.z_factor(p, t, f.gas_sg)
    assert np.all((z >= 0.3) & (z <= 1.2))


def test_mass_bookkeeping_exact():
    """Liquid plus free gas at any (p, t) carries exactly the standard mass per Sm3 of liquid."""
    p, t, f = envelope()
    st = pvt.state(p, t, f)
    np.testing.assert_allclose(st.m_l + st.m_g, pvt.mass_per_liquid_sc(f), rtol=1e-12)


def test_gas_ideal_limit():
    rho = pvt.gas_density(u.P_SC, u.T_SC, 0.7)
    assert abs(rho / pvt.rho_gas_sc(0.7) - 1) < 0.01


def test_mix_is_mass_exact():
    rng = np.random.default_rng(3)
    f = pvt.Fluid(api=rng.uniform(25, 45, 5), gas_sg=rng.uniform(0.6, 0.9, 5), gor=rng.uniform(50, 300, 5),
                  wc=rng.uniform(0, 0.7, 5), salinity=rng.uniform(0, 0.1, 5))
    q_o, q_g, q_w = choke.std_rates(rng.uniform(5, 20, 5), f)
    mf = pvt.mix(q_o, q_g, q_w, f)
    total = network.std_mass(q_o.sum(), q_g.sum(), q_w.sum(), mf)
    assert abs(total / network.std_mass(q_o, q_g, q_w, f).sum() - 1) < 1e-12


# ---------- choke ----------
def test_liquid_only_is_bernoulli():
    f = pvt.Fluid(api=35.0, gas_sg=0.75, gor=0.0, wc=0.3, salinity=0.04)
    p1, p2, t1, d = 50e5, 20e5, 330.0, 0.02
    m = choke.mass_rate(p1, p2, t1, f, 1.0, d, cd=0.8)
    rho_l = pvt.state(p1, t1, f).rho_l
    expected = 0.8 * np.pi / 4 * d ** 2 * np.sqrt(2 * rho_l * (p1 - p2))
    assert abs(m / expected - 1) < 1e-9


def test_gas_only_critical_ratio():
    k = 1.28
    yc = choke.critical_ratio(1.0, 1e-3, 0.02, k)
    assert abs(yc - (2 / (k + 1)) ** (k / (k - 1))) < 1e-6


def test_critical_ratio_matches_flux_maximiser():
    """The derivative-root bisection finds the same ratio as a direct search for the flux maximum."""
    rng = np.random.default_rng(4)
    x = np.concatenate([rng.uniform(1e-4, 1.0, 3000), [0.0, 1e-6, 1.0]])
    v_l = rng.uniform(1 / 1100, 1 / 600, x.size)
    v_g1 = rng.uniform(1 / 200, 1 / 5, x.size)
    fast = choke.critical_ratio(x, v_l, v_g1)
    ref = choke.critical_ratio_golden(x, v_l, v_g1, iters=80)
    assert np.max(np.abs(fast - ref)) < 1e-6
    f_fast = choke.flux_shape(fast, x, v_l, v_g1)
    f_ref = choke.flux_shape(ref, x, v_l, v_g1)
    assert np.all(f_fast >= f_ref * (1 - 1e-10))


def test_continuity_at_critical(fluid):
    p1, t1 = 60e5, 340.0
    _, info = choke.mass_rate(p1, 0.1 * p1, t1, fluid, 0.7, 0.025, info=True)
    yc = float(info['yc'])
    below = choke.mass_rate(p1, yc * (1 - 1e-7) * p1, t1, fluid, 0.7, 0.025)
    above = choke.mass_rate(p1, yc * (1 + 1e-7) * p1, t1, fluid, 0.7, 0.025)
    assert abs(below / above - 1) < 1e-3


def test_choke_monotone(fluid):
    t1 = 340.0
    opening = np.linspace(0, 1, 101)
    m = choke.mass_rate(60e5, 20e5, t1, fluid, opening, 0.025, trim_gamma=1.3)
    assert np.all(np.diff(m) >= -1e-12 * m.max())
    p2 = np.linspace(1e5, 59.9e5, 200)
    m = choke.mass_rate(60e5, p2, t1, fluid, 0.6, 0.025)
    assert np.all(np.diff(m) <= 1e-12 * m.max())
    p1 = np.linspace(21e5, 200e5, 200)
    m = choke.mass_rate(p1, 20e5, t1, fluid, 0.6, 0.025)
    assert np.all(np.diff(m) >= -1e-12 * m.max())


def test_gilbert_unit_conversion():
    """A = 10.00 with GLR in scf/stb must match Gilbert's original 435 with GLR in Mscf/bbl (within rounding)."""
    p, s, glr = 800.0, 32.0, 900.0
    original = p * s ** 1.89 / (435.0 * (glr / 1000.0) ** 0.546)
    assert abs(choke.gilbert_rate(p, s, glr, 'gilbert') / original - 1) < 0.005


# ---------- well ----------
def test_ipr_inverse(wells):
    w = wells.take((0, 0))
    pb = min(float(w.p_b), float(w.p_res))
    p = np.linspace(0.05 * pb, float(w.p_res), 200)
    q = well.ipr_rate(p, w.p_res, w.p_b, w.pi)
    assert abs(float(well.ipr_rate(w.p_res, w.p_res, w.p_b, w.pi))) < 1e-12
    np.testing.assert_allclose(well.ipr_pwf(q, w.p_res, w.p_b, w.pi), p, rtol=1e-6, atol=1.0)


def test_operating_point_consistent(wells):
    table = well.vlp_table(wells)
    p2 = np.full(wells.pi.shape, 20e5)
    opening = np.full(wells.pi.shape, 0.6)
    op = well.solve_rate(wells, table, p2, opening)
    assert np.all(op['q_l'] > 0)
    m_choke = choke.mass_rate(op['p_wh'], p2, op['t_wh'], wells.fluid, opening, wells.d_choke, wells.cd,
                              wells.trim_gamma)
    q_choke = m_choke / pvt.mass_per_liquid_sc(wells.fluid)
    np.testing.assert_allclose(q_choke, op['q_l'], rtol=1e-3)


def test_rate_rises_with_opening(wells):
    table = well.vlp_table(wells)
    p2 = np.full(wells.pi.shape, 20e5)
    q = [well.solve_rate(wells, table, p2, np.full(wells.pi.shape, o))['q_l'] for o in (0.2, 0.4, 0.6, 0.8, 1.0)]
    assert np.all(np.diff(np.stack(q), axis=0) >= -1e-9)


# ---------- network ----------
def _conservation(out):
    lhs = out['m'].sum(1)
    rhs = out['leak_m'] + out['sep_m']
    return np.abs(lhs - rhs) / lhs


def test_network_no_leak(wells, line):
    table = well.vlp_table(wells)
    out = network.solve(wells, table, np.full(wells.pi.shape, 0.6), line)
    assert np.all(_conservation(out) < 1e-9)
    assert np.all(np.abs(out['p_man'] - out['p_man_implied']) < 5e3)
    assert np.all(out['leak_m'] == 0)
    assert np.all(out['p_man'] > line.p_sep)


def test_jumper_leak_signature(wells, line):
    table = well.vlp_table(wells)
    leak = network.Leak(kind=np.array([network.JUMPER, network.NONE]), well=np.array([1, 0]),
                        pos=np.zeros(2), d_hole=np.array([0.008, 0.0]))
    out = network.solve(wells, table, np.full(wells.pi.shape, 0.6), line, leak)
    assert np.all(_conservation(out) < 1e-9)
    assert out['leak_m'][0] > 0 and out['leak_m'][1] == 0
    f = wells.fluid.take((0, 1))
    assert abs(out['leak_q_g'][0] / out['leak_q_o'][0] / f.gor - 1) < 1e-9
    wc_leak = out['leak_q_w'][0] / (out['leak_q_o'][0] + out['leak_q_w'][0])
    assert abs(wc_leak - f.wc) < 1e-9


def test_flowline_leak_signature(wells, line):
    table = well.vlp_table(wells)
    opening = np.full(wells.pi.shape, 0.6)
    base = network.solve(wells, table, opening, line)
    leak = network.Leak(kind=np.array([network.NONE, network.FLOWLINE]), well=np.zeros(2, int),
                        pos=np.array([0.0, 0.5]), d_hole=np.array([0.0, 0.01]))
    out = network.solve(wells, table, opening, line, leak)
    assert np.all(_conservation(out) < 1e-9)
    assert out['leak_m'][1] > 0
    gor_leak = out['leak_q_g'][1] / out['leak_q_o'][1]
    assert abs(gor_leak / out['sep_fluid'].gor[1] - 1) < 1e-9
    assert out['sep_q_o'][1] < base['sep_q_o'][1]
    assert out['p_man'][1] < base['p_man'][1]        # less flow downstream of the hole: lower manifold pressure


def test_riser_leak_conserves(wells, line):
    table = well.vlp_table(wells)
    leak = network.Leak(kind=np.array([network.RISER, network.MANIFOLD]), well=np.zeros(2, int),
                        pos=np.array([0.3, 0.0]), d_hole=np.array([0.01, 0.006]))
    out = network.solve(wells, table, np.full(wells.pi.shape, 0.6), line, leak)
    assert np.all(_conservation(out) < 1e-9)
    assert np.all(out['leak_m'] > 0)


# ---------- MPFM ----------
def test_mpfm_exact_inversion():
    rng = np.random.default_rng(7)
    n = 500
    alpha = rng.dirichlet([2, 2, 2], n)
    mu = mpfm.mu_linear(rng.uniform(700, 900, n), rng.uniform(1000, 1100, n), rng.uniform(30, 200, n),
                        rng.uniform(0, 0.1, n))
    d = np.full(n, 0.08)
    np.testing.assert_allclose(mpfm.invert(mpfm.attenuation(alpha, mu, d), mu, d), alpha, atol=1e-9)


def test_mpfm_read_noise_free_is_exact(fluid):
    q_l = 0.01
    cfg = mpfm.Config(salinity=fluid.salinity)
    r = mpfm.read(40e5, 330.0, fluid, q_l, 0.08, cfg)
    q_o, q_g, q_w = choke.std_rates(q_l * pvt.mass_per_liquid_sc(fluid), fluid)
    for got, want in ((r['q_o'], q_o), (r['q_g'], q_g), (r['q_w'], q_w)):
        assert abs(got / want - 1) < 1e-9


def test_mpfm_stale_salinity_overreads_water(fluid):
    q_l = 0.01
    fresh_cfg = mpfm.Config(salinity=fluid.salinity - 0.03)
    r = mpfm.read(40e5, 330.0, fluid, q_l, 0.08, fresh_cfg)
    wc_meter = r['q_w'] / (r['q_o'] + r['q_w'])
    assert wc_meter > fluid.wc
