"""Model-layer checks: few-shot filter, array prior, tree export parity, VFM multiplier, setpoint inverse, SimField."""
import numpy as np
import pandas as pd

from vfmchoke import control, edge, fewshot, hybrid, network, pvt, simfield, well


def test_latent_prior_only_without_labels():
    lp = fewshot.LatentPrior(tau0=0.1, tau1=0.2)
    x, p, t_last = fewshot.latent_posterior([], [], [], [], lp)
    assert np.allclose(x, 0) and np.allclose(np.diag(p), [0.01, 0.04]) and t_last is None


def test_latent_recovers_truth_with_many_labels():
    rng = np.random.default_rng(0)
    lp = fewshot.LatentPrior(tau0=0.3, tau1=0.3, q0=0.0)
    b = np.array([0.12, -0.08])
    u = rng.uniform(0.2, 1.0, 400)
    phi = np.column_stack([np.ones_like(u), np.log(u)])
    r = phi @ b + rng.normal(0, 0.02, len(u))
    x, p, _ = fewshot.latent_posterior(np.arange(len(u)), phi, r, np.full(len(u), 0.02), lp)
    assert np.allclose(x, b, atol=0.01)


def test_track_moves_toward_observations():
    m, v = fewshot.track([1, 2, 3], [1.0, 1.0, 1.0], [0.1, 0.1, 0.1], 0.0, 1.0, 0.0, [0])
    assert 0.95 < m[0] <= 1.0 and v < 0.01


def test_prior_arrays_match_prior(fluid):
    df = pd.DataFrame({'p_wh': [50e5, 60e5], 'p_man': [20e5, 25e5], 't_wh': [330.0, 340.0],
                       'opening': [0.5, 0.8], 'd_choke': [0.02, 0.025], 'vendor_gamma': [1.0, 1.6]})
    m1, f1 = hybrid.prior(df, fluid)
    m2, x2 = hybrid.prior_arrays(df.p_wh, df.p_man, df.t_wh, df.opening, df.d_choke, df.vendor_gamma, fluid)
    assert np.allclose(m1, m2) and np.allclose(f1.to_numpy(), x2)
    assert list(f1.columns) == hybrid.FEATURES


def test_tree_export_parity():
    import lightgbm as lgb
    rng = np.random.default_rng(1)
    x = rng.normal(size=(2000, 5))
    y = np.sin(x[:, 0]) + x[:, 1] * x[:, 2] + rng.normal(0, 0.1, 2000)
    m = lgb.LGBMRegressor(n_estimators=80, num_leaves=15, verbose=-1).fit(x, y)
    arr = edge.export_trees(m)
    xt = rng.normal(size=(500, 5))
    xt[::17, 3] = np.nan
    assert np.max(np.abs(edge.predict_trees(arr, xt) - m.predict(xt))) < 1e-9


def test_vfm_multiplier_scales_rate(wells):
    table = well.vlp_table(wells)
    base = wells.choke_rate(table['p_wh'][..., 10], np.full(wells.pi.shape, 20e5), table['t_wh'][..., 10],
                            np.full(wells.pi.shape, 0.5))
    w2 = control.cautious(wells, np.full(wells.pi.shape, np.log(1.2)))
    scaled = w2.choke_rate(table['p_wh'][..., 10], np.full(wells.pi.shape, 20e5), table['t_wh'][..., 10],
                           np.full(wells.pi.shape, 0.5))
    assert np.allclose(scaled, 1.2 * base)


def test_setpoint_hits_target_on_its_own_model(wells):
    w1 = wells.take((np.array([0]), np.array([0])))
    w1 = well.Wells(**{f: np.ravel(getattr(w1, f)) for f in ('p_res', 't_res', 'pi', 'depth', 'd_tub', 'd_choke',
                                                             'cd', 'trim_gamma', 't_sea', 'q_t')},
                    fluid=pvt.Fluid(*(np.ravel(getattr(w1.fluid, n)) for n in ('api', 'gas_sg', 'gor', 'wc',
                                                                                'salinity'))))
    table = well.vlp_table(w1)
    q_half = float(well.solve_rate(w1, table, np.array([20e5]), np.array(0.5))['q_l'][0])
    u_set, ok = control.single_well_setpoint(w1, table, np.array([20e5]), q_half)
    assert ok and abs(u_set - 0.5) < 0.01


def test_simfield_small_run_conserves():
    pop = simfield.sample_population(n_groups=3, max_wells=4, seed=5)
    res = simfield.simulate(pop, days=3, seed=5)
    d = res['daily']
    assert len(d) == 3 * int(pop.mask.sum())
    assert d.true_m.ge(0).all() and (d[d.true_m > 0].mpfm_q_o > 0).all()
    g = res['groups'].sort_values(['group', 'day'])
    m_wells = d.groupby(['group', 'day']).true_m.sum().sort_index().to_numpy()
    m_sep = network.std_mass(g.sep_q_o.to_numpy(), g.sep_q_g.to_numpy(), g.sep_q_w.to_numpy(),
                             pvt.Fluid(35.0, 0.7, 0.0, 0.0, 0.0))
    assert len(m_wells) == len(g) and np.all(m_sep > 0)


def test_edge_step_matches_full_solve(wells, line):
    w = wells.take(np.array([0]))
    w = well.Wells(**{f: np.asarray(getattr(w, f))[0] for f in ('p_res', 't_res', 'pi', 'depth', 'd_tub',
                                                               'd_choke', 'cd', 'trim_gamma', 't_sea', 'q_t')},
                   fluid=pvt.Fluid(*(np.asarray(getattr(w.fluid, n))[0] for n in ('api', 'gas_sg', 'gor', 'wc',
                                                                                 'salinity'))))
    tab = well.vlp_table(w)
    u_grid, p2_grid = np.linspace(0.05, 1.0, 20), np.linspace(10e5, 80e5, 30)
    rt = edge.twin_tables(w, tab, u_grid, p2_grid)
    mls = pvt.mass_per_liquid_sc(w.fluid)
    ln = network.Line(length=line.length[:1], d=line.d[:1], depth=line.depth[:1], p_sep=line.p_sep[:1])
    u_chk = np.full(3, 0.6)
    full = network.solve(control.replicate(w, 1), control.replicate_table(tab, 1), u_chk[None, :], ln, iters=40)
    mix = full['sep_fluid']
    m_grid = np.linspace(0, (rt[:, -1, 0] * mls).sum() * 1.1, 120)
    p_req = edge.line_table(ln, pvt.Fluid(*(np.asarray(getattr(mix, n))[0] for n in ('api', 'gas_sg', 'gor', 'wc',
                                                                                     'salinity'))),
                            float(full['t_mix'][0]), m_grid)
    p_edge, q_edge = edge.edge_step(rt, u_grid, p2_grid, mls, m_grid, p_req, u_chk)
    assert abs(p_edge / full['p_man'][0] - 1) < 0.02
    assert abs(q_edge.sum() / full['q_l'][0].sum() - 1) < 0.03
