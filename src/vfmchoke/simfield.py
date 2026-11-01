"""SimField: a synthetic population of subsea wells on manifolds, simulated day by day.

Each group has one manifold and one flowline + riser to a separator. Each well has a reservoir, a tubing string,
and a choke whose discharge coefficient and trim exponent are hidden (the vendor trim type is known, the as-built
deviation is not). Over time reservoir pressure declines, water cut rises, GOR drifts, the choke erodes, operators
move chokes and shut wells in, and produced-water salinity drifts while the gamma meter keeps its configured value.
Every well has a gamma MPFM (daily readings with their own errors) and a test-separator well test about monthly.
Leaks can be scheduled per group to build slick scenarios.

Truth columns (prefixed true_ or hidden) are for evaluation only; models see design data, sensors and labels.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import choke, mpfm, network, pvt, units as u, well


@dataclass
class Population:
    wells: well.Wells          # truth at day 0, arrays (G, W)
    mask: np.ndarray           # real well slots
    line: network.Line
    vendor_gamma: np.ndarray   # trim exponent on the vendor datasheet
    decline: np.ndarray        # reservoir pressure decline, Pa/day
    wc_rate: np.ndarray        # water-cut rise per day
    gor_drift: np.ndarray      # GOR log-drift per day
    erosion: np.ndarray        # relative Cd growth per day
    sal_target: np.ndarray     # salinity the produced water drifts toward (seawater breakthrough)
    sal_rate: np.ndarray       # fraction of the gap closed per day
    meter_sal: np.ndarray      # salinity configured in the MPFM (stale)
    meter_cd: np.ndarray       # MPFM venturi Cd ratio
    meter_d: np.ndarray        # MPFM throat/pipe diameter, m
    test_phase: np.ndarray     # day offset of the monthly test cycle
    well_id: np.ndarray        # global id per slot (-1 for padding)


def _choke_size(w, line_p_sep, frac=0.6):
    """Bean diameter that passes about `frac` of open flow at full opening against a typical manifold pressure."""
    q_des = frac * well.aof(w.p_res, w.p_b, w.pi)
    m_des = q_des * pvt.mass_per_liquid_sc(w.fluid)
    p1 = np.maximum(line_p_sep + 25e5, 40e5)
    m_ref = choke.mass_rate(p1, 0.5 * p1, u.c_to_k(60.0), w.fluid, 1.0, 0.0254, choke.CD_NOMINAL)
    d = 0.0254 * np.sqrt(m_des / np.maximum(m_ref, 1e-9))
    return np.round(np.clip(d, 0.008, 0.064) / (u.INCH / 64.0)) * (u.INCH / 64.0)


def sample_population(n_groups=200, max_wells=10, seed=0, choke_frac=0.6,
                      line_d=(0.1524, 0.2032, 0.254, 0.3048), depth=(30, 400), p_sep=(8e5, 25e5),
                      length=(1000, 15000), flow_margin=30e5):
    """Defaults reproduce the M2 v1 population; the keyword ranges let a variant change the network."""
    rng = np.random.default_rng(seed)
    g, w = n_groups, max_wells
    shape = (g, w)
    n_wells = rng.integers(4, w + 1, g)
    mask = np.arange(w)[None, :] < n_wells[:, None]
    fl = pvt.Fluid(api=rng.uniform(25, 45, shape), gas_sg=rng.uniform(0.62, 0.9, shape),
                   gor=np.exp(rng.uniform(np.log(50), np.log(350), shape)),
                   wc=np.where(rng.random(shape) < 0.3, 0.0, rng.uniform(0.0, 0.5, shape)),
                   salinity=rng.uniform(0.02, 0.15, shape))
    vendor_gamma = rng.choice([1.0, 1.6], shape)
    line = network.Line(length=rng.uniform(*length, g), d=rng.choice(list(line_d), g),
                        depth=rng.uniform(*depth, g), p_sep=rng.uniform(*p_sep, g))
    wells = well.Wells(p_res=rng.uniform(200e5, 360e5, shape), t_res=u.c_to_k(rng.uniform(70, 130, shape)),
                       pi=np.exp(rng.uniform(np.log(2), np.log(30), shape)) * well.PI_FIELD,
                       depth=rng.uniform(1500, 3200, shape), d_tub=rng.choice([0.0762, 0.1005, 0.1213], shape),
                       d_choke=np.full(shape, 0.0254),
                       cd=np.clip(rng.normal(0.82, 0.04, shape), 0.68, 0.95),
                       trim_gamma=vendor_gamma * np.exp(rng.normal(0.0, 0.08, shape)),
                       t_sea=np.repeat(u.c_to_k(rng.uniform(4, 25, g))[:, None], w, 1),
                       q_t=np.full(shape, 0.002), fluid=fl)
    # every real well must be able to flow: shut-in wellhead pressure at least 30 bar above the separator
    for _ in range(4):
        shut_in = well.vlp_table(wells, n_q=8)['p_wh'][..., 0]
        deficit = (line.p_sep[:, None] + flow_margin) - shut_in
        if not np.any(mask & (deficit > 0)):
            break
        wells.p_res = np.where(mask & (deficit > 0), wells.p_res + deficit + 5e5, wells.p_res)
    wells.d_choke = _choke_size(wells, line.p_sep[:, None], choke_frac)
    # padding slots: no inflow at all
    wells.p_res = np.where(mask, wells.p_res, u.P_SC)
    wells.pi = np.where(mask, wells.pi, 1e-30)
    ids = np.cumsum(mask.ravel()).reshape(shape) - 1
    return Population(
        wells=wells, mask=mask, line=line, vendor_gamma=vendor_gamma,
        decline=rng.uniform(0, 15e5, shape) / 365.0,
        wc_rate=rng.uniform(0, 0.0015, shape),
        gor_drift=rng.normal(0, 0.0005, shape),
        erosion=rng.uniform(0, 0.05, shape) / 365.0,
        sal_target=np.where(rng.random(shape) < 0.4, 0.035, fl.salinity),
        sal_rate=rng.uniform(0.0, 0.01, shape),
        meter_sal=fl.salinity.copy(),
        meter_cd=rng.normal(1.0, 0.015, shape),
        meter_d=np.clip(np.sqrt(wells.d_choke * wells.d_tub) * 1.6, 0.05, 0.12),
        test_phase=rng.integers(0, 30, shape),
        well_id=np.where(mask, ids, -1),
    )


def state_on(pop, day):
    """Truth well parameters on a given day."""
    w0 = pop.wells
    f0 = w0.fluid
    fl = pvt.Fluid(api=f0.api, gas_sg=f0.gas_sg, gor=f0.gor * np.exp(pop.gor_drift * day),
                   wc=np.clip(f0.wc + pop.wc_rate * day, 0.0, 0.95),
                   salinity=pop.sal_target + (f0.salinity - pop.sal_target) * (1 - pop.sal_rate) ** day)
    return well.Wells(p_res=np.where(pop.mask, np.maximum(w0.p_res - pop.decline * day, 60e5), w0.p_res),
                      t_res=w0.t_res, pi=w0.pi, depth=w0.depth, d_tub=w0.d_tub, d_choke=w0.d_choke,
                      cd=w0.cd * (1.0 + pop.erosion * day), trim_gamma=w0.trim_gamma, t_sea=w0.t_sea, q_t=w0.q_t,
                      fluid=fl)


def choke_schedule(pop, days, seed=0, p_move=0.1, p_shut=0.005):
    """Operator choke positions per day, shape (days, G, W): piecewise constant moves and short shut-ins."""
    rng = np.random.default_rng(seed + 1)
    g, w = pop.mask.shape
    op = np.empty((days, g, w))
    cur = rng.uniform(0.3, 0.9, (g, w))
    shut = np.zeros((g, w), int)
    for t in range(days):
        move = rng.random((g, w)) < p_move
        cur = np.where(move, np.clip(cur + rng.normal(0, 0.15, (g, w)), 0.15, 1.0), cur)
        start = (shut == 0) & (rng.random((g, w)) < p_shut)
        shut = np.where(start, rng.integers(1, 6, (g, w)), np.maximum(shut - 1, 0))
        op[t] = np.where(shut > 0, 0.0, cur) * pop.mask
    return op


def simulate(pop, days=180, seed=0, leak=None, leak_start=None, log_every=0, n_q=40):
    """Run the field day by day. leak: network.Leak (G,) active from day leak_start (G,) onwards.
    Returns DataFrames: daily (well-days), tests (separator tests), groups (group-days)."""
    rng = np.random.default_rng(seed + 2)
    g, w = pop.mask.shape
    openings = choke_schedule(pop, days, seed)
    rows, test_rows, group_rows = [], [], []
    gi, wi = np.nonzero(pop.mask)
    none = network.Leak.none(g)
    for t in range(days):
        st = state_on(pop, t)
        table = well.vlp_table(st, n_q=n_q)
        lk = none
        if leak is not None:
            active = t >= leak_start
            lk = network.Leak(kind=np.where(active, leak.kind, 0), well=leak.well, pos=leak.pos,
                              d_hole=np.where(active, leak.d_hole, 0.0))
        out = network.solve(st, table, openings[t], pop.line, lk, iters=32)
        info = choke.mass_rate(out['p_wh'], out['p_man'][:, None] * np.ones((1, w)), out['t_wh'], st.fluid,
                               openings[t], st.d_choke, st.cd, st.trim_gamma, info=True)[1]
        cfg = mpfm.Config(salinity=pop.meter_sal, cd_ratio=pop.meter_cd, noise_l=0.02, noise_g=0.04)
        rd = mpfm.read(out['p_wh'], out['t_wh'], st.fluid, out['q_l'], pop.meter_d, cfg, rng=rng)
        flowing = out['q_l'] > 0
        rows.append(pd.DataFrame({
            'day': t, 'group': gi, 'slot': wi, 'well_id': pop.well_id[gi, wi],
            'opening': openings[t][gi, wi], 'd_choke': st.d_choke[gi, wi], 'vendor_gamma': pop.vendor_gamma[gi, wi],
            'p_wh': out['p_wh'][gi, wi], 't_wh': out['t_wh'][gi, wi], 'p_man': out['p_man'][gi],
            'p_sep': pop.line.p_sep[gi],
            'mpfm_q_o': np.where(flowing, rd['q_o'], 0.0)[gi, wi], 'mpfm_q_g': np.where(flowing, rd['q_g'], 0.0)[gi, wi],
            'mpfm_q_w': np.where(flowing, rd['q_w'], 0.0)[gi, wi],
            'true_q_o': out['q_o'][gi, wi], 'true_q_g': out['q_g'][gi, wi], 'true_q_w': out['q_w'][gi, wi],
            'true_m': out['m'][gi, wi], 'true_critical': info['critical'][gi, wi], 'true_y': info['y'][gi, wi],
            'true_gor': st.fluid.gor[gi, wi], 'true_wc': st.fluid.wc[gi, wi], 'true_sal': st.fluid.salinity[gi, wi],
            'hidden_cd': st.cd[gi, wi], 'hidden_trim_gamma': st.trim_gamma[gi, wi],
            'p_res': st.p_res[gi, wi],
        }))
        testing = pop.mask & ((t + pop.test_phase) % 30 == 0) & flowing
        if testing.any():
            tg, tw = np.nonzero(testing)
            noise = lambda: np.exp(rng.normal(0.0, 0.015, len(tg)))
            test_rows.append(pd.DataFrame({'day': t, 'group': tg, 'slot': tw, 'well_id': pop.well_id[tg, tw],
                                           'test_q_o': out['q_o'][tg, tw] * noise(),
                                           'test_q_g': out['q_g'][tg, tw] * noise(),
                                           'test_q_w': out['q_w'][tg, tw] * noise()}))
        group_rows.append(pd.DataFrame({'day': t, 'group': np.arange(g), 'p_man': out['p_man'],
                                        'p_riser_base': out['p_riser_base'], 'p_sep': pop.line.p_sep,
                                        'sep_q_o': out['sep_q_o'], 'sep_q_g': out['sep_q_g'],
                                        'sep_q_w': out['sep_q_w'], 'leak_m': out['leak_m'],
                                        'leak_q_o': out['leak_q_o'], 'leak_q_g': out['leak_q_g'],
                                        'leak_q_w': out['leak_q_w'], 'leak_kind': lk.kind}))
        if log_every and t % log_every == 0:
            print(f'day {t}: flowing {int(flowing[pop.mask].sum())}/{int(pop.mask.sum())}, '
                  f'oil {u.si_to_stb_d(out["sep_q_o"].sum()):.0f} stb/d', flush=True)
    return {'daily': pd.concat(rows, ignore_index=True),
            'tests': pd.concat(test_rows, ignore_index=True) if test_rows else pd.DataFrame(),
            'groups': pd.concat(group_rows, ignore_index=True)}


def wells_frame(pop):
    """Static per-well data: design values a greenfield well has on day 1, plus hidden truth."""
    gi, wi = np.nonzero(pop.mask)
    w, f = pop.wells, pop.wells.fluid
    return pd.DataFrame({
        'well_id': pop.well_id[gi, wi], 'group': gi, 'slot': wi,
        'design_api': f.api[gi, wi], 'design_gas_sg': f.gas_sg[gi, wi], 'design_gor': f.gor[gi, wi],
        'design_wc': f.wc[gi, wi], 'design_sal': f.salinity[gi, wi],
        'd_choke': w.d_choke[gi, wi], 'vendor_gamma': pop.vendor_gamma[gi, wi], 'd_tub': w.d_tub[gi, wi],
        'depth': w.depth[gi, wi], 'meter_d': pop.meter_d[gi, wi],
        'hidden_cd0': w.cd[gi, wi], 'hidden_trim_gamma': w.trim_gamma[gi, wi], 'hidden_pi': w.pi[gi, wi],
        'hidden_p_res0': w.p_res[gi, wi], 'line_length': pop.line.length[gi], 'line_depth': pop.line.depth[gi],
    })


def summarize(res):
    d = res['daily']
    fl = d[d.true_m > 0]
    q_l = u.si_to_stb_d(fl.true_q_o + fl.true_q_w)
    mp_err = (fl.mpfm_q_o + fl.mpfm_q_w) / (fl.true_q_o + fl.true_q_w) - 1
    pct = lambda s: {k: float(np.percentile(s, k)) for k in (1, 10, 50, 90, 99)}
    return {
        'well_days': int(len(d)), 'flowing_fraction': float(len(fl) / max(len(d), 1)),
        'wells': int(d.well_id.nunique()), 'groups': int(d.group.nunique()), 'days': int(d.day.nunique()),
        'liquid_stb_d': pct(q_l), 'p_wh_bar': pct(fl.p_wh / 1e5), 'p_man_bar': pct(fl.p_man / 1e5),
        'pressure_ratio_y': pct(fl.true_y), 'critical_fraction': float(fl.true_critical.mean()),
        'water_cut': pct(fl.true_wc), 'gor_scf_stb': pct(u.si_to_scf_stb(fl.true_gor)),
        'mpfm_liquid_rel_error': pct(mp_err), 'tests': int(len(res['tests'])),
    }
