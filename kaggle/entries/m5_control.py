# M5: choke setpoints and a multi-well optimizer on the calibrated VFM twin, applied to the SimField truth.
# Fold 0 groups are held out of training (same fold assignment as M3). Control happens on day 120; calibration
# labels come from days < 90, so the twin is 30+ days stale, as in practice.
import json
from pathlib import Path
import pickle
import sys
import time

sys.path.insert(0, 'src')
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from vfmchoke import choke, control, fewshot, fleet, hybrid, network, pvt, simfield, well  # noqa: E402

T0 = time.time()
import glob  # noqa: E402
IN = str(Path(sorted(glob.glob('/kaggle/input/**/simfield_daily.parquet', recursive=True), key=lambda p: ('m2b' not in p))[0]).parent)
print('simulation input:', IN, flush=True)
with open(f'{IN}/simfield_pop.pkl', 'rb') as fh:
    pop = pickle.load(fh)
daily, tests, wells = fleet.prepare(pd.read_parquet(f'{IN}/simfield_daily.parquet'),
                                    pd.read_parquet(f'{IN}/simfield_tests.parquet'),
                                    pd.read_parquet(f'{IN}/simfield_wells.parquet'))
DAY, CAL_END, N_GROUPS, KS = 120, 90, 8, (0, 3, 30)
RNG = np.random.default_rng(0)


def log(m):
    print(f'[{time.time() - T0:7.0f}s] {m}', flush=True)


groups = np.sort(daily.group.unique())
np.random.default_rng(0).shuffle(groups)
fold_of = {gid: i % 5 for i, gid in enumerate(groups)}
daily['fold'] = daily.group.map(fold_of)
trained = fleet.train(daily[(daily.fold != 0) & daily.flowing], tests)
log(f'trained; prior {trained.prior.to_dict()}')

openings_all = simfield.choke_schedule(pop, 180, seed=2026)
truth = simfield.state_on(pop, DAY)
open_today = (openings_all[DAY] > 0).sum(axis=1)
sel_groups = [int(g) for g in groups if fold_of[g] == 0 and open_today[g] >= 4][:N_GROUPS]
log(f'groups {sel_groups}')


def truth_solve(g, u_vec):
    w = truth.take(np.array([g]))
    tab = well.vlp_table(w)
    line = network.Line(*(np.asarray(getattr(pop.line, f))[[g]] for f in ('length', 'd', 'depth', 'p_sep')))
    out = network.solve(w, tab, u_vec[None, :], line, iters=32)
    return out


def twin_for(g, k):
    """Twin wells of group g (shape (W,)) and the per-well sd of the log-rate, calibrated with k MPFM labels."""
    W = pop.mask.shape[1]
    w0 = pop.wells.take(np.array([g]))       # design geometry (truth day 0)
    p_res, pi = w0.p_res[0].copy(), w0.pi[0].copy()
    rng = np.random.default_rng(1000 + g)
    p_res = p_res * (1 + rng.normal(0, 0.03, W))
    pi = pi * np.exp(rng.normal(0, 0.25, W))
    k_mult, k_exp, sd = np.ones(W), np.zeros(W), np.full(W, 0.1)
    api, gsg, gor, wc, sal = (np.array(w0.fluid.api[0]), np.array(w0.fluid.gas_sg[0]), np.array(w0.fluid.gor[0]),
                              np.array(w0.fluid.wc[0]), np.array(w0.fluid.salinity[0]))
    for s in range(W):
        if not pop.mask[g, s]:
            continue
        wid = int(pop.well_id[g, s])
        hist = daily[(daily.well_id == wid) & daily.flowing & (daily.day < CAL_END)].sort_values('day').head(k)
        cal = fleet.calibrate(hist, wells.loc[wid], trained, 'mpfm')
        wc[s], gor[s] = cal.wc, cal.gor
        row = daily[(daily.well_id == wid) & (daily.day == DAY)]
        e = fleet.with_est(row, cal.wc, cal.gor)
        _, fe = hybrid.prior(e, hybrid.fluid_from(e))
        f_now = float(trained.model.correction(fe)[0])
        phi = np.array([[1.0, np.log(max(float(row.opening.iloc[0]), 1e-3))]])
        _, var = fewshot.latent_predict(cal.x, cal.p, cal.t_last, [DAY], phi, trained.prior)
        k_mult[s], k_exp[s], sd[s] = np.exp(f_now + cal.x[0]), cal.x[1], float(np.sqrt(var[0]))
        if k and len(hist):      # deliverability correction: shift reservoir pressure to match observed WHP
            tw = well.Wells(p_res=p_res[[s]], t_res=w0.t_res[0, [s]], pi=pi[[s]], depth=w0.depth[0, [s]],
                            d_tub=w0.d_tub[0, [s]], d_choke=w0.d_choke[0, [s]], cd=np.array([choke.CD_NOMINAL]),
                            trim_gamma=pop.vendor_gamma[g, [s]], t_sea=w0.t_sea[0, [s]], q_t=w0.q_t[0, [s]],
                            fluid=pvt.Fluid(api[[s]], gsg[[s]], gor[[s]], wc[[s]], sal[[s]]))
            tab = well.vlp_table(tw)
            q_lab = (hist.mpfm_q_o + hist.mpfm_q_w).to_numpy()
            whp_twin = np.array([well.interp({kk: v for kk, v in tab.items()}, np.array([q]))[0][0] for q in q_lab])
            p_res[s] += float(np.mean(hist.p_wh.to_numpy() - whp_twin))
    fl = pvt.Fluid(api, gsg, gor, wc, sal)
    tw = well.Wells(p_res=np.where(pop.mask[g], p_res, 1e5), t_res=w0.t_res[0], pi=np.where(pop.mask[g], pi, 1e-30),
                    depth=w0.depth[0], d_tub=w0.d_tub[0], d_choke=w0.d_choke[0],
                    cd=np.full(W, choke.CD_NOMINAL), trim_gamma=pop.vendor_gamma[g], t_sea=w0.t_sea[0],
                    q_t=w0.q_t[0], fluid=fl, k_mult=k_mult, k_exp=k_exp)
    return tw, sd


single, multi = [], []
for g in sel_groups:
    u_now = openings_all[DAY][g].copy()
    base = truth_solve(g, u_now)
    line_g = network.Line(*(np.asarray(getattr(pop.line, f))[[g]] for f in ('length', 'd', 'depth', 'p_sep')))
    for k in KS:
        tw, sd = twin_for(g, k)
        tab = well.vlp_table(tw)
        # single-well setpoints: cut each open well to 80% of its current rate, and raise it by 10%
        for s in np.nonzero((u_now > 0) & pop.mask[g])[0]:
            q_now = float(base['q_l'][0, s])
            if q_now <= fleet.MIN_Q:
                continue
            w1 = tw.take(np.array([s]))
            t1 = {kk: v[[s]] for kk, v in tab.items()}
            for factor in (0.8, 1.1):
                u_set, ok = control.single_well_setpoint(w1, t1, np.array([base['p_man'][0]]), factor * q_now)
                u_new = u_now.copy()
                u_new[s] = u_set
                real = truth_solve(g, u_new)
                single.append({'group': int(g), 'slot': int(s), 'k': k, 'factor': factor, 'achievable': ok,
                               'u_now': float(u_now[s]), 'u_set': float(u_set),
                               'target': factor * q_now, 'realised': float(real['q_l'][0, s]),
                               'rel_err': float(real['q_l'][0, s] / (factor * q_now) - 1)})
        # multi-well: caps at 85% of today's gas and 90% of today's water
        gas_cap, water_cap = 0.85 * float(base['q_g'].sum()), 0.90 * float(base['q_w'].sum())
        hi = control.cautious(tw, 1.645 * sd)
        u_heur = control.scale_heuristic(tw, hi, tab, line_g, u_now, gas_cap, water_cap)
        u_opt, info = control.optimize_group(tw, hi, tab, line_g, u_heur, (u_now > 0) & pop.mask[g], gas_cap,
                                             water_cap, maxiter=20)
        rh, ro = truth_solve(g, u_heur), truth_solve(g, u_opt)
        multi.append({'group': int(g), 'k': k, 'gas_cap': gas_cap, 'water_cap': water_cap,
                      'oil_now': float(base['q_o'].sum()),
                      'oil_heur': float(rh['q_o'].sum()), 'oil_opt': float(ro['q_o'].sum()),
                      'gas_heur': float(rh['q_g'].sum()), 'gas_opt': float(ro['q_g'].sum()),
                      'water_heur': float(rh['q_w'].sum()), 'water_opt': float(ro['q_w'].sum()),
                      'opt_success': info['success'], 'opt_nit': info['nit']})
        log(f'group {g} k={k}: oil heur {multi[-1]["oil_heur"]:.4f} opt {multi[-1]["oil_opt"]:.4f}')

sdf, mdf = pd.DataFrame(single), pd.DataFrame(multi)
sdf.to_csv('m5_single_well.csv', index=False)
mdf.to_csv('m5_multi_well.csv', index=False)
summary = {}
for k in KS:
    s = sdf[(sdf.k == k) & sdf.achievable]
    m = mdf[mdf.k == k]
    gain = (m.oil_opt / m.oil_heur - 1).to_numpy()
    boots = [np.mean(RNG.choice(gain, len(gain))) for _ in range(2000)]
    viol = lambda col, cap: float(np.mean(m[col] > m[cap] * 1.0001))
    summary[k] = {
        'single_within_5pct': float(np.mean(np.abs(s.rel_err) <= 0.05)), 'single_median_abs_err': float(
            np.median(np.abs(s.rel_err))), 'single_trials': int(len(s)),
        'multi_oil_gain_mean': float(gain.mean()), 'multi_oil_gain_ci95': [float(np.percentile(boots, 2.5)),
                                                                           float(np.percentile(boots, 97.5))],
        'violations_gas_opt': viol('gas_opt', 'gas_cap'), 'violations_water_opt': viol('water_opt', 'water_cap'),
        'violations_gas_heur': viol('gas_heur', 'gas_cap'), 'violations_water_heur': viol('water_heur', 'water_cap'),
    }
summary['seconds'] = round(time.time() - T0, 1)
with open('m5_summary.json', 'w') as fh:
    json.dump(summary, fh, indent=1)
print(json.dumps(summary, indent=1))
