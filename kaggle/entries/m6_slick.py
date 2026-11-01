# M6: slick attribution on simulated leaks. Held-out (fold 0) groups are copied into scenarios: leaks of random kind,
# place and size start on days 25-40; a slick is detected 1-5 days later. Controls have no leak and a slick from
# elsewhere (vessel, seep). Evidence uses per-well MPFM readings, noisy separator rates, manifold and riser-base
# pressures, the slick's position and an uncertain volume estimate.
import json
from pathlib import Path
import pickle
import sys
import time

sys.path.insert(0, 'src')
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from vfmchoke import leak, network, simfield, units as u, well  # noqa: E402

T0 = time.time()
import glob  # noqa: E402
IN = str(Path(sorted(glob.glob('/kaggle/input/**/simfield_daily.parquet', recursive=True), key=lambda p: ('m2b' not in p))[0]).parent)
print('simulation input:', IN, flush=True)
with open(f'{IN}/simfield_pop.pkl', 'rb') as fh:
    pop = pickle.load(fh)
DAYS, N_LEAK, N_CTRL, REF = 60, 300, 100, (5, 25)
rng = np.random.default_rng(7)


def log(m):
    print(f'[{time.time() - T0:7.0f}s] {m}', flush=True)


groups = np.arange(pop.mask.shape[0])
np.random.default_rng(0).shuffle(groups)
fold0 = np.array([g for i, g in enumerate(groups) if i % 5 == 0 and pop.mask[g].sum() >= 3])
S = N_LEAK + N_CTRL
base = rng.choice(fold0, S)


def sub_population(p, idx):
    w = p.wells.take(idx)
    line = network.Line(*(np.asarray(getattr(p.line, f))[idx] for f in ('length', 'd', 'depth', 'p_sep')))
    kw = {f: getattr(p, f)[idx] for f in ('mask', 'vendor_gamma', 'decline', 'wc_rate', 'gor_drift', 'erosion',
                                          'sal_target', 'sal_rate', 'meter_sal', 'meter_cd', 'meter_d', 'test_phase',
                                          'well_id')}
    return simfield.Population(wells=w, line=line, **kw)


ps = sub_population(pop, base)
W = ps.mask.shape[1]
kind = np.zeros(S, int)
kind[:N_LEAK] = rng.choice([network.JUMPER, network.MANIFOLD, network.FLOWLINE, network.RISER], N_LEAK,
                           p=[0.4, 0.15, 0.3, 0.15])
slot = np.array([rng.choice(np.nonzero(ps.mask[i])[0]) for i in range(S)])
pos = np.where(kind == network.FLOWLINE, rng.uniform(0.05, 0.95, S), np.where(kind == network.RISER,
                                                                             rng.uniform(0.05, 0.8, S), 0.0))
d_hole = np.where(kind > 0, np.exp(rng.uniform(np.log(0.002), np.log(0.015), S)), 0.0)
onset = rng.integers(25, 41, S)
detect = onset + rng.integers(1, 6, S)
lk = network.Leak(kind=kind, well=slot, pos=pos, d_hole=d_hole)
log(f'simulating {S} scenarios x {DAYS} days')
res = simfield.simulate(ps, days=DAYS, seed=11, leak=lk, leak_start=onset, log_every=10)
d, g = res['daily'], res['groups']
openings = simfield.choke_schedule(ps, DAYS, seed=11)
log('simulated')

# separator measurements with 1% daily noise
for ph in ('o', 'g', 'w'):
    g[f'sepm_{ph}'] = g[f'sep_q_{ph}'] * np.exp(rng.normal(0, 0.01, len(g)))
wsum = d.groupby(['group', 'day'])[['mpfm_q_o', 'mpfm_q_g', 'mpfm_q_w']].sum().reset_index()
g = g.merge(wsum, on=['group', 'day'])

# pressure sensitivities and the no-leak model pressure at each scenario's detection day
hyp_rows = []          # (scenario, hypothesis spec) to solve
lay = {}
for i in range(S):
    lrng = np.random.default_rng(1000 + i)
    well_xy, riser_xy = leak.layout(W, float(ps.line.length[i]), lrng)
    hs = leak.hypotheses(np.nonzero(ps.mask[i])[0], well_xy, riser_xy)      # every real well, open or not
    lay[i] = (well_xy, riser_xy, hs)
    for j, h in enumerate(hs):
        kd = {'none': 0, 'jumper': 1, 'manifold': 2, 'flowline': 3, 'riser': 4}[h.kind]
        hyp_rows.append((i, j, kd, max(h.well, 0), h.pos, 0.0 if kd == 0 else 0.005))
hr = pd.DataFrame(hyp_rows, columns=['scen', 'hyp', 'kind', 'well', 'pos', 'd_hole'])
sens = {}
for t_d in np.unique(detect):
    st = simfield.state_on(ps, int(t_d))
    sub = hr[detect[hr.scen.to_numpy()] == t_d]
    for c0 in range(0, len(sub), 800):
        ch = sub.iloc[c0:c0 + 800]
        idx = ch.scen.to_numpy()
        w = st.take(idx)
        line = network.Line(*(np.asarray(getattr(ps.line, f))[idx] for f in ('length', 'd', 'depth', 'p_sep')))
        out = network.solve(w, well.vlp_table(w), openings[t_d][idx], line,
                            network.Leak(kind=ch.kind.to_numpy(), well=ch.well.to_numpy(), pos=ch.pos.to_numpy(),
                                         d_hole=ch.d_hole.to_numpy()), iters=32)
        for r, (si, hj) in enumerate(zip(idx, ch.hyp.to_numpy())):
            sens[(int(si), int(hj))] = (float(out['p_man'][r]), float(out['p_riser_base'][r]),
                                        float(out['leak_q_o'][r]))
log('sensitivities done')

records, demo = [], None
P_ERR, P_NOISE = 0.3e5, 0.05e5
for i in range(S):
    gi = g[g.group == i].sort_values('day')
    r = np.column_stack([gi.mpfm_q_o - gi.sepm_o, gi.mpfm_q_g - gi.sepm_g, gi.mpfm_q_w - gi.sepm_w])
    ref = r[REF[0]:REF[1]]
    bias, cov_day = ref.mean(0), np.cov(ref.T) + np.diag([1e-12, 1e-10, 1e-12])
    alarm, start = leak.cusum_onset(r[:, 0] / max(np.mean(gi.mpfm_q_o), 1e-12), REF[1])
    t_d = int(detect[i])
    t_s = start if (alarm is not None and alarm <= t_d) else max(t_d - 3, REF[1])
    win = slice(t_s, t_d + 1)
    n_win = t_d + 1 - t_s
    r_mean = r[win].mean(0) - bias
    r_cov = cov_day / n_win
    # compositions: pre-onset MPFM ratios per well, and the commingled mix
    di = d[(d.group == i) & d.day.between(REF[0], REF[1] - 1)]
    per_slot = di.groupby('slot')[['mpfm_q_o', 'mpfm_q_g', 'mpfm_q_w']].mean()
    mix = per_slot.sum().to_numpy()
    well_xy, riser_xy, hs = lay[i]
    comps = []
    for h in hs:
        if h.kind == 'none':
            comps.append(None)
        elif h.kind == 'jumper':
            v = per_slot.loc[h.well].to_numpy() if h.well in per_slot.index else mix
            comps.append(v / max(v[0], 1e-12))
        else:
            comps.append(mix / max(mix[0], 1e-12))
    # pressures: observed minus the no-leak model on the detection day, with a model error
    p_nl = sens[(i, 0)]
    obs_day = gi[gi.day == t_d].iloc[0]
    model_bias = rng.normal(0, P_ERR, 2)
    dp_obs = np.array([obs_day.p_man - p_nl[0], obs_day.p_riser_base - p_nl[1]]) + model_bias \
        + rng.normal(0, P_NOISE, 2)
    dp_cov = np.eye(2) * (P_ERR ** 2 + P_NOISE ** 2)
    def unit_sens(j):
        dq = sens[(i, j)][2]
        if dq < 1e-9:          # the reference hole would not leak there (line below ambient): no pressure signature
            return (0.0, 0.0)
        return ((sens[(i, j)][0] - p_nl[0]) / dq, (sens[(i, j)][1] - p_nl[1]) / dq)

    dp_sens = [None] + [unit_sens(j) for j in range(1, len(hs))]
    # slick: up-current end of the slick back-tracked with an uncertain drift
    depth = float(ps.line.depth[i])
    cur_dir = rng.uniform(0, 2 * np.pi)
    true_drift = 4.0 * depth * np.array([np.cos(cur_dir), np.sin(cur_dir)])
    if kind[i] == network.JUMPER:
        src = well_xy[slot[i]] * 0.5
    elif kind[i] == network.FLOWLINE:
        src = np.array([pos[i] * ps.line.length[i], 0.0])
    elif kind[i] == network.RISER:
        src = riser_xy
    elif kind[i] == network.MANIFOLD:
        src = np.zeros(2)
    else:
        src = rng.uniform(-5000, 5000, 2)          # a slick from elsewhere
    slick_xy = src + true_drift + rng.normal(0, 150, 2)
    drift_est = true_drift * (1 + rng.normal(0, 0.3))
    loc_sd = float(np.sqrt(150 ** 2 + (0.3 * np.linalg.norm(true_drift)) ** 2))
    released_true = float(gi[gi.day.between(onset[i], t_d)].leak_q_o.sum()) * u.DAY * 0.7
    slick_vol = max(released_true, 0.05) * float(np.exp(rng.normal(0, 0.5)))
    post = leak.posterior(hs, r_mean, r_cov, comps, dp_obs, dp_cov, dp_sens, slick_xy, loc_sd, drift_est,
                          lambda j, lam: lam * n_win * u.DAY * 0.7, slick_vol)
    # which hypothesis is "the truth"
    if kind[i] == 0:
        truth = 0
    elif kind[i] == network.JUMPER:
        truth = next(j for j, h in enumerate(hs) if h.kind == 'jumper' and h.well == slot[i])
    elif kind[i] == network.MANIFOLD:
        truth = next(j for j, h in enumerate(hs) if h.kind == 'manifold')
    else:
        name = 'flowline' if kind[i] == network.FLOWLINE else 'riser'
        cands = [(abs(h.pos - pos[i]), j) for j, h in enumerate(hs) if h.kind == name]
        truth = min(cands)[1]
    ranked = [hs.index(o['hypothesis']) for o in post]
    leak_frac = float(gi[gi.day == t_d].leak_m.iloc[0] / max(d[(d.group == i) & (d.day == t_d)].true_m.sum(), 1e-12))
    rec = {'scenario': i, 'kind': int(kind[i]), 'leak_frac': leak_frac, 'alarm': alarm, 'onset': int(onset[i]),
           'detect': t_d, 'top1': ranked[0] == truth, 'top3': truth in ranked[:3],
           'p_truth': next(o['posterior'] for o in post if hs.index(o['hypothesis']) == truth),
           'p_none': next(o['posterior'] for o in post if o['hypothesis'].kind == 'none'),
           'top_kind': post[0]['hypothesis'].kind, 'kind_correct': post[0]['hypothesis'].kind == hs[truth].kind}
    records.append(rec)
    if demo is None and kind[i] == network.JUMPER and 0.01 < leak_frac < 0.04:
        demo = {'scenario': i, 'truth': f'jumper of well slot {slot[i]}', 'leak_frac': leak_frac,
                'onset_true': int(onset[i]), 'onset_cusum': t_s, 'detect': t_d,
                'table': [{'hypothesis': f"{o['hypothesis'].kind}"
                                         + (f" slot {o['hypothesis'].well}" if o['hypothesis'].kind == 'jumper' else '')
                                         + (f" pos {o['hypothesis'].pos:.1f}" if o['hypothesis'].kind in
                                            ('flowline', 'riser') else ''),
                           'posterior': round(o['posterior'], 4), 'leak_oil_stb_d': round(float(
                               u.si_to_stb_d(o['lambda'])), 1),
                           'll_phase': round(o['ll_phase'], 2), 'll_pressure': round(o['ll_dp'], 2),
                           'll_location': round(o['ll_loc'], 2), 'll_volume': round(o['ll_vol'], 2)}
                          for o in post[:8]]}

rec = pd.DataFrame(records)
rec['alarm'] = pd.to_numeric(rec.alarm)
rec.to_csv('m6_scenarios.csv', index=False)
lk_rows = rec[rec.kind > 0]
bins = pd.cut(lk_rows.leak_frac, [0, 0.01, 0.02, 0.05, 1.0])
summary = {
    'leaks': int(len(lk_rows)), 'controls': int((rec.kind == 0).sum()),
    'top1': float(lk_rows.top1.mean()), 'top3': float(lk_rows.top3.mean()),
    'kind_correct': float(lk_rows.kind_correct.mean()),
    'by_kind': lk_rows.groupby('kind')[['top1', 'top3', 'kind_correct']].mean().round(3).to_dict('index'),
    'by_size': lk_rows.groupby(bins, observed=True)[['top1', 'top3']].mean().round(3).astype(float).to_dict('index'),
    'alarm_rate_leaks': float(lk_rows.alarm.notna().mean()),
    'alarm_delay_median_days': float((lk_rows.alarm - lk_rows.onset).median()),
    'false_alarm_rate_controls': float(rec[rec.kind == 0].alarm.notna().mean()),
    'controls_none_top1': float(rec[rec.kind == 0].top1.mean()),
    'target_top3_ge_2pct': float(lk_rows[lk_rows.leak_frac >= 0.02].top3.mean()),
    'demo': demo, 'seconds': round(time.time() - T0, 1),
}
summary['by_size'] = {str(k): v for k, v in summary['by_size'].items()}
with open('m6_summary.json', 'w') as fh:
    json.dump(summary, fh, indent=1, default=str)
print(json.dumps(summary, indent=1, default=str))
