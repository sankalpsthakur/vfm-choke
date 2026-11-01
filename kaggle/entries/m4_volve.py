# B1: Volve walk-forward. Protocol in BENCHMARKS.md ("B1 run protocol").
import json
import sys
import time

sys.path.insert(0, 'src')
sys.path.insert(0, '.')
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402

from benchmarks import volve  # noqa: E402
from vfmchoke import choke, fewshot, network, pvt, units as u  # noqa: E402

T0 = time.time()
R = 30            # recalibration interval, days
START = 60        # first predicted day of each well
D_NOM = 0.0508    # nominal bean, m (the latent absorbs the real size)
API, GSG, SAL = 30.0, 0.75, 0.035
PRIOR = fewshot.LatentPrior(tau0=2.0, tau1=0.5, q0=1e-4, sig_mpfm=0.10, sig_test=0.10, sig_model=0.05)

df = volve.load()
print(df.groupby('well').agg(days=('day', 'size'), start=('date', 'min'), end=('date', 'max')).to_string())
print('records (5 producers):', len(df), '| 4-well set:', int(df.well.isin(volve.FOUR).sum()))


def fluid(wc, gor):
    return pvt.Fluid(api=API, gas_sg=GSG, gor=gor, wc=wc, salinity=SAL)


def prior_mass(row, fl):
    p1 = (row.p_wh_barg + 1.01325) * 1e5
    p2 = p1 - row.dp_choke_bar * 1e5
    return float(choke.mass_rate(p1, p2, row.t_wh_c + 273.15, fl, row.opening, D_NOM, 0.80, 1.0))


def walk_forward(d, anchor_only):
    """Physics prior + per-well latent (b0, b1) and ratio trackers, frozen between recalibrations."""
    x, p = np.zeros(2), np.diag([PRIOR.tau0 ** 2, PRIOR.tau1 ** 2])
    zw, vw = fewshot.logit(0.05), 2.0 ** 2           # water cut, logit space
    zg, vg = np.log(150.0), 1.0 ** 2                 # GOR, log space
    t_last = None
    frozen = None
    next_recal = START
    used_windows = set()
    preds = []
    for row in d.itertuples(index=False):
        while row.day >= next_recal:
            frozen = (x.copy(), p.copy(), t_last, zw, zg)
            next_recal += R
        fl_live = fluid(float(fewshot.expit(zw)), float(np.exp(zg)))
        if frozen is not None and row.day >= START:
            fx, fp, ft, fzw, fzg = frozen
            fl = fluid(float(fewshot.expit(fzw)), float(np.exp(fzg)))
            m0 = prior_mass(row, fl)
            phi = np.array([1.0, np.log(max(row.opening, 1e-3))])
            mean, _ = fewshot.latent_predict(fx, fp, ft, [row.day], phi[None, :], PRIOR)
            q_o, _, _ = choke.std_rates(m0 * np.exp(mean[0]), fl)
            preds.append({'well': row.well, 'date': row.date, 'day': row.day, 'pred_q_o': float(q_o),
                          'q_o': row.q_o})
        # the day's label becomes available for later recalibrations
        window = row.day // R
        if anchor_only and window in used_windows:
            continue
        used_windows.add(window)
        wc_obs = row.q_w / (row.q_o + row.q_w)
        gor_obs = row.q_g / row.q_o
        dt = row.day - (t_last if t_last is not None else row.day)
        vw, vg = vw + 1e-3 * dt, vg + 1e-3 * dt
        k = vw / (vw + 0.1 ** 2)
        zw, vw = zw + k * (fewshot.logit(min(max(wc_obs, 1e-4), 0.999)) - zw), (1 - k) * vw
        k = vg / (vg + 0.1 ** 2)
        zg, vg = zg + k * (np.log(max(gor_obs, 1.0)) - zg), (1 - k) * vg
        label_m = float(network.std_mass(row.q_o, row.q_g, row.q_w, fl_live))
        r = np.log(label_m / max(prior_mass(row, fl_live), 1e-9))
        h = np.array([1.0, np.log(max(row.opening, 1e-3))])
        if t_last is not None:
            p = p + np.diag([PRIOR.q0 * (row.day - t_last), 0.0])
        s = h @ p @ h + PRIOR.sig_mpfm ** 2
        kk = p @ h / s
        x = x + kk * (r - h @ x)
        p = p - np.outer(kk, h) @ p
        t_last = row.day
    return pd.DataFrame(preds)


def metrics(pr):
    e = np.abs(pr.pred_q_o - pr.q_o) / pr.q_o
    out = {'mape_pct': float(e.mean() * 100), 'r2': float(1 - ((pr.q_o - pr.pred_q_o) ** 2).sum()
                                                         / ((pr.q_o - pr.q_o.mean()) ** 2).sum()),
           'rmse_sm3_d': float(np.sqrt(((pr.q_o - pr.pred_q_o) ** 2).mean())), 'n': int(len(pr))}
    out['per_well_mape_pct'] = {w: float((np.abs(g.pred_q_o - g.q_o) / g.q_o).mean() * 100)
                                for w, g in pr.groupby('well')}
    return out


results = {}
for mode, anchor in (('physics_latent_all_past', False), ('physics_latent_anchor_30d', True)):
    pr = pd.concat([walk_forward(g, anchor) for _, g in df.groupby('well')], ignore_index=True)
    pr.to_csv(f'b1_{mode}.csv', index=False)
    results[mode] = {'five_wells': metrics(pr), 'four_wells': metrics(pr[pr.well.isin(volve.FOUR)])}
    print(mode, json.dumps(results[mode]['four_wells'], indent=1), flush=True)

# replicate of the published data-driven approach: pooled gradient boosting on log rates, walk-forward, 30-day refits
four = df[df.well.isin(volve.FOUR)].copy()
four['well_code'] = four.well.astype('category').cat.codes
four['t_days'] = (four.date - four.date.min()).dt.days
feat = ['p_wh_barg', 'opening', 't_wh_c', 'well_code', 'day']
preds = []
cut = four.date.min() + pd.Timedelta(days=START)
while cut <= four.date.max():
    trn = four[four.date < cut]
    tst = four[(four.date >= cut) & (four.date < cut + pd.Timedelta(days=R)) & (four.day >= START)]
    if len(tst) and len(trn) > 50:
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0)
        m.fit(trn[feat], np.log(trn.q_o))
        preds.append(tst.assign(pred_q_o=np.exp(m.predict(tst[feat]))))
    cut += pd.Timedelta(days=R)
pr = pd.concat(preds)
pr.to_csv('b1_gbm_walk_forward_replicate.csv', index=False)
results['gbm_walk_forward_replicate'] = {'four_wells': metrics(pr)}

# head-to-head on identical rows, with a block bootstrap over (well, 30-day window) blocks
ours_pr = pd.read_csv('b1_physics_latent_all_past.csv', parse_dates=['date'])
both = ours_pr.merge(pr[['well', 'date', 'pred_q_o']].rename(columns={'pred_q_o': 'pred_gbm'}), on=['well', 'date'])
both['ape_ours'] = np.abs(both.pred_q_o - both.q_o) / both.q_o
both['ape_gbm'] = np.abs(both.pred_gbm - both.q_o) / both.q_o
both['block'] = both.well + '_' + (both.day // R).astype(str)
blocks = both.groupby('block')[['ape_ours', 'ape_gbm']].agg(['sum', 'count'])
rng = np.random.default_rng(0)
bs = []
names = blocks.index.to_numpy()
for _ in range(2000):
    pick = blocks.loc[rng.choice(names, len(names), replace=True)]
    n = pick[('ape_ours', 'count')].sum()
    bs.append((pick[('ape_ours', 'sum')].sum() / n * 100, pick[('ape_gbm', 'sum')].sum() / n * 100))
bs = np.array(bs)
results['head_to_head_identical_rows'] = {
    'n': int(len(both)), 'ours_mape_pct': float(both.ape_ours.mean() * 100),
    'ours_ci95': [float(np.percentile(bs[:, 0], 2.5)), float(np.percentile(bs[:, 0], 97.5))],
    'gbm_replicate_mape_pct': float(both.ape_gbm.mean() * 100),
    'gbm_ci95': [float(np.percentile(bs[:, 1], 2.5)), float(np.percentile(bs[:, 1], 97.5))],
    'diff_ci95': [float(np.percentile(bs[:, 0] - bs[:, 1], 2.5)), float(np.percentile(bs[:, 0] - bs[:, 1], 97.5))],
}
print(json.dumps(results['head_to_head_identical_rows'], indent=1))
ours_all = ours_pr[ours_pr.well.isin(volve.FOUR)].assign(ape=lambda d: np.abs(d.pred_q_o - d.q_o) / d.q_o,
                                                         block=lambda d: d.well + '_' + (d.day // R).astype(str))
ob = ours_all.groupby('block').ape.agg(['sum', 'count'])
obs = []
for _ in range(2000):
    pick = ob.loc[rng.choice(ob.index.to_numpy(), len(ob), replace=True)]
    obs.append(pick['sum'].sum() / pick['count'].sum() * 100)
results['physics_latent_all_past']['four_wells']['mape_ci95'] = [float(np.percentile(obs, 2.5)),
                                                                 float(np.percentile(obs, 97.5))]
# frozen temporal split: train on each well's first 70 %, never update
four['rank'] = four.groupby('well').date.rank(pct=True)
trn, tst = four[four['rank'] <= 0.7], four[four['rank'] > 0.7]
m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0).fit(trn[feat], np.log(trn.q_o))
results['gbm_frozen_replicate'] = {'four_wells': metrics(tst.assign(pred_q_o=np.exp(m.predict(tst[feat]))))}
print(json.dumps({k: v['four_wells'] for k, v in results.items() if 'four_wells' in v}, indent=1))

published = {'walk_forward_mape_pct': 18.0, 'walk_forward_r2': 0.93, 'walk_forward_rmse': 333.0,
             'frozen_mape_pct': 66.0}
rep = results['gbm_walk_forward_replicate']['four_wells']['mape_pct']
ours = results['physics_latent_all_past']['four_wells']
results['gate'] = {
    'protocol_reconstruction_ok': abs(rep - 18.0) <= 5.0,
    'beat_outright_walk_forward': ours['mape_ci95'][1] < 18.0,
    'anchor_only_below_18': results['physics_latent_anchor_30d']['four_wells']['mape_pct'] < 18.0,
    'published': published, 'seconds': round(time.time() - T0, 1),
}
with open('b1_results.json', 'w') as fh:
    json.dump(results, fh, indent=1, default=str)
print(json.dumps(results['gate'], indent=1, default=str))
