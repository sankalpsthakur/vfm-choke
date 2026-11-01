# M3: the general model with few-shot calibration against baselines, on wells held out by manifold group.
# Labels for calibration come only from days < 90; every method is scored on days 90-179 against the truth.
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, 'src')
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import lightgbm as lgb  # noqa: E402

from vfmchoke import choke, fleet, hybrid, pvt, units as u  # noqa: E402

T0 = time.time()
import glob  # noqa: E402
IN = str(Path(sorted(glob.glob('/kaggle/input/**/simfield_daily.parquet', recursive=True), key=lambda p: ('m2b' not in p))[0]).parent)
print('simulation input:', IN, flush=True)
daily, tests, wells = fleet.prepare(pd.read_parquet(f'{IN}/simfield_daily.parquet'),
                                    pd.read_parquet(f'{IN}/simfield_tests.parquet'),
                                    pd.read_parquet(f'{IN}/simfield_wells.parquet'))
CAL_END = 90
KS = {'mpfm': [0, 1, 2, 3, 5, 10, 30], 'test': [0, 1, 2, 3]}
FOLDS = 5
RNG = np.random.default_rng(0)
RAW = ['opening', 'd_choke_mm', 'vendor_gamma', 'p1_bar', 'p2_bar', 't1_c', 'api', 'gor', 'wc']


def log(m):
    print(f'[{time.time() - T0:7.0f}s] {m}', flush=True)


def raw_feats(df, fl):
    return pd.DataFrame({'opening': df.opening.to_numpy(), 'd_choke_mm': df.d_choke.to_numpy() * 1e3,
                         'vendor_gamma': df.vendor_gamma.to_numpy(), 'p1_bar': df.p_wh.to_numpy() / 1e5,
                         'p2_bar': df.p_man.to_numpy() / 1e5, 't1_c': df.t_wh.to_numpy() - 273.15,
                         'api': fl.api, 'gor': fl.gor, 'wc': fl.wc}, index=df.index)


def gilbert_mass(df, fl):
    s_eff = u.m_to_choke64(df.d_choke.to_numpy()) * np.sqrt(choke.trim(df.opening.to_numpy(),
                                                                         df.vendor_gamma.to_numpy()))
    q_l = u.stb_d_to_si(choke.gilbert_rate(u.pa_to_psig(df.p_wh.to_numpy()), s_eff,
                                           u.si_to_scf_stb(fl.gor * (1 - fl.wc))))
    return q_l * pvt.mass_per_liquid_sc(fl)


def mape(pred, true):
    return float(np.mean(np.abs(pred - true) / true))


groups = np.sort(daily.group.unique())
RNG.shuffle(groups)
fold_of = {gid: i % FOLDS for i, gid in enumerate(groups)}
daily['fold'] = daily.group.map(fold_of)
records, priors = [], []
for fold in range(FOLDS):
    tr = daily[(daily.fold != fold) & daily.flowing]
    trained = fleet.train(tr, tests)
    priors.append({'fold': fold, **trained.prior.to_dict()})
    log(f'fold {fold}: trained on {tr.well_id.nunique()} wells; prior {trained.prior.to_dict()}')
    trf = fleet.with_est(tr, tr.trail_wc, tr.trail_gor)
    bb = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=31, min_child_samples=200,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.9, verbose=-1)
    bb.fit(raw_feats(trf, hybrid.fluid_from(trf))[RAW], np.log(trf.label_m.to_numpy()))
    te = daily[daily.fold == fold]
    for wid, tw in te.groupby('well_id'):
        ev = tw[(tw.day >= CAL_END) & tw.flowing]
        if len(ev) < 5:
            continue
        drow = wells.loc[wid]
        for regime, ks in KS.items():
            src = tw[tw.flowing & (tw.day < CAL_END)] if regime == 'mpfm' else \
                tests[(tests.well_id == wid) & (tests.day < CAL_END)]
            src = src.sort_values('day')
            for k in ks:
                cal_rows = src.head(k)
                cal = fleet.calibrate(cal_rows, drow, trained, regime)
                m_h, lo, hi, fl = fleet.predict(ev, cal, trained)
                truth_m, truth_l = ev.true_m.to_numpy(), ev.true_q_l.to_numpy()
                preds = {'hybrid': m_h}
                e = fleet.with_est(ev, cal.wc, cal.gor)
                m_phys, _ = hybrid.prior(e, fl)
                m_gil = gilbert_mass(e, fl)
                m_bb = np.exp(bb.predict(raw_feats(e, fl)[RAW]))
                if len(cal_rows):           # a well can have fewer labels than k (e.g. no test before day 90)
                    c = fleet.with_est(cal_rows, cal.wc, cal.gor)
                    flc = hybrid.fluid_from(c)
                    lab = c.label_m.to_numpy()
                    mult = lambda m_cal: np.exp(np.mean(np.log(lab / m_cal)))
                    m_phys = m_phys * mult(hybrid.prior(c, flc)[0])
                    m_gil = m_gil * mult(gilbert_mass(c, flc))
                    m_bb = m_bb * mult(np.exp(bb.predict(raw_feats(c, flc)[RAW])))
                    if regime == 'mpfm':
                        if len(cal_rows) >= 5:
                            pw = lgb.LGBMRegressor(n_estimators=60, learning_rate=0.1, num_leaves=7,
                                                   min_child_samples=2, verbose=-1)
                            cols = ['opening', 'p_wh', 'p_man', 't_wh']
                            pw.fit(c[cols], np.log(lab))
                            preds['perwell_gbm'] = np.exp(pw.predict(e[cols]))
                        else:
                            preds['perwell_gbm'] = np.full(len(e), np.exp(np.mean(np.log(lab))))
                preds.update({'physics': m_phys, 'gilbert': m_gil, 'pooled_blackbox': m_bb})
                for meth, m_hat in preds.items():
                    q_o, q_g, q_w = choke.std_rates(m_hat, fl)
                    rec = {'fold': fold, 'regime': regime, 'k': k, 'k_used': int(len(cal_rows)), 'method': meth,
                           'well_id': int(wid),
                           'mape_liquid': mape(q_o + q_w, truth_l), 'mape_mass': mape(m_hat, truth_m),
                           'mape_oil': mape(q_o, np.maximum(ev.true_q_o.to_numpy(), fleet.MIN_Q)),
                           'mape_gas': mape(q_g, np.maximum(ev.true_q_g.to_numpy(), 1e-6)),
                           'err_water_over_liquid': float(np.mean(np.abs(q_w - ev.true_q_w.to_numpy()) / truth_l)),
                           'n_days': int(len(ev))}
                    if meth == 'hybrid':
                        rec['coverage90'] = float(np.mean((truth_m >= lo) & (truth_m <= hi)))
                    records.append(rec)
    log(f'fold {fold} evaluated')

res = pd.DataFrame(records)
res.to_parquet('m3_per_well.parquet', index=False)
pd.DataFrame(priors).to_csv('m3_eb_prior.csv', index=False)


def boot_median(x, n=2000):
    x = np.asarray(x)
    meds = np.median(RNG.choice(x, (n, len(x)), replace=True), axis=1)
    return float(np.median(x)), float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


rows = []
for (regime, k, meth), gdf in res.groupby(['regime', 'k', 'method']):
    row = {'regime': regime, 'k': int(k), 'method': meth, 'wells': int(len(gdf))}
    for ph in ('liquid', 'mass', 'oil', 'gas'):
        row[f'{ph}_median'], row[f'{ph}_lo'], row[f'{ph}_hi'] = boot_median(gdf['mape_' + ph])
    row['water_err_median'] = float(gdf.err_water_over_liquid.median())
    if meth == 'hybrid':
        row['coverage90_mean'] = float(gdf.coverage90.mean())
    rows.append(row)
summary = pd.DataFrame(rows).sort_values(['regime', 'k', 'liquid_median'])
summary.to_csv('m3_summary.csv', index=False)
print(summary.to_string())

s = summary[summary.regime == 'mpfm'].set_index(['method', 'k'])
gate = {}
for base in ('gilbert', 'physics'):
    gate[f'k0_beats_{base}'] = bool(s.loc[('hybrid', 0)].liquid_hi < s.loc[(base, 0)].liquid_lo)
ref_name = min(('perwell_gbm', 'physics'), key=lambda mth: s.loc[(mth, 30)].liquid_median)
ref = float(s.loc[(ref_name, 30)].liquid_median)
gate['reference'] = f'{ref_name} at k=30: {ref:.4f}'
for k in (1, 2, 3):
    gate[f'k{k}_hybrid_over_reference'] = float(s.loc[('hybrid', k)].liquid_median / ref)
    gate[f'k{k}_within_1.25x'] = bool(s.loc[('hybrid', k)].liquid_median <= 1.25 * ref)
cov = summary[summary.method == 'hybrid'].coverage90_mean
gate['coverage_by_k'] = summary[summary.method == 'hybrid'][['regime', 'k', 'coverage90_mean']].to_dict('records')
gate['coverage_85_95_all_k'] = bool(((cov >= 0.85) & (cov <= 0.95)).all())
gate['seconds'] = round(time.time() - T0, 1)
with open('m3_gate.json', 'w') as fh:
    json.dump(gate, fh, indent=1)
print(json.dumps(gate, indent=1))
