# B2: Dabiri et al. 2024, 565 wellhead-choke tests. Protocol in BENCHMARKS.md ("B2 run protocol").
import json
import sys
import time

sys.path.insert(0, 'src')
sys.path.insert(0, '.')
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import lightgbm as lgb  # noqa: E402
from sklearn.ensemble import AdaBoostRegressor  # noqa: E402
from sklearn.model_selection import KFold, train_test_split  # noqa: E402
from sklearn.preprocessing import MinMaxScaler  # noqa: E402
from sklearn.svm import SVR  # noqa: E402
from scipy import stats  # noqa: E402

from benchmarks import dabiri  # noqa: E402
from vfmchoke import choke, pvt, units as u  # noqa: E402

T0 = time.time()
data, url = dabiri.download()
rows = dabiri.tables(data)[0]
df = pd.DataFrame(rows[1:], columns=['no', 'p_psi', 'glr', 'd64', 'qliq']).astype(float)
assert len(df) == 565, len(df)
print('rows', len(df), 'from', url)


def aapre(pred, true):
    return float(np.mean(np.abs(pred - true) / true) * 100)


def r2(pred, true):
    return float(1 - np.sum((true - pred) ** 2) / np.sum((true - true.mean()) ** 2))


def physics(p_psi, d64, glr):
    """Zero-shot mechanistic choke rate (stb/d), pre-registered fluid defaults, critical flow."""
    f = pvt.Fluid(api=30.0, gas_sg=0.75, gor=u.scf_stb_to_si(glr), wc=0.0, salinity=0.0)
    m = choke.mass_rate(u.psig_to_pa(p_psi), u.P_SC, u.c_to_k(60.0), f, 1.0, u.choke64_to_m(d64), 0.80)
    return u.si_to_stb_d(m / pvt.mass_per_liquid_sc(f))


P, D, GLR, Q = df.p_psi.to_numpy(), df.d64.to_numpy(), df.glr.to_numpy(), df.qliq.to_numpy()
q_phys = physics(P, D, GLR)
results = {'zero_shot': {}}
results['zero_shot']['physics'] = {'aapre': aapre(q_phys, Q), 'r2': r2(q_phys, Q),
                                   'median_ratio': float(np.median(q_phys / Q))}
for name in choke.GILBERT_FAMILY:
    q = choke.gilbert_rate(P, D, GLR, name)
    results['zero_shot'][name] = {'aapre': aapre(q, Q), 'r2': r2(q, Q)}
print(json.dumps(results['zero_shot'], indent=1))

X_log = np.column_stack([np.log(P), np.log(D), np.log(GLR)])
resid_all = np.log(Q / q_phys)


def feats_gbm(idx):
    return np.column_stack([X_log[idx], np.log(q_phys[idx])])


def fit_lin(idx):
    a = np.column_stack([np.ones(len(idx)), X_log[idx]])
    beta, *_ = np.linalg.lstsq(a, resid_all[idx], rcond=None)
    return lambda j: q_phys[j] * np.exp(np.column_stack([np.ones(len(j)), X_log[j]]) @ beta)


def fit_gbm(idx, seed=0):
    m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=7, min_child_samples=10, subsample=0.8,
                          subsample_freq=1, random_state=seed, verbose=-1)
    m.fit(feats_gbm(idx), resid_all[idx])
    return lambda j: q_phys[j] * np.exp(m.predict(feats_gbm(j)))


def fit_power(idx):
    a = np.column_stack([np.ones(len(idx)), X_log[idx]])
    beta, *_ = np.linalg.lstsq(a, np.log(Q[idx]), rcond=None)
    return lambda j: np.exp(np.column_stack([np.ones(len(j)), X_log[j]]) @ beta)


def fit_adaboost_svr(idx, seed=0):
    xs, ys = MinMaxScaler(), MinMaxScaler()
    x = xs.fit_transform(df[['p_psi', 'glr', 'd64']].to_numpy()[idx])
    y = ys.fit_transform(Q[idx, None]).ravel()
    m = AdaBoostRegressor(estimator=SVR(C=1.0, epsilon=0.1, kernel='rbf', gamma='scale'), random_state=seed)
    m.fit(x, y)
    return lambda j: ys.inverse_transform(m.predict(xs.transform(df[['p_psi', 'glr', 'd64']].to_numpy()[j]))[:, None]
                                          ).ravel()


def fit_gbm_raw(idx, seed=0):
    m = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=15, min_child_samples=5, subsample=0.8,
                          subsample_freq=1, random_state=seed, verbose=-1)
    m.fit(X_log[idx], np.log(Q[idx]))
    return lambda j: np.exp(m.predict(X_log[j]))


def choose_ours(idx, seed):
    """Pick hybrid-lin or hybrid-gbm by 5-fold CV inside the training part only."""
    scores = {'hybrid_lin': [], 'hybrid_gbm': []}
    for tr, va in KFold(5, shuffle=True, random_state=seed).split(idx):
        scores['hybrid_lin'].append(aapre(fit_lin(idx[tr])(idx[va]), Q[idx[va]]))
        scores['hybrid_gbm'].append(aapre(fit_gbm(idx[tr], seed)(idx[va]), Q[idx[va]]))
    best = min(scores, key=lambda k: np.mean(scores[k]))
    return best, (fit_lin(idx) if best == 'hybrid_lin' else fit_gbm(idx, seed))


per_seed = []
all_idx = np.arange(len(df))
for seed in range(20):
    tr, te = train_test_split(all_idx, test_size=0.2, random_state=seed)
    row = {'seed': seed}
    choice, ours = choose_ours(tr, seed)
    row['ours_choice'] = choice
    row['ours'] = aapre(ours(te), Q[te])
    row['hybrid_lin'] = aapre(fit_lin(tr)(te), Q[te])
    row['hybrid_gbm'] = aapre(fit_gbm(tr, seed)(te), Q[te])
    row['power_law_refit'] = aapre(fit_power(tr)(te), Q[te])
    row['adaboost_svr_replicate'] = aapre(fit_adaboost_svr(tr, seed)(te), Q[te])
    row['gbm_raw'] = aapre(fit_gbm_raw(tr, seed)(te), Q[te])
    row['physics_zero_shot_on_test'] = aapre(q_phys[te], Q[te])
    per_seed.append(row)
    print(row, flush=True)
ps = pd.DataFrame(per_seed)
ps.to_csv('b2_per_seed.csv', index=False)


def mean_ci(x):
    x = np.asarray(x, float)
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return {'mean': float(x.mean()), 'lo': float(x.mean() - h), 'hi': float(x.mean() + h), 'sd': float(x.std(ddof=1))}


results['split_80_20'] = {c: mean_ci(ps[c]) for c in ps.columns if c not in ('seed', 'ours_choice')}
results['split_80_20']['ours_choice_counts'] = ps.ours_choice.value_counts().to_dict()

# extra (not the gate): leakage-safe 80/20 where exact-duplicate rows stay on the same side
from sklearn.model_selection import GroupShuffleSplit  # noqa: E402
from sklearn.neighbors import KNeighborsRegressor  # noqa: E402
dup_group = df.groupby(['p_psi', 'glr', 'd64', 'qliq']).ngroup().to_numpy()
safe = []
for seed in range(20):
    tr, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=seed).split(all_idx, groups=dup_group))
    row = {'seed': seed}
    choice, ours = choose_ours(tr, seed)
    row['ours'] = aapre(ours(te), Q[te])
    row['gbm_raw'] = aapre(fit_gbm_raw(tr, seed)(te), Q[te])
    row['adaboost_svr_replicate'] = aapre(fit_adaboost_svr(tr, seed)(te), Q[te])
    row['power_law_refit'] = aapre(fit_power(tr)(te), Q[te])
    safe.append(row)
sf = pd.DataFrame(safe)
results['split_80_20_dedup_groups'] = {c: mean_ci(sf[c]) for c in sf.columns if c != 'seed'}
# how much a pure memoriser gains from duplicates under the published random protocol
memo = []
for seed in range(20):
    tr, te = train_test_split(all_idx, test_size=0.2, random_state=seed)
    knn = KNeighborsRegressor(n_neighbors=1).fit(X_log[tr], np.log(Q[tr]))
    memo.append(aapre(np.exp(knn.predict(X_log[te])), Q[te]))
results['one_nearest_neighbour_random_split'] = mean_ci(memo)
results['duplicate_rows'] = int((pd.Series(dup_group).map(pd.Series(dup_group).value_counts()) > 1).sum())

rng = np.random.default_rng(0)
few = {}
for k in (0, 1, 3, 10):
    vals = []
    for _ in range(50 if k else 1):
        cal = rng.choice(all_idx, k, replace=False) if k else np.array([], int)
        rest = np.setdiff1d(all_idx, cal)
        mult = np.exp(np.mean(resid_all[cal])) if k else 1.0
        vals.append(aapre(q_phys[rest] * mult, Q[rest]))
    few[k] = {'mean': float(np.mean(vals)), 'p10': float(np.percentile(vals, 10)), 'p90': float(np.percentile(vals, 90))}
results['few_shot_global_multiplier'] = few

published = {'adaboost_svr_test': 4.57, 'best_correlation_whole_set': 17.20, 'gilbert_whole_set': 22.36}
s = results['split_80_20']
results['gate'] = {
    'zero_shot_beats_all_published_correlations': results['zero_shot']['physics']['aapre'] < 17.20,
    'beat_outright_80_20': s['ours']['hi'] < 4.57,
    'few_shot_k3_parity': few[3]['mean'] <= 4.57,
    'published': published,
    'seconds': round(time.time() - T0, 1),
}
with open('b2_results.json', 'w') as fh:
    json.dump(results, fh, indent=1, default=str)
pd.DataFrame({'p_psi': P, 'd64': D, 'glr': GLR, 'qliq': Q, 'q_physics_zero_shot': q_phys}).to_csv(
    'b2_zero_shot_predictions.csv', index=False)
print(json.dumps(results, indent=1, default=str))
