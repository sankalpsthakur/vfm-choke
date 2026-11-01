# %% [markdown]
# # Reproducing Two Virtual Flow Metering Benchmarks
#
# **TL;DR**
# - **Dabiri et al. 2024 (*Sci Rep*), 565 wellhead-choke tests.** The released data match the paper exactly (all 40
#   entries of Table 3; four of the five correlation errors in Table 7). The headline result does not reproduce:
#   Adaboost-SVR with the published settings averages 20.3% test AAPRE at best over 20 random 80/20 splits, against
#   the reported 4.57%. Across 200 splits, the best model we tried (LightGBM) never goes below 7.1%. 124 of the 565
#   rows sit in 44 groups of exact duplicates, so about 20% of a random test set has a copy in training.
# - **D'Souza et al. 2024 (SIMS EUROSIM), two simulated gas-lifted wells.** The CSVs behind Table 1 are not released,
#   and neither released copy of the simulator reproduces them. In both, the test window returns to the first step's
#   exact state, where the paper's figures show a new level, so all ten of the paper's models score below 1% MAPE on
#   regenerated data (Table 1 reports 1.5-16.7%). As released, the simulation script also asks a helper for 20
#   outputs when it declares 17, which is an error in MATLAB and Octave.
# - Our own first pass got the D'Souza part wrong; section 2.4 says how. Questions for the authors are at the end.
#
# We wanted public benchmarks for a wellhead-choke flow model, so before treating these published numbers as bars we
# tried to reproduce them. Both papers released data or code, which is what made the check possible. Every number
# below is computed in this notebook from sources it downloads at run time.

# %% [markdown]
# ## Setup

# %%
import io
import json
import re
import subprocess
import time
import urllib.request
import warnings
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 30)
pd.set_option('display.max_colwidth', 60)
T0 = time.time()
SRC = Path('/tmp/sources')      # third-party files: fetched at run time, never written to the notebook's outputs
SRC.mkdir(parents=True, exist_ok=True)
RESULTS = {}
UA = {'User-Agent': 'Mozilla/5.0 (benchmark reproducibility check)'}


def fetch(url, timeout=120):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
        return r.read()


def num(s):
    """A number as printed in a paper or DOCX table: unicode minus, thousands separators, stray spaces."""
    try:
        return float(str(s).replace('−', '-').replace(',', '').replace(' ', ''))
    except ValueError:
        return np.nan


def aapre(pred, true):
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    return float(np.mean(np.abs(pred - true) / np.abs(true)) * 100)


def rmse(pred, true):
    return float(np.sqrt(np.mean((np.asarray(pred, float) - np.asarray(true, float)) ** 2)))


def r2(pred, true):
    """Coefficient of determination, 1 - SSE/SST."""
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    return float(1 - np.sum((true - pred) ** 2) / np.sum((true - true.mean()) ** 2))


def r2_corr(pred, true):
    """Squared Pearson correlation, the other common reading of 'R2'."""
    return float(np.corrcoef(np.asarray(pred, float), np.asarray(true, float))[0, 1] ** 2)


def mean_ci(x):
    x = np.asarray(x, float)
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return {'mean': float(x.mean()), 'ci_lo': float(x.mean() - h), 'ci_hi': float(x.mean() + h),
            'min': float(x.min()), 'max': float(x.max())}


def save():
    Path('audit_results.json').write_text(json.dumps(RESULTS, indent=1, default=float))

# %% [markdown]
# ## 1. Dabiri et al. 2024: 565 wellhead-choke tests
#
# Dabiri, M.-S. et al. (2024). Modeling liquid rate through wellhead chokes using machine learning techniques.
# *Scientific Reports* 14, 6945. [doi:10.1038/s41598-024-54010-2](https://doi.org/10.1038/s41598-024-54010-2)
# (CC BY 4.0).
#
# The paper fits six machine-learning models and a new correlation to 565 liquid-rate measurements compiled from
# eight literature sources. Inputs are wellhead pressure (psi), gas-liquid ratio (scf/stb) and choke size (1/64 in);
# the target is liquid rate (stb/d). Its best model, Adaboost-SVR, reports a test-set AAPRE of 4.57% on a random
# 80/20 split (Table 6). The data are in the supplementary DOCX.

# %%
DOCX_URLS = [
    'https://static-content.springer.com/esm/art%3A10.1038%2Fs41598-024-54010-2/MediaObjects/'
    '41598_2024_54010_MOESM1_ESM.docx',
    'https://pmc.ncbi.nlm.nih.gov/articles/instance/10960849/bin/41598_2024_54010_MOESM1_ESM.docx',
]
WNS = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'


def docx_tables(blob):
    root = ET.fromstring(zipfile.ZipFile(io.BytesIO(blob)).read('word/document.xml'))
    return [[[''.join(t.text or '' for t in tc.iter(WNS + 't')).strip() for tc in tr.iter(WNS + 'tc')]
             for tr in tbl.iter(WNS + 'tr')] for tbl in root.iter(WNS + 'tbl')]


docx = None
for url in DOCX_URLS:
    try:
        blob = fetch(url)
    except Exception as e:
        print('mirror failed:', url, e)
        continue
    if blob[:2] == b'PK':
        docx = blob
        print(f'supplementary file: {url} ({len(blob):,} bytes)')
        break
supp = docx_tables(docx)
print(len(supp), 'tables; table S1 header:', supp[0][0])
df = pd.DataFrame([[num(c) for c in r] for r in supp[0][1:]], columns=['no', 'p_psi', 'glr', 'd64', 'qliq'])
assert len(df) == 565 and df.notna().all().all()
P, GLR, D, Q = (df[c].to_numpy() for c in ('p_psi', 'glr', 'd64', 'qliq'))
idx = np.arange(len(df))
df.drop(columns='no').describe().round(1)

# %% [markdown]
# The paper's own tables come from its open-access XML on Europe PMC.

# %%
paper = fetch('https://www.ebi.ac.uk/europepmc/webservices/rest/PMC10960849/fullTextXML').decode('utf-8')


def jats_table(tab_id):
    block = re.search(r'<table-wrap[^>]*id="%s".*?</table-wrap>' % tab_id, paper, re.S).group(0)
    return [[re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', ' ', cell)).strip()
             for cell in re.findall(r'<t[dh](?:\s[^>]*)?>(.*?)</t[dh]>', tr, re.S)]
            for tr in re.findall(r'<tr(?:\s[^>]*)?>(.*?)</tr>', block, re.S)]


print('tables found:', re.findall(r'<table-wrap[^>]*id="(Tab\d+)"', paper))

# %% [markdown]
# ### 1.1 Does our copy of the data match the paper?
# Table 3 of the paper gives descriptive statistics of the 565 rows, and Table 7 gives the error of classic
# Gilbert-type correlations on all rows. If our parse and our error metric are right, both should match.

# %%
t3 = jats_table('Tab3')
paper_t3 = pd.DataFrame({r[0]: [num(v) for v in r[1:5]] for r in t3[1:]}, index=['p_psi', 'glr', 'd64', 'qliq']).T


def first_mode(s):
    """Spreadsheet MODE: the first value, in data order, among the most frequent."""
    counts = s.map(s.value_counts())
    return s[counts == counts.max()].iloc[0]


ours_t3 = pd.DataFrame({c: {'Mean': df[c].mean(), 'Standard Error': df[c].std() / np.sqrt(len(df)),
                            'Median': df[c].median(), 'Mode': first_mode(df[c]), 'Standard Deviation': df[c].std(),
                            'Sample Variance': df[c].var(), 'Kurtosis': df[c].kurt(), 'Skewness': df[c].skew(),
                            'Minimum': df[c].min(), 'Maximum': df[c].max()}
                        for c in ('p_psi', 'glr', 'd64', 'qliq')}).reindex(paper_t3.index)
diff3 = (ours_t3.round(3) - paper_t3).abs()
RESULTS['dabiri_table3_max_abs_diff'] = float(diff3.max().max())
print('largest |this notebook - paper| over all 40 entries (paper prints 3 decimals):',
      RESULTS['dabiri_table3_max_abs_diff'])
pd.concat({'paper, Table 3': paper_t3, 'this notebook': ours_t3.round(3)}, axis=1)

# %%
GILBERT_FAMILY = {  # P_wh = A * q * GLR**B / D**C   (P in psi, q in stb/d, GLR in scf/stb, D in 1/64 in)
    'Gilbert': (10.00, 0.546, 1.89), 'Ros': (17.40, 0.500, 2.00), 'Baxendell': (9.56, 0.546, 1.93),
    'Achong': (3.82, 0.650, 1.88), 'Pilehvari': (46.67, 0.313, 2.11)}
paper_t7 = {r[0]: [num(v) for v in r[1:6]] for r in jats_table('Tab7')[1:]}       # APRE, AAPRE, RMSE, SD, R2
rows = []
for name, (a, b, c) in GILBERT_FAMILY.items():
    q = P * D ** c / (a * GLR ** b)
    rows.append({'correlation': name, 'AAPRE paper': paper_t7[name][1], 'AAPRE ours': aapre(q, Q),
                 'RMSE paper': paper_t7[name][2], 'RMSE ours': rmse(q, Q),
                 'R2 paper': paper_t7[name][4], 'R2 ours, 1-SSE/SST': r2(q, Q), 'R2 ours, corr^2': r2_corr(q, Q),
                 'APRE paper': paper_t7[name][0], 'APRE ours, (meas-pred)/meas': float(np.mean((Q - q) / Q) * 100)})
t7 = pd.DataFrame(rows).set_index('correlation')
RESULTS['dabiri_correlations'] = t7.round(4).to_dict('index')
t7.round(3)

# %% [markdown]
# **Result.** All 40 entries of Table 3 match to the printed precision. For Gilbert, Ros, Baxendell and Achong in
# Table 7, AAPRE and APRE (measured minus predicted) match to the printed precision and RMSE to within 0.04 stb/d. The
# paper's R² column is the squared correlation (within 0.004), not 1 - SSE/SST. Our Pilehvari constants give 122% against the paper's 51.82%; we leave that open.
# So the data and the error metric are the paper's.

# %% [markdown]
# ### 1.2 Duplicate rows
# Literature compilations can contain the same test twice. Exact copies matter under a random split: a copy in the
# training set lets a model answer its twin in the test set from memory.

# %%
from sklearn.model_selection import train_test_split  # noqa: E402

record = df[['p_psi', 'glr', 'd64', 'qliq']].apply(tuple, axis=1)
size = record.map(record.value_counts())
group_id = record.astype('category').cat.codes.to_numpy()
inputs = df[['p_psi', 'glr', 'd64']].apply(tuple, axis=1)
n_rates = df.groupby(inputs)['qliq'].nunique()
leak = []
for seed in range(20):
    tr, te = train_test_split(idx, test_size=0.2, random_state=seed)
    leak.append(np.isin(group_id[te], group_id[tr]).mean() * 100)
dups = {'rows': len(df), 'distinct_records': int(record.nunique()),
        'rows_in_duplicate_groups': int((size > 1).sum()), 'duplicate_groups': int((record.value_counts() > 1).sum()),
        'largest_group': int(size.max()), 'input_triples_with_more_than_one_rate': int((n_rates > 1).sum()),
        'test_rows_with_an_exact_copy_in_training_pct_20_random_splits': mean_ci(leak)}
RESULTS['dabiri_duplicates'] = dups
print(json.dumps(dups, indent=1))
print('most repeated record (P, GLR, D, Q):', record.value_counts().index[0], 'x', record.value_counts().iloc[0])
print('rows whose inputs (P, GLR, D) appear with different rates:')
df[inputs.isin(n_rates[n_rates > 1].index)].sort_values(['p_psi', 'glr', 'd64'])[['p_psi', 'glr', 'd64', 'qliq']]

# %% [markdown]
# **Result.** 124 rows (22%) sit in 44 groups of identical rows, the largest with 10 copies, and 6 input triples carry
# two or more different rates. On a random 80/20 split, 19.7% of test rows have an exact copy in training (9.7-25.7%
# over 20 seeds).

# %% [markdown]
# ### 1.3 What the paper reports
# Table 6 gives each model's error on the training set, the test set and all rows.

# %%
t6 = jats_table('Tab6')
models6, paper_t6, part = t6[0][-6:], {}, None
for r in t6[1:]:
    if len(r) == 1:
        part = r[0]
        continue
    paper_t6[f'{part} | {r[0]}'] = [num(v) for v in r[-6:]]
paper_t6 = pd.DataFrame(paper_t6, index=models6).T
RESULTS['dabiri_table6'] = paper_t6.to_dict()
paper_t6

# %% [markdown]
# 4.57% is the test-set figure; the abstract's 5.15% is the whole set. The training-set error (5.3%) is above the
# test-set error.

# %% [markdown]
# ### 1.4 Re-running the protocol
# The protocol is a random 80/20 split with no seed given, so we use 20 seeds and report the spread. Every model
# sees the same 20 splits.
#
# The replicate follows Table S3: `AdaBoostRegressor` over `SVR(C=1.0, epsilon=0.1, kernel='rbf', gamma='scale')`,
# boosting at scikit-learn defaults. The paper doesn't say how inputs and target were scaled, and with
# `epsilon=0.1` that choice matters (on a min-max-scaled target, 0.1 is a tenth of the whole rate range). So we run
# all ten combinations of input scaling (min-max, standard) and target transform (none, min-max, standard, log,
# log then standard), plus a version whose SVR settings are tuned by 5-fold cross-validation inside each training
# set, and a single tuned SVR without boosting.
#
# For context we add four generic models: LightGBM, a random forest, a 1-nearest-neighbour lookup (pure memory) and a
# Gilbert-form power law refit by least squares. These, and the tuned SVRs, use log inputs and a log target.

# %%
import lightgbm as lgb  # noqa: E402
from sklearn.compose import TransformedTargetRegressor  # noqa: E402
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.model_selection import GridSearchCV, GroupShuffleSplit, KFold  # noqa: E402
from sklearn.neighbors import KNeighborsRegressor  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, StandardScaler  # noqa: E402
from sklearn.svm import SVR  # noqa: E402

X_NAT = df[['p_psi', 'glr', 'd64']].to_numpy()
X_LOG = np.log(X_NAT)
SVR_TABLE_S3 = dict(C=1.0, epsilon=0.1, kernel='rbf', gamma='scale')
X_SCALERS = {'minmax': MinMaxScaler, 'standard': StandardScaler}
GRID = {'C': [1, 10, 100, 1000], 'epsilon': [0.01, 0.1], 'gamma': ['scale', 1, 3]}


def with_target(reg, how):
    if how == 'none':
        return reg
    if how == 'log':
        return TransformedTargetRegressor(reg, func=np.log, inverse_func=np.exp, check_inverse=False)
    tf = {'minmax': MinMaxScaler(), 'standard': StandardScaler(),
          'log+standard': make_pipeline(FunctionTransformer(np.log, np.exp, check_inverse=False), StandardScaler())}
    return TransformedTargetRegressor(reg, transformer=tf[how], check_inverse=False)


def adaboost_svr(xs, ys, seed):
    return with_target(make_pipeline(X_SCALERS[xs](), AdaBoostRegressor(estimator=SVR(**SVR_TABLE_S3),
                                                                        random_state=seed)), ys)


def tuned(seed, boosted):
    if boosted:
        pipe = make_pipeline(StandardScaler(), AdaBoostRegressor(estimator=SVR(kernel='rbf'), random_state=seed))
        grid = {f'adaboostregressor__estimator__{k}': v for k, v in GRID.items()}
    else:
        pipe = make_pipeline(StandardScaler(), SVR(kernel='rbf'))
        grid = {f'svr__{k}': v for k, v in GRID.items()}
    search = GridSearchCV(pipe, grid, cv=KFold(5, shuffle=True, random_state=seed), scoring='neg_mean_absolute_error',
                          n_jobs=-1)
    return with_target(search, 'log')


MODELS = {f'Adaboost-SVR (Table S3), X {xs}, y {ys}': (lambda s, xs=xs, ys=ys: adaboost_svr(xs, ys, s), X_NAT)
          for xs in X_SCALERS for ys in ('none', 'minmax', 'standard', 'log', 'log+standard')}
MODELS.update({
    'Adaboost-SVR, tuned by inner CV': (lambda s: tuned(s, True), X_LOG),
    'SVR, tuned by inner CV': (lambda s: tuned(s, False), X_LOG),
    'LightGBM': (lambda s: with_target(lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=15,
                                                         min_child_samples=5, subsample=0.8, subsample_freq=1,
                                                         random_state=s, verbose=-1), 'log'), X_LOG),
    'Random forest': (lambda s: with_target(RandomForestRegressor(500, random_state=s, n_jobs=-1), 'log'), X_LOG),
    '1-nearest neighbour (memory)': (lambda s: with_target(KNeighborsRegressor(1), 'log'), X_LOG),
    'Power law refit (Gilbert form)': (lambda s: with_target(LinearRegression(), 'log'), X_LOG),
})


def run(splits, models):
    out = []
    for seed, (tr, te) in enumerate(splits):
        for name, (make, X) in models.items():
            m = make(seed).fit(X[tr], Q[tr])
            p_te, p_tr = m.predict(X[te]), m.predict(X[tr])
            out.append({'method': name, 'seed': seed, 'test AAPRE': aapre(p_te, Q[te]),
                        'train AAPRE': aapre(p_tr, Q[tr]), 'test RMSE': rmse(p_te, Q[te]),
                        'test R2': r2(p_te, Q[te]), 'test R2 (corr^2)': r2_corr(p_te, Q[te])})
    return pd.DataFrame(out)


def summarize(runs):
    out = []
    for name, x in runs.groupby('method'):
        c = mean_ci(x['test AAPRE'])
        out.append({'method': name, 'test AAPRE, mean': c['mean'], '95% CI': '[%.2f, %.2f]' % (c['ci_lo'], c['ci_hi']),
                    'best split': c['min'], 'worst split': c['max'], 'train AAPRE, mean': x['train AAPRE'].mean(),
                    'test RMSE, mean': x['test RMSE'].mean(), 'test R2, mean': x['test R2'].mean(),
                    'test corr^2, mean': x['test R2 (corr^2)'].mean()})
    return pd.DataFrame(out).set_index('method').sort_values('test AAPRE, mean')


t1 = time.time()
random_splits = [train_test_split(idx, test_size=0.2, random_state=s) for s in range(20)]
runs = run(random_splits, MODELS)
runs.to_csv('dabiri_runs_random_split.csv', index=False)
table = summarize(runs)
RESULTS['dabiri_random_split'] = table.round(4).reset_index().to_dict('records')
save()
print(f'{len(runs)} model fits in {time.time() - t1:.0f} s')
table.round(2)

# %%
order = table.index.tolist()
fig, ax = plt.subplots(figsize=(11, 0.45 * len(order) + 1.2))
for i, name in enumerate(order):
    x = runs.loc[runs.method == name, 'test AAPRE'].to_numpy()
    ax.scatter(x, i + np.random.default_rng(i).uniform(-0.15, 0.15, len(x)), s=14, alpha=0.7)
    ax.plot([x.mean()] * 2, [i - 0.3, i + 0.3], color='black', lw=1.5)
ax.axvline(4.57, color='crimson', ls='--', lw=1.2)
ax.text(4.57, -0.8, ' 4.57%: paper, Adaboost-SVR, test set', color='crimson', fontsize=9, va='center')
ax.set_yticks(range(len(order)))
ax.set_yticklabels(order, fontsize=8)
ax.set_ylim(len(order) - 0.5, -1.2)
ax.set_xscale('log')
ax.set_xlabel('test AAPRE, % (log scale). Dots: 20 random 80/20 splits; bar: mean')
ax.set_title('Dabiri et al. 2024 data: test error by model')
plt.tight_layout()
plt.savefig('dabiri_test_aapre.png', dpi=130)
plt.show()

# %% [markdown]
# **Result.**
# - **The Table S3 replicate.** The best of the ten preprocessing combinations (min-max inputs, log target) averages
#   20.3% [18.3, 22.3]. The lowest single split of any Table S3 variant is 13.5%. The log-target variants' training
#   error is 16-18%, against the paper's 5.3%.
# - **Tuning.** Tuning the SVR inside each training set helps a single SVR (13.5%) but makes the boosted version
#   unstable (23.6%, with a few very large misses).
# - **Generic models.** These land at 12-19%; LightGBM is best at 12.2% [10.8, 13.6].
# - **Pure memory.** Looking up the nearest training row scores 13.7%, so the duplicates alone don't produce
#   numbers near 4.57%.
#
# A single lucky split could still give a low number, so the cheapest strong model gets 200 splits.

# %%
t1 = time.time()
many = []
for s in range(200):
    tr, te = train_test_split(idx, test_size=0.2, random_state=s)
    many.append(aapre(MODELS['LightGBM'][0](s).fit(X_LOG[tr], Q[tr]).predict(X_LOG[te]), Q[te]))
many = np.array(many)
RESULTS['dabiri_lightgbm_200_splits'] = {'mean': float(many.mean()), 'min': float(many.min()),
                                         'p5': float(np.percentile(many, 5)), 'max': float(many.max())}
print(f'LightGBM, 200 random 80/20 splits ({time.time() - t1:.0f} s):', RESULTS['dabiri_lightgbm_200_splits'])

# %% [markdown]
# **Result.** Over 200 random splits LightGBM averages 11.9%; its best split is 7.1% and its 5th percentile 9.0%.

# %% [markdown]
# ### 1.5 The same comparison with duplicates kept on one side
# Each group of identical rows now goes wholly to training or wholly to test (`GroupShuffleSplit`, 20 seeds).

# %%
group_splits = [next(GroupShuffleSplit(1, test_size=0.2, random_state=s).split(idx, groups=group_id))
                for s in range(20)]
best_s3 = [m for m in table.index if m.startswith('Adaboost-SVR (Table S3)')][0]
keep = [best_s3, 'Adaboost-SVR, tuned by inner CV', 'LightGBM', 'Random forest', '1-nearest neighbour (memory)',
        'Power law refit (Gilbert form)']
runs_g = run(group_splits, {k: MODELS[k] for k in keep})
runs_g.to_csv('dabiri_runs_group_split.csv', index=False)
table_g = summarize(runs_g)
RESULTS['dabiri_group_split'] = table_g.round(4).reset_index().to_dict('records')
save()
table_g.round(2)

# %% [markdown]
# **Result.** With duplicates kept together, LightGBM rises to 13.9% and the nearest-row lookup to 15.2%; the order
# of the models doesn't change.

# %% [markdown]
# ### 1.6 Two further observations
# **External validation (Table 8).** The paper also tests Adaboost-SVR on 28 points from another source, which it
# describes as outside the training and test data. Table 8 lists measured and predicted rates but not the inputs,
# so only the pairs can be examined.

# %%
TABLE8 = [(5250.7, 4207.7), (400.0, 400.0), (5220.5, 4207.7), (500.0, 700.0), (1890.2, 1895.2), (540.0, 540.0),
          (1900.2, 1900.2), (570.0, 770.1), (1350.1, 1700.0), (690.1, 927.0), (6500.8, 6758.01), (940.1, 1158.04),
          (2500.3, 2500.3), (1100.1, 1205.05), (1300.1, 1300.1), (1380.1, 1380.1), (711.0, 711.0),
          (1680.1, 1532.067), (800.1, 749.86), (1900.5, 1680.1), (260.0, 330.0), (2050.3, 1900.26), (290.0, 330.0),
          (2250.3, 2006.325), (330.0, 340.0), (3000.4, 2919.822), (360.0, 360.0), (3500.5, 3000.4)]
pairs = []
for r in jats_table('Tab8')[1:]:
    v = [num(c) for c in r]
    if len(v) >= 2 and np.isfinite(v).all():
        pairs += [(v[i], v[i + 1]) for i in range(0, len(v) - 1, 2)]
print('Table 8 parsed from the paper XML matches the copy above:', sorted(pairs) == sorted(TABLE8))
t8 = pd.DataFrame(pairs or TABLE8, columns=['measured', 'predicted'])
t8['identical'] = t8.measured == t8.predicted
t8['equals another measured value in Table 8'] = t8.predicted.isin(t8.measured) & ~t8.identical
RESULTS['dabiri_table8'] = {'pairs': len(t8), 'identical': int(t8.identical.sum()),
                            'equal_to_another_measured_value': int(t8['equals another measured value in Table 8'].sum()),
                            'aapre_over_table8_pct': aapre(t8.predicted, t8.measured)}
print(json.dumps(RESULTS['dabiri_table8'], indent=1))
t8

# %% [markdown]
# **Result.**
# - **Identical values.** 8 of the 28 predictions equal the measured value to the printed precision (1900.2 and
#   1900.2, for example).
# - **Another row's value.** 4 more equal a different measured value in the same table (1900.5 predicted as 1680.1,
#   for example).
#
# Exact agreement is unusual for a regression model on inputs it hasn't seen. Without the inputs we can't re-run
# these points, so we list this as a question for the authors, not as a finding. The 28 pairs give an AAPRE of 11.3%.
#
# **Model settings (Table S3 and Table 4).** The supplementary Table S3 lists the scikit-learn settings; the paper's
# Table 4 lists the MLP and RBF network settings.

# %%
from sklearn.neural_network import MLPRegressor  # noqa: E402

print('Table S3 (supplementary):')
for r in supp[3]:
    print('   ', ' | '.join(r))
print('\nTable 4 (paper):')
for r in jats_table('Tab4'):
    print('   ', ' | '.join(r))
s3 = {r[0]: r[1] for r in supp[3][1:]}
try:
    MLPRegressor(solver='scg').fit(np.zeros((4, 1)), np.zeros(4))
    scg = 'accepted'
except Exception as e:
    scg = f'rejected ({type(e).__name__})'
RESULTS['dabiri_settings'] = {
    'Table S3: RBF row equals the Adaboost-SVR row': s3.get('RBF') == s3.get('Adaboost_SVR'),
    'Table S3: MLP-LM row equals the MLP-BR row': s3.get('MLP-lM') == s3.get('MLP-BR'),
    "scikit-learn MLPRegressor(solver='scg')": scg}
save()
print('\n' + json.dumps(RESULTS['dabiri_settings'], indent=1))

# %% [markdown]
# **Result.** Table S3 doesn't pin down the models behind Table 6:
# - the RBF row repeats the Adaboost-SVR base learner's settings;
# - the MLP-LM and MLP-BR rows are identical;
# - scikit-learn has no `scg` solver;
# - Table 4's two hidden layers and 250-neuron RBF network don't appear in Table S3.

# %% [markdown]
# ## 2. D'Souza, Pfeiffer & Mirlekar 2024: two simulated gas-lifted wells
#
# D'Souza, N. A., Pfeiffer, C. & Mirlekar, G. (2024). Assessment of data-driven techniques for flow rate predictions
# in sub-sea oil production. *Proceedings of SIMS EUROSIM 2024*.
# [doi:10.3384/ecp212.014](https://doi.org/10.3384/ecp212.014)
#
# The paper simulates two gas-lifted wells with the open-loop model of Janatian et al. (2022), averages the output
# to 5,762 one-minute samples per well, splits each series in time (training, validation, then the last 30% as
# test) and compares ten regressors that predict three mass rates from gas-lift rate, bottomhole pressure and
# wellhead pressure. Table 1 reports test MAPE; the best is an LSTM (1.96% for well 1, 1.53% for well 2).
#
# Two copies of the simulator are public: the GitHub repository linked from the paper and a DataverseNO deposit.

# %%
GH_RAW = 'https://raw.githubusercontent.com/dsouzaneville/FMH606-1-Masters-Thesis/main/'
try:
    tree = json.loads(fetch('https://api.github.com/repos/dsouzaneville/FMH606-1-Masters-Thesis/git/trees/HEAD'
                            '?recursive=1'))
    gh_files = [t['path'] for t in tree['tree'] if t['type'] == 'blob']
except Exception as e:
    gh_files = f'listing failed: {e}'
try:
    dv = json.loads(fetch('https://dataverse.no/api/datasets/:persistentId/versions/:latest/files'
                          '?persistentId=doi:10.18710/KIJEWJ'))
    dv_files = {f['dataFile']['filename']: f['dataFile']['id'] for f in dv['data']}
except Exception as e:
    dv_files = f'listing failed: {e}'
RESULTS['dsouza_released_files'] = {'github': gh_files, 'dataverse': dv_files}
print('GitHub:', gh_files)
print('DataverseNO:', dv_files)

# %%
nb_facts = {}
for name in ('ML_dev_part1.ipynb', 'ML_dev_part2.ipynb'):
    nb = json.loads(fetch(GH_RAW + name))
    code = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
    lines = [l.strip() for s in code for l in s.splitlines()]
    printed = ' '.join(''.join(o.get('text', '')) for c in nb['cells'] for o in c.get('outputs', []))
    rows_seen = re.search(r'RangeIndex: (\d+) entries', printed)
    nb_facts[name] = {'reads': re.findall(r'load_data\("([^"]+)"\)', '\n'.join(code)),
                      'rows in the CSV (saved .info() output)': rows_seen.group(1) if rows_seen else None,
                      'scaling': [l for l in lines if 'fit_transform' in l],
                      'splits': [l for l in lines if 'train_test_split(' in l and '=' in l],
                      'MLP': [l for l in lines if 'hidden_layer_sizes' in l],
                      'computes MAPE anywhere': any(re.search(r'percentage_error|mape', l, re.I) for l in lines)}
RESULTS['dsouza_notebooks'] = nb_facts
print(json.dumps(nb_facts, indent=1))

# %% [markdown]
# **Result.**
# - **Data.** Neither deposit contains `Well1_average_data.csv` or `Well2_average_data.csv`. Both notebooks read the
#   well-1 file, which had 5,762 rows according to their saved output, and no released code computes the MAPE values
#   in Table 1.
# - **Settings.** Three details differ from the paper:
#   - the notebook's MLP has one hidden layer of two neurons (`hidden_layer_sizes=(2)`), where the paper describes
#     two hidden layers;
#   - the scaler is fit on all rows, where the paper says the training set;
#   - the abstract gives 1.56% for well 2's LSTM, where Table 1 gives 1.53%.

# %% [markdown]
# ### 2.1 Regenerating the data
# Octave runs the MATLAB code. The GitHub copy is a single-well version (`Theta_nom = [2.51 0.05 0.2]`, all of the
# gas-lift supply to the well); the DataverseNO copy is the original two-well model (30% of the supply to each
# well). Both use the same schedule: 40,000, then 20,000, then 40,000 Sm³/h, in three equal steps.

# %%
if subprocess.run('command -v octave', shell=True).returncode != 0:
    subprocess.run('apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq octave >/dev/null',
                   shell=True, check=True, timeout=1800)
print(subprocess.run(['octave', '--version'], capture_output=True, text=True).stdout.splitlines()[0])

GH_DIR, DV_DIR = SRC / 'github', SRC / 'dataverse'
GH_DIR.mkdir(exist_ok=True)
DV_DIR.mkdir(exist_ok=True)
for f in ('openloop_simulation.m', 'oil_field_model.m', 'make_w_gc.m', 'update_states.m', 'calculate_output.m'):
    (GH_DIR / f).write_bytes(fetch(GH_RAW + 'Simulator/' + f))
DV_IDS = {'oil_field_model.m': 268788, 'make_w_gc.m': 268786, 'update_states.m': 268787,
          'openloop_simulation.m': 268789, 'calculate_output.m': 268790}
if isinstance(dv_files, dict):
    DV_IDS.update({k: v for k, v in dv_files.items() if k in DV_IDS})
for f, fid in DV_IDS.items():
    (DV_DIR / f).write_bytes(fetch(f'https://dataverse.no/api/access/datafile/{fid}'))


def outputs_of(line):
    return [v.strip() for v in re.search(r'\[(.*?)\]', line).group(1).split(',') if v.strip()]


gh_script = (GH_DIR / 'openloop_simulation.m').read_text().replace('\r', '')
declared = outputs_of((GH_DIR / 'calculate_output.m').read_text().replace('\r', '').splitlines()[0])
requested = outputs_of(next(l for l in gh_script.splitlines() if '= calculate_output(' in l))
RESULTS['dsouza_github_script'] = {
    'calculate_output declares outputs': len(declared), 'openloop_simulation requests outputs': len(requested),
    'requested but not declared': [v for v in requested if v not in declared],
    'csv writing lines commented out': all(l.lstrip().startswith('%') for l in gh_script.splitlines()
                                           if 'fopen' in l or 'writematrix' in l),
    'csv header written by the script': re.search(r"^header = \{(.*)\};", gh_script, re.M).group(1)}
print(json.dumps(RESULTS['dsouza_github_script'], indent=1))

# %% [markdown]
# As released, the GitHub script asks `calculate_output` for more outputs than it declares (an error in MATLAB as
# well as Octave), and its CSV-writing lines are commented out. So we keep the script's settings and loop and take
# the outputs straight from `oil_field_model`, which is all `calculate_output` does. For the DataverseNO copy we
# make `oil_field_model` return its intermediate variables in the same way and run 96 h, the length of the paper's
# series (the deposited script runs 100 h).

# %%
GEN_GITHUB = """
Theta_nom = [2.51 0.05 0.2]; dt = 20; timespan = 96;
t = 0:dt:timespan*60*60;
w_gc = make_w_gc('2step', [40000 1; 20000 1; 40000 1], t);
w_ga = 100/100 .* w_gc;
m0 = [16422, 1669, 15229];
m = zeros(length(t), length(m0)); m(1,:) = m0;
for k = 1:length(t)-1
  m(k+1,:) = update_states(m(k,:), w_ga(k,:), Theta_nom, dt);
end
Theta_OL = Theta_nom .* ones(length(t),1);
[~, w_op, w_gop, P_a, P_ainj, rho_l, V_G, rho_m, P_tinj, Y_2, rho_ga, w_ginj, P_wf, w_l, w_o, w_g, P_wh, Y_3, w_gp, w_lp, w_wp] = oil_field_model(m, zeros(length(t),1), Theta_OL);
dlmwrite('series_well1.csv', [t'/3600, w_ga, P_wf, P_wh, w_op, w_wp, w_gp, w_lp, w_gop, w_ginj], 'precision', 12);
"""
GEN_DATAVERSE = """
Theta_nom = [2.51 1.63 0.08 0.07 0.15 0.15]; dt = 20; timespan = 96;
t = 0:dt:timespan*60*60;
w_gc = make_w_gc('2step', [40000 1; 20000 1; 40000 1], t);
w_ga = [30 30]/100 .* w_gc;
m0 = [16422 15445 1669 1517 15229 13458];
m = zeros(length(t), 6); m(1,:) = m0;
for k = 1:length(t)-1
  m(k+1,:) = update_states(m(k,:), w_ga(k,:), Theta_nom, dt);
end
Theta_OL = Theta_nom .* ones(length(t),1);
[~, w_op, w_gop, P_a, P_ainj, rho_l, V_G, rho_m, P_tinj, Y_2, rho_ga, w_ginj, P_wf, w_l, w_o, w_g, P_wh, Y_3, w_gp, w_lp, w_wp] = oil_field_model(m, zeros(length(t),2), Theta_OL);
w_ga = [w_ga; w_ga(end,:)];
for wi = 1:2
  dlmwrite(sprintf('series_well%d.csv', wi), [t'/3600, w_ga(:,wi), P_wf(:,wi), P_wh(:,wi), w_op(:,wi), w_wp(:,wi), w_gp(:,wi), w_lp(:,wi), w_gop(:,wi), w_ginj(:,wi)], 'precision', 12);
end
"""
dv_model = (DV_DIR / 'oil_field_model.m').read_text().replace('\r', '').splitlines()
dv_model[0] = ('function  [dm, w_op, w_gop, P_a, P_ainj, rho_l, V_G, rho_m, P_tinj, Y_2, rho_ga, w_ginj, P_wf, '
               'w_l, w_o, w_g, P_wh, Y_3, w_gp, w_lp, w_wp] = oil_field_model(m, w_ga, Theta)')
(DV_DIR / 'oil_field_model.m').write_text('\n'.join(l.replace('%     w_wp = w_lp -w_op;', '    w_wp = w_lp - w_op;')
                                                    for l in dv_model) + '\n')
for d, gen in ((GH_DIR, GEN_GITHUB), (DV_DIR, GEN_DATAVERSE)):
    (d / 'regenerate.m').write_text(gen)
    t1 = time.time()
    r = subprocess.run(['octave', '--no-gui', '--quiet', 'regenerate.m'], cwd=d, capture_output=True, text=True,
                       timeout=3600)
    print(d.name, 'exit', r.returncode, f'in {time.time() - t1:.0f} s', (r.stdout + r.stderr)[-400:])
    assert r.returncode == 0

COLS = ['t_h', 'w_ga', 'P_wf', 'P_wh', 'w_op', 'w_wp', 'w_gp', 'w_lp', 'w_gop', 'w_ginj']


def minute_average(path):
    raw = pd.read_csv(path, header=None, names=COLS)
    return raw.groupby(np.arange(len(raw)) // 3).mean()       # 20 s steps to 1-minute rows


series = {'GitHub copy, well 1': minute_average(GH_DIR / 'series_well1.csv'),
          'DataverseNO copy, well 1': minute_average(DV_DIR / 'series_well1.csv'),
          'DataverseNO copy, well 2': minute_average(DV_DIR / 'series_well2.csv')}
RESULTS['dsouza_rows'] = {k: len(s) for k, s in series.items()}
print('one-minute rows per series (the notebooks read 5,762):', RESULTS['dsouza_rows'])

# %% [markdown]
# ### 2.2 Is the test window a new operating point?
# The notebooks train on rows 0-2,303, validate on rows 2,304-4,032 and test on rows 4,033 onward (from 67.2 h), so
# the test window lies in the third supply step, which repeats the first step's supply. In the paper's Figs. 6 and
# 7 the measured rates in the test window settle at levels different from the first step's. If the test window
# instead repeats the first step's state, a model that has learned the first step scores almost perfectly.

# %%
WINDOWS = {'step 1 (20-30 h)': (20, 30), 'step 2 (52-62 h)': (52, 62), 'test window (85-95 h)': (85, 95)}
level_rows = []
for k, s in series.items():
    for col in ('w_lp', 'w_op', 'w_wp', 'w_gp', 'w_ga', 'P_wf', 'P_wh'):
        r = {'series': k, 'quantity': col}
        for w, (a, b) in WINDOWS.items():
            r[w] = s.loc[(s.t_h >= a) & (s.t_h < b), col].mean()
        r['test window vs step 1, %'] = (r['test window (85-95 h)'] / r['step 1 (20-30 h)'] - 1) * 100
        level_rows.append(r)
levels = pd.DataFrame(level_rows)
levels.to_csv('dsouza_window_levels.csv', index=False)
RESULTS['dsouza_window_levels'] = levels.round(4).to_dict('records')
step1 = {k: s[(s.t_h >= 20) & (s.t_h < 30)] for k, s in series.items()}
RESULTS['dsouza_oil_to_liquid_step1'] = {k: float(s.w_op.mean() / s.w_lp.mean()) for k, s in step1.items()}
save()
print('oil / liquid mass ratio in step 1:', {k: round(v, 3) for k, v in RESULTS['dsouza_oil_to_liquid_step1'].items()})
levels.round(3)

# %%
fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharey=True)
for ax, (k, s) in zip(axes, series.items()):
    for col, lab in (('w_lp', 'liquid'), ('w_op', 'oil'), ('w_wp', 'water'), ('w_gp', 'gas')):
        ax.plot(s.t_h, s[col], lw=1, label=lab)
    ax.axvline(s.t_h.iloc[4033], color='k', ls=':', lw=1)
    ax.set_title(k)
    ax.set_xlabel('time, h (dotted: start of the test rows)')
axes[0].set_ylabel('mass rate, kg/s')
axes[0].legend()
plt.tight_layout()
plt.savefig('dsouza_regenerated_series.png', dpi=120)
plt.show()

# %% [markdown]
# **Result.** In both copies the test window returns exactly to the first step's state, with a 0.0% change in every
# rate and pressure, and the series are smooth. The published figures differ (values read off the figures, so
# approximate):
# - **Level.** In the paper's Fig. 6 the largest well-1 curve sits near 78 kg/s in the first step and near 71 kg/s in
#   the test window.
# - **Oscillation.** The model predictions oscillate during the second step, so the inputs did too.
# - **The GitHub copy.** Its oil-to-liquid ratio (0.762, from water cut 0.2) matches the figure's (about 60/78 =
#   0.77), but its levels don't (liquid 92.5 kg/s).
#
# Neither copy reproduces the published series.

# %% [markdown]
# ### 2.3 The paper's models on each regenerated series
# Settings from the GitHub notebooks (well 1) and the paper's text (well 2), inputs standardised on all rows and
# the split rows exactly as in the notebooks. MAPE is scikit-learn's, averaged over the three targets. The released
# script writes oil, water and gas, while the notebooks call their targets oil, gas and liquid, so both target sets
# are scored.

# %%
from sklearn.cross_decomposition import PLSRegression  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.ensemble import GradientBoostingRegressor  # noqa: E402
from sklearn.metrics import mean_absolute_percentage_error  # noqa: E402
from sklearn.multioutput import MultiOutputRegressor  # noqa: E402
from sklearn.tree import DecisionTreeRegressor  # noqa: E402
from xgboost import XGBRegressor  # noqa: E402

TABLE1 = {'LSTM': (1.96, 1.53), 'MLP NN': (2.43, 5.49), 'MV linear regression': (2.14, 7.57), 'SVR': (5.04, 4.31),
          'KNN': (8.05, 5.41), 'Decision tree': (9.26, 5.43), 'Gradient boosting': (4.95, 5.55),
          'XGBoost': (4.23, 5.56), 'PLS': (9.54, 7.57), 'PCR': (9.52, 16.69)}
TR, VA = 2304, 4033


def paper_models(well):
    w1 = well == 1
    return {
        'MLP NN': MLPRegressor(hidden_layer_sizes=2, activation='identity' if w1 else 'logistic', solver='lbfgs',
                               alpha=0.0005, max_iter=1000, random_state=42),
        'MV linear regression': MultiOutputRegressor(LinearRegression()),
        'SVR': MultiOutputRegressor(SVR(kernel='rbf', C=1.0, gamma=0.1) if w1 else SVR(kernel='rbf', C=10.0,
                                                                                       gamma=0.01)),
        'KNN': KNeighborsRegressor(n_neighbors=8, p=2),
        'Decision tree': DecisionTreeRegressor(max_depth=1, max_features='sqrt', min_samples_leaf=1,
                                               min_samples_split=2, random_state=0) if w1 else
        DecisionTreeRegressor(max_depth=10, max_features='log2', min_samples_leaf=1, min_samples_split=10,
                              random_state=0),
        'Gradient boosting': MultiOutputRegressor(GradientBoostingRegressor(n_estimators=100, max_depth=3,
                                                                            learning_rate=0.1 if w1 else 0.05)),
        'XGBoost': MultiOutputRegressor(XGBRegressor(objective='reg:squarederror', n_estimators=300 if w1 else 100,
                                                     max_depth=7 if w1 else 3, learning_rate=0.01 if w1 else 0.1,
                                                     random_state=42)),
        'PLS': PLSRegression(n_components=1),
        'PCR': make_pipeline(StandardScaler(), PCA(n_components=1), LinearRegression())}


def lstm_mape(X, Y, well):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(0)
    inp = tf.keras.Input(shape=(1, 3))
    out = tf.keras.layers.Dense(3, activation='linear')(tf.keras.layers.LSTM(32 if well == 1 else 40)(inp))
    model = tf.keras.Model(inp, out)
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X[:TR, None, :], Y[:TR], validation_data=(X[TR:VA, None, :], Y[TR:VA]), epochs=100, batch_size=32,
              verbose=0, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)])
    return mean_absolute_percentage_error(Y[VA:], model.predict(X[VA:, None, :], verbose=0)) * 100


TARGETS = {'oil, gas, liquid': ['w_op', 'w_gp', 'w_lp'], 'oil, water, gas': ['w_op', 'w_wp', 'w_gp']}
t1 = time.time()
fid = []
for k, s in series.items():
    well = 2 if 'well 2' in k else 1
    X = StandardScaler().fit_transform(s[['w_ga', 'P_wf', 'P_wh']])
    for tname, tcols in TARGETS.items():
        Y = s[tcols].to_numpy()
        row = {'series': k, 'targets': tname}
        for mname, model in paper_models(well).items():
            row[mname] = mean_absolute_percentage_error(Y[VA:], model.fit(X[:TR], Y[:TR]).predict(X[VA:])) * 100
        try:
            row['LSTM'] = lstm_mape(X, Y, well)
        except Exception as e:
            row['LSTM'] = np.nan
            print('LSTM failed:', e)
        fid.append(row)
fid = pd.DataFrame(fid)
fid.to_csv('dsouza_model_mape.csv', index=False)
comp = pd.DataFrame({'paper, well 1': {m: v[0] for m, v in TABLE1.items()}})
for _, r in fid.iterrows():
    comp[f"{r['series']} ({r['targets']})"] = [r[m] for m in TABLE1]
comp['paper, well 2'] = [v[1] for v in TABLE1.values()]
RESULTS['dsouza_model_mape'] = comp.round(4).to_dict()
RESULTS['seconds'] = round(time.time() - T0)
save()
print(f'{time.time() - t1:.0f} s; whole notebook {RESULTS["seconds"]} s')
comp.round(3)

# %% [markdown]
# **Result.** On every regenerated series, every one of the paper's ten models scores below 1% MAPE (the largest is
# 0.83%), against 1.5-16.7% in Table 1. When the test window repeats the training conditions, the task is
# interpolation. The published numbers come from data we could not regenerate.
#
# ### 2.4 A correction to our own first pass
# Our first, private pass regenerated only the DataverseNO copy. We read its sub-1% scores as a sign that the
# released simulator produces trivially clean data, and that the paper's CSVs must carry processing missing from the
# release. That was wrong. The scores are low because the test window repeats the first step. The paper's figures
# show a test window that doesn't, so its data came from a configuration that isn't in the release.

# %% [markdown]
# ## Summary
#
# | | Reproduces | Does not reproduce |
# |---|---|---|
# | Dabiri et al. 2024 | The data (Table 3, exactly); Table 7's AAPRE, RMSE and APRE for 4 of the 5 classic correlations | Adaboost-SVR's 4.57% test AAPRE: 20.3% at best with the stated settings, and 7.1% on the best of 200 splits for the best model we tried; Pilehvari's 51.82% |
# | D'Souza et al. 2024 | The simulator runs, once its outputs are taken from `oil_field_model` | The data behind Table 1: in both released copies the test window repeats the first step, so every model scores below 1% |
#
# ## Questions for the authors
# 1. **Dabiri et al.:** which split (seed or row indices) and which input and target scaling produced Table 6?
# 2. **Dabiri et al.:** how were the Table 8 predictions produced? 8 of 28 equal the measured value exactly, and 4
#    equal another measured value in the table.
# 3. **Dabiri et al.:** which Pilehvari constants give 51.82%? With A = 46.67, B = 0.313, C = 2.11 we get 122%.
# 4. **D'Souza et al.:** could `Well1_average_data.csv` and `Well2_average_data.csv` be released, or the simulator
#    settings that produced them?
#
# Corrections are welcome in the comments. If a missing piece turns up, this notebook can be re-run against it.
#
# ## What we may have wrong
# - **Dabiri et al.:**
#   - **Splits.** The paper's split and seed are unknown, so we use 20 seeds (200 for LightGBM). A test AAPRE of
#     4.57% would sit far outside every spread we measured.
#   - **Scaling.** The inputs and target scaling aren't stated, so we tried ten combinations plus tuned versions. A
#     setting we didn't try could behave differently.
# - **D'Souza et al.:**
#   - **Levels.** The levels we compare against are read off the published figures, so they are approximate.
#   - **Missing configuration.** The paper's data may come from settings the release doesn't show, such as a
#     different gas-lift share or initial state. We ran each released copy with its own settings.
#   - **Models.** The paper's models follow its notebooks for well 1 and its text for well 2. Our LSTM is seeded;
#     theirs was not.
#
# ## Reproduce
# - **This notebook:** CPU, internet on, about 2.4 hours. Most of the time goes to the tuned SVR grids.
# - **Source, outputs and write-up:** [github.com/sankalpsthakur/reproducing-vfm-benchmarks](https://github.com/sankalpsthakur/reproducing-vfm-benchmarks).
#
# ## Sources and licences
# - **Downloaded at run time:**
#   - the Dabiri et al. supplementary data (CC BY 4.0);
#   - the paper's open-access XML from Europe PMC;
#   - the D'Souza et al. simulator and notebooks from GitHub (no licence stated) and from DataverseNO
#     (doi:10.18710/KIJEWJ, CC0).
# - **Saved as outputs:** only this notebook's own results. Third-party files are not redistributed.
# - **Credit:** both papers' authors released enough to make this check possible.
