# M7: the edge package. Single-threaded numpy inference measured on a Kaggle CPU core as the proxy device.
import os

for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[v] = '1'
import json
from pathlib import Path  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import tracemalloc  # noqa: E402

sys.path.insert(0, 'src')
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from vfmchoke import choke, control, edge, fleet, hybrid, network, pvt, units as u, well  # noqa: E402

T0 = time.time()
import glob  # noqa: E402
IN = str(Path(sorted(glob.glob('/kaggle/input/**/simfield_daily.parquet', recursive=True), key=lambda p: ('m2b' not in p))[0]).parent)
print('simulation input:', IN, flush=True)
daily, tests, wells = fleet.prepare(pd.read_parquet(f'{IN}/simfield_daily.parquet'),
                                    pd.read_parquet(f'{IN}/simfield_tests.parquet'),
                                    pd.read_parquet(f'{IN}/simfield_wells.parquet'))
groups = np.sort(daily.group.unique())
np.random.default_rng(0).shuffle(groups)
fold_of = {gid: i % 5 for i, gid in enumerate(groups)}
daily['fold'] = daily.group.map(fold_of)
trained = fleet.train(daily[(daily.fold != 0) & daily.flowing], tests)
trees = edge.export_trees(trained.model.model)
print('trees', len(trees['roots']), 'nodes', len(trees['feature']), 'leaves', len(trees['leaf']), flush=True)

# 1. tree parity on 20k rows
sample = daily[daily.flowing].sample(20000, random_state=0)
e = fleet.with_est(sample, sample.trail_wc, sample.trail_gor)
_, fe = hybrid.prior(e, hybrid.fluid_from(e))
x = fe[[str(s) for s in trees['feature_names']]].to_numpy()
tree_parity = float(np.max(np.abs(edge.predict_trees(trees, x) - trained.model.correction(fe))))

# 2. 30 calibrated test wells (k = 3 MPFM labels), their edge parameters, and VFM parity
test_wells = daily[(daily.fold == 0)].well_id.unique()[:30]
wp_rows, parity, rows_for_latency = [], [], []
for wid in test_wells:
    tw = daily[daily.well_id == wid]
    cal_rows = tw[tw.flowing & (tw.day < 90)].sort_values('day').head(3)
    cal = fleet.calibrate(cal_rows, wells.loc[wid], trained, 'mpfm')
    ev = tw[(tw.day >= 90) & tw.flowing].head(20)
    if not len(ev):
        continue
    m_full, _, _, _ = fleet.predict(ev, cal, trained)
    wrow = wells.loc[wid]
    wp = {'api': float(wrow.design_api), 'gas_sg': float(wrow.design_gas_sg), 'gor': cal.gor, 'wc': cal.wc,
          'sal': float(wrow.design_sal), 'd_choke': float(wrow.d_choke), 'vendor_gamma': float(wrow.vendor_gamma),
          'b0': float(cal.x[0]), 'b1': float(cal.x[1])}
    m_edge = edge.vfm_rate(trees, wp, ev.p_wh.to_numpy(), ev.p_man.to_numpy(), ev.t_wh.to_numpy(),
                           ev.opening.to_numpy())
    parity.append(float(np.max(np.abs(m_edge / m_full - 1))))
    wp_rows.append(wp)
    rows_for_latency.append(ev.iloc[0])
well_params = {k: np.array([w[k] for w in wp_rows]) for k in wp_rows[0]}
size = edge.artifact_bytes(trees, well_params)
edge.save('vfmchoke_edge.npz', trees, well_params)
trees2, wp2 = edge.load('vfmchoke_edge.npz')

# 3. latency: one well, one evaluation, after loading the artifact (no tracing while timing); then RAM by
#    tracemalloc in a separate pass; then a batched scan of every exported well in one call
lat, scan = [], []
for rep in range(1000):
    j = rep % len(wp_rows)
    r = rows_for_latency[j]
    wpj = {k: v[j] for k, v in wp2.items()}
    t = time.perf_counter()
    edge.vfm_rate(trees2, wpj, r.p_wh, r.p_man, r.t_wh, r.opening)
    lat.append((time.perf_counter() - t) * 1e3)
tracemalloc.start()
for rep in range(50):
    j = rep % len(wp_rows)
    r = rows_for_latency[j]
    edge.vfm_rate(trees2, {k: v[j] for k, v in wp2.items()}, r.p_wh, r.p_man, r.t_wh, r.opening)
_, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
P1 = np.array([r.p_wh for r in rows_for_latency])
P2 = np.array([r.p_man for r in rows_for_latency])
T1 = np.array([r.t_wh for r in rows_for_latency])
U = np.array([r.opening for r in rows_for_latency])
for rep in range(300):
    t = time.perf_counter()
    edge.vfm_rate(trees2, wp2, P1, P2, T1, U)
    scan.append((time.perf_counter() - t) * 1e3)
out1 = edge.vfm_rate(trees2, {k: v[0] for k, v in wp2.items()}, rows_for_latency[0].p_wh,
                     rows_for_latency[0].p_man, rows_for_latency[0].t_wh, rows_for_latency[0].opening)
out2 = edge.vfm_rate(trees2, {k: v[0] for k, v in wp2.items()}, rows_for_latency[0].p_wh,
                     rows_for_latency[0].p_man, rows_for_latency[0].t_wh, rows_for_latency[0].opening)

# 4. a 30-well optimizer step on edge tables (twin wells: three 10-slot groups merged on one flowline)
import pickle  # noqa: E402
with open(f'{IN}/simfield_pop.pkl', 'rb') as fh:
    pop = pickle.load(fh)
gsel = [g for g in groups if fold_of[g] == 0 and pop.mask[g].sum() >= 6][:5]
w_all = [pop.wells.take(np.array([g])) for g in gsel]
cat = lambda name: np.concatenate([np.asarray(getattr(w, name))[0] for w in w_all])[:30]
fl = pvt.Fluid(*(np.concatenate([np.asarray(getattr(w.fluid, n))[0] for w in w_all])[:30]
                 for n in ('api', 'gas_sg', 'gor', 'wc', 'salinity')))
tw = well.Wells(p_res=cat('p_res'), t_res=cat('t_res'), pi=cat('pi'), depth=cat('depth'), d_tub=cat('d_tub'),
                d_choke=cat('d_choke'), cd=np.full(30, choke.CD_NOMINAL),
                trim_gamma=np.concatenate([pop.vendor_gamma[g] for g in gsel])[:30], t_sea=cat('t_sea'),
                q_t=cat('q_t'), fluid=fl)
tab = well.vlp_table(tw)
t_build = time.perf_counter()
u_grid = np.linspace(0.05, 1.0, 39)
p2_grid = np.linspace(10e5, 100e5, 61)
rate_tab = edge.twin_tables(tw, tab, u_grid, p2_grid)
mls = pvt.mass_per_liquid_sc(fl)
line = network.Line(length=np.array([8000.0]), d=np.array([0.3048]), depth=np.array([150.0]),
                    p_sep=np.array([12e5]))
q_ref = rate_tab[:, 10, 4]
mix = pvt.mix(*choke.std_rates(q_ref * mls, fl), fl)
m_grid = np.linspace(0.0, (rate_tab[:, -1, 0] * mls).sum() * 1.1, 80)
p_req = edge.line_table(line, mix, 330.0, m_grid)
build_s = time.perf_counter() - t_build
step_lat = []
rng = np.random.default_rng(0)
for rep in range(500):
    u_vec = rng.uniform(0.1, 1.0, 30)
    t = time.perf_counter()
    edge.edge_step(rate_tab, u_grid, p2_grid, mls, m_grid, p_req, u_vec)
    step_lat.append((time.perf_counter() - t) * 1e3)
# edge step against the full network solve on the same twin (pressure agreement)
u_chk = np.full(30, 0.6)
p_edge, q_edge = edge.edge_step(rate_tab, u_grid, p2_grid, mls, m_grid, p_req, u_chk)
full = network.solve(control.replicate(tw, 1), control.replicate_table(tab, 1), u_chk[None, :], line, iters=40)

pc = lambda a, q: float(np.percentile(a, q))
summary = {
    'tree_parity_max_abs': tree_parity, 'vfm_parity_max_rel': float(max(parity)), 'wells_exported': len(wp_rows),
    'artifact_bytes': int(size), 'peak_traced_bytes_during_inference': int(peak),
    'well_eval_ms': {'p50': pc(lat, 50), 'p99': pc(lat, 99), 'max': float(max(lat))},
    'scan_all_wells_ms': {'wells': len(wp_rows), 'p50': pc(scan, 50), 'p99': pc(scan, 99),
                          'per_well_p99': pc(scan, 99) / len(wp_rows)},
    'optimizer_step_ms_30_wells': {'p50': pc(step_lat, 50), 'p99': pc(step_lat, 99)},
    'edge_tables_build_s': round(build_s, 1),
    'edge_vs_full_manifold_pressure_bar': [float(p_edge / 1e5), float(full['p_man'][0] / 1e5)],
    'edge_vs_full_total_liquid_rel': float((q_edge.sum() - full['q_l'][0].sum()) / full['q_l'][0].sum()),
    'deterministic': bool(np.array_equal(out1, out2)),
    'numpy_only_inference': True,
}
summary['gate'] = {
    'artifact_le_5MB': summary['artifact_bytes'] <= 5e6,
    'ram_le_100MB': summary['peak_traced_bytes_during_inference'] <= 100e6,
    'well_eval_p99_le_5ms': summary['well_eval_ms']['p99'] <= 5.0,
    'optimizer_step_p99_le_100ms': summary['optimizer_step_ms_30_wells']['p99'] <= 100.0,
    'deterministic': summary['deterministic'],
    'parity': summary['vfm_parity_max_rel'] < 1e-6 and summary['tree_parity_max_abs'] < 1e-9,
}
summary['seconds'] = round(time.time() - T0, 1)
with open('m7_summary.json', 'w') as fh:
    json.dump(summary, fh, indent=1)
print(json.dumps(summary, indent=1))
