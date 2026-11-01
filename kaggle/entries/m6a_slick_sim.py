# M6a: simulate the slick scenarios and the pressure sensitivities; save them for M6b. Held-out (fold 0) groups are copied into scenarios: leaks of random kind,
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
DAYS, N_LEAK, N_CTRL, REF = 46, 240, 80, (5, 25)
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


# keep everything M6b needs, so attribution can be rerun without re-simulating
import pickle as _pk  # noqa: E402
d.to_parquet('m6_daily.parquet', index=False)
g.to_parquet('m6_groups.parquet', index=False)
np.save('m6_openings.npy', openings)
with open('m6_state.pkl', 'wb') as fh:
    _pk.dump({'kind': kind, 'slot': slot, 'pos': pos, 'd_hole': d_hole, 'onset': onset, 'detect': detect,
              'base': base, 'line_length': np.asarray(ps.line.length), 'line_depth': np.asarray(ps.line.depth),
              'mask': ps.mask, 'W': W, 'sens': sens, 'lay': lay, 'REF': REF, 'S': S}, fh)
log('saved scenario data')
