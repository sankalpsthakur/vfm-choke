# %% [markdown]
# # VFM-Choke: Few-Shot Flow Metering for New Wells
#
# **TL;DR**
# - **One choke flow model for every well.** It uses no well identity, so it works on a well it has never seen,
#   starting from design data alone.
# - **Accuracy on simulated wells.** On 1,219 simulated wells it never saw, the median liquid-rate error is 7.6%
#   from design data. One to three labels (gamma-meter readings or separator tests) bring it within 1.05-1.11× of a
#   per-well model fitted on 30 labels.
# - **Built on top of it:**
#   - choke setpoints;
#   - a multi-well optimiser;
#   - a ranked answer to "which source fed this oil slick";
#   - a numpy-only edge package (233 KB, p99 4.1 ms per well).
# - **The bar was missed.** The pre-registered "excels on two public benchmarks" bar was not met. Volve beats its
#   published 18% (12.1%), but the other two benchmarks could not be reproduced
#   ([audit](https://www.kaggle.com/code/sankalpsthakur/reproducing-two-virtual-flow-metering-benchmarks)).
#   Setpoint accuracy and slick attribution also miss their pre-set targets. All of it is reported here.
#
# This notebook explains the approach end to end and runs a small version of it live. The full-scale numbers are
# loaded from the run outputs committed to the public repository:
# [github.com/sankalpsthakur/vfm-choke](https://github.com/sankalpsthakur/vfm-choke).

# %% [markdown]
# ## The problem
# A subsea field may have dozens of wells, and each usually gets its own virtual flow meter (VFM): a model fitted to
# that well's tests, with its own discharge coefficient, gas-oil ratio and water cut. A new (greenfield) well has none
# of that history. This project asks whether one general model can meter a new well from its design data, then
# tighten after one to three labels. The labels can be test-separator results or readings from a gamma-ray
# multiphase meter (MPFM).
#
# **Approach in five pieces.**
# 1. **Physics prior from design data.** A mechanistic two-phase choke model (critical and subcritical flow) with
#    black-oil PVT, at the measured wellhead and downstream pressures, opening and bean size. No field fitting.
# 2. **Pooled correction.** A monotone LightGBM model learns a multiplicative correction across many wells from inputs
#    that mean the same thing everywhere: pressure ratio, critical-ratio margin, gas volume fraction, water cut, GOR
#    and opening. It sees no well identity.
# 3. **Few-shot calibration.** Each well has a two-number latent (a log multiplier and a trim-exponent error) with an
#    empirical-Bayes prior learned from other wells. Each label updates it in closed form, with its own noise level.
# 4. **Control.** Setpoints come from a bisection on the monotone model; a constrained optimiser handles a whole
#    manifold.
# 5. **Slick attribution.** A posterior over leak sources from the mass-balance deficit's phase signature, pressure
#    shifts, the slick's drift-corrected position and its volume.

# %% [markdown]
# ## Setup
# The package comes from the public repository, pinned to the release commit.

# %%
import json
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

T0 = time.time()
REPO, COMMIT = Path('/tmp/vfm-choke'), '696c290'
if not REPO.exists():
    subprocess.run(['git', 'clone', '-q', 'https://github.com/sankalpsthakur/vfm-choke', str(REPO)], check=True)
subprocess.run(['git', '-C', str(REPO), 'checkout', '-q', COMMIT], check=True)
sys.path.insert(0, str(REPO / 'src'))
from vfmchoke import choke, fleet, hybrid, pvt, simfield, units as u  # noqa: E402

RUNS = REPO / 'reports' / 'runs'
print('vfm-choke at', subprocess.run(['git', '-C', str(REPO), 'rev-parse', '--short', 'HEAD'], capture_output=True,
                                     text=True).stdout.strip())

# %% [markdown]
# ## 1. The physics prior: one choke, critical and subcritical
# The rate through a choke rises as the downstream pressure falls, until the flow turns critical; below that the
# downstream pressure no longer matters. Gilbert-type correlations assume critical flow everywhere. A subsea choke on a
# long tie-back often runs subcritical.

# %%
fluid = pvt.Fluid(api=32.0, gas_sg=0.75, gor=u.scf_stb_to_si(600.0), wc=0.2, salinity=0.035)
p1 = u.psig_to_pa(1200.0)
p2 = np.linspace(u.P_SC, 0.98 * p1, 120)
m = choke.mass_rate(np.full_like(p2, p1), p2, u.c_to_k(60.0), fluid, 1.0, u.choke64_to_m(32.0), 0.80)
q_mech = u.si_to_stb_d(m / pvt.mass_per_liquid_sc(fluid))
q_gil = choke.gilbert_rate(1200.0, 32.0, 600.0 * (1 - 0.2), 'gilbert')
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(p2 / p1, q_mech, label='mechanistic (this prior)')
ax.axhline(q_gil, color='grey', ls='--', label='Gilbert (critical flow assumed)')
ax.set_xlabel('downstream / upstream pressure')
ax.set_ylabel('liquid rate, stb/d')
ax.set_title('One 32/64 in choke at 1,200 psig, GOR 600 scf/stb, water cut 0.2')
ax.legend()
plt.tight_layout()
plt.show()
print(f'rate at p2/p1 = 0.2: {q_mech[np.argmin(abs(p2 / p1 - 0.2))]:,.0f} stb/d; '
      f'at 0.9: {q_mech[np.argmin(abs(p2 / p1 - 0.9))]:,.0f} stb/d; Gilbert: {float(q_gil):,.0f} stb/d')

# %% [markdown]
# ## 2. A small fleet, end to end
# SimField simulates manifold groups of subsea wells. Each well → jumper → manifold → flowline → riser → separator
# chain is solved as one network, with each well's design data (reservoir pressure, productivity, tubing, choke
# size and trim, fluid) and its true parameters differing from the design. The simulation includes:
# - daily operation with choke moves;
# - decline and a rising water cut;
# - monthly separator tests;
# - gamma-meter readings with realistic errors.
#
# The full study used 200 groups (1,406 wells) and a second population of 120 groups (843 wells). This live run uses
# 16 groups for 120 days, so it finishes in a few minutes.

# %%
t1 = time.time()
pop = simfield.sample_population(n_groups=16, max_wells=10, seed=11, choke_frac=0.45,
                                 line_d=(0.254, 0.3048, 0.3556), depth=(30, 250), p_sep=(5e5, 15e5),
                                 length=(1000, 8000), flow_margin=50e5)
res = simfield.simulate(pop, days=120, seed=11, log_every=40)
for name, frame in (('daily', res['daily']), ('tests', res['tests']), ('wells', simfield.wells_frame(pop))):
    frame.to_parquet(f'/tmp/sim_{name}.parquet', index=False)
daily, tests, wells = fleet.prepare(pd.read_parquet('/tmp/sim_daily.parquet'), pd.read_parquet('/tmp/sim_tests.parquet'),
                                    pd.read_parquet('/tmp/sim_wells.parquet'))
print(f'{int(pop.mask.sum())} wells in {pop.mask.shape[0]} groups simulated in {time.time() - t1:.0f} s; '
      f'{len(daily):,} well-days, {len(tests):,} separator tests')

# %% [markdown]
# **Train on half the groups, meter the other half.** The general model is trained on 8 groups. For every well in
# the other 8, it predicts days 90-119 from:
# - design data alone (k = 0);
# - then 1 and 3 gamma-meter readings taken before day 90.
#
# Two baselines get the same labels, as one global multiplier each: the physics prior alone, and Gilbert.

# %%
CAL_END = 90
groups = np.sort(daily.group.unique())
train_groups = set(groups[::2])
trained = fleet.train(daily[daily.group.isin(train_groups) & daily.flowing], tests)


def gilbert_mass(df, fl):
    s_eff = u.m_to_choke64(df.d_choke.to_numpy()) * np.sqrt(choke.trim(df.opening.to_numpy(),
                                                                         df.vendor_gamma.to_numpy()))
    q_l = u.stb_d_to_si(choke.gilbert_rate(u.pa_to_psig(df.p_wh.to_numpy()), s_eff,
                                           u.si_to_scf_stb(fl.gor * (1 - fl.wc))))
    return q_l * pvt.mass_per_liquid_sc(fl)


rows = []
for wid, tw in daily[~daily.group.isin(train_groups)].groupby('well_id'):
    ev = tw[(tw.day >= CAL_END) & tw.flowing]
    if len(ev) < 5:
        continue
    src = tw[tw.flowing & (tw.day < CAL_END)].sort_values('day')
    for k in (0, 1, 3):
        cal_rows = src.head(k)
        cal = fleet.calibrate(cal_rows, wells.loc[wid], trained, 'mpfm')
        m_h, lo, hi, fl = fleet.predict(ev, cal, trained)
        e = fleet.with_est(ev, cal.wc, cal.gor)
        m_phys, m_gil = hybrid.prior(e, fl)[0], gilbert_mass(e, fl)
        if len(cal_rows):
            c = fleet.with_est(cal_rows, cal.wc, cal.gor)
            flc, lab = hybrid.fluid_from(c), c.label_m.to_numpy()
            m_phys = m_phys * np.exp(np.mean(np.log(lab / hybrid.prior(c, flc)[0])))
            m_gil = m_gil * np.exp(np.mean(np.log(lab / gilbert_mass(c, flc))))
        for method, m_hat in (('general model', m_h), ('physics prior', m_phys), ('Gilbert', m_gil)):
            q_o, q_g, q_w = choke.std_rates(m_hat, fl)
            rows.append({'well': wid, 'k': k, 'method': method,
                         'liquid_ape': float(np.mean(np.abs(q_o + q_w - ev.true_q_l.to_numpy()) / ev.true_q_l.to_numpy()))})
live = pd.DataFrame(rows)
table = live.pivot_table(index='method', columns='k', values='liquid_ape', aggfunc='median') * 100
print(f'held-out wells: {live.well.nunique()}; median liquid error, % ({time.time() - t1:.0f} s so far)')
table.round(1)

# %% [markdown]
# ## 3. The full-scale result (from the committed runs)
# The same evaluation on the full SimField population, M3 v3, scores 1,219 held-out wells in 5 group folds; the
# bracketed ranges are bootstrap 95% intervals. The reference is the better per-well model fitted on 30 labels.

# %%
m3 = pd.read_csv(RUNS / 'vfm-choke-m3-fewshot-v3' / 'm3_summary.csv')
mp = m3[m3.regime == 'mpfm']
fig, ax = plt.subplots(figsize=(8, 4.5))
for method, label in (('hybrid', 'general model'), ('physics', 'physics prior + multiplier'),
                      ('gilbert', 'Gilbert + multiplier'), ('pooled_blackbox', 'pooled black box + multiplier'),
                      ('perwell_gbm', 'per-well model')):
    s = mp[mp.method == method].sort_values('k')
    if len(s):
        ax.plot(s.k.clip(lower=0.5), s.liquid_median * 100, marker='o', label=label)
gate = json.loads((RUNS / 'vfm-choke-m3-fewshot-v3' / 'm3_gate.json').read_text())
ax.set_xscale('log')
ax.set_xticks([0.5, 1, 2, 3, 5, 10, 30])
ax.set_xticklabels(['0', '1', '2', '3', '5', '10', '30'])
ax.set_xlabel('gamma-meter labels on the new well (k)')
ax.set_ylabel('median liquid-rate error, %')
ax.set_title('1,219 held-out simulated wells (M3 v3)')
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()
print('reference:', gate['reference'], '| general model at k = 1, 2, 3 over the reference:',
      [round(gate[f'k{k}_hybrid_over_reference'], 3) for k in (1, 2, 3)])
mp[mp.k.isin([0, 1, 3, 30])].pivot_table(index='method', columns='k', values='liquid_median').mul(100).round(2)

# %% [markdown]
# ## 4. Public benchmarks

# %%
b1 = json.loads((RUNS / 'vfm-choke-b1-volve-v3' / 'b1_results.json').read_text())
b2 = json.loads((RUNS / 'vfm-choke-b2-dabiri-v2' / 'b2_results.json').read_text())
b3 = json.loads((RUNS / 'vfm-choke-b3-dsouza-v1' / 'b3_results.json').read_text())
ours_b1 = b1['physics_latent_all_past']['four_wells']
bench = pd.DataFrame([
    {'benchmark': 'Volve, 4 wells, 30-day walk-forward (B1)', 'published best': 'MAPE 18%',
     'ours': f"{ours_b1['mape_pct']:.1f}% [{ours_b1['mape_ci95'][0]:.1f}, {ours_b1['mape_ci95'][1]:.1f}]",
     'verdict': 'beats it; protocol reconstructed from the abstract'},
    {'benchmark': 'Dabiri et al. 2024, 565 choke tests (B2)', 'published best': 'AAPRE 4.57%',
     'ours': f"{b2['split_80_20']['ours']['mean']:.1f}% (20 random 80/20 splits)",
     'verdict': 'misses; the 4.57% could not be reproduced'},
    {'benchmark': "D'Souza et al. 2024, 2 simulated wells (B3)", 'published best': 'LSTM 1.96 / 1.53%',
     'ours': 'not scored',
     'verdict': f"excluded: on our regeneration even linear regression scores "
                f"{b3['well1']['mv_linear']:.4f}%, so the paper's data could not be regenerated"},
])
bench

# %% [markdown]
# - **Volve.** The paper's protocol had to be reconstructed from its abstract. Our own gradient-boosting replicate of
#   the published method scores 14.6% on the same rows, and the margin over it is not significant.
# - **Dabiri and D'Souza.** Both are covered in the separate
#   [reproducibility audit](https://www.kaggle.com/code/sankalpsthakur/reproducing-two-virtual-flow-metering-benchmarks).
#
# So only one benchmark counts, and the pre-registered bar of two is not met.

# %% [markdown]
# ## 5. Choke setpoints, the multi-well optimiser, slick attribution and the edge package (from the committed runs)

# %%
m5 = json.loads((RUNS / 'vfm-choke-m5-control-v2' / 'm5_summary.json').read_text())
control = pd.DataFrame({f'k = {k}': {
    'single-well setpoint within ±5%': f"{m5[k]['single_within_5pct']:.0%}",
    'median setpoint error': f"{m5[k]['single_median_abs_err']:.1%}",
    'optimiser oil gain vs equal scaling': f"{m5[k]['multi_oil_gain_mean']:+.0%}",
    'groups breaking the water cap': f"{m5[k]['violations_water_opt']:.0%}"} for k in ('0', '3', '30')})
control

# %% [markdown]
# **Control (M5 v2, 8 held-out manifold groups).**
# - **Setpoints.** A single-well setpoint lands within ±5% of its target only about a third of the time. The weak
#   link is the twin's deliverability, built from design data with errors, not the choke model.
# - **Optimiser.** It recovers a third more oil than scaling every choke equally. It breaks the water cap in 3 of 8
#   groups because its water cut was a month stale.
# - **Fixes.** Close the loop, feed it fresh meter ratios, and carry water-cut uncertainty into the constraint.

# %%
m6 = json.loads((RUNS / 'vfm-choke-m6b-slick-attr-v1' / 'm6_summary.json').read_text())
by_size = pd.DataFrame(m6['by_size']).T.rename_axis('leak, fraction of line flow')
fig, ax = plt.subplots(figsize=(7, 3.5))
by_size[['top1', 'top3']].mul(100).plot.bar(ax=ax, rot=0)
ax.axhline(80, color='crimson', ls='--', lw=1)
ax.set_ylabel('% of leaks')
ax.set_title(f"Source ranking over {m6['leaks']} simulated leaks (target: top-3 >= 80% for leaks >= 2%)")
plt.tight_layout()
plt.show()
print(f"top-1 {m6['top1']:.0%}, top-3 {m6['top3']:.0%}; leaks >= 2% of flow: top-3 {m6['target_top3_ge_2pct']:.0%}; "
      f"false alarms on no-leak controls {m6['false_alarm_rate_controls']:.1%}")

# %% [markdown]
# **Slick attribution (M6, 240 leaks and 80 no-leak controls).**
# - **Where it works.** Mass balance localises a leak to the line well. Big leaks are ranked well.
# - **Where it doesn't.** Telling neighbouring wells apart needs their ratios to differ, and often they don't.
# - **Against the targets.** For leaks of at least 2% of flow, top-3 reaches 77% against a target of 80%. The
#   early-warning alarm fires falsely on 22.5% of controls against a 5% target.

# %%
m7 = json.loads((RUNS / 'vfm-choke-m7-edge-v5' / 'm7_summary.json').read_text())
pd.Series({'artifact size': f"{m7['artifact_bytes'] / 1e3:.0f} KB ({m7['wells_exported']} wells + trees)",
           'one-well evaluation, p99': f"{m7['well_eval_ms']['p99']:.1f} ms",
           'all-well scan, p99 per well': f"{m7['scan_all_wells_ms']['per_well_p99']:.2f} ms",
           '30-well optimiser step, p99': f"{m7['optimizer_step_ms_30_wells']['p99']:.2f} ms",
           'tree export parity': f"{m7['tree_parity_max_abs']:.0e}",
           'numpy-only, deterministic': str(m7['numpy_only_inference'] and m7['deterministic'])}, name='M7 v5').to_frame()

# %% [markdown]
# **Edge.** Inference needs only numpy. The criteria were an artifact of at most 5 MB and a p99 of at most 5 ms per
# well on one CPU core, and all of them are met. They were measured on a Kaggle Xeon core; an ARM edge box may be
# slower.

# %% [markdown]
# ## What this does not show
# - **Simulation flatters the model.** SimField's physics comes from the same family as the prior, so the simulated
#   errors are optimistic. The real-data checks are Volve (one small field, allocated rates) and the literature choke
#   tests.
# - **Gamma-meter labels carry their own error.** That error is modelled here, but not from field data.
# - **Mass balance stops at the line.** It localises a leak to the line; telling wells apart needs distinct ratios,
#   pressure signatures or oil fingerprinting.
#
# ## Reproduce, licence and credit
# - **Reproduce.** Every full-scale number above comes from a Kaggle run whose outputs are committed under
#   `reports/runs/`. The repository's README explains how to re-run one.
# - **Licence.** Code is under Apache-2.0; text, figures and results under CC BY 4.0.
# - **Please credit this work** if it helps yours. The citation is in the repository's `CITATION.cff`:
#   Thakur, S. (2026), *vfm-choke*, https://github.com/sankalpsthakur/vfm-choke.

# %%
print(f'notebook finished in {time.time() - T0:.0f} s')
