# Documentation: milestone status and decisions

## Status
| Milestone | Status | Evidence |
|---|---|---|
| M0 | done, Nov 1 2026 | BENCHMARKS.md, RESEARCH.md |
| M1 | done, Nov 1 2026 | `sankalpsthakur/vfm-choke-m1-physics-tests` v1: 19/19; v4 on the final code: 29/29 (physics + model checks) |
| M2 | done, realism gate partly missed | `sankalpsthakur/vfm-choke-m2-simfield` v1, 6,677 s (reports/runs/vfm-choke-m2-simfield-v1/m2_summary.json) |
| M2b | done, all realism ranges met | `sankalpsthakur/vfm-choke-m2b-simfield` v1: 843 wells, 120 groups; flowing 0.985, critical 0.368, median liquid 5,353 stb/d, median WHP 55 bar, MPFM error median +0.15% |
| M3 | done: few-shot gate met; physics-margin and coverage gates missed (coverage fixed post hoc in M3c) | `vfm-choke-m3-fewshot` v3, `vfm-choke-m3b-fewshot` v1, `vfm-choke-m3c-fewshot` v1 |
| B1 | run, beat outright (protocol reconstructed) | `sankalpsthakur/vfm-choke-b1-volve` v3 (reports/runs/vfm-choke-b1-volve-v3/b1_results.json) |
| B2 | run, gate not met | `sankalpsthakur/vfm-choke-b2-dabiri` v2 (reports/runs/vfm-choke-b2-dabiri-v2/b2_results.json) |
| B3 | run, not counted (fidelity check failed) | `sankalpsthakur/vfm-choke-b3-dsouza` v1 (reports/runs/vfm-choke-b3-dsouza-v1/b3_results.json) |
| Audit | B2/B3 reproducibility audit public on Kaggle + GitHub; B3 explanation corrected | `sankalpsthakur/reproducing-two-virtual-flow-metering-benchmarks` v1 (private) / v2 (public) |
| M5 | done: both targets missed | `vfm-choke-m5-control` v2 |
| M6 | done: top-3 77% for leaks ≥ 2% (target 80%), false alarms 22.5% (target 5%) | `vfm-choke-m6a-slick-sim` v1, `vfm-choke-m6b-slick-attr` v1 |
| M7 | done: all edge criteria met (v5) | `vfm-choke-m7-edge` v5 |
| M8 | REPORT.md written; publish gate not met, project stays private | REPORT.md |

### M2 result (Nov 1 2026, v1), against the ranges written before the run
- 1,406 wells in 200 groups, 180 days, 253,080 well-days, 6,975 separator tests.
- Met: median liquid 4,941 stb/d (range 500-15,000); median wellhead pressure 67.8 bar (20-120); MPFM liquid error
  median +0.1%, 10th-90th percentile −3.1 / +3.4% (within ±5 / ±10%).
- **Missed:** flowing fraction 0.825 (bar > 0.90) and critical-flow fraction 0.063 (bar 0.10-0.90). Cause: high
  manifold back-pressure (median 51 bar; long, deep, narrow tie-backs), so most chokes run subcritical and some wells
  cannot open against the manifold.
- Decision: M3, M5, M6 and M7 run on this population (most real subsea chokes on long tie-backs run subcritical too),
  and M2b repeats the simulation with wider, shorter, shallower lines, lower separator pressure, smaller chokes and a
  50-bar flow margin (`sample_population` keyword ranges; defaults reproduce v1). M3 is re-run on M2b as a check.

### B1 result (Nov 1 2026, v3), against the bar written before the run
- 4 wells (F-11, F-12, F-14, F-15 D), 6,750 predicted days, 30-day walk-forward recalibration.
- Ours (physics prior + per-well latent + ratio tracking, all past labels): MAPE 12.12% [10.55, 13.84] (block
  bootstrap), R² 0.944, RMSE 330.7 Sm³/d. Published: MAPE 18%, R² 0.93, RMSE 333. CI upper below 18%: **beat
  outright** under the reconstructed protocol.
- Protocol check: our gradient-boosting replicate on the identical rows scores 14.62% [11.69, 19.25], within ±5 of
  18%, so B1 counts. Head-to-head against that replicate: difference −2.5 points, CI [−7.45, +0.70], not significant;
  the replicate has the lower RMSE (304.6) and higher R² (0.952). So: better MAPE than the published model, on par
  with a strong data-driven model refit monthly on all wells' history.
- One anchor label per 30 days instead of all daily labels: 21.61%, above 18%.
- Frozen 70/30 split replicate: 88% (published 66%), confirming that recalibration dominates.

### B3 result (Nov 1 2026, v1), against the bar written before the run
- Regenerated with the published simulator in Octave 8.4 (two-well model, 96 h, 1-minute averages).
- Every method is near-perfect on this data. Test MAPE, Well 1 / Well 2:
  - linear regression 0.0014 / 0.0021%;
  - MLP 0.0013 / 0.0126%;
  - XGBoost 0.12 / 0.03%;
  - LSTM 0.24 / 0.27%;
  - SVR 0.38 / 0.62%.
- The paper reports 1.53-16.69%, so its CSVs must differ from what the published simulator produces (noise or
  processing not in the released code). The fidelity check fails, so **B3 does not count**.
- For the record, on this regeneration: our physics-feature regression scores 0.011 / 0.017%; the physics prior with
  deliberately wrong design values scores 12.8 / 12.5% zero-shot and 1.14 / 1.42% after 3 labels.

### Correction to B3, and the B2/B3 audit (Nov 1 2026, after the report)

**The B3 explanation above was wrong.** The B3 v1 regeneration used only the DataverseNO two-well copy.
- **Why the scores were low.** Its test window (rows 4,033 onward) repeats the first supply step's state: well 1's
  rates at 90 h equal those at 20 h to every printed digit (64.38 / 7.92 / 78.58 kg/s). Every model scored < 1% for
  that reason, not because the simulator output is trivially clean.
- **What the paper shows instead.** The paper's Figs. 6-7, and the plots saved in the authors' GitHub notebooks, show
  the test window settling at a new level: well 1's largest curve is about 78 kg/s in step 1 and about 71 kg/s in the
  test window. Our own pre-run caveat in BENCHMARKS.md ("the test window is a new steady operating point") said so.
- **Corrected statement.** We could not regenerate the paper's data from either released copy. B3 stays excluded, and
  the "excels" verdict is unchanged.

**Audit run.** `sankalpsthakur/reproducing-two-virtual-flow-metering-benchmarks` v1, private, 8,614 s; outputs in
`reports/runs/reproducing-two-virtual-flow-metering-benchmarks-v1/`. It fetches every source at run time.

**B2 (Dabiri et al.):**
- **What matches.**
  - Table 3 matches exactly: all 40 entries, max |diff| 0.
  - Table 7 matches for Gilbert, Ros, Baxendell and Achong: AAPRE and APRE to the printed precision, RMSE within
    0.04. The paper's R² is the squared correlation.
  - Pilehvari stays unresolved: 122% against 51.82%.
- **Duplicates.** 124 rows sit in 44 groups (largest 10), and 6 input triples carry more than one rate. 19.7% of a
  random test set has an exact copy in training.
- **Table S3 replicate, 10 scalings.** The best is min-max inputs with a log target, at 20.30% [18.28, 22.32]; the
  best single split is 13.48%. The B2 run's 40.08% was the min-max/min-max cell of this grid.
- **Tuned models.** A tuned single SVR scores 13.54%; tuned Adaboost-SVR scores 23.64% (unstable).
- **Generic models.** LightGBM 12.23%, random forest 14.20%, 1-NN 13.69%, power law 18.73%.
- **LightGBM over 200 splits.** Mean 11.88%, best 7.09%, 5th percentile 8.95%.
- **Grouped splits.** LightGBM 13.92%, 1-NN 15.21%.
- **Table 8.** 8 of 28 predictions equal the measured value, and 4 equal another measured value; AAPRE 11.33%.
- **Settings tables.** In Table S3 the RBF row equals the Adaboost-SVR row and the MLP-LM row equals MLP-BR;
  scikit-learn rejects `solver='scg'`.

**B3 (D'Souza et al.):**
- **Data not released.** Neither deposit has the CSVs. The notebooks read a 5,762-row well-1 file and never compute
  MAPE.
- **GitHub script.** It requests 20 outputs from `calculate_output`, which declares 17 (`rho_l`, `V_G` and `rho_m` are
  missing). Its CSV writer is commented out.
- **Both copies (GitHub single-well; DataverseNO two-well).** The test window returns to step 1 (0.0% change), and all
  ten of the paper's models score below 1% (largest 0.83%).
- **GitHub copy.** Its oil/liquid ratio is 0.762 (water cut 0.2), which matches the figure's ratio, but its levels
  don't (liquid 92.5 kg/s).

**Published (Sankalp's go).**
- **Kaggle:** v2 is public with markdown only (code cells identical to v1, checked against the pulled v1 source), with
  the tags energy, regression, lightgbm, data cleaning and data visualization.
- **GitHub:** `sankalpsthakur/reproducing-vfm-benchmarks`, commit 03b5717. It holds the audit only, with v1's
  results.

### M7 v2 (Nov 1 2026): edge package, first measurement
- Met:
  - tree export parity 7.8e-16 (absolute); edge VFM against the full model 4.4e-16 (relative);
  - artifact 233 KB (29 wells + 500 trees);
  - peak traced memory 144 KB;
  - 30-well optimizer step on edge tables p99 1.7 ms;
  - deterministic.
- **Missed:** single-well evaluation p99 26.3 ms against ≤ 5 ms.
  - tracemalloc was running inside the timing loop, which inflates every numpy allocation.
  - v3 times without tracing, adds a batched scan of all wells, and refines the edge tables: the v2 table step was
    2.3% off the full network solve in manifold pressure and 3.1% in total liquid.

### M3 result (Nov 1 2026, v3), against the margins written before the run
- Protocol:
  - 1,219 held-out wells in 5 group folds; calibration labels from days < 90, scored on days 90-179 against truth;
  - median-well MAPE with bootstrap 95% CI;
  - reports/runs/vfm-choke-m3-fewshot-v3/m3_summary.csv.
- Liquid MAPE with MPFM labels, at k = 0 / 1 / 3 / 30:

  | Model | k = 0 | k = 1 | k = 3 | k = 30 |
  |---|---|---|---|---|
  | Hybrid (ours) | 7.64 [7.19, 8.16] | 5.73 | 5.43 | 4.67 |
  | Physics prior with per-well multiplier | 8.33 | 5.98 | 5.76 | 5.18 |
  | Pooled black box | 20.35 | 10.18 | 10.10 | 9.54 |
  | Per-well LightGBM | — | 18.92 | 18.97 | 13.09 |
  | Gilbert | 36.49 | 23.55 | 23.37 | 21.94 |

- Separator-test labels, hybrid: k = 1 / 2 / 3 give 5.04 / 4.31 / 3.28 [3.05, 3.56]; physics gives 5.39 / 4.70 / 4.19.
- Gates:
  - k = 0 beats Gilbert: **met**.
  - k = 0 beats pure physics with non-overlapping CIs: **not met** (7.64 [7.19, 8.16] vs 8.33 [7.83, 8.92]).
  - k ≤ 3 within 1.25× of the best per-well model at k = 30 (physics with per-well multiplier, 5.18): **met** (1.11×,
    1.08×, 1.05×).
  - 90% coverage within 85-95%: **not met** (78% at k = 0, 53-67% at k ≥ 1). The intervals leave out water-cut and
    choke drift after the last label, and the stale ratio estimates.
- Oil error (8.5-11%) is dominated by water-cut drift between the last label and the scoring window, for every model.
- Caveat: SimField's truth uses the same choke-physics family as the prior, so the physics prior is privileged here.
  The real-data benchmarks (B1, B2) are the check on model-form error.

### M7 v3-v4 (Nov 1 2026): latency without tracing, faster solver
- v3 (timing without tracemalloc):
  - single well p50 5.39 ms, p99 7.07 ms;
  - batched scan of 29 wells p99 21.4 ms (0.74 ms per well);
  - optimizer step p99 2.1 ms.
- v4 (derivative-bisection critical ratio, 40 steps, PVT state computed once):
  - single well p50 3.64 ms, **p99 5.19 ms (bar 5 ms: just missed)**;
  - batched scan 0.70 ms per well;
  - optimizer step p99 1.5 ms;
  - artifact 233 KB, peak traced memory 32 KB, parity 4e-16, deterministic.
- The edge table step stays 2.6% below the full solve in manifold pressure and 3.3% above it in liquid, even with
  finer grids. The edge flowline table freezes the commingled composition and temperature: a modelling shortcut,
  not resolution.
- v5 uses 32 bisection steps (2e-10 precision). That is the last latency change; whatever it measures is reported.
- **v5 result:** single well p50 3.08 ms, p99 4.14 ms; batched scan 0.59 ms per well; optimizer step p99 1.26 ms;
  artifact 233 KB; peak traced memory 32 KB; parity 4e-16; deterministic. **All edge criteria met** on a Kaggle CPU
  core (reports/runs/vfm-choke-m7-edge-v5). An ARM edge box may be 2-4× slower per core; the batched scan leaves the
  most headroom.

### M6 result (Nov 1 2026; M6a v1 simulation 1,861 s + sensitivities 950 s; M6b v1 attribution 5 s)
- 240 leaks: top-1 23.3%, top-3 61.7%; kind correct 46%.
- By size (top-1 / top-3):
  - ≤ 1%: 15 / 60%;
  - 1-2%: 37 / 60%;
  - 2-5%: 44 / 74%;
  - > 5%: 75 / 83%.
- By kind, top-3: jumper 52%, manifold 42%, flowline 68%, riser 91%.
- Targets: top-3 for leaks ≥ 2%: 76.9% (target ≥ 80%, **not met**). False alarm rate on controls 22.5% (target ≤ 5%,
  **not met**).
- Controls: "none" ranked first in 100%. CUSUM alarm before detection on 39.6% of leaks, median delay 3 days.
- Demo: a 1.96% jumper leak on slot 3 ranked 4th (posterior 0.08); slots 2 and 1 have similar ratios.

### M5 result (Nov 1 2026, v2, 7,772 s), against the targets written before the run
- Single-well setpoint within ±5%: 16% / 35% / 36% of achievable trials at k = 0 / 3 / 30 (target ≥ 90% at k = 3:
  **not met**). Median miss 13.4 / 7.7 / 7.5%. The +10% targets were achievable in 56-70% of trials.
- Multi-well:
  - Oil gain over the equal-scaling heuristic +33% [23, 43] / +37% [24, 51] / +34% [23, 47].
  - Gas-cap violations 0.
  - Water-cap violations 38 / 38 / 50% of groups (target ≤ 10%: **not met**), by 2-19%.
- Water cut stale by 30+ days and outside the cautious margin. SLSQP hit its 20-iteration cap in 14 of 24 runs.

### M6 v2 failure and restructure (Nov 1 2026)
- v2 simulated 400 scenarios × 60 days (5,361 s) and the sensitivities (1,797 s). The attribution loop then hit
  a zero phase direction: a well shut in through the whole reference window. Nothing had been saved.
- Fixes:
  - `leak.phase_loglik` returns λ = 0 for an unusable direction;
  - such a jumper hypothesis uses the commingled mix direction.
- Split into M6a (`vfm-choke-m6a-slick-sim`: simulate, sensitivities, save) and M6b (`vfm-choke-m6b-slick-attr`:
  attribution from M6a's files).
- Smaller run: 240 leaks + 80 controls over 46 days (detection is at most day 45).

### M3b (Nov 1 2026, on M2b)
- k = 0: hybrid 8.51 [8.01, 9.01], physics 9.70 [9.01, 10.35], black box 14.13, Gilbert 19.93.
- With labels:
  - k = 1 / 3 / 30: hybrid 6.33 / 6.35 / 5.19;
  - per-well reference (physics at k = 30) 6.08, so ratios 1.04 / 1.03 / 1.05;
  - 3 separator tests: 3.70 [3.44, 3.96] vs physics 4.87.
- Coverage: 85% at k = 0, 59-75% after labels.
- Conclusions unchanged from M3.

### Interval calibration (decided Nov 1 2026 after M3 v3 under-covered)
- `fleet.train(..., interval_scales=True)` fits a split-conformal width per regime and label count on training
  wells only. Each training well goes through the same label-then-forecast protocol as held-out wells, with the inner
  model that never saw its group; the errors are measured against MPFM labels.
- The held-out wells were not used to fit it. The decision to add it was prompted by the held-out coverage result,
  so M3c's coverage is a post-hoc fix and is labelled as one.
- Run: `vfm-choke-m3c-fewshot` (same data and folds as M3 v3).
- Result (v1):
  - Coverage of the 90% intervals: MPFM k = 0, 1, 2, 3, 5, 10, 30 → 89.1, 86.7, 86.0, 85.7, 85.6, 85.1, 84.8%;
    tests k = 1-3 → 87.4, 86.7, 86.6%. Up from 53-78%.
  - Point predictions unchanged. The pre-set band 85-95% is met everywhere except k = 30 (84.8%).
  - Scales 1.07-2.69. They are largest in folds 1 and 3, where the empirical-Bayes τ1 hit its floor (0.001).

### Critical-ratio solver change (Nov 1 2026, after M3-M6 were launched)
- `choke.critical_ratio` now bisects on the analytic derivative of the mass flux (dE/dy = −D makes it one expression)
  instead of golden-section search on the flux.
- Same answer: the new test `test_critical_ratio_matches_flux_maximiser` agrees with an 80-step golden search to
  < 1e-6 over 3,003 random states, and its flux is never lower.
- About 2× fewer evaluations; the test suite went from 147 s to 81 s.
- Runs launched before the change (M3, M5, M6, M2b) used the golden version. The difference is below solver tolerance.
- Test suite v3: 29/29 passed.

### "Excels" verdict
Counting benchmarks: B1 (beat outright, protocol reconstructed). B2 fails its bar; B3 is excluded by its fidelity
check. One of the required two, so the publish condition is **not met**; the project stays private per the plan.

### B2 result (Nov 1 2026, v1), against the bar written before the run
- Parsing check: our Gilbert, Ros, Baxendell and Achong AAPRE on all 565 rows are 22.36, 21.32, 20.36 and 21.05%,
  identical to the paper's Table 7. Our Pilehvari constants do not reproduce its 51.82% (we get 122%); excluded.
- Zero-shot physics (no fitting): AAPRE 20.14%, R² 0.816. Better than Gilbert, Ros, Baxendell, Achong, Al-Attar and
  Pilehvari as published; worse than the two correlations fitted to data (Beiranvand 19.03%, the paper's refit 17.20%).
  Bar (< 17.20%) **not met**.
- 80/20 random splits, 20 seeds: our hybrid (physics prior × LightGBM residual, chosen by inner CV in 20/20 splits)
  11.49% [10.39, 12.59]; raw LightGBM 12.23%; power-law refit 18.73%; Adaboost-SVR replicate with the paper's
  stated defaults 40.08%. The paper's 4.57% could not be reproduced. Bar (CI upper < 4.57%) **not met**.
- Few-shot global multiplier: k = 1, 3, 10 give 25.3, 20.7, 19.1%. Bar **not met**.
- Data finding: 124 of 565 rows (22%) belong to exact-duplicate groups (44 distinct records; one repeated 10
  times), so random splits leak copies from train to test.
- Extra, v2 (not the gate), on 80/20 splits that keep duplicate groups on one side: ours 12.64% [11.29, 13.99]; raw
  LightGBM 13.95%; power law 18.63%; Adaboost-SVR replicate 42.30%. A pure 1-nearest-neighbour memoriser scores 13.69%
  on the published random split, so duplicates alone do not explain the paper's 4.57%.

## Decisions
- **Nov 1 2026, scope (Sankalp):** public + simulated data; VFM with choke inverse plus multi-well optimizer; gamma MPFM
  readings as labels; oil-slick attribution; private project and report first. Publish publicly only if the model
  excels on public benchmarks and is edge-deployable, and only after an explicit go.
- **Nov 1 2026, benchmarks (M0 gate):** kept B1 Volve (Ebrahimzadeh 2026 baseline), B2 Dabiri et al. 2024 (565 choke
  tests), B3 D'Souza et al. 2024 (two simulated gas-lifted subsea wells). Dropped Akbari et al. 2025 (no public
  data), HVFM (no dataset download; needs OLGA), 3W 2.0 (no comparable published baseline, no leak events). B1 counts
  toward "excels" only if its protocol can be matched from the paper.
- **Nov 1 2026, after the report (Sankalp):** "commit it and do the reproducibility audit write-up", then "publish on
  kaggle and github once it's done". Initial commit 24e21b5 (authored by Sankalp, no agent trailer). The audit covers
  B2 and B3 only and runs as the Kaggle notebook `sankalpsthakur/reproducing-two-virtual-flow-metering-benchmarks`
  (source `kaggle/audit/reproducing_vfm_benchmarks.py`), private until its numbers are checked. On GitHub it goes to
  its own repository with the audit only; this project's code stays private, since its publish gate was not met and
  `reports/runs/*b3-probe*/github_*.m` are copies of an unlicensed repository.
- **Nov 1 2026, ~17:00 UTC (Sankalp):** "audit only is fine, go ahead". Audit v1 (private, 8,614 s) completed with
  every section; v2 changes only markdown (code cells checked identical to v1) and is pushed public with tags energy,
  regression, lightgbm, data-cleaning, data-visualization via `kaggle/publish.py`.
