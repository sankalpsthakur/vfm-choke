# vfm-choke: report (Nov 1 2026)

Every number below comes from a private Kaggle kernel; the run is named next to it and its outputs are under
`reports/runs/<kernel>-v<version>/`. The bars were written in BENCHMARKS.md and Plans.md before each run.

## Bottom line
- **The greenfield question:** on 1,219 simulated wells the general model never saw, design data alone gives a
  median liquid-rate error of 7.6%. One to three labels bring it to 5.7-5.4% (MPFM) or 5.0-3.3% (separator tests),
  within 5-11% of a per-well model fitted on 30 labels. The physics prior carries most of that: the learned
  correction adds 0.3-0.7 points, and the k = 0 gain over pure physics is not significant by the pre-set test.
- **Public benchmarks:**
  - Volve (real field): 12.1% MAPE against the published 18%, under a protocol reconstructed from the paper's
    abstract. Against our own monthly-refit gradient-boosting replicate on the same rows (14.6%) the margin is not
    significant (CI −7.45 to +0.70), and the replicate has the better RMSE and R².
  - Dabiri et al. 2024 (565 literature choke tests): the bar is not met; the published best number could not be
    reproduced.
  - D'Souza et al. 2024 (simulated gas-lift wells): excluded, because the published simulator does not reproduce the
    published data.
  - So the pre-registered "excels on two benchmarks" condition is **not met**, and nothing is published.
- **Edge:** all criteria met on a Kaggle Xeon core in the final run (v5): 233 KB artifact, p99 4.1 ms per well,
  0.6 ms per well in a batched scan, 1.3 ms per 30-well optimizer step.
  - It passed after three logged latency fixes (v2-v5).
  - An ARM edge box may be slower.
  - The edge table step is about 3% off the full network solve.
- **Control:**
  - Open-loop single-well setpoints land within ±5% only 35% of the time with 3 labels. The twin's deliverability
    is the weak link, not the choke model.
  - The multi-well optimiser recovers much more oil than an equal-scaling heuristic (+33-37%), but breaks the water
    cap in 3-4 of 8 groups because its water cut was a month stale.
  - Both pre-set targets are missed; the fixes are known (closed loop, fresh MPFM ratios, water-cut uncertainty in
    the constraint).
- **Slick attribution:**
  - The true source is in the top 3 for 77% of leaks ≥ 2% of line flow (target 80%), and the "no leak" hypothesis
    wins every control.
  - Single-well attribution is limited when neighbouring wells have similar GOR and water cut, which is the physical
    limit mass balance runs into.

## 1. Research (RESEARCH.md)
- The closest prior art is multi-task / few-shot VFM: Løvland, Grimstad & Imsland, arXiv 2309.15828 (80 wells; 1-3
  points often enough for a new well).
- Hybrid physics + ML is where industry is (SPE 233457, Oct 2026; SLB Olga Online).
- Validation design dominates reported accuracy: Volve random split R² 0.98, frozen split MAPE 66%, walk-forward
  18%.
- No public foundation model for VFM exists. No public method ties a SAR slick to one well; mass balance localises
  to the line.

## 2. Physics core and simulated fleet (M1, M2)
- 29/29 invariants pass on the final code (`vfm-choke-m1-physics-tests` v4), including:
  - mass conservation through leaks;
  - monotone choke rate;
  - continuity at the critical ratio;
  - exact gamma-MPFM inversion without noise;
  - Gilbert unit conversion;
  - tree-export parity.
- SimField (`vfm-choke-m2-simfield` v1): 1,406 wells, 200 manifolds, 180 days, 6,975 separator tests.
  - Met the rate, pressure and MPFM-error ranges.
  - **Missed** two realism bars: flowing fraction 0.825 (> 0.90 wanted) and critical-flow fraction 0.063 (0.10-0.90
    wanted). High manifold back-pressure leaves most chokes subcritical.
  - A variant population (M2b: wider, shorter, shallower lines; smaller chokes) re-checks M3: section 3b.

## 3. The general model on unseen wells (M3, `vfm-choke-m3-fewshot` v3)
Median per-well MAPE of liquid rate, days 90-179, labels only from days < 90, 95% bootstrap CI:

| Labels | Ours | Physics + per-well multiplier | Pooled black box | Per-well LightGBM | Gilbert |
|---|---|---|---|---|---|
| 0 (design data) | **7.64** [7.19, 8.16] | 8.33 | 20.35 | — | 36.49 |
| 1 MPFM day | **5.73** | 5.98 | 10.18 | 18.92 | 23.55 |
| 3 MPFM days | **5.43** | 5.76 | 10.10 | 18.97 | 23.37 |
| 30 MPFM days | **4.67** | 5.18 | 9.54 | 13.09 | 21.94 |
| 3 separator tests | **3.28** [3.05, 3.56] | 4.19 | 8.38 | — | 18.72 |

- Gates:
  - beats Gilbert at k = 0: **met**;
  - beats pure physics at k = 0 with non-overlapping CIs: **not met**;
  - k ≤ 3 within 1.25× of the best per-well model at k = 30: **met** (1.11×, 1.08×, 1.05×);
  - 90% interval coverage 85-95%: **not met** (53-78%). The intervals leave out post-label drift and stale
    water-cut estimates.
- Post-hoc fix (`vfm-choke-m3c-fewshot` v1): split-conformal interval scales per label count, fitted on training
  wells under the same protocol.
  - Coverage becomes 89.1, 86.7, 86.0, 85.7, 85.6, 85.1, 84.8% for MPFM k = 0-30, and 87.4, 86.7, 86.6% for 1-3
    separator tests. Point predictions are unchanged.
  - One cell (k = 30, 84.8%) is a hair under the 85% bound. The fix was added after seeing the miss, so treat it as
    post-hoc.
- Oil-rate error (8.5-11%) is dominated by water cut rising after the last label, for every model.
- Caveat: SimField's truth uses the same choke physics family as the prior. The real benchmarks below are the check
  on model-form error.

### 3b. Robustness on the M2b population (`vfm-choke-m3b-fewshot` v1)
- Population: 843 wells; 37% of flowing well-days critical; all realism ranges met. Same code and protocol.

| Labels | Ours | Physics + per-well multiplier | Pooled black box | Gilbert |
|---|---|---|---|---|
| 0 | **8.51** [8.01, 9.01] | 9.70 [9.01, 10.35] | 14.13 | 19.93 |
| 1 MPFM day | **6.33** | 6.79 | 7.77 | 10.19 |
| 3 MPFM days | **6.35** | 6.63 | 7.61 | 10.39 |
| 30 MPFM days | **5.19** | 6.08 | 7.50 | 9.49 |
| 3 separator tests | **3.70** [3.44, 3.96] | 4.87 [4.53, 5.17] | 6.67 | 8.40 |

- Same verdicts as M3:
  - k ≤ 3 within 1.25× of the per-well reference (1.04×, 1.03×, 1.05×);
  - k = 0 better than pure physics, with the CIs touching at 9.01;
  - coverage 85% at k = 0, 66-75% after labels.
- With separator tests the gain over physics is clear (3.70 vs 4.87, non-overlapping CIs).
- Gilbert improves a lot here because more chokes are critical, which is the regime it assumes.

## 4. Public benchmarks (M4)

| Benchmark | Published best | Ours | Verdict |
|---|---|---|---|
| B1 Volve, 4 wells, walk-forward 30 d | MAPE 18%, R² 0.93, RMSE 333 | **12.12%** [10.55, 13.84], R² 0.944, RMSE 331 | beat (protocol reconstructed) |
| B2 Dabiri, 565 choke tests, 80/20 × 20 seeds | Adaboost-SVR 4.57% | 11.49% [10.39, 12.59] | not met |
| B2 zero-shot (no fitting) | best fitted correlation 17.20% | 20.14% | not met |
| B3 D'Souza, 2 gas-lift wells | LSTM 1.96 / 1.53% | 0.011 / 0.017% on regenerated data | excluded (fidelity) |

- **B1:** our gradient-boosting replicate of the published approach scores 14.62% on the identical 6,750 rows, so the
  protocol check passes. Against that replicate the difference is −2.5 points, CI [−7.45, +0.70]: not significant,
  and the replicate has the lower RMSE.
- **B2:**
  - Our hybrid is the best model we could reproduce under both the published random split and a duplicate-safe split
    (12.64%).
  - The paper's Adaboost-SVR re-run with its stated settings scores 40%.
  - 22% of rows are exact duplicates; even a pure 1-nearest-neighbour memoriser only reaches 13.7%.
  - Our zero-shot physics (no fitting at all) beats Gilbert, Ros, Baxendell, Achong, Al-Attar and Pilehvari as
    published.
- **B3:** the published simulator, run in Octave, produces data on which every method scores < 1% (linear regression
  0.0014%). The paper reports 1.5-16.7%, so its CSVs carry processing that isn't in the release.

## 5. Choke setpoints and multi-well optimisation (M5, `vfm-choke-m5-control` v2)
- Setup: 8 held-out groups on day 120. The twin is calibrated with k MPFM labels from days < 90 (30+ days stale).
  Its deliverability comes from design data with errors (reservoir pressure ±3%, productivity ±25%), plus a
  pressure shift fitted to the labels. Every recommendation is applied to the simulated truth.
- **Single-well setpoints** (cut each well to 80% of its rate, or raise it 10%; open loop):

  | k | Realised within ±5% of target | Median miss |
  |---|---|---|
  | 0 | 16% | 13.4% |
  | 3 | 35% | 7.7% |
  | 30 | 36% | 7.5% |

  Target ≥ 90% at k = 3: **not met**. The miss comes from the twin's deliverability (stale, design-based), not
  the choke model. A single open-loop move cannot beat the VFM's own ~5% accuracy; the realistic mode is a
  closed loop that moves, reads the VFM, and corrects.
- **Multi-well optimiser** (gas cap 85% and water cap 90% of today's rates, cautious twin for constraints):
  - Oil above the equal-scaling heuristic: +33% [23, 43] at k = 0; +37% [24, 51] at k = 3; +34% [23, 47] at k = 30.
  - The heuristic itself over-corrects on the cautious twin: it cuts oil 20-43% below today and stays 11-52%
    under the water cap.
  - No gas-cap violations by either method.
  - **Water-cap violations: 3-4 of 8 groups** (38-50%) for the optimiser, by 2-19% (median 14%); none for the
    heuristic. Target ≤ 10%: **not met**.
  - Cause: the water cut was estimated from labels 30+ days old while the true water cut kept rising. The cautious
    margin inflates the total rate, not water-cut uncertainty.
- Fix for deployment: use current MPFM ratios (daily), carry water-cut uncertainty into the water constraint, and
  run setpoints as a closed loop with VFM feedback.

## 6. Slick attribution (M6, `vfm-choke-m6a-slick-sim` v1 + `vfm-choke-m6b-slick-attr` v1)
- Setup:
  - 240 simulated leaks (jumper 40%, manifold 15%, flowline 30%, riser 15%; holes 2-15 mm, mostly 0.1-5% of line
    flow) on held-out groups, plus 80 controls whose slick comes from elsewhere.
  - Evidence: per-well MPFM, separator rates with 1% noise, manifold and riser-base pressures with a 0.3 bar model
    error, the slick's back-tracked position (drift known to ±30%), and a volume estimate known to a factor ~1.6.
  - About 16 hypotheses per scenario: random top-3 ≈ 19%.

| | Top-1 | Top-3 |
|---|---|---|
| All leaks | 23% | 62% |
| Leak ≤ 1% of flow | 15% | 60% |
| 1-2% | 37% | 60% |
| 2-5% | 44% | 74% |
| > 5% | 75% | 83% |
| Jumper (single well) | 23% | 52% |
| Flowline segment | 29% | 68% |
| Riser | 15% | 91% |
| Controls ranked "no leak" first | 100% | |

- Targets:
  - top-3 ≥ 80% for leaks ≥ 2%: **77%**, just short;
  - false alarms ≤ 5%: **22.5%**, missed.
  - The mass-balance alarm alone fired on only 40% of leaks before the slick was seen (median 3 days after onset).
- Telling which well leaked depends on the wells' ratios differing. In the worked example (2% leak on well 3's jumper)
  wells 2 and 1 carry near-identical GOR and water cut, and the truth ranks 4th with 8%. Mass balance localises to
  the line, as the literature says. Separating drill centre from flowline from riser works better (location and
  pressure signatures).
- What would help: oil fingerprinting where wells produce distinct oils; per-well pressure downstream of each choke;
  thresholds calibrated on separate no-leak periods.

## 7. Edge (M7, `vfm-choke-m7-edge` v5)
- Artifact 233 KB (500 trees + 29 calibrated wells); numpy-only inference; peak traced memory 32 KB.
- One well p50 3.08 ms, p99 4.14 ms; batched scan of 29 wells 0.59 ms per well; 30-well optimizer step p99 1.26 ms.
- Parity with the full model 4e-16; deterministic.
- The edge table step is 2.6% off the full network solve in manifold pressure and 3.3% in liquid. The edge flowline
  table freezes the commingled composition and temperature.
- Measured on a Kaggle CPU core; an ARM edge box may be slower.

## 8. Publish gate (M8)
- "Excels" needs two counting benchmarks. Only B1 counts and passes.
- The edge criteria are met.
- Per the plan the project stays private: no public notebooks or repository without a new decision from Sankalp.

## 9. Next steps
1. Make the interval calibration standard: it is in `fleet.train(interval_scales=True)`. The empirical-Bayes
   trim-error variance collapses to its floor in 2 of 5 folds; a hierarchical fit across folds would stabilise it.
2. Real multi-well data with well tests: the method's claim rests on SimField and one small real field.
3. B2: ask the authors for their split or code; B3: ask for the CSVs.
4. Replace the frozen-composition edge flowline table with a two-axis table (mass rate × GOR).
5. Control:
   - run setpoints as a closed loop (move, read the VFM, correct);
   - feed the optimiser the latest daily MPFM ratios, and carry water-cut uncertainty into the water constraint;
   - give SLSQP more iterations (14 of 24 runs hit the 20-iteration cap).
6. Slick attribution: calibrate the CUSUM thresholds on separate no-leak periods; add laboratory oil fingerprinting
   (CEN diagnostic ratios) where wells produce distinct oils; use per-well pressures downstream of each choke where
   they exist.
