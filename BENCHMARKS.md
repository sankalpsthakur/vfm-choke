# Benchmarks: access, baselines, protocols, and the pre-registered bar

Written Nov 1 2026, before any model was run. The numbers below are the published ones, copied from the sources.
Anything we report later is compared against this page as it stands now; changes to it after results exist must
be logged in Documentation.md with the reason.

## Kept (data access confirmed, sourced baseline)

### B1 Volve (real field, daily allocated rates)
- **Data:** Equinor Volve production data. Kaggle copy `lamyalbert/volve-production-data`
  (`Volve production data.xlsx`, 2.3 MB). Licence: Equinor Open Data Licence (attribution).
- **Published baseline:** Ebrahimzadeh (2026), "Toward Deployable Machine Learning Virtual Flow Metering: A
  Reproducible Volve-Field Benchmark on Recalibration and Validation Design", 7th Int. Conf. on Petroleum
  Engineering, Gas Geology and Petrochemical Industries ([civilica 2662147](https://civilica.com/doc/2662147/)).
  - 4 producing wells, 7,437 daily records (2008-02-12 to 2016-09-17), target: daily oil rate.
  - Random split: gradient boosting R² 0.98 (the author calls it optimistic).
  - Frozen temporal split: R² 0.80, MAPE 66%.
  - Walk-forward with periodic recalibration: R² 0.93, RMSE 333 Sm³/d, MAPE 18%; per-well MAPE under 20%.
- **Comparability limit:** only the abstract is visible (well names, cleaning thresholds and recalibration interval
  are not). We reconstruct the protocol and say so. **B1 counts toward "excels" only if the protocol can be matched;
  otherwise it is supporting evidence.** We also add a well-held-out protocol the paper does not run.
- **Label caveat:** Volve volumes are allocated daily volumes, not test-separator measurements.

### B2 Dabiri et al. 2024 (565 literature wellhead-choke tests)
- **Data:** supplementary DOCX of Dabiri et al., *Sci Rep* 14:6945 (2024),
  [PMC10960849](https://pmc.ncbi.nlm.nih.gov/articles/PMC10960849/), doi:10.1038/s41598-024-54010-2. CC BY 4.0.
  Inputs: wellhead pressure Pwh (psi, 50–4045), choke size Dc (1/64 in, 16–80), GLR (scf/stb, 36–5706.6).
  Target: liquid rate (stb/d, 205–25,878). Eight literature sources (refs 12, 20, 23–28 of the paper).
- **Protocol:** random 80/20 split (no seed published; we report the mean and spread over 20 seeds).
- **Published baselines (test set):** Adaboost-SVR AAPRE 4.57%, R² 0.9827 (best ML); MLP-BR 7.92%; RBF 8.45%;
  MLP-LM 9.19%; MLP-SCG 10.9%; MARS 12.14%.
- **Published correlations (whole set):** their new equation 17.20% (AAPRE-optimised), Beiranvand 19.03%,
  Baxendell 20.36%, Achong 21.05%, Ros 21.32%, Gilbert 22.36%, Al-Attar 36.97%, Pilehvari 51.82%.
- **Known source inconsistencies** (text vs tables) are listed in RESEARCH.md; we use the table numbers.

### B3 D'Souza, Pfeiffer & Mirlekar 2024 (two simulated gas-lifted subsea wells)
- **Data:** DataverseNO, doi:10.18710/KIJEWJ, CC0; code and simulator:
  https://github.com/dsouzaneville/FMH606-1-Masters-Thesis. Simulator: Janatian, Jayamanne & Sharma (2022).
  5,762 samples per well; inputs: bottomhole and wellhead pressures and temperatures (up- and downstream of the
  choke), choke opening; targets: oil and gas (and liquid) mass rates.
- **Paper:** "Assessment of Data-Driven Techniques for Flow Rate Predictions in Sub-sea Oil Production",
  SIMS EUROSIM 2024, doi:10.3384/ecp212.014.
- **Protocol:** 70/30 time-series split per well; MAPE as in sklearn (eq. 2 of the paper).
- **Published baselines, Table 1 (MAPE %, Well 1 / Well 2):** LSTM 1.96 / 1.53 (abstract says 1.56 for Well 2);
  MV linear regression 2.14 / 7.57; MLP 2.43 / 5.49; XGBoost 4.23 / 5.56; gradient boosting 4.95 / 5.55;
  SVR 5.04 / 4.31; KNN 8.05 / 5.41; decision tree 9.26 / 5.43; PLS 9.54 / 7.57; PCR 9.52 / 16.69.
- **Noise variants:** impulse noise, LSTM 5.98 / 4.67; with a 3-point median filter, LSTM 1.87 / 2.86.
- **Caveat:** simulated data; the test window is a new steady operating point after a step change.

### B2 run protocol (written Nov 1 2026 after parsing the file, before any B2 prediction)
- **Data:** the supplementary DOCX from Springer's static mirror (the PMC link serves a JavaScript page), table 0:
  565 rows of P (psi), GLR (scf/stb), D (64ths of an inch), Qliq (stb/d). No source column, so few-shot is measured
  with k labelled points drawn from the whole set (stricter than per source).
- **Zero-shot physics (no fitting):** mechanistic choke with API 30, gas SG 0.75, 60 °C, water cut 0, salinity 0,
  Cd 0.80, critical flow (downstream 1 atm), P read as psig, bean area π/4 (D/64 in)².
- **Published-correlation check:** Gilbert, Ros, Baxendell, Achong and Pilehvari with literature constants on all 565
  rows; ours should land near the paper's Table 7 if parsing and units are right.
- **80/20 protocol:** seeds 0-19, random 80/20 splits. Our model per split is chosen by 5-fold CV on the training part
  only, between (a) physics prior × power-law residual (OLS on log P, log D, log GLR) and (b) physics prior × LightGBM
  residual. Same splits for every method.
- **Adaboost-SVR replicate:** sklearn AdaBoostRegressor(SVR(C=1.0, epsilon=0.1, kernel='rbf', gamma='scale')) with
  default boosting settings, inputs and target min-max scaled on the training part (the paper's Table S3 lists SVR
  defaults; scaling is not stated). Also the paper's power law refit on each training part.
- **Statistic:** mean test AAPRE over the 20 seeds with a t-based 95% CI; "beat outright" needs the CI upper bound
  below 4.57%.
- **Few-shot:** k ∈ {0, 1, 3, 10} random labelled points (50 draws each) fit one global multiplier on the physics
  prior; AAPRE on the remaining rows.

### B1 run protocol (written Nov 1 2026, before any B1 prediction)
- **Data:** Daily Production Data sheet; producers F-1 C, F-11, F-12, F-14, F-15 D; days with ≥ 20 on-stream hours,
  oil > 0, choke > 1%, a positive choke pressure drop below the wellhead pressure. Rates scaled to 24 h. Headline
  numbers on the 4-well set F-11, F-12, F-14, F-15 D (the paper's well names are not visible); 5-well numbers also
  reported.
- **Walk-forward:** recalibration every 30 days; each well predicted from its day 60; only labels strictly before the
  recalibration point are used.
- **Ours:** mechanistic choke prior (wellhead pressure; downstream = wellhead − choke dp; opening; nominal 2 in bean,
  Cd 0.80, linear trim; API 30, gas SG 0.75, salinity 0.035) with water cut and GOR tracked from labels (random walk,
  q = 1e-3/day, obs sd 0.1) and the per-well latent (τ0 = 2.0, τ1 = 0.5, q0 = 1e-4/day, label sd 0.10). Two label
  regimes: all past daily labels, and one anchor label per 30 days.
- **Replicates:** pooled gradient boosting on log oil rate (wellhead pressure, opening, wellhead temperature, well code,
  day), refit every 30 days on all past data; and a frozen 70/30 temporal split.
- **Protocol check:** B1 counts toward "excels" only if the walk-forward replicate lands within ±5 points of the
  paper's 18% MAPE.

### B3 run protocol (written Nov 1 2026, before any B3 data existed)
- **Data regeneration:** the published dataset holds the MATLAB simulator, not the CSVs the notebooks load
  (`Well{1,2}_average_data.csv`). We run the simulator in Octave 8.4: the DataverseNO two-well model (Θ = [2.51 1.63
  0.08 0.07 0.15 0.15], 30% of the gas-lift supply to each well, two-step supply 40,000 → 20,000 → 40,000 Sm³/h, RK4
  at dt = 20 s) with the output list of the GitHub version, over 96 h (the paper's Figure 6 steps at 32 h and 64 h),
  averaged to 1-minute rows. Columns as the notebook reads them: time, W_ga, P_wf, P_wh → W_op, W_gp, W_lp.
- **Split:** the notebook's absolute rows: train 0-2303, validation 2304-4032, test 4033-end (≈ 67.2 h, as the paper's
  plot line). MAPE as sklearn's uniform average over the three targets.
- **Fidelity check:** we re-run their LSTM (1-step input, 32/40 units, adam, MSE, 100 epochs, batch 32, early stopping
  on validation, patience 10), linear regression, XGBoost (W1: 300 trees, depth 7, lr 0.01; W2: 100, 3, 0.1), MLP and
  SVR with the paper's stated settings (inputs standardised on all rows, as in their notebook). **B3 counts toward
  "excels" only if our LSTM and linear-regression MAPEs land within ±30% (relative) of Table 1 on both wells.**
- **Ours:** physics-feature regression from the lumped gas-lift structure with design geometry and reservoir pressure:
  features (P_r − P_wf), W_ga and Y3·√(ρ_m·max(P_wh − P_s, 0)), with ρ_m from the tubing hydrostatics. Fitted on the
  training rows only. Zero-shot variant: the physics prior with deliberately wrong design parameters (PI +20%, GOR −20%,
  water cut +0.05). Few-shot variant: that prior with one multiplier per target fitted on k = 3 training rows.

## Dropped at the M0 gate
- **Akbari et al. 2025** ([PMC12717276](https://pmc.ncbi.nlm.nih.gov/articles/PMC12717276/)): data not public
  ("cited and referenced" only; derived from Ghorbani et al. 2019, Reshadat field). Reported metrics are on a
  normalised scale and the split may leak.
- **HVFM / BiVFM-Fleet** ([PyPI](https://pypi.org/project/HVFM), [GitHub](https://github.com/mmmahhhhe/HVFM)): code is
  MIT, but no dataset download exists; generation needs OLGA (commercial).
- **3W 2.0** (Petrobras, data CC BY 4.0): no published baseline for a comparable task, and no leak events. Kept as a
  possible later test of the diagnosis engine, not part of "excels".

## Pre-registered bar

**"Excels"** = on at least 2 of the benchmarks that count (B2, B3, and B1 only if its protocol is matched), under the
published protocol, with bootstrap 95% confidence intervals, at least one of:
1. **Beat outright:** our model's error is below the best published number, and the CI upper bound is below it too.
2. **Few-shot parity:** with zero-shot (no training on the benchmark) or at most 3 labelled points per well or per
   source, our error is at or below the best published number.

Concretely:
- **B2:** zero-shot AAPRE below 17.20% beats every published correlation without fitting; at most 3 points per
  literature source reaching 4.57% is few-shot parity; under the 80/20 protocol, below 4.57% (mean over 20 seeds,
  CI upper bound below 4.57%) is beat outright.
- **B3:** below 1.96% (Well 1) and 1.53% (Well 2) is beat outright; at most 3 training points per well reaching those
  numbers is few-shot parity.
- **B1:** below 18% MAPE walk-forward (with the matched protocol) is beat outright.

Parity is reported as parity, together with the calibration advantage. A miss is reported as a miss.

**Edge criteria:** numpy-only inference module plus a tree export (treelite or ONNX); artifact at most 5 MB; RAM at
most 100 MB; one CPU core: p99 at most 5 ms per well evaluation and at most 100 ms per 30-well optimizer step;
deterministic; works offline. Measured inside a Kaggle CPU kernel as the proxy device.
