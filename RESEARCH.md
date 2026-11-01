# Virtual flow metering: where the field is (Nov 2026), and what that means for a greenfield choke model

Scope: what a virtual flow meter (VFM) can do for wellhead choke control, how well VFMs transfer to wells they were
not trained on, what gamma-ray multiphase meters (MPFMs) contribute as labels, and what is known about tracing a
sea-surface oil slick back to a subsea source. Claims are tied to sources; anything not verified is marked.

## 1. Three families of VFM
- **First principles.** Well, choke and flowline models (inflow performance, vertical lift, choke equations,
  transient multiphase codes such as OLGA). Strong extrapolation, but each well needs tuning: choke discharge
  coefficients, productivity, fluid ratios. The standard review is Bikmukhametov & Jäschke (2020), *J. Pet. Sci.
  Eng.* 184, 106487.
- **Data-driven.** Neural networks (AL-Qutami et al. 2018, ensembles), LSTMs (Andrianov 2018), tree ensembles.
  Accurate inside the training envelope; brittle at new operating points and on new wells.
- **Hybrid.** Physics model plus a learned correction, now the industry direction:
  - SPE 233457 (summarised in JPT, Oct 2026): a hybrid physics/ML VFM for ESP-lifted fields that combines flow
    physics, well-performance relationships and ML, reported as deployed in the field.
  - SLB Olga Online, extended to Johan Sverdrup Phase 3, runs live multiphase models for both VFM and leak detection.
  - Vendors (Sensia, Turbulent Flux, Cognite, Xenoss) describe hybrids that use physics to fill gaps in labelled
    data; TÜV SÜD (NFOGM, Sep 2026) uses a virtual flow tool to monitor and troubleshoot MPFMs.

## 2. The question that matters here: does a VFM transfer to a new well?
- **Multi-task learning** (Sandnes et al. 2021, [arXiv 2103.08713](https://arxiv.org/pdf/2103.08713)): one network
  shared across wells, with a small set of well-specific parameters.
- **Bayesian neural networks** (Grimstad et al. 2021, [arXiv 2102.01391](https://arxiv.org/pdf/2102.01391)): about
  66,000 points from 60 wells in five fields; calibrated uncertainty.
- **Few-shot** (Løvland, Grimstad & Imsland, [arXiv 2309.15828](https://arxiv.org/abs/2309.15828v3), v3 Sep 2025):
  multi-well models learned from 80 wells allow few-shot learning on new wells; "1-3 data points often leads to high
  performance on new wells". This is the closest prior art to the greenfield goal. The data is proprietary, so the
  numbers cannot be reproduced; only the idea can.
- **Nonstationarity** (Hotvedt et al., [arXiv 2202.03236](https://arxiv.org/pdf/2202.03236)): wells drift (water cut
  rises, pressure declines), so a VFM needs ongoing updates.
- **Validation design dominates reported accuracy** (Ebrahimzadeh 2026, Volve): random splits gave R² 0.98; a frozen
  temporal split gave MAPE 66%; walk-forward recalibration gave MAPE 18%. "Recalibration is the single most important
  factor." Random splits inflate field performance.
- **Not found:** no published foundation model or pretrained model for VFM. A "ManyWells" simulated dataset could not
  be verified, so it is not cited.

**Consequence for the design.** A general model must (a) carry physics so it extrapolates to new operating points,
(b) take only inputs that mean the same thing on every well (pressure ratios, gas and water fractions, choke opening
and size), not well identities, and (c) have an explicit, cheap per-well update so 1-3 tests or MPFM readings
calibrate it.

## 3. Choke models
- **Empirical (Gilbert family):** q = P·S^b / (a·GLR^c), from Gilbert (1954), Ros, Baxendell, Achong, Pilehvari,
  Al-Attar and others. Assumes critical flow (rate independent of downstream pressure). Field-specific constants.
  On 565 literature tests (Dabiri et al. 2024), these correlations score 19–52% AAPRE; the authors' refit power law
  scores 17.20%.
- **Mechanistic:** Sachdeva et al. (1986) and Perkins (1993) model two-phase critical and subcritical flow from fluid
  properties and the pressure ratio across the choke. They need the downstream pressure and fluid ratios, but no
  field fit beyond a discharge coefficient.
- **ML on choke data:**
  - Dabiri et al. 2024 (*Sci Rep*): Adaboost-SVR test AAPRE 4.57% on a random 80/20 split of 565 points.
  - Akbari et al. 2025 (*Sci Rep*): an MLP on 182 Reshadat well tests; metrics on a normalised scale and a possibly
    leaky split, data not public.
  - KFUPM (2022): a workflow that attributes most rate error to sparse well tests and empirical correlations.
  - Hassi Messaoud (Interpore 2026 poster): correlations and ML refined on more than 61,000 field well tests.
  Every one of these fits to field data; none claims calibration-free transfer.

## 4. Gamma-ray multiphase meters as labels
- An MPFM measures oil, gas and water rates without separating them. A typical design combines a venturi
  (total flow) with a gamma-ray densitometer: the mixture's absorption at two or more energies gives the phase
  fractions (Scheers 2000, multi-energy gamma composition meter).
- Neural-network inversion of gamma data has a long history: dual-energy densitometry with neural networks in
  *Nucl. Instrum. Methods A* 327 (1990s); Salgado et al. (2010), gamma attenuation with neural networks for volume
  fractions; CNN/LSTM flow-regime classification from gamma plus venturi data (WIT Press 2022).
- MPFM uncertainty is several percent per phase and depends on the flow regime; water-salinity changes shift the
  water attenuation and bias the water cut. So MPFM readings are good labels, but noisy ones; test-separator
  measurements are sparser and more accurate. The model should weight them differently.

## 5. Tracing a slick to its source
- **Detection:** satellite radar (SAR) shows slicks as dark patches (damped capillary waves). Equinor's Krafla base
  case pairs SAR with a mass-balance system and charts monthly when conditions allow SAR to see a release.
- **Mass balance localises to the line, not the well:** a Rigzone comparison puts mass balance at line-level
  localisation, against tens of metres for pressure-wave methods and metre scale for fibre and ROV surveys. With a
  venturi on every subsea well, a statistical volume balance flagged a 0.5% imbalance in about 12 minutes on a
  roughly 120,000 m³/d crude line (Atmos, *World Pipelines*, May 2025).
- **Well-level attribution needs more evidence:** pressure signatures predicted by a network model, the leaked
  phases' ratios versus each well's gas-oil ratio and water cut, slick drift back-tracking, and, where wells produce
  chemically distinct oils, laboratory fingerprinting with CEN diagnostic ratios (a ratio matches when r95% < 14%,
  NewFields 2024 report; Daling et al. 2002; Wang & Christensen).
- **Not found:** no public method that ties a SAR-detected slick to one specific well. That is the gap the slick
  module addresses, and it will be evaluated on simulated scenarios only.

## 6. What this project bets on
A physics prior that needs only day-1 design data, a monotone learned correction on transferable ratios, a closed-form
per-well update from 1-3 labels, and control and leak attribution built on that same calibrated model. The bet is
tested against three public benchmarks (BENCHMARKS.md) and a simulated field, with the bar written down first.

## References
- Bikmukhametov, T. & Jäschke, J. (2020). First principles and machine learning virtual flow metering: a literature
  review. *J. Pet. Sci. Eng.* 184, 106487. doi:10.1016/j.petrol.2019.106487
- Sandnes, A. T. et al. (2021). Multi-task learning for virtual flow metering. [arXiv 2103.08713](https://arxiv.org/pdf/2103.08713)
- Grimstad, B. et al. (2021). Bayesian neural networks for virtual flow metering: an empirical study.
  [arXiv 2102.01391](https://arxiv.org/pdf/2102.01391)
- Hotvedt, M. et al. Passive learning to address nonstationarity in virtual flow metering applications.
  [arXiv 2202.03236](https://arxiv.org/pdf/2202.03236)
- Løvland, A., Grimstad, B. & Imsland, L. Multi-task and few-shot learning in virtual flow metering.
  [arXiv 2309.15828](https://arxiv.org/abs/2309.15828v3) (v3, Sep 2025)
- Ebrahimzadeh, A. (2026). Toward deployable ML virtual flow metering: a reproducible Volve-field benchmark on
  recalibration and validation design. [civilica 2662147](https://civilica.com/doc/2662147/)
- Dabiri, M.-S. et al. (2024). Modeling liquid rate through wellhead chokes using machine learning techniques.
  *Sci Rep* 14, 6945. [PMC10960849](https://pmc.ncbi.nlm.nih.gov/articles/PMC10960849/)
- Akbari et al. (2025). Machine learning-based prediction of well performance parameters for wellhead choke flow
  optimization. *Sci Rep*. [PMC12717276](https://pmc.ncbi.nlm.nih.gov/articles/PMC12717276/)
- D'Souza, N. A., Pfeiffer, C. & Mirlekar, G. (2024). Assessment of data-driven techniques for flow rate predictions in
  sub-sea oil production. SIMS EUROSIM 2024. doi:10.3384/ecp212.014; data doi:10.18710/KIJEWJ
- Janatian, N., Jayamanne, K. & Sharma, R. (2022). Model based control and analysis of gas lifted oil field for
  optimal operation. SIMS EUROSIM 2021. doi:10.3384/ecp21185241
- AL-Qutami, T. A. et al. (2018). Virtual multiphase flow metering using diverse neural network ensemble and adaptive
  simulated annealing. *Expert Syst. Appl.* doi:10.1016/j.eswa.2017.10.014
- Andrianov, N. (2018). A machine learning approach for virtual flow metering and forecasting. *IFAC-PapersOnLine*.
  doi:10.1016/j.ifacol.2018.06.376
- SPE 233457, hybrid physics/ML framework for virtual flowmetering (JPT Technology Focus, Oct 2026).
  [JPT](https://jpt.spe.org/hybrid-physics-ml-framework-for-virtual-flowmetering-optimizes-production-restricted)
- Petrobras 3W Dataset 2.0.0. [arXiv 2507.01048](https://arxiv.org/abs/2507.01048); [repo](https://github.com/petrobras/3W)
- HVFM / BiVFM-Fleet. [PyPI](https://pypi.org/project/HVFM); [GitHub](https://github.com/mmmahhhhe/HVFM)
- Equinor, Krafla subsea leak detection philosophy.
  [pdf](https://cdn.equinor.com/files/h61q9gi9/global/b7b683f3a613f9e3082bb4024cb570968abd1397.pdf)
- Olga Online at Johan Sverdrup Phase 3. [Ocean News](https://oceannews.com/news/energy/olga-online-extends-real-time-leak-detection-at-johan-sverdrup-phase-3/)
- Atmos International, *World Pipelines*, May 2025. [pdf](https://www.atmosi.com/media/x3gck542/atmos-world-pipelines-may-2025.pdf)
- Rigzone, How are offshore pipelines inspected for leaks?
  [link](https://www.rigzone.com/insights/how-it-works-3/how-are-offshore-pipelines-inspected-for-leaks-585)
- NewFields (2024). Oil fingerprinting report, Westlake Sulphur Dome (CEN protocol, 30 diagnostic ratios).
  [pdf](https://www.dnr.louisiana.gov/assets/OC/SulphurMines/Sampling/OilFingerprinting/NF_Fingerprinting_Report_040424-Westlake.pdf)
- Classic choke correlations: Gilbert (1954), Ros (1960), Baxendell (1958), Achong (1961); mechanistic: Sachdeva et
  al. (1986, SPE 15657), Perkins (1993).
