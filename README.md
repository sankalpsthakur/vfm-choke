# vfm-choke

A general virtual flow meter (VFM) for wellhead chokes that works on a new well from its design data and tightens
with 1-3 labels; choke setpoints and a multi-well optimiser on top of it; and a ranked answer to "which source fed
this oil slick". Numpy-only inference for edge devices.

Private research project. Results, with every number traced to a Kaggle kernel version, are in REPORT.md; the bar
for each benchmark was written in BENCHMARKS.md before the run.

## How it works
1. **Physics prior from design data.** A mechanistic two-phase choke model (critical and subcritical flow) at the
   measured wellhead and downstream pressures, wellhead temperature, opening, bean size and vendor trim curve, with
   black-oil PVT from the well's design fluid. No field fitting.
2. **Pooled correction on transferable ratios.** A LightGBM model learns a multiplicative correction across many
   wells from inputs that mean the same thing everywhere (pressure ratio, critical-ratio margin, gas quality and
   volume fraction, water cut, GOR, opening). No well identity, so it applies to a well it has never seen. Monotone in
   opening and pressure drop, so the choke inverse is a bisection with one answer.
3. **Few-shot calibration.** Each well has a latent pair (log multiplier, trim-exponent error) with an
   empirical-Bayes prior learned from other wells and a random walk for drift. MPFM readings (gamma densitometer +
   venturi) and test-separator results update it in closed form, each with its own noise; water cut and GOR are
   tracked the same way.
4. **Control.** A twin of each well (design deliverability + the calibrated VFM choke) gives the opening for a target
   rate and, across a manifold, the openings that maximise oil within separator gas and water caps, using cautious
   (upper-quantile) rates for the constraints. Advisory only.
5. **Slick attribution.** A posterior over leak sources (each well's jumper, manifold, flowline segments, riser) from
   the phase signature of the mass-balance deficit, pressure shifts, the slick's back-tracked position and its volume.

## Layout
- `src/vfmchoke/`: `pvt`, `choke`, `well`, `network`, `mpfm` (physics); `simfield` (synthetic fleet); `hybrid`,
  `fewshot`, `fleet` (the general model); `control`; `leak`; `edge`.
- `benchmarks/`: loaders for the public benchmarks (Volve, Dabiri et al. 2024; the D'Souza et al. simulator runs inside
  its kernel).
- `kaggle/`: notebook builder (`build.py`), guarded private push (`push.py`) and the entry script of every run.
- `tests/`: physics invariants and model checks, run in a Kaggle kernel.
- Project docs: Prompt, Plans, Architecture, Implement, Documentation; RESEARCH.md (literature), BENCHMARKS.md.

## Running
Nothing runs on the laptop. Build a notebook and push it as a private Kaggle CPU kernel:

```bash
python3 kaggle/build.py m1_tests.py vfm-choke-m1-physics-tests "VFM Choke M1 Physics Tests" --tests
python3 kaggle/push.py vfm-choke-m1-physics-tests
```

## Data and licences
Public data and simulation only: Equinor Volve (Equinor Open Data Licence), Dabiri et al. 2024 supplementary data
(CC BY 4.0), D'Souza et al. simulator (CC0). AI assistance (Claude) was used to write code and documents; every result
comes from code in this repository run on Kaggle.
