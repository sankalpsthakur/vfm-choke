# Architecture

## Principles
- **Physics first, learning second.** The physics model is the prior a greenfield well has on day 1. Learning only
  corrects it, multiplicatively, from inputs that mean the same thing on every well.
- **No well identities as features.** Wells differ through their ratios (pressure ratio across the choke, gas volume
  fraction, water-liquid ratio, gas-liquid ratio, choke opening and size), not through IDs. That is what lets the
  model run on a well it has never seen.
- **Monotone where physics is monotone.** Rate never decreases with choke opening or pressure drop, so the choke
  inverse is a bisection with a unique answer.
- **Per-well calibration is one closed-form update.** An empirical-Bayes offset per well, with label-specific noise
  (MPFM wider than test separator). No retraining on site.
- **Advisory control.** Setpoints are recommendations for an operator and the control system, never direct writes.
- **Every number has a receipt.** Results come from Kaggle kernel outputs, fetched right after each run, and the report
  cites the kernel version.

## Modules (`src/vfmchoke/`)
| Module | Role |
|---|---|
| `units.py` | SI internally; field-unit conversions at the edges (psi, bbl/d, scf/stb, 1/64 in) |
| `pvt.py` | Black-oil PVT: Standing Rs and Bo, Beggs-Robinson viscosity, gas Z (Papay), water density |
| `choke.py` | Sachdeva mechanistic two-phase choke (critical and subcritical), trim curves, Gilbert-family correlations |
| `well.py` | Vogel/linear IPR, drift-flux tubing pressure drop, well operating point |
| `network.py` | Wells, manifold, flowline, separator: operating point by nested root finding |
| `mpfm.py` | Dual-energy gamma densitometer + venturi emulator and its inversion to phase fractions and rates |
| `simfield.py` | Population generator: fields, wells, time series, tests, MPFM readings, leak injection |
| `hybrid.py` | Physics prior + monotone LightGBM residual on transferable features |
| `fewshot.py` | Empirical-Bayes per-well calibration, online ratio tracking |
| `control.py` | Single-well inverse, multi-well optimizer |
| `leak.py` | Mass-balance residuals, CUSUM onset, hypothesis likelihoods, drift prior, ranked posterior |
| `edge.py` | Numpy-only inference path and exported tree evaluation |

Benchmarks live in `benchmarks/` (loaders and protocols per dataset). Tests live in `tests/`.

## Execution model
- The laptop only edits files and builds notebooks. `kaggle/build.py` writes a self-contained notebook: one cell per
  source file (`%%writefile`), then the entry script (tests, experiments, benchmarks). The notebook is the exact code
  that ran.
- Each run is a private Kaggle CPU kernel with metadata built from an allowlist (fresh id, `is_private: true`, never a
  pulled `id_no`). Large artifacts stay on Kaggle and pass between kernels through `kernel_sources`.

## Constraints
- Public data and simulation only.
- Python 3.10+, numpy, scipy, pandas, lightgbm (all in the Kaggle image). The edge path needs numpy only.
