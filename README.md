# vfm-choke

A general virtual flow meter (VFM) for wellhead chokes that works on a new well from its design data and tightens
with 1-3 labels; choke setpoints and a multi-well optimiser on top of it; and a ranked answer to "which source fed
this oil slick". Numpy-only inference for edge devices.

Research project by Sankalp Thakur, released Nov 1 2026. Results are in [REPORT.md](REPORT.md), and every number
there is traced to a Kaggle kernel version whose outputs are under `reports/runs/`. The bar for each benchmark was
written in [BENCHMARKS.md](BENCHMARKS.md) before the run.

## Results in brief
- **New wells from design data.** Tested on 1,219 simulated wells the model never saw:

  | Labels on the new well | Median liquid-rate error |
  |---|---|
  | None (design data only) | 7.6% |
  | 1-3 labels | within 1.05-1.11× of a per-well model fitted on 30 labels |

  The simulator shares the model's physics family, which flatters it.
- **Public benchmarks.**
  - **Volve (real field):** 12.1% MAPE against the published 18%, under a protocol reconstructed from the paper's
    abstract. The margin over our own replicate of the published method is not significant.
  - **Dabiri et al. 2024 and D'Souza et al. 2024:** their published numbers could not be reproduced. See the audit
    below.
- **Edge.** The edge package meets every criterion: 233 KB, p99 4.1 ms per well on one CPU core.
- **Control and slick attribution.** Both miss their pre-set targets. REPORT.md lists the fixes.
- **The bar.** The pre-registered "excels on two benchmarks" bar was **not** met. The work is public anyway, so
  others can check it and build on it.

The reproducibility audit of the two benchmarks is a separate repository,
[reproducing-vfm-benchmarks](https://github.com/sankalpsthakur/reproducing-vfm-benchmarks), with a
[Kaggle notebook](https://www.kaggle.com/code/sankalpsthakur/reproducing-two-virtual-flow-metering-benchmarks).

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
Every result here came from a Kaggle CPU kernel. To re-run one:
1. Set `OWNER` in `kaggle/build.py` and `kaggle/push.py` to your Kaggle username.
2. Build the notebook and push it as a private kernel. `push.py` waits for the run and saves its outputs.

```bash
python3 kaggle/build.py m1_tests.py vfm-choke-m1-physics-tests "VFM Choke M1 Physics Tests" --tests
python3 kaggle/push.py vfm-choke-m1-physics-tests
```

## Licence and credit
- **Code:** [Apache License 2.0](LICENSE): the Python package, the tests, the Kaggle builders and the entry scripts.
  Apache 2.0 requires keeping the copyright and [NOTICE](NOTICE) attribution in copies and derived works.
- **Everything else:** [CC BY 4.0](LICENSE-CC-BY-4.0): documents, figures, reports and run outputs. You may reuse
  them if you credit the source.
- **Please credit this work** if it helps yours, whether you are an author, a researcher or a practitioner. Cite
  it as in [CITATION.cff](CITATION.cff) (GitHub's "Cite this repository" button), or as:

  > Thakur, S. (2026). *vfm-choke: a general few-shot virtual flow meter for wellhead chokes, with choke control and
  > oil-slick source attribution.* https://github.com/sankalpsthakur/vfm-choke

## Data
Public data and simulation only. The third-party data keep their own licences:
- Equinor Volve: Equinor Open Data Licence, attribution required;
- Dabiri et al. 2024 supplementary data: CC BY 4.0;
- D'Souza et al. simulator, DataverseNO copy: CC0.

Files from the unlicensed GitHub copy of that simulator are not included; the kernels download them at run time.

AI assistance (Claude) was used to write the code and documents. Every result comes from code in this repository run
on Kaggle.
