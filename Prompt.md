# Prompt

## Goal
A choke-control model for oilfield wellheads. Today each well carries its own virtual flow meter (VFM) with its own
fitted ratios (gas-oil ratio, water cut, choke discharge coefficient). Build one general model that also works on a
greenfield well with little data or calibration, and use it to recommend choke settings.

## Spec
1. **General VFM.** Predict oil, gas and water rates from choke opening, choke size, pressures and temperatures, and
   fluid ratios. Zero-shot on a new well from design data; tightened by 1-3 well tests or MPFM readings.
2. **Labels.** Gamma-ray MPFM readings (phase fractions and rates) and test-separator well tests, each with its own
   noise level.
3. **Choke inverse.** The choke opening that hits a target rate within drawdown, gas-oil ratio and water-cut limits,
   with uncertainty bands. Advisory only.
4. **Multi-well optimizer.** Choke settings across wells on a shared manifold that maximise oil within separator gas,
   water and liquid capacity.
5. **Slick attribution.** For one oil slick detected at sea, a ranked posterior over leak sources (well jumper,
   manifold, flowline segment, riser) with onset and leak-rate estimates.
6. **Edge deployable.** Numpy-only inference, small artifact, millisecond latency on one CPU core.

## Deliverables
- Private: this repository, RESEARCH.md, BENCHMARKS.md, REPORT.md with every number traced to a Kaggle kernel output.
- If the model excels on the public benchmarks and meets the edge criteria (both pre-registered in BENCHMARKS.md):
  public Kaggle notebooks and a public GitHub repository with the approach documented end to end, after Sankalp's
  explicit go.

## Constraints
- Public data and simulation only.
- No local runs on the laptop: all execution in private Kaggle CPU kernels. No paid compute.
- Commits authored by Sankalp only; no agent co-author trailers.
