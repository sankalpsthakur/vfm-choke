# Plans: milestones and validations

The bar for "excels" and the edge criteria live in BENCHMARKS.md. The gates below were written Nov 1 2026, before any
model ran. Changing a gate after results exist needs a logged reason in Documentation.md.

| # | Milestone | Gate |
|---|---|---|
| M0 | Skeleton, five docs, RESEARCH.md, BENCHMARKS.md | Every kept benchmark has data access and a sourced baseline (done: B1, B2, B3 kept; 3 dropped) |
| M1 | Physics core + unit tests (Kaggle CPU kernel) | All invariants pass (list below) |
| M2 | SimField + leak injector (kernel outputs) | Rate, pressure and critical-flow distributions in realistic ranges |
| M3 | General model + few-shot calibration vs baselines | k=0 beats Gilbert and pure physics; k≤3 within margin of per-well k=30 |
| M4 | Public benchmarks B1-B3 | "Excels" per BENCHMARKS.md |
| M5 | Choke inverse + multi-well optimizer, closed loop | Violations ≤10% at 90% targets; oil gain vs heuristic with CI |
| M6 | Slick attribution, ~500 leak + 200 no-leak scenarios, one-slick demo | Targets below; honest reporting |
| M7 | Edge package + latency benchmark | Edge criteria in BENCHMARKS.md |
| M8 | REPORT.md, then the publish gate | Pass: prepare notebooks + repo privately, ask for go. Fail: report the gap |

## M1 invariants
- Mass conservation through choke, manifold and flowline (relative error < 1e-6 at solve tolerance).
- Choke rate is non-decreasing in opening and in upstream pressure, non-increasing in downstream pressure.
- Continuity at the critical/subcritical boundary (no jump larger than 1e-3 relative).
- MPFM inversion recovers phase fractions exactly (< 1e-9) with zero counting noise.
- Gilbert correlation reproduces a hand-computed textbook value to 4 significant figures.
- PVT outputs inside physical bounds (Rs ≥ 0, Bo ≥ 1, viscosities > 0) over the sampled envelope.

## M2 realism ranges (written before the run)
- Flowing fraction of well-days > 0.90; median liquid rate 500-15,000 stb/d; median wellhead pressure 20-120 bar.
- Both choke regimes present: critical fraction between 0.10 and 0.90 of flowing well-days.
- MPFM liquid error: median within ±5%, 10th-90th percentile within ±10%.

## M3 pre-set margins
- Evaluation: wells grouped into 5 folds (no well in both train and test); k ∈ {0, 1, 2, 3, 5, 10, 30};
  metrics per phase: MAPE and median APE, 90% interval coverage; 95% CIs by bootstrapping wells.
- k=0: the general model's median-well MAPE is below both zero-shot baselines (Gilbert with literature constants;
  pure physics with nominal discharge coefficient), with non-overlapping CIs.
- k≤3: the general model's median-well MAPE is at most 1.25 × that of per-well models trained on k=30 points.
- Coverage: 90% intervals cover 85-95% of held-out points at every k.

## M5 targets
- Single-well inverse: realised rate within ±5% of target in ≥ 90% of trials at k=3; drawdown/GOR/WC limit
  violations ≤ 10% when solving for the 90% quantile.
- Multi-well: realised oil gain over the equal-opening heuristic reported with a 95% CI at k = 0, 3, 30; capacity
  violations ≤ 10%.

## M6 targets (not gates; reported as measured)
- Leaks ≥ 2% of line oil rate with ≥ 24 h of data: top-3 source accuracy ≥ 80%.
- No-leak controls: false alarm rate ≤ 5%.
- Detection delay, top-1/top-3 accuracy vs leak size and MPFM noise, reported as curves.
- Well-level attribution claimed only where the posterior separates the hypotheses.
