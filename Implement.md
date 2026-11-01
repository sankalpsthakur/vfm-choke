# Implement: execution brief

Follows Plans.md milestone by milestone. Each milestone ends with its gate, measured in a Kaggle kernel, and a dated
entry in Documentation.md.

## Run procedure (every milestone)
1. Edit code locally (no local execution).
2. `python3 kaggle/build.py <entry> <slug>` writes `kaggle/<slug>/` with the notebook and allowlisted metadata
   (`is_private: true`, CPU, internet only when a benchmark download needs it).
3. Before pushing, check the account's running CPU sessions (cap 5, shared with the competition thread).
4. `kaggle kernels push -p kaggle/<slug>`; confirm the reply names `sankalpsthakur/<slug>`.
5. Wait for COMPLETE, then immediately fetch outputs and the log into `reports/runs/<slug>-v<N>/`
   (versioned refs return the latest version, so fetch before the next push).
6. Record the kernel version, gate metrics and decision in Documentation.md.

## Milestone notes
- **M1:** units, pvt, choke, well, network, mpfm + `tests/` (pytest). Entry: run pytest, write `test_report.json`.
- **M2:** simfield. Entry: generate the population and scenario sets to parquet in the kernel output; write summary
  statistics for the gate.
- **M3:** hybrid + fewshot. Entry: grouped folds, k-curves, baselines, bootstrap CIs; write `m3_metrics.json`.
- **M4:** `benchmarks/volve.py`, `benchmarks/dabiri.py`, `benchmarks/dsouza.py`. Internet on for downloads (DOCX via
  PMC, DataverseNO/GitHub for B3); Volve via the attached Kaggle dataset.
- **M5:** control. Entry: closed-loop trials on SimField truth.
- **M6:** leak. Entry: scenario sweep + the one-slick demo.
- **M7:** edge. Entry: export, artifact size, RAM, p50/p99 latency on one core.
- **M8:** REPORT.md from the fetched outputs; publish gate.
