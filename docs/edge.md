# Edge deployment

## What runs at the edge
- `vfmchoke_edge.npz`: the pooled correction as flat tree arrays plus, per well, the design fluid, choke size and
  vendor trim, and the calibration (water cut, GOR, b0, b1). Measured size and latency are in REPORT.md (M7).
- `vfmchoke.edge.vfm_rate`: numpy only. Physics choke rate × exp(tree correction + b0 + b1·log opening).
- `vfmchoke.edge.edge_step`: one setpoint-optimisation step on lookup tables (each well's rate on an opening ×
  downstream-pressure grid, and the flowline's required inlet pressure against total mass rate). The tables are built
  centrally from the twin and shipped with the artifact.

## Inputs (per well, per scan)
| Tag (example name) | Meaning | Unit |
|---|---|---|
| `WELLxx.PT_UPSTREAM` | wellhead pressure upstream of the choke | bar g |
| `WELLxx.TT_UPSTREAM` | wellhead temperature upstream of the choke | °C |
| `WELLxx.PT_DOWNSTREAM` (or the manifold PT) | pressure downstream of the choke | bar g |
| `WELLxx.ZT_CHOKE` | choke position | % open |
| `WELLxx.MPFM_Q_OIL/GAS/WATER` (optional) | gamma MPFM rates, used as labels | Sm³/d |
| `WELLxx.TEST_Q_OIL/GAS/WATER` (optional) | test separator results, used as labels | Sm³/d |
| `MANIFOLD.PT`, `RISERBASE.PT`, `SEP.Q_OIL/GAS/WATER` | network pressures and separator rates (leak evidence) | bar g, Sm³/d |

## Outputs
- Per well: oil, gas and water rate estimates with a 90% interval; the critical/subcritical flag.
- Advisory: choke opening for a requested rate, and a manifold setpoint set under separator caps, each with the
  constraint margins it used.
- Leak watch: per-phase mass-balance residual, CUSUM state, and the ranked source posterior when a slick is reported.

## Placement and integration
- Purdue level 2/3: an industrial PC or edge gateway beside the historian, reading tags over OPC UA (client) and
  publishing results over MQTT (Sparkplug B payloads) or back to OPC UA as soft tags.
- Calibration updates (new labels → b0, b1, ratios) are closed-form and can run on the edge; retraining the pooled
  correction happens centrally and ships a new artifact.
- Where a one-way data diode separates the control network from the business network, the edge publishes outward
  only; nothing the edge computes writes back into the control system.

## Safety boundary
Setpoints are advisory. An operator and the control system (with its own interlocks and management of change)
decide whether to move a choke. The edge never writes a setpoint.
