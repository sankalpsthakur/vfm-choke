"""Which source fed one oil slick? A ranked posterior over leak hypotheses for one manifold group.

Hypotheses: no leak; the jumper of well k (each well); the manifold; flowline segments; riser positions.
Evidence, each a log-likelihood:
  1. Phase signature of the mass-balance deficit. The per-phase residual r = sum of well-level rates (MPFM or VFM)
     minus the separator, averaged after the detected onset and with its pre-onset bias removed, should point along
     the leaked fluid's composition: well k's own (oil, gas, water) for its jumper, the commingled mix downstream.
     Profile likelihood over the leak size lambda >= 0.
  2. Pressure signature: manifold and riser-base pressure shifts after onset against each hypothesis' predicted
     sensitivity (from the network model) times lambda.
  3. Slick location: the up-current end of the slick, back-tracked for drift, against each hypothesis' seabed
     position.
  4. Volume consistency: oil released between onset and detection (after weathering) against the slick volume range.
Mass balance localises to the line; separating wells needs their ratios to differ. The posterior says how well the
evidence separates them, rather than forcing a single answer.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class Hypothesis:
    kind: str          # 'none', 'jumper', 'manifold', 'flowline', 'riser'
    well: int          # slot for jumper hypotheses, else -1
    pos: float         # fraction along flowline / riser
    xy: tuple          # seabed position, m (manifold at the origin, flowline along +x)


def layout(n_slots, line_length, rng):
    """Well positions on a drill-centre ring around the manifold, flowline along +x to the riser base."""
    ang = rng.uniform(0, 2 * np.pi, n_slots)
    rad = rng.uniform(40, 180, n_slots)
    return np.column_stack([rad * np.cos(ang), rad * np.sin(ang)]), np.array([line_length, 0.0])


def hypotheses(open_slots, well_xy, riser_xy, fl_positions=(0.1, 0.3, 0.5, 0.7, 0.9), riser_positions=(0.2, 0.6)):
    hs = [Hypothesis('none', -1, 0.0, (np.nan, np.nan))]
    hs += [Hypothesis('jumper', int(s), 0.0, tuple(well_xy[s] * 0.5)) for s in open_slots]
    hs.append(Hypothesis('manifold', -1, 0.0, (0.0, 0.0)))
    hs += [Hypothesis('flowline', -1, p, (p * riser_xy[0], 0.0)) for p in fl_positions]
    hs += [Hypothesis('riser', -1, p, tuple(riser_xy)) for p in riser_positions]
    return hs


def cusum_onset(x, ref_end, k_sd=0.5, h_sd=6.0):
    """One-sided CUSUM on a deficit series. Baseline mean and sd from x[:ref_end].
    Returns (alarm index or None, onset estimate or None)."""
    mu, sd = np.mean(x[:ref_end]), max(np.std(x[:ref_end]), 1e-12)
    s, start = 0.0, None
    for i in range(ref_end, len(x)):
        s_new = max(0.0, s + (x[i] - mu) / sd - k_sd)
        if s == 0.0 and s_new > 0:
            start = i
        s = s_new
        if s > h_sd:
            return i, start
    return None, None


def phase_loglik(r_mean, r_cov, direction):
    """Max over lambda >= 0 of the Gaussian log-likelihood of r_mean = lambda * direction."""
    inv = np.linalg.inv(r_cov)
    c = np.asarray(direction, dtype=float)
    denom = float(c @ inv @ c)
    if not np.isfinite(denom) or denom <= 0.0:          # no usable direction (e.g. a well shut in all along)
        return -0.5 * float(r_mean @ inv @ r_mean), 0.0
    lam = max(float(c @ inv @ r_mean) / denom, 0.0)
    e = r_mean - lam * c
    return -0.5 * float(e @ inv @ e), lam


def posterior(hs, r_mean, r_cov, compositions, dp_obs, dp_cov, dp_sens, slick_xy, loc_sd, drift_xy,
              released_vol, slick_vol, vol_sd_log=0.7, prior_none=0.2):
    """Combine the evidence into a posterior over hypotheses.
    compositions[i]: phase direction (oil, gas, water) per hypothesis (None for 'none').
    dp_sens[i]: predicted (d p_man, d p_riser_base) per unit leak oil rate for hypothesis i.
    released_vol(i, lam): oil volume released by detection for hypothesis i at leak oil rate lam."""
    out = []
    inv_dp = np.linalg.inv(dp_cov)
    for i, h in enumerate(hs):
        if h.kind == 'none':
            ll_phase = -0.5 * float(r_mean @ np.linalg.inv(r_cov) @ r_mean)
            ll_dp = -0.5 * float(dp_obs @ inv_dp @ dp_obs)
            ll_loc = np.log(1e-3)          # a slick with no subsea source: passing vessel, seep; flat, small
            ll_vol, lam = 0.0, 0.0
        else:
            ll_phase, lam = phase_loglik(r_mean, r_cov, compositions[i])
            e = dp_obs - lam * np.asarray(dp_sens[i])
            ll_dp = -0.5 * float(e @ inv_dp @ e)
            d = np.asarray(slick_xy) - (np.asarray(h.xy) + np.asarray(drift_xy))
            ll_loc = -0.5 * float(d @ d) / loc_sd ** 2
            v = max(released_vol(i, lam), 1e-9)
            ll_vol = -0.5 * ((np.log(v) - np.log(slick_vol)) / vol_sd_log) ** 2
        out.append({'hypothesis': h, 'lambda': lam, 'll_phase': ll_phase, 'll_dp': ll_dp, 'll_loc': ll_loc,
                    'll_vol': ll_vol})
    n_src = len(hs) - 1
    logp = np.array([o['ll_phase'] + o['ll_dp'] + o['ll_loc'] + o['ll_vol']
                     + np.log(prior_none if o['hypothesis'].kind == 'none' else (1 - prior_none) / n_src)
                     for o in out])
    p = np.exp(logp - logp.max())
    p /= p.sum()
    for o, pi in zip(out, p):
        o['posterior'] = float(pi)
    return sorted(out, key=lambda o: -o['posterior'])
