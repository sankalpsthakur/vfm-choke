"""Per-well calibration in closed form.

Rate model for one well: log m = log m_prior + f(features) + b0 + b1 * log(opening) + noise.
(b0, b1) is a latent pair per well: b0 absorbs a wrong discharge coefficient or bean size, b1 a wrong trim curve.
Its prior N(0, diag(tau0^2, tau1^2)) is learned from training wells (empirical Bayes); b0 may drift slowly
(erosion) as a random walk with variance q0 per day. Labels carry their own noise: MPFM readings wider than
test-separator tests. With no labels the prediction is the general model itself; each label tightens it.

Fluid ratios (water cut in logit space, GOR in log space) are tracked the same way: a scalar random walk that
starts at the design value.
"""
from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class LatentPrior:
    tau0: float = 0.10       # sd of b0 across wells
    tau1: float = 0.10       # sd of b1 across wells
    q0: float = 1e-5         # b0 random-walk variance per day
    sig_mpfm: float = 0.04   # sd of an MPFM label (log scale), model error included
    sig_test: float = 0.02   # sd of a separator-test label (log scale), model error included
    sig_model: float = 0.02  # sd of day-to-day model error at prediction time (log scale)

    def to_dict(self):
        return asdict(self)


def latent_posterior(t_obs, phi_obs, r_obs, sd_obs, prior):
    """Kalman filter over the observations (time order). Returns the state mean (2,), covariance (2, 2) and the
    time of the last observation. phi rows are (1, log opening)."""
    x = np.zeros(2)
    p = np.diag([prior.tau0 ** 2, prior.tau1 ** 2])
    t_last = None
    for t, phi, r, sd in sorted(zip(t_obs, phi_obs, r_obs, sd_obs), key=lambda z: z[0]):
        if t_last is not None:
            p = p + np.diag([prior.q0 * (t - t_last), 0.0])
        h = np.asarray(phi, dtype=float)
        s = h @ p @ h + sd ** 2
        k = p @ h / s
        x = x + k * (r - h @ x)
        p = p - np.outer(k, h) @ p
        t_last = t
    return x, p, t_last


def latent_predict(x, p, t_last, t_pred, phi_pred, prior):
    """Mean and variance of the log-rate correction at prediction times (after the last observation)."""
    phi_pred = np.asarray(phi_pred, dtype=float)
    dt = np.maximum(np.asarray(t_pred, dtype=float) - (t_last if t_last is not None else 0.0), 0.0)
    drift = prior.q0 * dt if t_last is not None else 0.0
    mean = phi_pred @ x
    var = np.einsum('ni,ij,nj->n', phi_pred, p, phi_pred) + drift + prior.sig_model ** 2
    return mean, var


def track(t_obs, y_obs, sd_obs, m0, sd0, q, t_pred):
    """Scalar random-walk Kalman filter: posterior mean at each prediction time (after the observations)."""
    m, v, t_last = m0, sd0 ** 2, 0.0
    for t, y, sd in sorted(zip(t_obs, y_obs, sd_obs), key=lambda z: z[0]):
        v = v + q * max(t - t_last, 0.0)
        k = v / (v + sd ** 2)
        m = m + k * (y - m)
        v = (1 - k) * v
        t_last = t
    return np.full(np.shape(t_pred), m), v


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def expit(z):
    return 1.0 / (1.0 + np.exp(-z))


def fit_prior(per_well, sig_floor=0.01):
    """Empirical-Bayes prior from training wells.
    per_well: list of dicts with 'b' (2,), 'cov' (2, 2) from each well's least-squares fit, 'resid_sd' (MPFM),
    'test_resid_sd' (may be nan) and 'b0_windows' (array of b0 fitted on consecutive 30-day windows)."""
    b = np.array([w['b'] for w in per_well])
    cov = np.array([w['cov'] for w in per_well])
    tau2 = np.maximum(b.var(axis=0) - cov[:, [0, 1], [0, 1]].mean(axis=0), 1e-6)
    diffs = np.concatenate([np.diff(w['b0_windows']) for w in per_well if len(w['b0_windows']) > 1] or [np.zeros(1)])
    q0 = max(float(np.var(diffs)) / 30.0 - 2 * float(np.median(cov[:, 0, 0])) / 30.0, 1e-8)
    sig_mpfm = max(float(np.median([w['resid_sd'] for w in per_well])), sig_floor)
    tests = [w['test_resid_sd'] for w in per_well if np.isfinite(w.get('test_resid_sd', np.nan))]
    sig_test = max(float(np.median(tests)) if tests else sig_mpfm / 2, sig_floor)
    return LatentPrior(tau0=float(np.sqrt(tau2[0])), tau1=float(np.sqrt(tau2[1])), q0=q0, sig_mpfm=sig_mpfm,
                       sig_test=sig_test, sig_model=max(sig_test, sig_floor))


def ols_well(phi, r):
    """Least-squares (b0, b1) of one well with its covariance and residual sd."""
    phi = np.asarray(phi, dtype=float)
    r = np.asarray(r, dtype=float)
    if len(r) < 3 or np.ptp(phi[:, 1]) < 1e-6:
        b0 = float(np.mean(r)) if len(r) else 0.0
        sd = float(np.std(r)) if len(r) > 1 else 0.05
        return np.array([b0, 0.0]), np.diag([sd ** 2 / max(len(r), 1), 1.0]), sd
    b, *_ = np.linalg.lstsq(phi, r, rcond=None)
    res = r - phi @ b
    sd = float(np.sqrt(res @ res / max(len(r) - 2, 1)))
    cov = sd ** 2 * np.linalg.inv(phi.T @ phi)
    return b, cov, sd
