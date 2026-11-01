"""Fleet-level machinery shared by the evaluation, control and leak milestones: label preparation, training the
general model and its empirical-Bayes prior, calibrating one well from k labels, and predicting with intervals.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import fewshot, hybrid, network, pvt, units as u

MIN_Q = u.stb_d_to_si(10.0)
LABEL_SD = {'mpfm': {'wc': 0.3, 'gor': 0.08}, 'test': {'wc': 0.1, 'gor': 0.03}}


def prepare(daily, tests, wells):
    """Add design data, mass labels, observed ratios and trailing ratio estimates (what a live system holds)."""
    wells = wells.set_index('well_id') if 'well_id' in wells.columns else wells
    daily = daily.copy()
    for c in ('api', 'gas_sg', 'gor', 'wc', 'sal'):
        daily['design_' + c] = wells.loc[daily.well_id, 'design_' + c].to_numpy()
    des = pvt.Fluid(api=daily.design_api, gas_sg=daily.design_gas_sg, gor=daily.design_gor, wc=daily.design_wc,
                    salinity=daily.design_sal)
    daily['label_m'] = network.std_mass(daily.mpfm_q_o, daily.mpfm_q_g, daily.mpfm_q_w, des)
    daily['true_q_l'] = daily.true_q_o + daily.true_q_w
    daily['flowing'] = (daily.opening > 0) & (daily.true_q_l > MIN_Q) & (daily.label_m > 0)
    daily['obs_wc'] = daily.mpfm_q_w / np.maximum(daily.mpfm_q_o + daily.mpfm_q_w, 1e-12)
    daily['obs_gor'] = daily.mpfm_q_g / np.maximum(daily.mpfm_q_o, 1e-12)
    daily = daily.sort_values(['well_id', 'day']).reset_index(drop=True)
    g = daily.where(daily.flowing).groupby(daily.well_id)
    daily['trail_wc'] = g.obs_wc.transform(lambda s: s.shift(1).rolling(7, min_periods=1).median())
    daily['trail_gor'] = g.obs_gor.transform(lambda s: s.shift(1).rolling(7, min_periods=1).median())
    daily['trail_wc'] = daily.trail_wc.fillna(daily.design_wc).clip(0, 0.98)
    daily['trail_gor'] = daily.trail_gor.fillna(daily.design_gor).clip(1, 3000)

    tests = tests.merge(daily[['well_id', 'day', 'opening', 'p_wh', 't_wh', 'p_man', 'd_choke', 'vendor_gamma']],
                        on=['well_id', 'day'], how='left')
    for c in ('api', 'gas_sg', 'gor', 'wc', 'sal'):
        tests['design_' + c] = wells.loc[tests.well_id, 'design_' + c].to_numpy()
    tdes = pvt.Fluid(api=tests.design_api, gas_sg=tests.design_gas_sg, gor=tests.design_gor, wc=tests.design_wc,
                     salinity=tests.design_sal)
    tests['label_m'] = network.std_mass(tests.test_q_o, tests.test_q_g, tests.test_q_w, tdes)
    tests['obs_wc'] = tests.test_q_w / np.maximum(tests.test_q_o + tests.test_q_w, 1e-12)
    tests['obs_gor'] = tests.test_q_g / np.maximum(tests.test_q_o, 1e-12)
    return daily, tests, wells


def with_est(df, wc, gor):
    out = df.copy()
    out['est_api'], out['est_gas_sg'], out['est_sal'] = df.design_api, df.design_gas_sg, df.design_sal
    out['est_wc'], out['est_gor'] = wc, gor
    return out


SCALE_KS = {'mpfm': (0, 1, 2, 3, 5, 10, 30), 'test': (0, 1, 2, 3)}


@dataclass
class Trained:
    model: hybrid.Hybrid
    prior: fewshot.LatentPrior
    scales: dict = None      # interval scale by regime and number of labels (conformal, from training wells)

    def scale(self, regime, k):
        table = (self.scales or {}).get(regime, {})
        keys = [kk for kk in table if kk <= k]
        return table[max(keys)] if keys else 1.0


def train(tr, tests, n_inner=3, seed=0, interval_scales=False, cal_end=90):
    """Fit the pooled correction on training rows (flowing, MPFM labels) and the empirical-Bayes latent prior from
    out-of-fold residuals. tr: prepared daily rows of training wells; tests: prepared tests (any wells).
    interval_scales: also calibrate the interval width by replaying the label-then-forecast protocol on training
    wells with the inner models that never saw them."""
    trf = with_est(tr, tr.trail_wc, tr.trail_gor)
    m_pr, feats = hybrid.prior(trf, hybrid.fluid_from(trf))
    ok = (m_pr > 0) & np.isfinite(m_pr)
    trf, feats, m_pr = trf[ok], feats[ok], m_pr[ok]
    resid = np.log(trf.label_m.to_numpy() / m_pr)
    inner = (trf.group.to_numpy().astype(int) * 2654435761 % 2 ** 32) % n_inner
    f_oof = np.zeros(len(trf))
    inner_models = []
    for j in range(n_inner):
        mdl = hybrid.Hybrid(n_estimators=300, seed=seed).fit(feats[inner != j], resid[inner != j])
        f_oof[inner == j] = mdl.correction(feats[inner == j])
        inner_models.append(mdl)
    model = hybrid.Hybrid(seed=seed).fit(feats, resid)
    trf = trf.assign(r=resid - f_oof, phi1=np.log(np.maximum(trf.opening.to_numpy(), 1e-3)))
    per_well = []
    tr_tests = tests[tests.well_id.isin(trf.well_id.unique())]
    for wid, gw in trf.groupby('well_id'):
        phi = np.column_stack([np.ones(len(gw)), gw.phi1.to_numpy()])
        b, cov, sd = fewshot.ols_well(phi, gw.r.to_numpy())
        win = [float(np.mean(gw.r.to_numpy()[m] - b[1] * gw.phi1.to_numpy()[m]))
               for m in [(gw.day.to_numpy() // 30) == k for k in range(12)] if m.sum() >= 10]
        tw = tr_tests[tr_tests.well_id == wid]
        t_sd = np.nan
        if len(tw) >= 2:
            twf = with_est(tw, tw.obs_wc.clip(0, 0.98), tw.obs_gor.clip(1, 3000))
            tm, tfe = hybrid.prior(twf, hybrid.fluid_from(twf))
            rr = np.log(tw.label_m.to_numpy() / tm) - model.correction(tfe) - (b[0] + b[1] * np.log(tw.opening.to_numpy()))
            t_sd = float(np.std(rr))
        per_well.append({'b': b, 'cov': cov, 'resid_sd': sd, 'test_resid_sd': t_sd, 'b0_windows': np.array(win)})
    trained = Trained(model=model, prior=fewshot.fit_prior(per_well))
    if interval_scales:
        inner_of_group = {int(gid): int(j) for gid, j in zip(trf.group.to_numpy(), inner)}
        trained.scales = _interval_scales(trf, tests, inner_models, inner_of_group, trained, cal_end, seed)
    return trained


def _interval_scales(trf, tests, inner_models, inner_of_group, trained, cal_end, seed, n_wells=300):
    """Split-conformal width per regime and label count: the 90th percentile of |standardised error| when training
    wells go through the deployment protocol (first k labels before cal_end, stale ratios, forecast after), with
    the inner model that did not see the well. Errors are against MPFM labels, so their noise is in the
    denominator."""
    rng = np.random.default_rng(seed)
    wids = trf.well_id.unique()
    wids = rng.choice(wids, min(n_wells, len(wids)), replace=False)
    lp = trained.prior
    zs = {(reg, k): [] for reg, ks in SCALE_KS.items() for k in ks}
    for wid in wids:
        gw = trf[trf.well_id == wid].sort_values('day')
        ev = gw[gw.day >= cal_end]
        if len(ev) < 5:
            continue
        sub = Trained(model=inner_models[inner_of_group[int(gw.group.iloc[0])]], prior=lp)
        design = gw.iloc[0]
        for reg, ks in SCALE_KS.items():
            src = gw[gw.day < cal_end] if reg == 'mpfm' else \
                tests[(tests.well_id == wid) & (tests.day < cal_end)].sort_values('day')
            for k in ks:
                c = calibrate(src.head(k), design, sub, reg)
                m_hat, lo, hi, _ = predict(ev, c, sub)
                sd = (np.log(hi) - np.log(m_hat)) / 1.645
                z = np.log(ev.label_m.to_numpy() / m_hat) / np.sqrt(sd ** 2 + lp.sig_mpfm ** 2)
                zs[(reg, k)].append(z)
    out = {reg: {} for reg in SCALE_KS}
    for (reg, k), parts in zs.items():
        if parts:
            out[reg][k] = float(max(np.quantile(np.abs(np.concatenate(parts)), 0.90) / 1.645, 1.0))
    return out


@dataclass
class Calibration:
    wc: float
    gor: float
    x: np.ndarray
    p: np.ndarray
    t_last: object
    regime: str = 'mpfm'
    k: int = 0


def calibrate(cal, design_row, trained, regime):
    """Fluid ratios and latent posterior of one well from its calibration labels (k rows, may be empty)."""
    sd = LABEL_SD[regime]
    mw, _ = fewshot.track(cal.day, fewshot.logit(cal.obs_wc.clip(0, 0.98)), [sd['wc']] * len(cal),
                          fewshot.logit(max(float(design_row.design_wc), 1e-3)), 1.5, 1e-4, [0])
    mg, _ = fewshot.track(cal.day, np.log(cal.obs_gor.clip(1, 3000)), [sd['gor']] * len(cal),
                          np.log(float(design_row.design_gor)), 0.15, 1e-5, [0])
    wc, gor = float(fewshot.expit(mw[0])), float(np.exp(mg[0]))
    lp = trained.prior
    if len(cal):
        c = with_est(cal, wc, gor)
        m_c, fe_c = hybrid.prior(c, hybrid.fluid_from(c))
        r = np.log(c.label_m.to_numpy() / m_c) - trained.model.correction(fe_c)
        phi = np.column_stack([np.ones(len(c)), np.log(np.maximum(c.opening.to_numpy(), 1e-3))])
        sd_lab = lp.sig_mpfm if regime == 'mpfm' else lp.sig_test
        x, p, t_last = fewshot.latent_posterior(c.day.to_numpy(), phi, r, np.full(len(c), sd_lab), lp)
    else:
        x, p, t_last = np.zeros(2), np.diag([lp.tau0 ** 2, lp.tau1 ** 2]), None
    return Calibration(wc=wc, gor=gor, x=x, p=p, t_last=t_last, regime=regime, k=int(len(cal)))


def predict(rows, cal, trained, z=1.645):
    """Mass rate with a two-sided interval and the estimated fluid, for rows of one well. The interval uses the
    conformal scale for this regime and label count when the model was trained with interval_scales."""
    e = with_est(rows, cal.wc, cal.gor)
    fl = hybrid.fluid_from(e)
    m0, fe = hybrid.prior(e, fl)
    base = m0 * np.exp(trained.model.correction(fe))
    phi = np.column_stack([np.ones(len(e)), np.log(np.maximum(e.opening.to_numpy(), 1e-3))])
    mean, var = fewshot.latent_predict(cal.x, cal.p, cal.t_last, e.day.to_numpy(), phi, trained.prior)
    sd = np.sqrt(var) * trained.scale(cal.regime, cal.k)
    return base * np.exp(mean), base * np.exp(mean - z * sd), base * np.exp(mean + z * sd), fl
