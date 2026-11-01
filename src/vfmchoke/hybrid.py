"""The general choke VFM: a physics prior from design data, corrected by a pooled monotone LightGBM model.

The prior is the mechanistic choke rate at the measured wellhead and manifold pressures, the wellhead temperature,
the opening, the bean size and the vendor trim curve, with the well's estimated fluid ratios and a nominal discharge
coefficient. The correction is a multiplicative factor exp(f(features)) learned across many wells from inputs that
mean the same thing on every well: pressure ratios, gas quality and volume fraction, water cut, GOR, opening.
No well identity enters, so the model applies unchanged to a well it has never seen.

f is constrained to rise with opening and pressure drop and to fall with the pressure ratio, so the corrected rate
keeps the physics' monotonicity and the choke inverse stays a bisection with one answer.
"""
import numpy as np
import pandas as pd

from . import choke, pvt

MONOTONE = {'opening': 1, 'log_opening': 1, 'dp_bar': 1, 'y': -1, 'y_over_yc': -1}


def fluid_from(df, prefix='est_'):
    return pvt.Fluid(api=df[prefix + 'api'].to_numpy(float), gas_sg=df[prefix + 'gas_sg'].to_numpy(float),
                     gor=df[prefix + 'gor'].to_numpy(float), wc=df[prefix + 'wc'].to_numpy(float),
                     salinity=df[prefix + 'sal'].to_numpy(float))


FEATURES = ['y', 'y_over_yc', 'quality', 'gvf', 'opening', 'log_opening', 'd_choke_mm', 'vendor_gamma', 'wc',
            'gor', 'api', 'rho_l', 'p1_bar', 'dp_bar']


def prior_arrays(p1, p2, t1, opening, d_choke, vendor_gamma, fluid, cd=choke.CD_NOMINAL):
    """Numpy-only core of prior(): mass rate (kg/s) and the feature matrix (n, len(FEATURES))."""
    p1, p2, t1, op = (np.atleast_1d(np.asarray(a, dtype=float)) for a in (p1, p2, t1, opening))
    d_choke = np.broadcast_to(np.asarray(d_choke, dtype=float), p1.shape)
    vg = np.broadcast_to(np.asarray(vendor_gamma, dtype=float), p1.shape)
    st = pvt.state(p1, t1, fluid)
    m, info = choke.mass_rate(p1, p2, t1, fluid, op, d_choke, cd, vg, info=True, st=st)
    cols = [info['y'], info['y'] / info['yc'], st.x, st.v_g / (st.v_g + st.v_l), op, np.log(np.maximum(op, 1e-3)),
            d_choke * 1e3, vg, fluid.wc, fluid.gor, fluid.api, st.rho_l, p1 / 1e5, (p1 - p2) / 1e5]
    return m, np.column_stack([np.broadcast_to(c, p1.shape) for c in cols])


def prior(df, fluid, cd=choke.CD_NOMINAL):
    """Physics-prior total mass rate (kg/s) and the transferable feature table.
    df needs p_wh, p_man, t_wh (SI), opening, d_choke (m), vendor_gamma."""
    m, x = prior_arrays(df['p_wh'].to_numpy(float), df['p_man'].to_numpy(float), df['t_wh'].to_numpy(float),
                        df['opening'].to_numpy(float), df['d_choke'].to_numpy(float),
                        df['vendor_gamma'].to_numpy(float), fluid, cd)
    return m, pd.DataFrame(x, columns=FEATURES, index=df.index)


class Hybrid:
    """Pooled correction model. fit() on (features, log(m_label / m_prior)); correction() returns f."""

    def __init__(self, n_estimators=500, learning_rate=0.05, num_leaves=31, min_child_samples=200, seed=0):
        self.params = dict(n_estimators=n_estimators, learning_rate=learning_rate, num_leaves=num_leaves,
                           min_child_samples=min_child_samples, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.9, random_state=seed, verbose=-1)
        self.model = None
        self.cols = None

    def fit(self, feats, residual, weight=None):
        import lightgbm as lgb
        self.cols = list(feats.columns)
        mono = [MONOTONE.get(c, 0) for c in self.cols]
        self.model = lgb.LGBMRegressor(monotone_constraints=mono, monotone_constraints_method='advanced',
                                       **self.params)
        self.model.fit(feats[self.cols], residual, sample_weight=weight)
        return self

    def correction(self, feats):
        return self.model.predict(feats[self.cols])
