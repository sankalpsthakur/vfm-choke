# B3: D'Souza, Pfeiffer & Mirlekar (SIMS EUROSIM 2024). Regenerate the two-well data with their MATLAB simulator in
# Octave, check fidelity by re-running their models, then score ours. Protocol: BENCHMARKS.md ("B3 run protocol").
import json
import os
import subprocess
import time
import urllib.request

import numpy as np
import pandas as pd

T0 = time.time()
DV = {'oil_field_model.m': 268788, 'make_w_gc.m': 268786, 'update_states.m': 268787}
for f, fid in DV.items():
    with urllib.request.urlopen(urllib.request.Request(f'https://dataverse.no/api/access/datafile/{fid}',
                                                       headers={'User-Agent': 'Mozilla/5.0'}), timeout=60) as r:
        src = r.read().decode()
    if f == 'oil_field_model.m':     # the two-well model, returning every intermediate (GitHub version's outputs)
        lines = src.splitlines()
        lines[0] = ('function  [dm, w_op, w_gop, P_a, P_ainj, rho_l, V_G, rho_m, P_tinj, Y_2, rho_ga, w_ginj, P_wf, '
                    'w_l, w_o, w_g, P_wh, Y_3, w_gp, w_lp, w_wp] = oil_field_model(m, w_ga, Theta)')
        src = '\n'.join(l.replace('%     w_wp = w_lp -w_op;', '    w_wp = w_lp - w_op;') for l in lines)
    with open(f, 'w') as fh:
        fh.write(src)
with open('gen.m', 'w') as fh:
    fh.write("""
Theta_nom = [2.51 1.63 0.08 0.07 0.15 0.15];
dt = 20; timespan = 96;
t = 0:dt:timespan*60*60;
w_gc = make_w_gc('2step', [40000 1; 20000 1; 40000 1], t);
w_ga = [30 30]/100 .* w_gc;
m0 = [16422 15445 1669 1517 15229 13458];
m = zeros(length(t), 6); m(1,:) = m0;
for k = 1:length(t)-1
  m(k+1,:) = update_states(m(k,:), w_ga(k,:), Theta_nom, dt);
end
Theta_OL = Theta_nom .* ones(length(t),1);
[dm, w_op, w_gop, P_a, P_ainj, rho_l, V_G, rho_m, P_tinj, Y_2, rho_ga, w_ginj, P_wf, w_l, w_o, w_g, P_wh, Y_3, w_gp, w_lp, w_wp] = oil_field_model(m, zeros(length(t),2), Theta_OL);
w_ga_full = [w_ga; w_ga(end,:)];
for wi = 1:2
  M = [t'/3600, w_ga_full(:,wi), P_wf(:,wi), P_wh(:,wi), w_op(:,wi), w_gp(:,wi), w_lp(:,wi)];
  dlmwrite(sprintf('raw_well%d.csv', wi), M, 'precision', 12);
end
""")
if subprocess.run('command -v octave', shell=True).returncode != 0:
    subprocess.run('apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq octave >/dev/null',
                   shell=True, check=True, timeout=1500)
r = subprocess.run(['octave', '--no-gui', '--quiet', 'gen.m'], capture_output=True, text=True, timeout=3000)
print(r.stdout[-2000:], r.stderr[-2000:])
print(f'generated in {time.time() - T0:.0f}s', flush=True)

COLS = ['Time [hrs]', 'W_ga [kg/s]', 'P_wf [bar]', 'P_wh [bar]', 'W_op [kg/s]', 'W_gp [kg/s]', 'W_lp [kg/s]']
X_COLS, Y_COLS = ['W_ga [kg/s]', 'P_wf [bar]', 'P_wh [bar]'], ['W_op [kg/s]', 'W_gp [kg/s]', 'W_lp [kg/s]']
data = {}
for wi in (1, 2):
    raw = pd.read_csv(f'raw_well{wi}.csv', header=None, names=COLS)
    avg = raw.groupby(np.arange(len(raw)) // 3).mean()          # 1-minute averages of the 20 s steps
    avg.to_csv(f'Well{wi}_average_data.csv', index=False)
    data[wi] = avg
    print(f'well {wi}: {len(avg)} rows; levels at 20/50/90 h:',
          avg.iloc[[1200, 3000, 5400]][Y_COLS].round(2).to_dict('records'), flush=True)

from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.metrics import mean_absolute_percentage_error as sk_mape  # noqa: E402
from sklearn.multioutput import MultiOutputRegressor  # noqa: E402
from sklearn.neural_network import MLPRegressor  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVR  # noqa: E402
from xgboost import XGBRegressor  # noqa: E402

TR, VA = 2304, 4033
PUBLISHED = {'lstm': (1.96, 1.53), 'mv_linear': (2.14, 7.57), 'mlp': (2.43, 5.49), 'xgboost': (4.23, 5.56),
             'svr': (5.04, 4.31)}
RHO_O, RHO_W, P_R, P_S, G = 800.0, 1000.0, 150.0, 30.0, 9.80665
GEOM = {1: {'L_vl': 2271.0, 'L_r': 114.0}, 2: {'L_vl': 2344.0, 'L_r': 67.0}}
THETA = {1: (2.51, 0.08, 0.15), 2: (1.63, 0.07, 0.15)}           # PI, GOR, WC used by the simulator


def mape(pred, true):
    return float(sk_mape(true, pred) * 100)


def physics_features(df, wi, wc):
    rho_l = RHO_W * wc + RHO_O * (1 - wc)
    p_wf, p_wh = df['P_wf [bar]'].to_numpy(), df['P_wh [bar]'].to_numpy()
    p_tinj = p_wf - rho_l * G * GEOM[wi]['L_r'] * 1e-5
    rho_m = np.maximum((p_tinj - p_wh) * 1e5 / (G * GEOM[wi]['L_vl']), 1.0)
    y3 = 1 - 0.66 * (p_wh - P_S) / np.maximum(p_wh, 1e-6)
    choke_feat = y3 * np.sqrt(rho_m * np.maximum(p_wh - P_S, 0.0))
    return np.column_stack([P_R - p_wf, df['W_ga [kg/s]'].to_numpy(), choke_feat])


def physics_prior(df, pi, gor, wc):
    rho_l = RHO_W * wc + RHO_O * (1 - wc)
    w_l = pi * np.maximum(P_R - df['P_wf [bar]'].to_numpy(), 0)
    w_o = RHO_O / rho_l * (1 - wc) * w_l
    return np.column_stack([w_o, gor * w_o + df['W_ga [kg/s]'].to_numpy(), w_l])


results = {}
for wi in (1, 2):
    df = data[wi]
    X = StandardScaler().fit_transform(df[X_COLS])                 # fit on all rows, as in their notebook
    Y = df[Y_COLS].to_numpy()
    te = slice(VA, None)
    res = {}
    res['mv_linear'] = mape(LinearRegression().fit(X[:TR], Y[:TR]).predict(X[te]), Y[te])
    xgb_kw = dict(n_estimators=300, max_depth=7, learning_rate=0.01) if wi == 1 else \
        dict(n_estimators=100, max_depth=3, learning_rate=0.1)
    res['xgboost'] = mape(MultiOutputRegressor(XGBRegressor(**xgb_kw)).fit(X[:TR], Y[:TR]).predict(X[te]), Y[te])
    mlp = MLPRegressor(hidden_layer_sizes=(100, 100), activation='identity' if wi == 1 else 'logistic',
                       solver='lbfgs', alpha=0.0005, max_iter=1000, random_state=0)
    res['mlp'] = mape(mlp.fit(X[:TR], Y[:TR]).predict(X[te]), Y[te])
    svr_kw = dict(kernel='rbf', gamma=0.1, C=1.0) if wi == 1 else dict(kernel='rbf', gamma=0.01, C=10.0)
    res['svr'] = mape(MultiOutputRegressor(SVR(**svr_kw)).fit(X[:TR], Y[:TR]).predict(X[te]), Y[te])
    try:
        import tensorflow as tf
        tf.random.set_seed(0)
        units = 32 if wi == 1 else 40
        inp = tf.keras.Input(shape=(1, 3))
        out = tf.keras.layers.Dense(3, activation='linear')(tf.keras.layers.LSTM(units)(inp))
        lstm = tf.keras.Model(inp, out)
        lstm.compile(loss='mean_squared_error', optimizer='adam')
        lstm.fit(X[:TR, None, :], Y[:TR], validation_data=(X[TR:VA, None, :], Y[TR:VA]), epochs=100, batch_size=32,
                 verbose=0, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)])
        res['lstm'] = mape(lstm.predict(X[te, None, :], verbose=0), Y[te])
    except Exception as e:      # report rather than hide a missing TensorFlow
        res['lstm'] = f'error: {e}'
    # ours: physics-feature regression, fitted on the training rows only
    F = physics_features(df, wi, THETA[wi][2])
    res['ours_physics_features'] = mape(LinearRegression().fit(F[:TR], Y[:TR]).predict(F[te]), Y[te])
    # zero-shot and few-shot: prior from deliberately wrong design values, then k = 3 multipliers
    pi, gor, wc = THETA[wi]
    prior = physics_prior(df, pi * 1.2, gor * 0.8, wc + 0.05)
    res['ours_zero_shot_wrong_design'] = mape(prior[te], Y[te])
    cal = [500, 1200, 2000]
    mult = np.exp(np.mean(np.log(Y[cal] / prior[cal]), axis=0))
    res['ours_few_shot_k3'] = mape(prior[te] * mult, Y[te])
    results[f'well{wi}'] = res
    print(wi, json.dumps(res, indent=1), flush=True)

fidelity = {}
for mdl in ('lstm', 'mv_linear'):
    for wi in (1, 2):
        ours = results[f'well{wi}'][mdl]
        pub = PUBLISHED[mdl][wi - 1]
        fidelity[f'{mdl}_well{wi}'] = isinstance(ours, float) and abs(ours / pub - 1) <= 0.30
results['fidelity'] = fidelity
results['fidelity_ok'] = all(fidelity.values())
best_pub = {1: 1.96, 2: 1.53}
results['gate'] = {
    'counts': results['fidelity_ok'],
    'beat_outright_both_wells': all(results[f'well{w}']['ours_physics_features'] < best_pub[w] for w in (1, 2)),
    'few_shot_k3_parity_both_wells': all(results[f'well{w}']['ours_few_shot_k3'] <= best_pub[w] for w in (1, 2)),
    'published_best': best_pub, 'seconds': round(time.time() - T0, 1),
}
with open('b3_results.json', 'w') as fh:
    json.dump(results, fh, indent=1)
print(json.dumps(results, indent=1))
