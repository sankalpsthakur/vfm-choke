"""Edge inference: numpy only, no LightGBM at run time.

- The pooled correction is exported from LightGBM to flat tree arrays and evaluated by array traversal.
- The choke physics is the same numpy code as everywhere else.
- For setpoint optimisation the twin is shipped as lookup tables: each well's liquid rate on a grid of opening and
  downstream pressure, and the flowline's required inlet pressure on a grid of total mass rate. One optimizer step
  is then a short bisection over table lookups.
"""
import io

import numpy as np

from . import choke, hybrid, network, pvt, well


def export_trees(lgbm):
    """Flatten a fitted LGBMRegressor into arrays: per node feature, threshold, children (negative = leaf),
    leaf values, default-left flag. Numerical splits only (all our features are numeric)."""
    dump = lgbm.booster_.dump_model()
    feats, thr, left, right, dleft, miss, leaf_vals, roots = [], [], [], [], [], [], [], []
    miss_code = {'None': 0, 'Zero': 1, 'NaN': 2}

    def walk(node):
        if 'leaf_index' in node or 'split_feature' not in node:
            leaf_vals.append(node.get('leaf_value', 0.0))
            return -len(leaf_vals)                     # leaf k is encoded as -(k+1)
        i = len(feats)
        feats.append(node['split_feature'])
        thr.append(node['threshold'])
        dleft.append(bool(node.get('default_left', True)))
        miss.append(miss_code.get(node.get('missing_type', 'None'), 0))
        left.append(0)
        right.append(0)
        left[i] = walk(node['left_child'])
        right[i] = walk(node['right_child'])
        return i

    for t in dump['tree_info']:
        roots.append(walk(t['tree_structure']))
    return {'feature': np.array(feats, np.int32), 'threshold': np.array(thr, np.float64),
            'left': np.array(left, np.int32), 'right': np.array(right, np.int32),
            'default_left': np.array(dleft, bool), 'missing': np.array(miss, np.int8),
            'leaf': np.array(leaf_vals, np.float64),
            'roots': np.array(roots, np.int32), 'feature_names': np.array(dump['feature_names'])}


def predict_trees(arr, x):
    """Sum of tree outputs for rows x (n, n_features), traversing all trees at once."""
    x = np.atleast_2d(np.asarray(x, dtype=float))
    n, t = x.shape[0], arr['roots'].shape[0]
    node = np.broadcast_to(arr['roots'], (n, t)).copy()
    out = np.zeros(n)
    active = node >= 0
    out += np.where(active, 0.0, arr['leaf'][np.where(active, 0, -node - 1)]).sum(1)
    rows = np.broadcast_to(np.arange(n)[:, None], (n, t))
    while active.any():
        nd = np.where(active, node, 0)
        v = x[rows, arr['feature'][nd]]
        mt = arr['missing'][nd]
        nan = np.isnan(v)
        # LightGBM: missing_type None turns NaN into 0; NaN sends NaN to the default side; Zero also sends ~0 there
        to_default = ((mt == 2) & nan) | ((mt == 1) & (nan | (np.abs(np.nan_to_num(v)) <= 1e-35)))
        v = np.where(nan & (mt == 0), 0.0, v)
        go_left = np.where(to_default, arr['default_left'][nd], v <= arr['threshold'][nd])
        nxt = np.where(go_left, arr['left'][nd], arr['right'][nd])
        node = np.where(active, nxt, node)
        newly_leaf = active & (node < 0)
        out += np.where(newly_leaf, arr['leaf'][np.where(newly_leaf, -node - 1, 0)], 0.0).sum(1)
        active = active & (node >= 0)
    return out


def save(path, arrays, well_params):
    buf = {f'tree_{k}': v for k, v in arrays.items()}
    buf.update({f'well_{k}': np.asarray(v) for k, v in well_params.items()})
    np.savez_compressed(path, **buf)


def load(path):
    z = np.load(path, allow_pickle=False)
    trees = {k[5:]: z[k] for k in z.files if k.startswith('tree_')}
    wp = {k[5:]: z[k] for k in z.files if k.startswith('well_')}
    return trees, wp


def vfm_rate(trees, wp, p1, p2, t1, opening):
    """Calibrated VFM mass rate (kg/s) on the edge, numpy only. wp holds per-well design and calibration values:
    api, gas_sg, gor, wc, sal, d_choke, vendor_gamma, b0, b1."""
    fl = pvt.Fluid(wp['api'], wp['gas_sg'], wp['gor'], wp['wc'], wp['sal'])
    m0, x = hybrid.prior_arrays(p1, p2, t1, opening, wp['d_choke'], wp['vendor_gamma'], fl)
    order = [hybrid.FEATURES.index(str(s)) for s in trees['feature_names']]
    f = predict_trees(trees, x[:, order])
    return m0 * np.exp(f + wp['b0'] + wp['b1'] * np.log(np.maximum(np.atleast_1d(opening), 1e-3)))


def twin_tables(w, table, u_grid, p2_grid):
    """Each twin well's liquid rate (Sm3/s) on an (opening, downstream pressure) grid: shape (W, n_u, n_p)."""
    W = np.shape(w.p_res)[0]
    out = np.zeros((W, len(u_grid), len(p2_grid)))
    for a, uu in enumerate(u_grid):
        for b, pp in enumerate(p2_grid):
            out[:, a, b] = well.solve_rate(w, table, np.full(W, pp), np.full(W, uu))['q_l']
    return out


def line_table(line, fluid_mix, t_mix, m_grid):
    """Required manifold pressure for a total mass rate of the commingled stream (no leak)."""
    p_rb = network.section_up(np.full(len(m_grid), float(line.p_sep[0])), m_grid, fluid_mix, t_mix,
                              float(line.d[0]), float(line.depth[0]), 1.0)
    return network.section_up(p_rb, m_grid, fluid_mix, t_mix, float(line.d[0]), float(line.length[0]), 0.0)


def edge_step(rate_tab, u_grid, p2_grid, mls, m_grid, p_req, u_vec, iters=30):
    """One optimizer step on the edge: manifold pressure and well rates for an opening vector, by bisection on
    the manifold pressure with bilinear table lookups."""
    W = rate_tab.shape[0]
    ui = np.interp(u_vec, u_grid, np.arange(len(u_grid)))
    u0 = np.clip(np.floor(ui).astype(int), 0, len(u_grid) - 2)
    wu = ui - u0
    lo, hi = p2_grid[0], p2_grid[-1]
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        pi_ = np.interp(mid, p2_grid, np.arange(len(p2_grid)))
        p0 = min(int(pi_), len(p2_grid) - 2)
        wp_ = pi_ - p0
        r = rate_tab[np.arange(W), u0]
        r1 = rate_tab[np.arange(W), u0 + 1]
        q = ((1 - wu) * ((1 - wp_) * r[:, p0] + wp_ * r[:, p0 + 1])
             + wu * ((1 - wp_) * r1[:, p0] + wp_ * r1[:, p0 + 1]))
        need = np.interp((q * mls).sum(), m_grid, p_req)
        lo, hi = (mid, hi) if mid < need else (lo, mid)
    return 0.5 * (lo + hi), q


def artifact_bytes(arrays, well_params):
    bio = io.BytesIO()
    np.savez_compressed(bio, **{f't_{k}': v for k, v in arrays.items()},
                        **{f'w_{k}': np.asarray(v) for k, v in well_params.items()})
    return bio.getbuffer().nbytes
