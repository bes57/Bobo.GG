"""
Phase 2B: Post-hoc calibration.

Inputs: baseline predictions from the winning margin variant (sqrt @ BETA=0.140).
Methods:
  (a) global Platt
  (b) per-bucket Platt with L2 toward global (6 buckets)
  (c) global isotonic regression
  (d) beta calibration (Kull et al)

70/30 temporal split — fit on first 70%, evaluate on held-out 30%.

Also report: equivalent BETA shift that would produce the same global Platt effect.
If global Platt is well-approximated by BETA' = BETA / slope, it's just BETA mis-tune.
"""
from __future__ import annotations
import json
import math
import sys
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, '/Users/benny_es1/PythonTest/artifacts')
import harness as H


def bucket_of(m):
    fmt = m['fmt']
    if m['intl']:
        if m['fav_region'] == 'CN' or m['dog_region'] == 'CN':
            rb = 'intl_CN'
        else:
            rb = 'intl_noCN'
    else:
        rb = 'domestic'
    return f'{fmt}_{rb}'


def platt_fit(p, o):
    return H.platt_slope(p, o)


def platt_apply(p, a, b):
    eps = 1e-9
    p = np.clip(np.asarray(p), eps, 1 - eps)
    z = np.log(p / (1 - p))
    return 1 / (1 + np.exp(-(a + b * z)))


def per_bucket_platt_fit(p, o, buckets, l2=1.0):
    """Fit a per-bucket Platt model with L2 regularization toward GLOBAL (a, b).

    Sklearn doesn't ship a clean L2-regularized Platt, so I'll write a small
    IRLS that adds (a_k - a_global)^2 + (b_k - b_global)^2 * l2 to the loss.

    Simpler version: independently fit each bucket, then weighted average with
    global fit, weight = n_k / (n_k + l2_anchor_n).
    """
    a_g, b_g = platt_fit(p, o)
    out = {'__global__': (a_g, b_g)}
    for bk in set(buckets):
        mask = np.array([x == bk for x in buckets])
        if mask.sum() < 20:
            out[bk] = (a_g, b_g)  # use global if too few samples
            continue
        a_k, b_k = platt_fit(p[mask], o[mask])
        # Shrink toward global, anchor weight = l2
        n = int(mask.sum())
        w = n / (n + l2)
        a_eff = w * a_k + (1 - w) * a_g
        b_eff = w * b_k + (1 - w) * b_g
        out[bk] = (a_eff, b_eff)
    return out


def per_bucket_platt_apply(p, buckets, params):
    out = np.empty_like(np.asarray(p), dtype=float)
    p = np.asarray(p)
    for bk in set(buckets):
        mask = np.array([x == bk for x in buckets])
        a, b = params.get(bk, params['__global__'])
        out[mask] = platt_apply(p[mask], a, b)
    return out


def beta_calibration_fit(p, o):
    """Beta calibration (Kull et al 2017). Two-parameter sigmoid on
    log(p), log(1-p) features. Fits b = (a, b, c) via logistic regression.

    log-odds = c + a*log(p) + b*log(1-p)

    Returns coefficients.
    """
    eps = 1e-9
    p = np.clip(np.asarray(p), eps, 1 - eps)
    o = np.asarray(o)
    x = np.column_stack([np.log(p), np.log(1 - p), np.ones_like(p)])
    # Newton-Raphson
    beta = np.zeros(3)
    for _ in range(80):
        eta = x @ beta
        pi = 1 / (1 + np.exp(-eta))
        W = pi * (1 - pi)
        g = x.T @ (o - pi)
        H_ = -x.T @ (x * W[:, None])
        try:
            d = np.linalg.solve(H_, -g)
        except np.linalg.LinAlgError:
            break
        beta += d
        if abs(d).max() < 1e-8:
            break
    return beta


def beta_calibration_apply(p, beta):
    eps = 1e-9
    p = np.clip(np.asarray(p), eps, 1 - eps)
    x = np.column_stack([np.log(p), np.log(1 - p), np.ones_like(p)])
    eta = x @ beta
    return 1 / (1 + np.exp(-eta))


def brier(p, o):
    return float(np.mean((np.asarray(p) - np.asarray(o)) ** 2))


# ───────────────────────────────────────────────────────────────────
# Setup
# ───────────────────────────────────────────────────────────────────
matches = H.load_matches()
matches.sort(key=lambda m: m['date'])
N = len(matches)
cutoff_idx = int(0.7 * N)
train_m = matches[:cutoff_idx]
test_m  = matches[cutoff_idx:]
print(f'train={len(train_m)}, test={len(test_m)}, cutoff={matches[cutoff_idx]["date"]}')

# Baseline predictions (sqrt, BETA=0.140, INTL_BONUS=0.22, CN_DOG_OFFSET=0.47)
p_train, o_train = H.predict_series(train_m)
p_test, o_test = H.predict_series(test_m)

base_test_brier = brier(p_test, o_test)
print(f'Uncalibrated test Brier: {base_test_brier:.5f}')

results = {'uncalibrated_test_brier': base_test_brier, 'methods': {}}

# (a) Global Platt
a_g, b_g = platt_fit(p_train, o_train)
p_test_a = platt_apply(p_test, a_g, b_g)
br_a = brier(p_test_a, o_test)
final_a, final_b = platt_fit(p_test_a, o_test)
results['methods']['global_platt'] = {
    'fit_a': a_g, 'fit_b': b_g,
    'test_brier': br_a, 'delta': br_a - base_test_brier,
    'final_test_platt_b': final_b,
}

# (b) Per-bucket Platt, L2=50 (≈ shrink toward global unless bucket has 50+ matches)
buckets_train = [bucket_of(m) for m in train_m]
buckets_test  = [bucket_of(m) for m in test_m]
params = per_bucket_platt_fit(np.array(p_train), np.array(o_train), buckets_train, l2=50.0)
p_test_b = per_bucket_platt_apply(p_test, buckets_test, params)
br_b = brier(p_test_b, o_test)
final_a_b, final_b_b = platt_fit(p_test_b, o_test)
results['methods']['per_bucket_platt'] = {
    'params': {bk: list(v) for bk, v in params.items()},
    'test_brier': br_b, 'delta': br_b - base_test_brier,
    'final_test_platt_b': final_b_b,
}

# (c) Isotonic regression (global)
iso = IsotonicRegression(out_of_bounds='clip', y_min=0.001, y_max=0.999)
iso.fit(p_train, o_train)
p_test_c = iso.predict(p_test)
br_c = brier(p_test_c, o_test)
final_a_c, final_b_c = platt_fit(p_test_c, o_test)
results['methods']['isotonic'] = {
    'test_brier': br_c, 'delta': br_c - base_test_brier,
    'final_test_platt_b': final_b_c,
}

# (d) Beta calibration
beta_params = beta_calibration_fit(p_train, o_train)
p_test_d = beta_calibration_apply(p_test, beta_params)
br_d = brier(p_test_d, o_test)
final_a_d, final_b_d = platt_fit(p_test_d, o_test)
results['methods']['beta_calibration'] = {
    'params': list(beta_params),
    'test_brier': br_d, 'delta': br_d - base_test_brier,
    'final_test_platt_b': final_b_d,
}

# Equivalent BETA shift for global Platt
# If Platt is b_g, that's roughly equivalent to scaling BETA by b_g.
# So a model with BETA = 0.140 / 1.0 * b_g would give the same effect.
# We test: does retuning BETA on training match the global-Platt benefit?
best_beta_shift = 0.140 * b_g
p_test_beta_shift, _ = H.predict_series(test_m, beta=best_beta_shift)
br_beta_shift = brier(p_test_beta_shift, o_test)
print(f'\nEquivalent BETA shift: BETA={best_beta_shift:.4f} → test Brier {br_beta_shift:.5f}')

results['equivalent_beta_shift'] = {
    'shifted_beta': best_beta_shift,
    'test_brier': br_beta_shift,
    'delta_vs_global_platt': br_beta_shift - br_a,
    'note': 'If close to 0, global Platt is just BETA mis-tune.',
}

# Paired bootstrap for each calibrator
for k in ['global_platt', 'per_bucket_platt', 'isotonic', 'beta_calibration']:
    p_calibrated = {
        'global_platt': p_test_a,
        'per_bucket_platt': p_test_b,
        'isotonic': p_test_c,
        'beta_calibration': p_test_d,
    }[k]
    btr = H.paired_bootstrap_brier(p_test, p_calibrated, o_test, n_boot=1000, seed=42)
    results['methods'][k]['paired_bootstrap_vs_uncalibrated'] = btr

# Winner
methods = ['global_platt', 'per_bucket_platt', 'isotonic', 'beta_calibration']
winner = min(methods, key=lambda k: results['methods'][k]['test_brier'])
results['winner'] = winner
results['winner_brier'] = results['methods'][winner]['test_brier']
results['winner_delta'] = results['methods'][winner]['test_brier'] - base_test_brier

print(f'\nWinner: {winner}')
print(f'  test Brier: {results["winner_brier"]:.5f} (Δ={results["winner_delta"]:+.5f})')
print(f'  paired bootstrap p: {results["methods"][winner]["paired_bootstrap_vs_uncalibrated"]["p_two_sided"]:.3f}')

Path('/Users/benny_es1/PythonTest/artifacts').mkdir(exist_ok=True)
json.dump(results, open('/Users/benny_es1/PythonTest/artifacts/calibration_report.json', 'w'),
          indent=2, default=float)
print('\nSaved → artifacts/calibration_report.json')

# Print summary table
print('\n' + '─' * 70)
print(f'{"Method":<22} {"Brier":>10} {"Δ vs base":>12} {"Platt b":>10}')
print('─' * 70)
print(f'{"(uncalibrated)":<22} {base_test_brier:>10.5f} {0:>+12.5f} {1.16:>10.4f}')
for k in methods:
    r = results['methods'][k]
    print(f'{k:<22} {r["test_brier"]:>10.5f} {r["delta"]:>+12.5f} {r["final_test_platt_b"]:>10.4f}')
