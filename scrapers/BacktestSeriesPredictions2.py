"""
Deep follow-up backtest:
  - Per-segment β optimization (same-region vs cross-region vs CN)
  - Cross-validation for cross_region_mult (current production = 0.664 is suspect)
  - Isotonic + Platt calibration
  - Brier-optimal temperature scaling
  - Variance/sensitivity analysis (|p-y| std, max error)
  - "Production current" vs "Proposed" head-to-head on the 2026 holdout
"""

import json, os, sys, math
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.optimize import minimize_scalar
from scipy.special import expit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# reuse helpers
sys.path.insert(0, os.path.join(ROOT, 'scrapers'))
from BacktestSeriesPredictions import (
    load_snapshots, load_match_data, find_snapshot_for, predict_series,
    gen_predictions, brier, logloss, expected_calibration_error,
    brier_decomposition, ORG_REGIONS
)


def main():
    print('Loading...')
    snaps = load_snapshots()
    matches = load_match_data()
    train = matches[matches['date'] < '2026-01-01'].copy()
    test  = matches[matches['date'] >= '2026-01-01'].copy()
    print(f'  Train: {len(train)}  Test 2026: {len(test)}')

    # ─── 1. Per-segment β optimization ───────────────────────────────────────
    print('\n━━━ 1. Per-segment β optimization on TRAIN ━━━')
    print('     Brier-optimal β per segment (and CV-fold-safe)')

    # Use a chronological 80/20 split *within* train for fold-CV
    train_sorted = train.sort_values('date').reset_index(drop=True)
    split = int(len(train_sorted) * 0.8)
    sub_train = train_sorted.iloc[:split]
    sub_val   = train_sorted.iloc[split:]
    print(f'  CV split: {len(sub_train)} train / {len(sub_val)} validate')

    def brier_at(beta, m, segment_fn=None, xmult=1.0):
        preds = gen_predictions(m, snaps, beta, cross_mult=xmult)
        if segment_fn is not None:
            preds = [d for d in preds if segment_fn(d)]
        if not preds:
            return float('nan'), 0
        ps = [d['p'] for d in preds]; ys = [d['y'] for d in preds]
        return brier(ps, ys), len(preds)

    segments = {
        'all':          lambda d: True,
        'same-region':  lambda d: not d['cross'],
        'cross-region': lambda d: d['cross'],
        'cn_involved':  lambda d: d['cn_involved'],
        'no_cn':        lambda d: not d['cn_involved'],
    }
    print(f'  {"segment":<18}  {"β*":>6}  {"train Brier":>12}  {"val Brier":>10}  {"n_val":>5}')
    for sname, fn in segments.items():
        best_b, best_v = None, 1.0
        for b in np.arange(0.05, 0.30, 0.01):
            bv, _ = brier_at(b, sub_val, fn)
            if not math.isnan(bv) and bv < best_v:
                best_v = bv; best_b = b
        bt, ntrain = brier_at(best_b, sub_train, fn)
        nv = sum(1 for d in gen_predictions(sub_val, snaps, best_b) if fn(d))
        print(f'  {sname:<18}  {best_b:>6.3f}  {bt:>12.5f}  {best_v:>10.5f}  {nv:>5}')

    # ─── 2. Cross-region mult sensitivity (rigorous) ────────────────────────
    print('\n━━━ 2. Cross-region mult sensitivity (production uses 0.664) ━━━')
    print('  Brier on cross-region subset only, varying xmult, β=0.17 base')
    print(f'  {"xmult":>7}  {"n_cross":>7}  {"x-Brier":>9}  {"all-Brier":>10}  {"x-ECE":>7}')
    for xm in [0.5, 0.6, 0.664, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5]:
        preds = gen_predictions(train, snaps, 0.17, cross_mult=xm)
        xpreds = [d for d in preds if d['cross']]
        ps = [d['p'] for d in xpreds]; ys = [d['y'] for d in xpreds]
        all_ps = [d['p'] for d in preds]; all_ys = [d['y'] for d in preds]
        xb = brier(ps, ys) if ps else float('nan')
        ab = brier(all_ps, all_ys)
        xe = expected_calibration_error(ps, ys) if ps else float('nan')
        print(f'  {xm:>7.3f}  {len(xpreds):>7}  {xb:.5f}    {ab:.5f}    {xe:.4f}')

    # ─── 3. Isotonic regression calibration ──────────────────────────────────
    print('\n━━━ 3. Isotonic regression calibration (β=0.17, xmult=1.0) ━━━')
    cal_preds = gen_predictions(sub_train, snaps, 0.17, cross_mult=1.0)
    cal_ps = np.array([d['p'] for d in cal_preds])
    cal_ys = np.array([d['y'] for d in cal_preds])
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(cal_ps, cal_ys)

    val_preds = gen_predictions(sub_val, snaps, 0.17, cross_mult=1.0)
    val_ps = np.array([d['p'] for d in val_preds])
    val_ys = np.array([d['y'] for d in val_preds])
    iso_val = iso.transform(val_ps)
    print(f'  Validation (held-out tail of train):')
    print(f'    raw      Brier={brier(val_ps, val_ys):.5f}  ECE={expected_calibration_error(val_ps, val_ys):.4f}')
    print(f'    isotonic Brier={brier(iso_val, val_ys):.5f}  ECE={expected_calibration_error(iso_val, val_ys):.4f}')

    # Test on 2026 holdout
    test_preds = gen_predictions(test, snaps, 0.17, cross_mult=1.0)
    test_ps = np.array([d['p'] for d in test_preds])
    test_ys = np.array([d['y'] for d in test_preds])
    iso_test = iso.transform(test_ps)
    print(f'  2026 holdout:')
    print(f'    raw      Brier={brier(test_ps, test_ys):.5f}  ECE={expected_calibration_error(test_ps, test_ys):.4f}')
    print(f'    isotonic Brier={brier(iso_test, test_ys):.5f}  ECE={expected_calibration_error(iso_test, test_ys):.4f}')

    # ─── 4. Platt scaling (logistic regression) ─────────────────────────────
    print('\n━━━ 4. Platt scaling (logistic on logits) ━━━')
    cal_logits = np.log(cal_ps / (1 - cal_ps)).reshape(-1, 1)
    val_logits = np.log(val_ps / (1 - val_ps)).reshape(-1, 1)
    test_logits = np.log(test_ps / (1 - test_ps)).reshape(-1, 1)
    lr = LogisticRegression().fit(cal_logits, cal_ys)
    print(f'  Platt slope = {lr.coef_[0,0]:.4f}, intercept = {lr.intercept_[0]:.4f}')
    print(f'    → multiplies logit by {lr.coef_[0,0]:.4f}')
    platt_val  = lr.predict_proba(val_logits)[:, 1]
    platt_test = lr.predict_proba(test_logits)[:, 1]
    print(f'  Validation: raw Brier={brier(val_ps, val_ys):.5f}  Platt={brier(platt_val, val_ys):.5f}')
    print(f'  2026 test:  raw Brier={brier(test_ps, test_ys):.5f}  Platt={brier(platt_test, test_ys):.5f}')

    # ─── 5. Brier-optimal temperature scaling ───────────────────────────────
    print('\n━━━ 5. Temperature scaling (Brier-optimal, not NLL) ━━━')
    def brier_T(T, logits, ys):
        z = expit(logits * T)
        return float(np.mean((z - ys) ** 2))
    res = minimize_scalar(lambda T: brier_T(T, cal_logits.ravel(), cal_ys), bounds=(0.1, 5.0), method='bounded')
    T = float(res.x)
    print(f'  Brier-optimal T = {T:.4f}')
    val_scaled  = expit(val_logits.ravel() * T)
    test_scaled = expit(test_logits.ravel() * T)
    print(f'  Validation: raw Brier={brier(val_ps, val_ys):.5f}  T-scaled={brier(val_scaled, val_ys):.5f}')
    print(f'  2026 test:  raw Brier={brier(test_ps, test_ys):.5f}  T-scaled={brier(test_scaled, test_ys):.5f}')

    # ─── 6. Sharpness vs calibration: variance shrinkage λ ──────────────────
    print('\n━━━ 6. Variance shrinkage — pull p toward 0.5 by factor λ ━━━')
    print('  q = 0.5 + λ * (p - 0.5)  [λ=1 → no shrinkage, λ<1 → toward 0.5]')
    print(f'  {"λ":>5}  {"val Brier":>10}  {"test Brier":>11}  {"val ECE":>9}  {"test ECE":>10}')
    for lam in [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.6, 0.5]:
        vq = 0.5 + lam * (val_ps - 0.5)
        tq = 0.5 + lam * (test_ps - 0.5)
        print(f'  {lam:>5.2f}  {brier(vq, val_ys):>10.5f}  {brier(tq, test_ys):>11.5f}  '
              f'{expected_calibration_error(vq, val_ys):>9.4f}  {expected_calibration_error(tq, test_ys):>10.4f}')

    # ─── 7. Variance/sensitivity analysis ───────────────────────────────────
    print('\n━━━ 7. Variance & sensitivity analysis on 2026 ━━━')
    preds = gen_predictions(test, snaps, 0.17, cross_mult=1.0)
    ps = np.array([d['p'] for d in preds]); ys = np.array([d['y'] for d in preds])
    errs = np.abs(ps - ys)
    print(f'  |p - y|: mean={errs.mean():.4f}  median={np.median(errs):.4f}  std={errs.std():.4f}  '
          f'p90={np.percentile(errs, 90):.4f}  max={errs.max():.4f}')

    # Sensitivity: perturb ratings by ±0.5, see how p changes
    deltas = []
    for d in preds:
        # numerical derivative of p w.r.t. (rA - rB) at current point
        h = 0.5
        pa = predict_series(d['ra'] + h, d['rb'] - h, 0.17, cross_region_mult=1.0, cross=d['cross'])
        pb = predict_series(d['ra'] - h, d['rb'] + h, 0.17, cross_region_mult=1.0, cross=d['cross'])
        deltas.append(abs(pa - pb))
    deltas = np.array(deltas)
    print(f'  ±0.5 rating perturbation → Δp:  mean={deltas.mean():.4f}  max={deltas.max():.4f}')
    print(f'   → "if I miss the rating by 0.5, my probability shifts by ~{deltas.mean():.2f} on average"')

    # ─── 8. Head-to-head: production vs proposed ────────────────────────────
    print('\n━━━ 8. Head-to-head: Production vs Proposed on 2026 holdout ━━━')
    # Production: β=0.17, cross_mult=0.664
    prod = gen_predictions(test, snaps, 0.17, cross_mult=0.664)
    # Proposed: β=0.17, cross_mult=1.0 (no cross-dampener; data says it hurts)
    prop = gen_predictions(test, snaps, 0.17, cross_mult=1.0)
    prod_ps = np.array([d['p'] for d in prod]); prod_ys = np.array([d['y'] for d in prod])
    prop_ps = np.array([d['p'] for d in prop]); prop_ys = np.array([d['y'] for d in prop])
    print(f'  Production (β=0.17, x={0.664}):  Brier={brier(prod_ps, prod_ys):.5f}  '
          f'ECE={expected_calibration_error(prod_ps, prod_ys):.4f}  LogLoss={logloss(prod_ps, prod_ys):.5f}')
    print(f'  Proposed   (β=0.17, x={1.0}):    Brier={brier(prop_ps, prop_ys):.5f}  '
          f'ECE={expected_calibration_error(prop_ps, prop_ys):.4f}  LogLoss={logloss(prop_ps, prop_ys):.5f}')
    delta = brier(prop_ps, prop_ys) - brier(prod_ps, prod_ys)
    print(f'  Δ Brier = {delta:+.5f}  ({"better" if delta < 0 else "worse"})')

    # Only cross-region subset
    pxp = [(d, e['p'], e['y']) for d, e in zip(prod, prop) if d['cross']]
    if pxp:
        prod_x_ps = np.array([d['p'] for d in prod if d['cross']])
        prod_x_ys = np.array([d['y'] for d in prod if d['cross']])
        prop_x_ps = np.array([d['p'] for d in prop if d['cross']])
        prop_x_ys = np.array([d['y'] for d in prop if d['cross']])
        print(f'  [Cross-region only n={len(prod_x_ps)}]')
        print(f'    Production: Brier={brier(prod_x_ps, prod_x_ys):.5f}')
        print(f'    Proposed:   Brier={brier(prop_x_ps, prop_x_ys):.5f}')


if __name__ == '__main__':
    main()
