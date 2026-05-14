"""
Phase 1: Sample-size-aware uncertainty experiments.

Two principled approaches:
  A. Bayesian rating shrinkage by evidence:
       r_eff = (n / (n+k)) * r        (pulls to 0 = league mean when n small)
  B. Variance inflation:
       β_eff = β / sqrt(1 + c/n_min)  (predictions less confident at low n)

Both sweep their hyperparameter via walk-forward CV.
"""
import json, sys, os, math
import numpy as np
from scipy.special import expit
sys.path.insert(0, 'scrapers')
from BacktestSeriesPredictions import (
    load_snapshots, load_match_data,
    brier, logloss, expected_calibration_error,
    ORG_REGIONS,
)


def load_teams_with_eg():
    """Snapshot data with (rating, effective_games) tuples."""
    with open('data/map_ratings.json') as f: mr = json.load(f)
    out = []  # (ref_date, year, snap, {team: (rating, eg)})
    for y, yblock in mr['ratings'].items():
        for snap, sdata in yblock['snapshots'].items():
            rd = sdata.get('ref_date')
            if not rd: continue
            d = {t: (v.get('overall_rating', 0.0), v.get('effective_games', 0.0))
                 for t, v in sdata.get('teams', {}).items()}
            out.append((rd, y, snap, d))
    out.sort(key=lambda x: x[0])
    return out


def find_snapshot_for(date, snaps):
    best = None
    for rd, y, s, d in snaps:
        if rd < date:
            best = (rd, y, s, d)
        else:
            break
    return best


def predict_bo3(p_map):
    return p_map**2 * (3 - 2*p_map)


def predict_bo5(p_map):
    return p_map**3 + 3*p_map**3*(1-p_map) + 6*p_map**3*(1-p_map)**2


def gen_preds(matches, snaps_eg, beta, shrink_k=None, var_c=None, fmt='bo3', cross_mult=1.0):
    """
    shrink_k: if set, r_eff = (n / (n + shrink_k)) * r (Bayesian shrinkage to 0)
    var_c:    if set, β_eff = β / sqrt(1 + var_c / n_min)
    """
    out = []
    for _, m in matches.iterrows():
        snap = find_snapshot_for(m['date'], snaps_eg)
        if not snap: continue
        _, _, _, d = snap
        if m['a'] not in d or m['b'] not in d: continue
        rA, nA = d[m['a']]
        rB, nB = d[m['b']]
        if shrink_k is not None:
            rA_eff = (nA / (nA + shrink_k)) * rA
            rB_eff = (nB / (nB + shrink_k)) * rB
        else:
            rA_eff, rB_eff = rA, rB
        b = beta
        if var_c is not None:
            n_min = max(0.5, min(nA, nB))
            b = beta / math.sqrt(1 + var_c / n_min)
        regA = ORG_REGIONS.get(m['a'], '?'); regB = ORG_REGIONS.get(m['b'], '?')
        cross = (regA != regB and regA != '?' and regB != '?')
        if cross:
            b *= cross_mult
        p_map = 1 / (1 + math.exp(-b * (rA_eff - rB_eff)))
        p_map = max(1e-6, min(1-1e-6, p_map))
        p = predict_bo3(p_map)  # most series are BO3
        out.append({
            'date': m['date'], 'a': m['a'], 'b': m['b'],
            'p': p, 'y': m['a_wins'],
            'nA': nA, 'nB': nB, 'n_min': min(nA, nB), 'rA': rA, 'rB': rB,
            'cross': cross,
        })
    return out


def cv_brier(matches, snaps_eg, beta, shrink_k=None, var_c=None):
    """Walk-forward CV Brier (5 folds, 2023-2026)."""
    splits = [
        ('2024-01-01', '2024-07-01'),
        ('2024-07-01', '2025-01-01'),
        ('2025-01-01', '2025-07-01'),
        ('2025-07-01', '2026-01-01'),
        ('2026-01-01', '2026-06-01'),
    ]
    total_b, total_n = 0.0, 0
    fold_briers = []
    for lo, hi in splits:
        fold = matches[(matches['date'] >= lo) & (matches['date'] < hi)]
        if len(fold) < 30: continue
        preds = gen_preds(fold, snaps_eg, beta, shrink_k=shrink_k, var_c=var_c)
        if not preds: continue
        ps = [d['p'] for d in preds]; ys = [d['y'] for d in preds]
        fb = brier(ps, ys)
        fold_briers.append((lo, fb, len(preds)))
        total_b += fb * len(preds); total_n += len(preds)
    return total_b / max(total_n, 1), fold_briers


def main():
    print('Loading...')
    snaps_eg = load_teams_with_eg()
    matches = load_match_data()
    print(f'  {len(matches)} matches, {len(snaps_eg)} snapshots')

    # ─── Baseline at β=0.17, no shrinkage ───
    print('\n━━━ Baseline (β=0.17, no shrink, no var inflation) ━━━')
    base_b, base_folds = cv_brier(matches, snaps_eg, 0.17)
    print(f'  Weighted CV Brier: {base_b:.5f}')
    for lo, fb, n in base_folds:
        print(f'    fold {lo} → Brier {fb:.5f}  (n={n})')

    # ─── Approach A: Bayesian shrinkage by sample size ───
    print('\n━━━ A. Bayesian shrinkage  r_eff = (n/(n+k)) * r ━━━')
    print(f'  {"k":>5}  {"CV Brier":>9}  {"Δ vs base":>10}')
    best_a_k, best_a_b = None, base_b
    for k in [0, 0.5, 1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 50, 100]:
        b, _ = cv_brier(matches, snaps_eg, 0.17, shrink_k=k)
        delta = b - base_b
        print(f'  {k:>5.1f}  {b:>9.5f}  {delta:>+10.5f}')
        if b < best_a_b:
            best_a_b = b; best_a_k = k

    # ─── Approach B: Variance inflation by sample size ───
    print('\n━━━ B. Variance inflation  β_eff = β / sqrt(1 + c/n_min) ━━━')
    print(f'  {"c":>5}  {"CV Brier":>9}  {"Δ vs base":>10}')
    best_b_c, best_b_b = None, base_b
    for c in [0, 0.5, 1, 2, 3, 5, 7, 10, 15, 25, 50, 100]:
        b, _ = cv_brier(matches, snaps_eg, 0.17, var_c=c)
        delta = b - base_b
        print(f'  {c:>5.1f}  {b:>9.5f}  {delta:>+10.5f}')
        if b < best_b_b:
            best_b_b = b; best_b_c = c

    # ─── Joint A+B ───
    print('\n━━━ A+B Joint sweep ━━━')
    print(f'  {"k":>4}  {"c":>5}  {"CV Brier":>9}')
    best_ab = (None, None, base_b)
    for k in [0, 1, 2, 5, 10, 20]:
        for c in [0, 1, 5, 10, 25]:
            if k == 0 and c == 0: continue
            b, _ = cv_brier(matches, snaps_eg, 0.17, shrink_k=k, var_c=c)
            marker = '  ←' if b < best_ab[2] else ''
            print(f'  {k:>4}  {c:>5}  {b:>9.5f}{marker}')
            if b < best_ab[2]:
                best_ab = (k, c, b)

    # ─── Joint with β sweep ───
    print(f'\n━━━ Best k={best_a_k}, sweep β to re-tune ━━━')
    print(f'  {"β":>5}  {"CV Brier":>9}')
    for bb in [0.10, 0.12, 0.14, 0.15, 0.16, 0.17, 0.18, 0.20, 0.22, 0.25]:
        b, _ = cv_brier(matches, snaps_eg, bb, shrink_k=best_a_k)
        print(f'  {bb:>5.3f}  {b:>9.5f}')

    # ─── Holdout test: 2026 ───
    print(f'\n━━━ 2026 holdout: baseline vs best ━━━')
    test = matches[matches['date'] >= '2026-01-01']
    for label, kw in [
        ('baseline β=0.17',  {}),
        (f'A: k={best_a_k}',  {'shrink_k': best_a_k}),
        (f'B: c={best_b_c}',  {'var_c': best_b_c}),
        (f'A+B: k={best_ab[0]} c={best_ab[1]}', {'shrink_k': best_ab[0], 'var_c': best_ab[1]}),
    ]:
        preds = gen_preds(test, snaps_eg, 0.17, **kw)
        ps = np.array([d['p'] for d in preds]); ys = np.array([d['y'] for d in preds])
        errs = np.abs(ps - ys)
        print(f'  {label:<28}  Brier={brier(ps,ys):.5f}  LL={logloss(ps,ys):.5f}  '
              f'ECE={expected_calibration_error(ps,ys):.4f}  |err|σ={errs.std():.4f}')


if __name__ == '__main__':
    main()
