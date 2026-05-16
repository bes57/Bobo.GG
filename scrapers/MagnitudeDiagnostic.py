"""
Diagnostic: is the BenPom model under-confident at the tails?

Test methodology:
  1. Walk-forward predict every historical series in match_results.csv
     using the snapshot in map_ratings.json that pre-dates the match.
  2. Build a reliability curve: bin predictions by probability, compute
     observed win rate in each bin.
  3. Fit isotonic + a logistic recalibration (Platt scaling) → measure
     the "miscalibration slope". Slope > 1.0 means the model is
     systematically too tame (probabilities pulled toward 0.5).
  4. Report: sharpness (mean |p-0.5|), calibration slope, Brier
     decomposition (rel / res / unc), and what β would look like AFTER
     Platt correction.

If slope >> 1.0, the user's intuition is mathematically right: the
model is under-confident, and the cure is either smaller β (sharper
predictions per unit rating diff) or less-compressive RD_TRANSFORM
(more rating spread per unit round-diff).
"""
import os, sys, math, json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scrapers'))

from BacktestSeriesPredictions import (
    load_match_data, load_snapshots, find_snapshot_for,
    brier, logloss, expected_calibration_error, brier_decomposition,
)


def predict_map(rA, rB, beta):
    return 1.0 / (1.0 + math.exp(-beta * (rA - rB)))


def predict_series(rA, rB, beta, fmt='bo3'):
    p = predict_map(rA, rB, beta)
    if fmt == 'bo3':
        return p**2 * (3 - 2*p)   # P(win Bo3) = P(2-0) + P(2-1)
    if fmt == 'bo5':
        return p**3 * (10 - 15*p + 6*p*p)
    return p


def gen(matches, snaps, beta, fmt='bo3'):
    out_p, out_y, out_diff = [], [], []
    for _, m in matches.iterrows():
        s = find_snapshot_for(m['date'], snaps)
        if not s: continue
        _, _, _, ratings = s
        a, b = m['a'], m['b']
        if a not in ratings or b not in ratings: continue
        rA = ratings[a].get('overall_rating') if isinstance(ratings[a], dict) else ratings[a]
        rB = ratings[b].get('overall_rating') if isinstance(ratings[b], dict) else ratings[b]
        p = predict_series(rA, rB, beta, fmt)
        out_p.append(p); out_y.append(m['a_wins']); out_diff.append(rA - rB)
    return np.array(out_p), np.array(out_y), np.array(out_diff)


def reliability_table(p, y, edges=None):
    if edges is None:
        edges = [0.0, 0.20, 0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80, 1.01]
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi)
        n = mask.sum()
        if n < 5: continue
        rows.append((lo, hi, n, p[mask].mean(), y[mask].mean()))
    return rows


def platt_fit(p, y):
    """Fit y = sigmoid(A * logit(p) + B). Return (A, B).
    A > 1 means model is under-confident; multiply rating diff by A to fix.
    """
    eps = 1e-6
    pc = np.clip(p, eps, 1-eps)
    z = np.log(pc / (1 - pc))   # logit of model probability
    # gradient descent: minimize log-loss of sigmoid(A*z + B)
    A, B = 1.0, 0.0
    lr = 0.05
    for _ in range(2000):
        pred = 1 / (1 + np.exp(-(A*z + B)))
        err = pred - y
        gA = (err * z).mean()
        gB = err.mean()
        A -= lr * gA
        B -= lr * gB
    return A, B


def isotonic_fit(p, y, n_bins=20):
    """Pool-adjacent-violators isotonic regression."""
    order = np.argsort(p)
    yo = y[order].astype(float)
    n = len(yo)
    # Bin first, then PAV on bin means
    edges = np.linspace(0, 1, n_bins+1)
    bin_idx = np.digitize(p, edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins-1)
    means = []
    counts = []
    centers = []
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() == 0: continue
        means.append(y[mask].mean())
        counts.append(mask.sum())
        centers.append(p[mask].mean())
    means = np.array(means); counts = np.array(counts, dtype=float); centers = np.array(centers)
    # PAV
    while True:
        viol = np.where(np.diff(means) < 0)[0]
        if len(viol) == 0: break
        i = viol[0]
        total = counts[i] + counts[i+1]
        merged = (means[i]*counts[i] + means[i+1]*counts[i+1]) / total
        means[i] = merged; counts[i] = total
        means = np.delete(means, i+1); counts = np.delete(counts, i+1)
        centers[i] = (centers[i] + centers[i+1]) / 2  # rough
        centers = np.delete(centers, i+1)
    return centers, means, counts


def main():
    print("="*78)
    print("BENPOM MAGNITUDE/CALIBRATION DIAGNOSTIC")
    print("="*78)

    matches = load_match_data()
    print(f"\nLoaded {len(matches)} historical series ({matches['date'].min()} → {matches['date'].max()})")

    snaps = load_snapshots()
    print(f"Loaded {len(snaps)} snapshots")

    beta = 0.25  # current production β
    p, y, diff = gen(matches, snaps, beta, fmt='bo3')
    print(f"\nUsable predictions: {len(p)}")
    print(f"Outcome base rate: {y.mean():.3f}  (alphabetical 'a_wins' — should be ~0.5)")

    print(f"\n── Sharpness & Calibration (β={beta}) ──")
    sharpness = np.abs(p - 0.5).mean()
    print(f"  Mean |p - 0.5|        : {sharpness:.4f}    (higher = more confident)")
    print(f"  Std of predictions    : {p.std():.4f}")
    print(f"  % predictions in 45-55: {((p>=0.45)&(p<=0.55)).mean()*100:.1f}%")
    print(f"  % predictions in 40-60: {((p>=0.40)&(p<=0.60)).mean()*100:.1f}%")
    print(f"  % predictions outside 30-70: {((p<0.30)|(p>0.70)).mean()*100:.1f}%")
    print(f"  max prediction        : {p.max():.4f}")
    print(f"  min prediction        : {p.min():.4f}")

    print(f"\n  Brier                 : {brier(p, y):.5f}")
    print(f"  Log-loss              : {logloss(p, y):.5f}")
    print(f"  ECE (10 bins)         : {expected_calibration_error(p, y):.5f}")
    rel, res, unc = brier_decomposition(p, y)
    print(f"  Brier decomp          : reliability={rel:.5f}  resolution={res:.5f}  uncertainty={unc:.5f}")
    print(f"     (reliability = miscalibration cost — lower is better)")
    print(f"     (resolution = how much pred varies with outcome — higher is better)")
    print(f"     (Brier = rel - res + unc)")

    print("\n── Reliability table (predicted probability vs observed win rate) ──")
    print(f"  {'range':>12}  {'n':>5}  {'pred mean':>10}  {'obs mean':>10}  {'gap':>7}")
    table = reliability_table(p, y)
    for lo, hi, n, pm, om in table:
        gap = om - pm
        sign = '↑' if gap > 0.02 else ('↓' if gap < -0.02 else '·')
        print(f"  [{lo:.2f}, {hi:.2f}) {n:>5}  {pm:>10.4f}  {om:>10.4f}  {gap:>+7.4f}  {sign}")
    print("\n  ↑ in low bins  + ↓ in high bins  →  model UNDER-CONFIDENT (predictions too tame)")
    print("  Reverse pattern                   →  model OVER-CONFIDENT")

    print("\n── Platt scaling (logistic recalibration) ──")
    A, B = platt_fit(p, y)
    print(f"  Recalibrated pred = sigmoid({A:.3f} * logit(p) + {B:+.3f})")
    print(f"  A = {A:.3f}  →  ", end='')
    if A > 1.05:
        print(f"model is UNDER-CONFIDENT by factor {A:.2f}.")
        print(f"            Fix: multiply rating-diff signal by ~{A:.2f}x")
        print(f"            (equivalent: raise β from {beta} → {beta*A:.3f})")
    elif A < 0.95:
        print(f"model is OVER-CONFIDENT by factor {1/A:.2f}.")
        print(f"            Fix: shrink rating-diff signal by ~{1/A:.2f}x")
        print(f"            (equivalent: lower β from {beta} → {beta*A:.3f})")
    else:
        print("model is roughly calibrated.")

    # What does the recalibrated model score?
    pc = np.clip(p, 1e-6, 1-1e-6)
    z = np.log(pc / (1-pc))
    p_recal = 1/(1+np.exp(-(A*z + B)))
    print(f"\n  After Platt recalibration:")
    print(f"     Brier            : {brier(p_recal, y):.5f}  (Δ {brier(p_recal,y)-brier(p,y):+.5f})")
    print(f"     Log-loss         : {logloss(p_recal, y):.5f}  (Δ {logloss(p_recal,y)-logloss(p,y):+.5f})")
    print(f"     Sharpness        : {np.abs(p_recal - 0.5).mean():.4f}  (vs {sharpness:.4f})")
    print(f"     % in [.45,.55]   : {((p_recal>=0.45)&(p_recal<=0.55)).mean()*100:.1f}%  (vs {((p>=0.45)&(p<=0.55)).mean()*100:.1f}%)")
    print(f"     max prediction   : {p_recal.max():.4f}  (vs {p.max():.4f})")

    # Stratify by era — recent ratings are more compressed (we showed sd dropped from 1.9 → 0.82)
    print("\n── Calibration by era ──")
    print(f"  {'era':>20}  {'n':>5}  {'sharp':>7}  {'Brier':>7}  {'A (Platt)':>10}")
    eras = [
        ('2023', matches['date'] < '2024-01-01'),
        ('2024', (matches['date'] >= '2024-01-01') & (matches['date'] < '2025-01-01')),
        ('2025', (matches['date'] >= '2025-01-01') & (matches['date'] < '2026-01-01')),
        ('2026', matches['date'] >= '2026-01-01'),
    ]
    for name, mask in eras:
        mm = matches[mask].reset_index(drop=True)
        pp, yy, _ = gen(mm, snaps, beta, fmt='bo3')
        if len(pp) < 30: continue
        sh = np.abs(pp - 0.5).mean()
        try:
            aa, bb = platt_fit(pp, yy)
        except Exception:
            aa = float('nan')
        print(f"  {name:>20}  {len(pp):>5}  {sh:.4f}  {brier(pp,yy):.5f}  {aa:>10.3f}")

    # Recent only — 2025 + 2026, most predictive of present model behavior
    recent_mask = matches['date'] >= '2025-01-01'
    pm, ym, _ = gen(matches[recent_mask].reset_index(drop=True), snaps, beta, fmt='bo3')
    print(f"\n── 2025-2026 only (n={len(pm)}, most representative of current model) ──")
    A2, B2 = platt_fit(pm, ym)
    print(f"  Platt slope A = {A2:.3f}  →  implied 'true' β ≈ {beta*A2:.3f}")
    print(f"  Sharpness         : {np.abs(pm-0.5).mean():.4f}")
    print(f"  % in [.45,.55]    : {((pm>=0.45)&(pm<=0.55)).mean()*100:.1f}%")
    print(f"  Reliability:")
    for lo, hi, n, ppm, oom in reliability_table(pm, ym):
        print(f"    [{lo:.2f}, {hi:.2f})  n={n:>4}  pred={ppm:.3f}  obs={oom:.3f}  gap={oom-ppm:+.3f}")


if __name__ == '__main__':
    main()
