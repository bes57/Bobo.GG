"""
Drill into the over-confidence finding:
  (1) Is over-confidence present at MAP level too, or only after the
      Bo3 amplification? Bo3 assumes map independence; if maps are
      correlated, the bo3 conversion over-amplifies.
  (2) What's actually in the broken [0.80, 1.01] series bin?
  (3) Map-level pred vs map-level outcomes — the cleanest signal.
"""
import os, sys, math, json
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scrapers'))

from BacktestSeriesPredictions import load_snapshots, find_snapshot_for


def predict_map(rA, rB, beta):
    return 1.0 / (1.0 + math.exp(-beta * (rA - rB)))


def load_map_outcomes():
    """Returns DataFrame with per-map outcomes: date, a (alphabetical), b, a_wins, MatchID, MapNum."""
    mr = pd.read_csv(os.path.join(ROOT, 'data/match_results.csv'))
    per_map = mr[mr['MapNum'] != 'all'].copy()
    per_map['MatchID'] = per_map['MatchID'].astype(int)

    with open(os.path.join(ROOT, 'data/match_dates.json')) as f:
        dates = json.load(f)

    # MatchID → (a, b) alphabetical
    teams_per = {}
    for fname in os.listdir(os.path.join(ROOT, 'data/maps')):
        if not fname.endswith('.csv'): continue
        try:
            mdf = pd.read_csv(os.path.join(ROOT, 'data/maps', fname), usecols=['MatchID', 'Org'])
        except Exception:
            continue
        for mid, grp in mdf.groupby('MatchID'):
            orgs = list(grp['Org'].dropna().unique())
            if len(orgs) == 2:
                a, b = sorted(orgs)
                teams_per[int(mid)] = (a, b)

    rows = []
    for _, r in per_map.iterrows():
        mid = int(r['MatchID'])
        d = dates.get(str(mid)) or dates.get(mid)
        if not d: continue
        pair = teams_per.get(mid)
        if not pair: continue
        a, b = pair
        winner = r['WinnerOrg']
        if winner not in (a, b): continue
        rows.append({
            'mid': mid, 'date': d, 'a': a, 'b': b,
            'a_wins': int(winner == a),
            'MapNum': r['MapNum'],
        })
    df = pd.DataFrame(rows)
    df.sort_values('date', inplace=True)
    return df.reset_index(drop=True)


def gen_map_preds(maps_df, snaps, beta):
    p_out, y_out = [], []
    for _, m in maps_df.iterrows():
        s = find_snapshot_for(m['date'], snaps)
        if not s: continue
        _, _, _, ratings = s
        a, b = m['a'], m['b']
        if a not in ratings or b not in ratings: continue
        rA = ratings[a].get('overall_rating', 0.0) if isinstance(ratings[a], dict) else ratings[a]
        rB = ratings[b].get('overall_rating', 0.0) if isinstance(ratings[b], dict) else ratings[b]
        p_out.append(predict_map(rA, rB, beta))
        y_out.append(m['a_wins'])
    return np.array(p_out), np.array(y_out)


def reliability(p, y, edges=None):
    if edges is None:
        edges = [0.0, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 1.01]
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi)
        n = int(m.sum())
        if n < 5: continue
        rows.append((lo, hi, n, float(p[m].mean()), float(y[m].mean())))
    return rows


def platt(p, y, iters=2000, lr=0.05):
    eps = 1e-6
    pc = np.clip(p, eps, 1-eps)
    z = np.log(pc/(1-pc))
    A, B = 1.0, 0.0
    for _ in range(iters):
        pred = 1/(1+np.exp(-(A*z + B)))
        err = pred - y
        A -= lr * (err * z).mean()
        B -= lr * err.mean()
    return float(A), float(B)


def brier(p, y): return float(np.mean((p-y)**2))
def logloss(p, y):
    p = np.clip(p, 1e-9, 1-1e-9); return float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p)))


def main():
    print("="*78)
    print("MAP-LEVEL CALIBRATION DIAGNOSTIC")
    print("="*78)

    snaps = load_snapshots()
    maps_df = load_map_outcomes()
    print(f"Loaded {len(maps_df)} per-map outcomes")

    beta = 0.25
    p, y = gen_map_preds(maps_df, snaps, beta)
    print(f"\nMap-level predictions: {len(p)}")
    print(f"Base rate (a_wins, alphabetical): {y.mean():.3f}")

    print(f"\n── Map-level sharpness & calibration (β={beta}) ──")
    print(f"  Sharpness         : {np.abs(p-0.5).mean():.4f}")
    print(f"  Std of pred       : {p.std():.4f}")
    print(f"  Brier             : {brier(p,y):.5f}")
    print(f"  Log-loss          : {logloss(p,y):.5f}")
    print(f"  % in [.45,.55]    : {((p>=0.45)&(p<=0.55)).mean()*100:.1f}%")
    print(f"  max p             : {p.max():.4f}")
    A, B = platt(p, y)
    print(f"  Platt slope A     : {A:.3f}    (1.0=calibrated, <1.0=over-confident, >1.0=under-confident)")
    print(f"  If A < 1: 'true' β ≈ {beta*A:.3f}")

    print(f"\n  Map reliability table:")
    print(f"  {'range':>12}  {'n':>5}  {'pred':>7}  {'obs':>7}  {'gap':>7}")
    for lo, hi, n, pm, om in reliability(p, y):
        sign = '↑' if (om-pm)>0.02 else ('↓' if (om-pm)<-0.02 else '·')
        print(f"  [{lo:.2f}, {hi:.2f}) {n:>5}  {pm:>7.4f}  {om:>7.4f}  {om-pm:>+7.4f}  {sign}")

    # 2025+ only
    recent = maps_df[maps_df['date'] >= '2025-01-01'].reset_index(drop=True)
    p2, y2 = gen_map_preds(recent, snaps, beta)
    print(f"\n── 2025+ map-level (n={len(p2)}) ──")
    A2, B2 = platt(p2, y2)
    print(f"  Sharpness         : {np.abs(p2-0.5).mean():.4f}")
    print(f"  Brier             : {brier(p2,y2):.5f}")
    print(f"  Platt slope A     : {A2:.3f}    → 'true' β ≈ {beta*A2:.3f}")
    print(f"  Map reliability:")
    for lo, hi, n, pm, om in reliability(p2, y2):
        sign = '↑' if (om-pm)>0.02 else ('↓' if (om-pm)<-0.02 else '·')
        print(f"  [{lo:.2f}, {hi:.2f}) {n:>5}  {pm:>7.4f}  {om:>7.4f}  {om-pm:>+7.4f}  {sign}")

    # Test multiple βs on map-level data to find the empirical best β
    print(f"\n── β sweep on map-level data (full history) ──")
    print(f"  {'β':>6}  {'Brier':>8}  {'LL':>8}  {'sharp':>7}  {'%[.45,.55]':>11}  {'maxP':>6}")
    for b in [0.07, 0.09, 0.10, 0.11, 0.12, 0.13, 0.15, 0.17, 0.20, 0.25, 0.30]:
        pp, yy = gen_map_preds(maps_df, snaps, b)
        print(f"  {b:>6.3f}  {brier(pp,yy):.5f}  {logloss(pp,yy):.5f}  "
              f"{np.abs(pp-0.5).mean():.4f}  {((pp>=0.45)&(pp<=0.55)).mean()*100:>9.1f}%  "
              f"{pp.max():.3f}")

    # Same on 2025+ only
    print(f"\n── β sweep on map-level data (2025+ only) ──")
    print(f"  {'β':>6}  {'Brier':>8}  {'LL':>8}  {'sharp':>7}  {'%[.45,.55]':>11}  {'maxP':>6}")
    for b in [0.07, 0.09, 0.10, 0.11, 0.12, 0.13, 0.15, 0.17, 0.20, 0.25, 0.30]:
        pp, yy = gen_map_preds(recent, snaps, b)
        print(f"  {b:>6.3f}  {brier(pp,yy):.5f}  {logloss(pp,yy):.5f}  "
              f"{np.abs(pp-0.5).mean():.4f}  {((pp>=0.45)&(pp<=0.55)).mean()*100:>9.1f}%  "
              f"{pp.max():.3f}")

    # Series level — what's actually in [0.80, 1.01]?
    print(f"\n── Drilling into series [0.80, 1.01] bin ──")
    from BacktestSeriesPredictions import load_match_data
    matches = load_match_data()
    psr, ysr = [], []
    rows_in_bin = []
    for _, m in matches.iterrows():
        s = find_snapshot_for(m['date'], snaps)
        if not s: continue
        _, _, _, ratings = s
        a, b = m['a'], m['b']
        if a not in ratings or b not in ratings: continue
        rA = ratings[a].get('overall_rating', 0.0) if isinstance(ratings[a], dict) else ratings[a]
        rB = ratings[b].get('overall_rating', 0.0) if isinstance(ratings[b], dict) else ratings[b]
        pm = predict_map(rA, rB, beta)
        pser = pm**2 * (3 - 2*pm)
        psr.append(pser); ysr.append(m['a_wins'])
        if pser >= 0.80:
            rows_in_bin.append((m['date'], a, b, rA, rB, pser, m['a_wins']))
    psr = np.array(psr); ysr = np.array(ysr)
    print(f"  Series in [0.80, 1.01]: n={len(rows_in_bin)}, observed win rate: {sum(r[6] for r in rows_in_bin)/len(rows_in_bin):.3f}")
    print(f"  Showing some samples (date, a, b, rA, rB, pred, a_won):")
    rows_in_bin.sort(key=lambda r: -r[5])
    for r in rows_in_bin[:15]:
        print(f"    {r[0]}  {r[1]:>6} vs {r[2]:<6}  rA={r[3]:+.2f}  rB={r[4]:+.2f}  pred={r[5]:.3f}  a_won={r[6]}")
    print(f"  Misses in that bin (a_wins=0 when pred high):")
    misses = [r for r in rows_in_bin if r[6] == 0]
    for r in sorted(misses, key=lambda r:-r[5])[:10]:
        print(f"    {r[0]}  {r[1]:>6} vs {r[2]:<6}  rA={r[3]:+.2f}  rB={r[4]:+.2f}  pred={r[5]:.3f}  ← LOST")


if __name__ == '__main__':
    main()
