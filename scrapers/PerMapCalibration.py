"""
Per-map prediction calibration. Fixes the load_snapshots() bug from the
final sweep — load the FULL team dict including maps, not just overall.
Sweep SHRINK_K to find optimum for per-map prediction Brier.
"""
import os, sys, math, json, importlib, contextlib, io
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scrapers'))

from BacktestSeriesPredictions import brier, expected_calibration_error


def load_map_outcomes():
    """Per-map outcomes: date, a (alphabetical), b, a_wins, MapName."""
    mr = pd.read_csv(os.path.join(ROOT, 'data/match_results.csv'))
    per_map = mr[mr['MapNum'] != 'all'].copy()
    per_map['MatchID'] = per_map['MatchID'].astype(int)
    with open(os.path.join(ROOT, 'data/match_dates.json')) as f:
        dates = json.load(f)
    teams_per = {}
    map_names = {}
    for fname in os.listdir(os.path.join(ROOT, 'data/maps')):
        if not fname.endswith('.csv'): continue
        try:
            mdf = pd.read_csv(os.path.join(ROOT, 'data/maps', fname),
                              usecols=['MatchID', 'Org', 'MapNum', 'MapName'])
        except Exception:
            continue
        for (mid, mn), grp in mdf.groupby(['MatchID', 'MapNum']):
            orgs = list(grp['Org'].dropna().unique())
            if len(orgs) == 2:
                a, b = sorted(orgs)
                teams_per[int(mid)] = (a, b)
                mns = grp['MapName'].dropna()
                if len(mns): map_names[(int(mid), str(mn))] = mns.iloc[0]
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
        mn = map_names.get((mid, str(r['MapNum'])))
        if not mn: continue
        rows.append({'mid': mid, 'date': d, 'a': a, 'b': b,
                     'a_wins': int(winner == a), 'MapName': mn})
    df = pd.DataFrame(rows)
    df.sort_values('date', inplace=True)
    return df.reset_index(drop=True)

DEPLOYED = dict(
    RD_TRANSFORM='power', RD_POWER=0.5, RD_SCALE=2.0,
    HALF_LIFE_WEEKS=6.0, ROSTER_PERSISTENCE=0.3,
    INTL_WIN_MULT=1.0, INTL_LOSS_MULT=1.0, CHAMPIONS_MULT=2.0,
    CN_PRIOR=-3.0, CN_INTL_K=4.0, CN_C_MIN=0.5,
    REGION_SPILLOVER_ALPHA=0.5,
    SHRINK_K=12,
)
BETA = 0.17


def predict(rA, rB, beta=BETA):
    return 1.0 / (1.0 + math.exp(-beta * (rA - rB)))


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


def load_full_snapshots():
    """Like load_snapshots but preserves full team dicts (with maps subkey)."""
    with open(os.path.join(ROOT, 'data/map_ratings.json')) as f:
        mr = json.load(f)
    out = []
    for y, yblock in mr['ratings'].items():
        for snap, sdata in yblock['snapshots'].items():
            rd = sdata.get('ref_date')
            if not rd: continue
            teams = sdata.get('teams', {})  # full dict including maps
            out.append((rd, y, snap, teams))
    out.sort(key=lambda x: x[0])
    return out


def find_snap_for(date, snaps):
    best = None
    for rd, y, sn, t in snaps:
        if rd < date: best = (rd, y, sn, t)
        else: break
    return best


def gen_map_preds(maps_df, snaps, beta=BETA, use_per_map=True):
    """Predict each map outcome.
       use_per_map=True → use per-map rating for that map, fall back to overall
       use_per_map=False → use overall rating only (ignore per-map info)
    """
    p, y = [], []
    for _, m in maps_df.iterrows():
        s = find_snap_for(m['date'], snaps)
        if not s: continue
        _, _, _, teams = s
        a, b = m['a'], m['b']
        if a not in teams or b not in teams: continue
        ta, tb = teams[a], teams[b]
        rA_overall = ta.get('overall_rating', 0.0)
        rB_overall = tb.get('overall_rating', 0.0)
        if use_per_map:
            ma = ta.get('maps', {}).get(m['MapName'], {})
            mb = tb.get('maps', {}).get(m['MapName'], {})
            rA = ma.get('rating', rA_overall) if isinstance(ma, dict) else rA_overall
            rB = mb.get('rating', rB_overall) if isinstance(mb, dict) else rB_overall
        else:
            rA, rB = rA_overall, rB_overall
        p.append(predict(rA, rB, beta)); y.append(m['a_wins'])
    return np.array(p), np.array(y)


def rebuild(**overrides):
    saved_argv = sys.argv[:]
    sys.argv = ['BuildMapRatings.py']
    try:
        import BuildMapRatings
        importlib.reload(BuildMapRatings)
        cfg = dict(DEPLOYED, **overrides)
        for k, v in cfg.items():
            setattr(BuildMapRatings, k, v)
        BuildMapRatings._apply_cn_shrinkage.__defaults__ = (
            cfg['CN_PRIOR'], cfg['CN_INTL_K'], cfg['CN_C_MIN']
        )
        with contextlib.redirect_stdout(io.StringIO()):
            BuildMapRatings.main()
    finally:
        sys.argv = saved_argv


def measure_map(maps_df, snaps, beta=BETA):
    p_pm, y_pm = gen_map_preds(maps_df, snaps, beta, use_per_map=True)
    p_ov, y_ov = gen_map_preds(maps_df, snaps, beta, use_per_map=False)
    # Hold-out: 2025+ only (most relevant)
    rec = maps_df[maps_df['date'] >= '2025-01-01'].reset_index(drop=True)
    p_pm_r, y_pm_r = gen_map_preds(rec, snaps, beta, use_per_map=True)
    p_ov_r, y_ov_r = gen_map_preds(rec, snaps, beta, use_per_map=False)
    A_pm, _ = platt(p_pm, y_pm)
    # Sample team's per-map spread
    teams26 = next((t for rd, y, sn, t in snaps
                    if y == '2026' and sn == 'after_santiago'), {})
    ns_maps = teams26.get('NS', {}).get('maps', {})
    ns_map_ratings = [v['rating'] for v in ns_maps.values() if 'rating' in v]
    ns_spread = (max(ns_map_ratings) - min(ns_map_ratings)) if ns_map_ratings else 0
    g2_maps = teams26.get('G2', {}).get('maps', {})
    g2_ratings = [v['rating'] for v in g2_maps.values() if 'rating' in v]
    g2_spread = (max(g2_ratings) - min(g2_ratings)) if g2_ratings else 0

    return {
        'pm_brier':    float(brier(p_pm, y_pm)),
        'ov_brier':    float(brier(p_ov, y_ov)),
        'pm_brier_r':  float(brier(p_pm_r, y_pm_r)),
        'ov_brier_r':  float(brier(p_ov_r, y_ov_r)),
        'pm_ece':      float(expected_calibration_error(p_pm, y_pm)),
        'pm_platt_A':  A_pm,
        'pm_sharp':    float(np.abs(p_pm-0.5).mean()),
        'ns_spread':   ns_spread,
        'g2_spread':   g2_spread,
        'lift_bp':     (float(brier(p_ov, y_ov)) - float(brier(p_pm, y_pm))) * 10000,
    }


def main():
    maps_df = load_map_outcomes()
    print(f"Loaded {len(maps_df)} per-map outcomes\n")

    # Deployed
    print("━━━ DEPLOYED BASELINE (SHRINK_K=12) ━━━")
    rebuild()
    snaps = load_full_snapshots()
    base = measure_map(maps_df, snaps)
    print(f"  per-map Brier (full):  {base['pm_brier']:.5f}")
    print(f"  per-map Brier (2025+): {base['pm_brier_r']:.5f}")
    print(f"  overall-only Brier:    {base['ov_brier']:.5f}  (lift from per-map: {base['lift_bp']:+.1f}bp)")
    print(f"  per-map ECE:           {base['pm_ece']:.4f}")
    print(f"  per-map Platt A:       {base['pm_platt_A']:.3f}")
    print(f"  per-map sharpness:     {base['pm_sharp']:.4f}")
    print(f"  NS map-spread:         {base['ns_spread']:.2f}")
    print(f"  G2 map-spread:         {base['g2_spread']:.2f}")
    print()

    # SHRINK_K sweep
    print("━━━ SHRINK_K SWEEP (per-map calibration) ━━━")
    print(f"  {'SK':>3}  {'pm_Br':>7}  {'pm_Br_r':>7}  {'lift':>5}  {'pm_A':>5}  {'pm_ECE':>6}  {'pm_sh':>5}  {'NS_sp':>5}  {'G2_sp':>5}")
    results = []
    for sk in [3, 5, 8, 12, 16, 20, 28, 40, 60]:
        rebuild(SHRINK_K=sk)
        snaps = load_full_snapshots()
        m = measure_map(maps_df, snaps)
        results.append((sk, m))
        print(f"  {sk:>3}  {m['pm_brier']:.5f}  {m['pm_brier_r']:.5f}  "
              f"{m['lift_bp']:>+5.1f}  {m['pm_platt_A']:>5.3f}  {m['pm_ece']:.4f}  "
              f"{m['pm_sharp']:.3f}  {m['ns_spread']:>5.2f}  {m['g2_spread']:>5.2f}", flush=True)

    # Best per-map Brier
    best_full = min(results, key=lambda x: x[1]['pm_brier'])
    best_rec  = min(results, key=lambda x: x[1]['pm_brier_r'])
    print(f"\n  Best SHRINK_K by full Brier:  {best_full[0]}  ({best_full[1]['pm_brier']:.5f})")
    print(f"  Best SHRINK_K by 2025+ Brier: {best_rec[0]}  ({best_rec[1]['pm_brier_r']:.5f})")
    print(f"  Deployed (12):                {base['pm_brier']:.5f} / {base['pm_brier_r']:.5f}")

    print(f"\n━━━ Restoring deployed ━━━")
    rebuild()
    print("Done.")


if __name__ == '__main__':
    main()
