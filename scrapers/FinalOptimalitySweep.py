"""
Final optimality investigation. Three parts:

PART 1 — Per-map predictions
  Build per-map Brier using per-map ratings (not overall). Sweep SHRINK_K.
  Find optimal per-map shrinkage at the new scale.

PART 2 — Interactions
  2D grids:
    (a) RD_SCALE × CN_PRIOR    — did the CN_PRIOR move free up RD_SCALE headroom?
    (b) RD_SCALE × ridge       — ridge controls overall spread, scale controls per-game
  Spot any interactions that change recommendations.

PART 3 — Integrated push
  Combine the best-known moves and ask: can we push magnitude higher with
  Brier still ≤ deployed?
"""
import os, sys, math, json, importlib, contextlib, io
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scrapers'))

from BacktestSeriesPredictions import (
    load_match_data, load_snapshots, find_snapshot_for, brier, logloss,
    expected_calibration_error,
)

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


def predict_series(rA, rB, beta=BETA, fmt='bo3'):
    p = predict(rA, rB, beta)
    return p**2 * (3 - 2*p)


def gen_series(matches, snaps, beta=BETA):
    p, y = [], []
    for _, m in matches.iterrows():
        s = find_snapshot_for(m['date'], snaps)
        if not s: continue
        _, _, _, ratings = s
        a, b = m['a'], m['b']
        if a not in ratings or b not in ratings: continue
        rA = ratings[a].get('overall_rating', 0.0) if isinstance(ratings[a], dict) else ratings[a]
        rB = ratings[b].get('overall_rating', 0.0) if isinstance(ratings[b], dict) else ratings[b]
        p.append(predict_series(rA, rB, beta)); y.append(m['a_wins'])
    return np.array(p), np.array(y)


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


def rebuild(**overrides):
    saved_argv = sys.argv[:]
    sys.argv = ['BuildMapRatings.py']
    try:
        import BuildMapRatings
        importlib.reload(BuildMapRatings)
        cfg = dict(DEPLOYED, **overrides)
        for k, v in cfg.items():
            setattr(BuildMapRatings, k, v)
        # Patch CN shrinkage defaults
        BuildMapRatings._apply_cn_shrinkage.__defaults__ = (
            cfg['CN_PRIOR'], cfg['CN_INTL_K'], cfg['CN_C_MIN']
        )
        # Note: ridge is hardcoded inside massey_ratings. To override we need
        # to monkeypatch the function. Skip for now — ridge sweep handled in Part 2.
        with contextlib.redirect_stdout(io.StringIO()):
            BuildMapRatings.main()
    finally:
        sys.argv = saved_argv


def load_map_outcomes():
    """Per-map outcomes: date, a (alphabetical), b, a_wins, MapName"""
    mr = pd.read_csv(os.path.join(ROOT, 'data/match_results.csv'))
    per_map = mr[mr['MapNum'] != 'all'].copy()
    per_map['MatchID'] = per_map['MatchID'].astype(int)
    with open(os.path.join(ROOT, 'data/match_dates.json')) as f:
        dates = json.load(f)
    teams_per = {}
    map_names = {}  # (MatchID, MapNum) -> map_name
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
                map_names[(int(mid), str(mn))] = grp['MapName'].dropna().iloc[0] if len(grp['MapName'].dropna()) else None
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
        if not mn or pd.isna(mn): continue
        rows.append({'mid': mid, 'date': d, 'a': a, 'b': b,
                     'a_wins': int(winner == a), 'MapName': mn})
    df = pd.DataFrame(rows)
    df.sort_values('date', inplace=True)
    return df.reset_index(drop=True)


def gen_map_preds_per_map(maps_df, snaps, beta=BETA):
    """Predict each map outcome using per-map ratings (not overall)."""
    p, y = [], []
    for _, m in maps_df.iterrows():
        s = find_snapshot_for(m['date'], snaps)
        if not s: continue
        _, _, _, ratings = s
        a, b = m['a'], m['b']
        if a not in ratings or b not in ratings: continue
        ta = ratings[a]; tb = ratings[b]
        if not isinstance(ta, dict): continue
        # per-map rating if available, else overall
        rA = (ta.get('maps', {}).get(m['MapName'], {}).get('rating')
              if isinstance(ta.get('maps'), dict) else None)
        if rA is None: rA = ta.get('overall_rating', 0.0)
        rB = (tb.get('maps', {}).get(m['MapName'], {}).get('rating')
              if isinstance(tb.get('maps'), dict) else None)
        if rB is None: rB = tb.get('overall_rating', 0.0)
        p.append(predict(rA, rB, beta)); y.append(m['a_wins'])
    return np.array(p), np.array(y)


def gen_map_preds_overall(maps_df, snaps, beta=BETA):
    """Predict each map outcome using OVERALL ratings (ignore per-map)."""
    p, y = [], []
    for _, m in maps_df.iterrows():
        s = find_snapshot_for(m['date'], snaps)
        if not s: continue
        _, _, _, ratings = s
        a, b = m['a'], m['b']
        if a not in ratings or b not in ratings: continue
        rA = ratings[a].get('overall_rating', 0.0) if isinstance(ratings[a], dict) else ratings[a]
        rB = ratings[b].get('overall_rating', 0.0) if isinstance(ratings[b], dict) else ratings[b]
        p.append(predict(rA, rB, beta)); y.append(m['a_wins'])
    return np.array(p), np.array(y)


def measure(matches, snaps, beta=BETA, also_maps_df=None):
    """Return all the metrics we care about for a given snapshot universe."""
    with open(os.path.join(ROOT, 'data/map_ratings.json')) as f:
        mr = json.load(f)
    teams26 = mr['ratings']['2026']['snapshots']['after_santiago']['teams']
    top26 = max(t['overall_rating'] for t in teams26.values())
    bot26 = min(t['overall_rating'] for t in teams26.values())

    # Per-map spread for top team — measure intra-team per-map variance
    NS_maps = teams26.get('NS', {}).get('maps', {})
    ns_map_spread = (max(m['rating'] for m in NS_maps.values()) -
                     min(m['rating'] for m in NS_maps.values())) if NS_maps else 0

    # Series-level
    p, y = gen_series(matches, snaps, beta)
    matches_te = matches[matches['date'] >= '2026-01-01'].reset_index(drop=True)
    p_te, y_te = gen_series(matches_te, snaps, beta)
    A, _ = platt(p, y)

    out = {
        'top26': top26, 'bot26': bot26, 'ns_map_spread': ns_map_spread,
        'series_brier': float(brier(p, y)),
        'series_brier_te': float(brier(p_te, y_te)) if len(p_te) >= 30 else float('nan'),
        'series_ece': float(expected_calibration_error(p, y)),
        'series_platt_A': A,
        'series_sharp': float(np.abs(p-0.5).mean()),
    }

    # Per-map level (if requested)
    if also_maps_df is not None:
        # Using per-map ratings
        pm_p, pm_y = gen_map_preds_per_map(also_maps_df, snaps, beta)
        pm_p_te, pm_y_te = gen_map_preds_per_map(
            also_maps_df[also_maps_df['date'] >= '2026-01-01'].reset_index(drop=True),
            snaps, beta
        )
        out['map_brier'] = float(brier(pm_p, pm_y))
        out['map_brier_te'] = float(brier(pm_p_te, pm_y_te)) if len(pm_p_te) >= 30 else float('nan')
        out['map_ece'] = float(expected_calibration_error(pm_p, pm_y))
        out['map_platt_A'], _ = platt(pm_p, pm_y)
        out['map_sharp'] = float(np.abs(pm_p-0.5).mean())
        # Using overall ratings (baseline — what if we IGNORED per-map?)
        ov_p, ov_y = gen_map_preds_overall(also_maps_df, snaps, beta)
        out['map_brier_overallonly'] = float(brier(ov_p, ov_y))
        # Lift from per-map info
        out['map_lift_bp'] = (out['map_brier_overallonly'] - out['map_brier']) * 10000
    return out


def fmt_series(label, m):
    return (f"  {label:<35}  top26={m['top26']:>+5.2f}  bot26={m['bot26']:>+5.2f}  "
            f"sBrier={m['series_brier']:.5f}  sBr_te={m['series_brier_te']:.5f}  "
            f"sA={m['series_platt_A']:>4.2f}  sECE={m['series_ece']:.4f}")


def fmt_both(label, m):
    return (f"  {label:<24}  top26={m['top26']:>+5.2f}  NSmap_sp={m['ns_map_spread']:>4.2f}  "
            f"sBrier={m['series_brier']:.5f}  sA={m['series_platt_A']:>4.2f}  "
            f"mBrier={m['map_brier']:.5f}  mA={m['map_platt_A']:>4.2f}  "
            f"map_lift={m['map_lift_bp']:>+5.1f}bp")


def main():
    matches = load_match_data()
    maps_df = load_map_outcomes()
    print(f"Loaded {len(matches)} series, {len(maps_df)} per-map outcomes\n")

    # ======================= BASELINE =======================
    print("━━━ DEPLOYED BASELINE ━━━")
    snaps = None
    rebuild()
    snaps = load_snapshots()
    base = measure(matches, snaps, BETA, also_maps_df=maps_df)
    print(fmt_both('(deployed)', base))
    print(f"  Detail: map ECE={base['map_ece']:.4f}  map sharp={base['map_sharp']:.3f}  "
          f"map_brier_overallonly={base['map_brier_overallonly']:.5f}")
    print()

    # ======================= PART 1: SHRINK_K SWEEP =======================
    # SHRINK_K only affects per-map ratings (overall_rating untouched).
    # Lower SHRINK_K = trust per-map data more (more magnitude, more variance).
    print("━━━ PART 1: SHRINK_K (per-map shrinkage) ━━━")
    print(f"  Goal: find SHRINK_K that minimizes per-map Brier without overfitting")
    sk_results = []
    for sk in [3, 5, 8, 12, 18, 24, 36]:
        rebuild(SHRINK_K=sk)
        snaps = load_snapshots()
        m = measure(matches, snaps, BETA, also_maps_df=maps_df)
        sk_results.append((sk, m))
        print(fmt_both(f'SHRINK_K={sk}', m), flush=True)
    print()

    # ======================= PART 2A: RD_SCALE × CN_PRIOR =======================
    print("━━━ PART 2A: RD_SCALE × CN_PRIOR (does deeper CN free up scale headroom?) ━━━")
    rs_cn = []
    for rs in [2.0, 2.5, 3.0]:
        # For each scale, find best β (proportional to 1/scale)
        for cp in [-2.5, -3.0, -3.5, -4.0]:
            # Adjust β proportionally
            beta_use = BETA * (2.0 / rs)
            rebuild(RD_SCALE=rs, CN_PRIOR=cp)
            snaps = load_snapshots()
            m = measure(matches, snaps, beta_use, also_maps_df=None)
            label = f'RS={rs} CP={cp} β={beta_use:.3f}'
            rs_cn.append((label, m))
            print(fmt_series(label, m), flush=True)
    print()

    # ======================= PART 2B: RD_POWER × CN_PRIOR =======================
    # Just to make sure RD_POWER 0.5 is still the right pick now
    print("━━━ PART 2B: RD_POWER variations with new CN_PRIOR ━━━")
    pwr_cn = []
    for pw, sc, bt in [(0.5, 2.0, 0.17), (0.6, 2.0, 0.14), (0.7, 2.0, 0.12),
                       (0.5, 2.5, 0.14), (0.6, 2.5, 0.11)]:
        rebuild(RD_POWER=pw, RD_SCALE=sc)
        snaps = load_snapshots()
        m = measure(matches, snaps, bt, also_maps_df=None)
        label = f'pow={pw} sc={sc} β={bt:.3f}'
        pwr_cn.append((label, m))
        print(fmt_series(label, m), flush=True)
    print()

    # ======================= PART 3: Integrated push =======================
    print("━━━ PART 3: Integrated magnitude push ━━━")
    # Try several combos pushing for max magnitude with calibration intact
    integrated_grid = [
        # (RD_POWER, RD_SCALE, β, CN_PRIOR, SPILL, label)
        (0.5, 2.0, 0.17, -3.0, 0.5, 'DEPLOYED'),
        (0.5, 2.5, 0.14, -3.0, 0.5, 'scale 2.5'),
        (0.5, 2.5, 0.14, -3.5, 0.5, 'scale 2.5 + CN deeper'),
        (0.5, 2.5, 0.14, -4.0, 0.5, 'scale 2.5 + CN -4'),
        (0.5, 3.0, 0.12, -3.5, 0.5, 'scale 3.0 + CN deeper'),
        (0.5, 3.0, 0.11, -4.0, 0.5, 'scale 3.0 + CN -4'),
        (0.5, 2.5, 0.14, -3.5, 0.6, 'scale 2.5 + CN deeper + SPILL 0.6'),
        (0.5, 2.5, 0.14, -3.0, 0.75, 'scale 2.5 + SPILL 0.75'),
        (0.5, 3.0, 0.11, -4.0, 0.75, 'scale 3 + CN -4 + SPILL 0.75'),
    ]
    integrated = []
    for pw, sc, bt, cp, sp, lbl in integrated_grid:
        rebuild(RD_POWER=pw, RD_SCALE=sc, CN_PRIOR=cp, REGION_SPILLOVER_ALPHA=sp)
        snaps = load_snapshots()
        m = measure(matches, snaps, bt, also_maps_df=None)
        integrated.append((lbl, m))
        print(fmt_series(lbl, m), flush=True)
    print()

    # ======================= SUMMARY =======================
    print("="*100)
    print("SUMMARY")
    print("="*100)

    # Part 1: best SHRINK_K
    best_sk = min(sk_results, key=lambda x: x[1]['map_brier'])
    print(f"\n  PART 1 — SHRINK_K:")
    print(f"    Deployed (12): map_Brier={base['map_brier']:.5f}  map_ECE={base['map_ece']:.4f}  NS_map_spread={base['ns_map_spread']:.2f}")
    print(f"    Best:    ({best_sk[0]:>2}): map_Brier={best_sk[1]['map_brier']:.5f}  map_ECE={best_sk[1]['map_ece']:.4f}  NS_map_spread={best_sk[1]['ns_map_spread']:.2f}")
    delta = (base['map_brier'] - best_sk[1]['map_brier']) * 10000
    print(f"    Δ Brier: {delta:+.1f}bp (positive = improvement)")

    # Part 3: best integrated push
    base_brier = base['series_brier']
    candidates = [(lbl, m) for lbl, m in integrated if m['series_brier'] <= base_brier * 1.01
                  and abs(m['series_platt_A'] - 1) <= 0.15]
    candidates.sort(key=lambda x: -x[1]['top26'])
    print(f"\n  PART 3 — Magnitude push candidates (Brier ≤ deployed × 1.01, Platt in [0.85, 1.15]):")
    for lbl, m in candidates[:8]:
        print(f"    {lbl:<40} top26={m['top26']:>+5.2f}  bot26={m['bot26']:>+5.2f}  "
              f"Brier={m['series_brier']:.5f}  Platt={m['series_platt_A']:>4.2f}  ECE={m['series_ece']:.4f}")

    print(f"\n━━━ Restoring deployed config ━━━")
    rebuild()
    print("Done.")


if __name__ == '__main__':
    main()
