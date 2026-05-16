"""
Per-year Brier evaluation for the remaining knob options.

Decisions to make:
  (A) SHRINK_K — per-map. Test 3, 5, 8, 12.
  (B) Main ratings magnitude push:
        - Stay (deployed): RD_SCALE=2.0, CN_PRIOR=-3.0, SPILL=0.5
        - Modest:          RD_SCALE=2.5, CN_PRIOR=-4.0, SPILL=0.5
        - Aggressive:      RD_SCALE=3.0, CN_PRIOR=-4.0, SPILL=0.75

For each: Brier broken out by year (2023, 2024, 2025, 2026).
Pick the option that's best on AVERAGE across years (or best worst-year),
not just on the 2026 holdout.
"""
import os, sys, math, json, importlib, contextlib, io
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scrapers'))

from BacktestSeriesPredictions import (
    load_match_data, load_snapshots, find_snapshot_for, brier,
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


def predict(rA, rB, beta):
    return 1.0 / (1.0 + math.exp(-beta * (rA - rB)))


def predict_series(rA, rB, beta, fmt='bo3'):
    p = predict(rA, rB, beta)
    return p**2 * (3 - 2*p)


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
        BuildMapRatings._apply_cn_shrinkage.__defaults__ = (
            cfg['CN_PRIOR'], cfg['CN_INTL_K'], cfg['CN_C_MIN']
        )
        with contextlib.redirect_stdout(io.StringIO()):
            BuildMapRatings.main()
    finally:
        sys.argv = saved_argv


def load_full_snapshots():
    with open(os.path.join(ROOT, 'data/map_ratings.json')) as f:
        mr = json.load(f)
    out = []
    for y, yblock in mr['ratings'].items():
        for snap, sdata in yblock['snapshots'].items():
            rd = sdata.get('ref_date')
            if not rd: continue
            teams = sdata.get('teams', {})
            out.append((rd, y, snap, teams))
    out.sort(key=lambda x: x[0])
    return out


def find_snap_for(date, snaps):
    best = None
    for rd, y, sn, t in snaps:
        if rd < date: best = (rd, y, sn, t)
        else: break
    return best


def load_map_outcomes():
    mr = pd.read_csv(os.path.join(ROOT, 'data/match_results.csv'))
    per_map = mr[mr['MapNum'] != 'all'].copy()
    per_map['MatchID'] = per_map['MatchID'].astype(int)
    with open(os.path.join(ROOT, 'data/match_dates.json')) as f:
        dates = json.load(f)
    teams_per, map_names = {}, {}
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
    return pd.DataFrame(rows).sort_values('date').reset_index(drop=True)


def series_brier_by_year(matches, snaps, beta):
    """Returns dict {year: Brier} for series-level."""
    out = {'2023': [], '2024': [], '2025': [], '2026': []}
    p, y = [], []
    for _, m in matches.iterrows():
        s = find_snap_for(m['date'], snaps)
        if not s: continue
        _, _, _, teams = s
        a, b = m['a'], m['b']
        if a not in teams or b not in teams: continue
        rA = teams[a].get('overall_rating', 0.0)
        rB = teams[b].get('overall_rating', 0.0)
        pred = predict_series(rA, rB, beta)
        actual = m['a_wins']
        year = m['date'][:4]
        if year in out:
            out[year].append((pred, actual))
        p.append(pred); y.append(actual)
    by_year = {}
    for yr, pairs in out.items():
        if not pairs:
            by_year[yr] = None; continue
        pp = np.array([x[0] for x in pairs])
        yy = np.array([x[1] for x in pairs])
        by_year[yr] = {'n': len(pp), 'brier': float(brier(pp, yy))}
    return by_year, np.array(p), np.array(y)


def map_brier_by_year(maps_df, snaps, beta):
    """Per-map Brier broken out by year."""
    out = {'2023': [], '2024': [], '2025': [], '2026': []}
    for _, m in maps_df.iterrows():
        s = find_snap_for(m['date'], snaps)
        if not s: continue
        _, _, _, teams = s
        a, b = m['a'], m['b']
        if a not in teams or b not in teams: continue
        ta, tb = teams[a], teams[b]
        ma = ta.get('maps', {}).get(m['MapName'], {})
        mb = tb.get('maps', {}).get(m['MapName'], {})
        rA = ma.get('rating', ta.get('overall_rating', 0.0)) if isinstance(ma, dict) else ta.get('overall_rating', 0.0)
        rB = mb.get('rating', tb.get('overall_rating', 0.0)) if isinstance(mb, dict) else tb.get('overall_rating', 0.0)
        pred = predict(rA, rB, beta)
        actual = m['a_wins']
        year = m['date'][:4]
        if year in out: out[year].append((pred, actual))
    by_year = {}
    for yr, pairs in out.items():
        if not pairs:
            by_year[yr] = None; continue
        pp = np.array([x[0] for x in pairs])
        yy = np.array([x[1] for x in pairs])
        by_year[yr] = {'n': len(pp), 'brier': float(brier(pp, yy))}
    return by_year


def evaluate_full(matches, maps_df, beta, **overrides):
    rebuild(**overrides)
    snaps = load_full_snapshots()
    series_by_year, p, y = series_brier_by_year(matches, snaps, beta)
    map_by_year = map_brier_by_year(maps_df, snaps, beta)
    A, _ = platt(p, y)
    # Magnitude metrics
    with open(os.path.join(ROOT, 'data/map_ratings.json')) as f:
        mr = json.load(f)
    teams26 = mr['ratings']['2026']['snapshots']['after_santiago']['teams']
    top26 = max(t['overall_rating'] for t in teams26.values())
    bot26 = min(t['overall_rating'] for t in teams26.values())
    teams24 = mr['ratings']['2024']['snapshots']['after_champions']['teams']
    top24 = max(t['overall_rating'] for t in teams24.values())
    g2_maps = teams26.get('G2', {}).get('maps', {})
    g2_ratings = [v['rating'] for v in g2_maps.values() if 'rating' in v]
    g2_spread = (max(g2_ratings) - min(g2_ratings)) if g2_ratings else 0
    return {
        'series': series_by_year, 'map': map_by_year, 'platt_A': A,
        'top26': top26, 'bot26': bot26, 'top24': top24, 'g2_spread': g2_spread,
    }


def fmt_year_row(label, r):
    s = r['series']; m = r['map']
    s_str = "  ".join(
        f"{yr}={s[yr]['brier']:.5f}({s[yr]['n']:>3})" if s[yr] else f"{yr}=---"
        for yr in ['2023','2024','2025','2026']
    )
    series_avg = np.mean([s[yr]['brier'] for yr in ['2024','2025','2026'] if s[yr]])
    map_str = "  ".join(
        f"{yr}={m[yr]['brier']:.5f}" if m[yr] else f"{yr}=---"
        for yr in ['2024','2025','2026']
    )
    map_avg = np.mean([m[yr]['brier'] for yr in ['2024','2025','2026'] if m[yr]])
    return (f"\n  {label}"
            f"\n     SERIES: {s_str}   avg(24-26)={series_avg:.5f}"
            f"\n     PER-MAP: {map_str}   avg={map_avg:.5f}"
            f"\n     top26={r['top26']:+.2f}  bot26={r['bot26']:+.2f}  top24={r['top24']:+.2f}  "
            f"G2_map_sp={r['g2_spread']:.2f}  Platt={r['platt_A']:.3f}")


def main():
    matches = load_match_data()
    maps_df = load_map_outcomes()
    print(f"Loaded {len(matches)} series, {len(maps_df)} per-map\n")

    print("="*100)
    print("PART A: SHRINK_K — per-map Brier by year")
    print("="*100)
    for sk in [3, 5, 8, 12]:
        r = evaluate_full(matches, maps_df, 0.17, SHRINK_K=sk)
        print(fmt_year_row(f'SHRINK_K={sk}', r), flush=True)

    print("\n" + "="*100)
    print("PART B: Main magnitude push — series Brier by year")
    print("="*100)

    # Stay (deployed)
    r = evaluate_full(matches, maps_df, 0.17,
                      RD_SCALE=2.0, CN_PRIOR=-3.0, REGION_SPILLOVER_ALPHA=0.5)
    print(fmt_year_row('STAY (RS=2.0, CN=-3.0, SPILL=0.5, β=0.17)', r), flush=True)

    # Modest
    r = evaluate_full(matches, maps_df, 0.136,
                      RD_SCALE=2.5, CN_PRIOR=-4.0, REGION_SPILLOVER_ALPHA=0.5)
    print(fmt_year_row('MODEST (RS=2.5, CN=-4.0, SPILL=0.5, β=0.136)', r), flush=True)

    # Modest variant — CN=-3.5
    r = evaluate_full(matches, maps_df, 0.136,
                      RD_SCALE=2.5, CN_PRIOR=-3.5, REGION_SPILLOVER_ALPHA=0.5)
    print(fmt_year_row('MODEST-mid (RS=2.5, CN=-3.5, SPILL=0.5, β=0.136)', r), flush=True)

    # Aggressive
    r = evaluate_full(matches, maps_df, 0.113,
                      RD_SCALE=3.0, CN_PRIOR=-4.0, REGION_SPILLOVER_ALPHA=0.75)
    print(fmt_year_row('AGGRESSIVE (RS=3.0, CN=-4.0, SPILL=0.75, β=0.113)', r), flush=True)

    # Aggressive variant — keep SPILL at 0.5
    r = evaluate_full(matches, maps_df, 0.113,
                      RD_SCALE=3.0, CN_PRIOR=-4.0, REGION_SPILLOVER_ALPHA=0.5)
    print(fmt_year_row('AGGRESSIVE-tight (RS=3.0, CN=-4.0, SPILL=0.5, β=0.113)', r), flush=True)

    print(f"\n━━━ Restoring deployed ━━━")
    rebuild()
    print("Done.")


if __name__ == '__main__':
    main()
