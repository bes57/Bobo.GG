"""
Refine the magnitude sweep:
  - Zoom in around (RD_POWER=0.5, RD_SCALE=1.5, β=0.22)
  - Test Kalshi backtest (only fair comparison vs market)
  - Sweep half-life and roster persistence too
"""
import os, sys, math, json, time, importlib, contextlib, io
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scrapers'))

from BacktestSeriesPredictions import (
    load_match_data, load_snapshots, find_snapshot_for, brier, logloss,
    expected_calibration_error,
)


def predict_map(rA, rB, beta):
    return 1.0 / (1.0 + math.exp(-beta * (rA - rB)))


def predict_series(rA, rB, beta, fmt='bo3'):
    p = predict_map(rA, rB, beta)
    if fmt == 'bo3': return p**2 * (3 - 2*p)
    if fmt == 'bo5': return p**3 * (10 - 15*p + 6*p*p)
    return p


def gen(matches, snaps, beta, fmt='bo3'):
    p, y = [], []
    for _, m in matches.iterrows():
        s = find_snapshot_for(m['date'], snaps)
        if not s: continue
        _, _, _, ratings = s
        a, b = m['a'], m['b']
        if a not in ratings or b not in ratings: continue
        rA = ratings[a].get('overall_rating', 0.0) if isinstance(ratings[a], dict) else ratings[a]
        rB = ratings[b].get('overall_rating', 0.0) if isinstance(ratings[b], dict) else ratings[b]
        p.append(predict_series(rA, rB, beta, fmt)); y.append(m['a_wins'])
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
        for k, v in overrides.items():
            setattr(BuildMapRatings, k, v)
        with contextlib.redirect_stdout(io.StringIO()):
            BuildMapRatings.main()
    finally:
        sys.argv = saved_argv


def winner_ranks(mr_dict):
    SNAPS = [(2024,'after_madrid','SEN'),(2024,'after_shanghai','GEN'),
             (2024,'after_champions','EDG'),(2025,'after_bangkok','T1'),
             (2025,'after_toronto','PRX'),(2025,'after_champions','NRG'),
             (2026,'after_santiago','NS')]
    rks = []
    for year, snap, winner in SNAPS:
        s = mr_dict['ratings'][str(year)]['snapshots'].get(snap, {}).get('teams', {})
        if not s: rks.append(50); continue
        items = sorted(s.items(), key=lambda x:-x[1]['overall_rating'])
        rk = next((i+1 for i,(t,_) in enumerate(items) if t==winner), 50)
        rks.append(rk)
    return rks


def kalshi_test(p_dict):
    """Compare to Kalshi market. p_dict maps (date, a, b) -> model_p_a.

    Returns metrics for both Kalshi and model on the matched subset.
    Note: Kalshi data uses team_a/team_b in alphabetical order too (after AnalyzeKalshiVsModel.py).
    """
    df = pd.read_csv(os.path.join(ROOT, 'data/kalshi_vs_model.csv'))
    pks, pms, ys = [], [], []
    for _, row in df.iterrows():
        # team_a/team_b in csv may not match alphabetical sort used in our matches
        # Try both orderings
        a, b = sorted([row['team_a'], row['team_b']])
        # If a == row['team_a'], the kalshi_a_fair refers to A. If a == row['team_b'], swap.
        if a == row['team_a']:
            kal_p_a = row['kalshi_a_fair']
            a_won = row['a_won']
        else:
            kal_p_a = 1 - row['kalshi_a_fair']
            a_won = 1 - row['a_won']
        key = (row['date'], a, b)
        if key not in p_dict: continue
        pks.append(kal_p_a)
        pms.append(p_dict[key])
        ys.append(a_won)
    if not ys: return None
    pks, pms, ys = np.array(pks), np.array(pms), np.array(ys)
    return {
        'n': len(ys),
        'kalshi_brier': float(np.mean((pks-ys)**2)),
        'model_brier': float(np.mean((pms-ys)**2)),
        'kalshi_ll': float(-np.mean(ys*np.log(np.clip(pks,1e-9,1-1e-9)) + (1-ys)*np.log(np.clip(1-pks,1e-9,1-1e-9)))),
        'model_ll': float(-np.mean(ys*np.log(np.clip(pms,1e-9,1-1e-9)) + (1-ys)*np.log(np.clip(1-pms,1e-9,1-1e-9)))),
        'edge_mean': float((pms-pks).mean()),
        'edge_std': float((pms-pks).std()),
    }


def build_p_dict(matches, snaps, beta):
    """For Kalshi test: build (date, a, b) -> p map."""
    out = {}
    for _, m in matches.iterrows():
        s = find_snapshot_for(m['date'], snaps)
        if not s: continue
        _, _, _, ratings = s
        a, b = m['a'], m['b']
        if a not in ratings or b not in ratings: continue
        rA = ratings[a].get('overall_rating', 0.0) if isinstance(ratings[a], dict) else ratings[a]
        rB = ratings[b].get('overall_rating', 0.0) if isinstance(ratings[b], dict) else ratings[b]
        out[(m['date'], a, b)] = predict_series(rA, rB, beta, fmt='bo3')
    return out


def evaluate(matches, beta_grid, **cfg):
    rebuild(**cfg)
    snaps = load_snapshots()
    with open(os.path.join(ROOT, 'data/map_ratings.json')) as f:
        mr = json.load(f)

    top_2026 = max(t['overall_rating'] for t in
                   mr['ratings']['2026']['snapshots']['after_santiago']['teams'].values())
    top_2024 = max(t['overall_rating'] for t in
                   mr['ratings']['2024']['snapshots']['after_champions']['teams'].values())
    trophy_avg = float(np.mean(winner_ranks(mr)))

    out = {'cfg': cfg, 'top_2026': top_2026, 'top_2024': top_2024, 'trophy_avg': trophy_avg, 'betas': {}}
    for b in beta_grid:
        p, y = gen(matches, snaps, b)
        if len(p) < 100: continue
        A, _ = platt(p, y)
        out['betas'][b] = {
            'brier': float(brier(p, y)),
            'll': float(logloss(p, y)),
            'ece': float(expected_calibration_error(p, y)),
            'sharp': float(np.abs(p-0.5).mean()),
            'platt_A': A,
        }
        # 2026 only
        recent = matches[matches['date'] >= '2026-01-01'].reset_index(drop=True)
        p26, y26 = gen(recent, snaps, b)
        if len(p26) >= 30:
            out['betas'][b]['brier_2026'] = float(brier(p26, y26))
        else:
            out['betas'][b]['brier_2026'] = None
        # Kalshi test
        p_dict = build_p_dict(matches, snaps, b)
        kal = kalshi_test(p_dict)
        if kal:
            out['betas'][b]['kalshi_n'] = kal['n']
            out['betas'][b]['kalshi_model_brier'] = kal['model_brier']
            out['betas'][b]['kalshi_market_brier'] = kal['kalshi_brier']
            out['betas'][b]['kalshi_model_ll'] = kal['model_ll']
            out['betas'][b]['kalshi_market_ll'] = kal['kalshi_ll']
    return out


def main():
    matches = load_match_data()
    print(f"Loaded {len(matches)} historical series")
    print(f"Refining around (RD_POWER=0.5, RD_SCALE=1.5, β=0.22) and testing on Kalshi")

    # Baseline
    print(f"\n━━━ Baseline (current production) ━━━")
    base = evaluate(matches, [0.25],
                    RD_TRANSFORM='power', RD_POWER=0.35, RD_SCALE=1.25,
                    HALF_LIFE_WEEKS=6.0, INTL_WIN_MULT=1.0, CHAMPIONS_MULT=2.0,
                    ROSTER_PERSISTENCE=0.3)
    b = base['betas'][0.25]
    print(f"  cfg: pow=0.35 scale=1.25 HL=6 IW=1 CH=2 RP=0.3 β=0.25")
    print(f"  top26={base['top_2026']:+.2f} top24={base['top_2024']:+.2f} trophy={base['trophy_avg']:.2f}")
    print(f"  Brier_full={b['brier']:.5f} Brier_2026={b['brier_2026']:.5f}")
    print(f"  Platt A={b['platt_A']:.3f} sharp={b['sharp']:.4f}")
    if 'kalshi_model_brier' in b:
        print(f"  Kalshi: model={b['kalshi_model_brier']:.5f} market={b['kalshi_market_brier']:.5f} "
              f"(diff: {b['kalshi_model_brier']-b['kalshi_market_brier']:+.5f})")

    # Refined grid around optimum — rebuild ONCE per (pow, scale), sweep β inline
    print(f"\n━━━ Refined search around (pow=0.5, scale=1.5) ━━━")
    configs = []
    for pw in [0.45, 0.50, 0.55, 0.60, 0.70]:
        for rd in [1.2, 1.5, 1.8, 2.0]:
            configs.append((pw, rd))
    betas = [0.16, 0.18, 0.20, 0.22, 0.25]
    print(f"  Testing {len(configs)} (pow, scale) configs × {len(betas)} β values")
    print()
    print(f"  {'pow':>4}  {'sc':>4}  {'β':>4}  {'top26':>6}  {'top24':>6}  {'troph':>5}  "
          f"{'Brier':>7}  {'Br26':>7}  {'A':>5}  {'sharp':>5}  {'kModel':>7}  {'kMarket':>7}  {'edge':>6}")

    results = [base]
    base_metric = b['brier']  # full brier benchmark
    for pw, sc in configs:
        r = evaluate(matches, betas,
                     RD_TRANSFORM='power', RD_POWER=pw, RD_SCALE=sc,
                     HALF_LIFE_WEEKS=6.0, INTL_WIN_MULT=1.0, CHAMPIONS_MULT=2.0,
                     ROSTER_PERSISTENCE=0.3)
        for beta in betas:
            bm = r['betas'].get(beta, {})
            if not bm: continue
            kal_m = bm.get('kalshi_model_brier', float('nan'))
            kal_k = bm.get('kalshi_market_brier', float('nan'))
            edge = (kal_m - kal_k) if not (math.isnan(kal_m) or math.isnan(kal_k)) else float('nan')
            mark = '★' if (r['top_2026'] >= 3.5 and bm['brier'] <= base_metric and r['trophy_avg'] <= 2.0 and abs(bm['platt_A']-1)<=0.15) else ' '
            print(f"  {mark} {pw:>4}  {sc:>4}  {beta:>4.2f}  {r['top_2026']:>+6.2f}  {r['top_2024']:>+6.2f}  "
                  f"{r['trophy_avg']:>5.2f}  {bm['brier']:.5f}  {bm['brier_2026']:.5f}  "
                  f"{bm['platt_A']:>5.3f}  {bm['sharp']:>5.3f}  {kal_m:.5f}  {kal_k:.5f}  {edge:+.5f}",
                  flush=True)
        results.append(r)

    print(f"\n━━━ Restoring current config ━━━")
    rebuild()
    print("Done.")


if __name__ == '__main__':
    main()
