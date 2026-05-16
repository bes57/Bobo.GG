"""
Joint sweep for the magnitude question:
  - Find configs that give larger top ratings (user wants +3 minimum, ideally +4-5)
  - Equal-or-better Brier than current production
  - Closer-to-1.0 Platt slope (better calibration at the tails)
  - Maintain ranking order (top intl winners stay near #1)

Tests across (RD_POWER, RD_SCALE) for the rating-build side, then sweeps β
on each rebuild to find the best (config × β) combo.

Crucially, separates two questions:
  (A) Can we make ratings BIGGER without hurting predictions?
      Answer: yes if we also lower β proportionally.
  (B) Can we make predictions SHARPER (more confident) without hurting Brier?
      Answer: maybe not — calibration suggests current predictions are already
      too sharp at the tails. But this varies by which sub-population.

Output: a table sorted by composite goal score, plus the Pareto frontier
of (Brier × top_rating).
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


def gen(matches, snaps, beta, fmt='bo3', return_diff=False):
    p, y, d = [], [], []
    for _, m in matches.iterrows():
        s = find_snapshot_for(m['date'], snaps)
        if not s: continue
        _, _, _, ratings = s
        a, b = m['a'], m['b']
        if a not in ratings or b not in ratings: continue
        rA = ratings[a].get('overall_rating', 0.0) if isinstance(ratings[a], dict) else ratings[a]
        rB = ratings[b].get('overall_rating', 0.0) if isinstance(ratings[b], dict) else ratings[b]
        p.append(predict_series(rA, rB, beta, fmt)); y.append(m['a_wins']); d.append(rA-rB)
    if return_diff: return np.array(p), np.array(y), np.array(d)
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


def snapshot_summary():
    """Read map_ratings.json, return summary metrics for current ratings."""
    with open(os.path.join(ROOT, 'data/map_ratings.json')) as f:
        mr = json.load(f)
    snaps = []
    for y, yblock in mr['ratings'].items():
        for snap, sdata in yblock['snapshots'].items():
            ratings = [t['overall_rating'] for t in sdata.get('teams', {}).values()]
            if not ratings: continue
            snaps.append((y, snap, sdata.get('ref_date'), np.std(ratings), max(ratings), min(ratings)))
    return snaps


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


def evaluate_cfg(matches, **cfg):
    """Rebuild ratings with cfg overrides, then sweep β to find best fit.
    Returns the (cfg, best β by Brier, top_rating, full metrics)."""
    rebuild(**cfg)
    snaps = load_snapshots()
    with open(os.path.join(ROOT, 'data/map_ratings.json')) as f:
        mr = json.load(f)

    # Top rating across recent snapshots for visual "magnitude" metric
    top_2026 = max(t['overall_rating'] for t in
                   mr['ratings']['2026']['snapshots']['after_santiago']['teams'].values())
    top_2024 = max(t['overall_rating'] for t in
                   mr['ratings']['2024']['snapshots']['after_champions']['teams'].values())
    top_2025 = max(t['overall_rating'] for t in
                   mr['ratings']['2025']['snapshots']['after_champions']['teams'].values())

    sds = []
    for y in ['2024','2025','2026']:
        snap_key = 'after_santiago' if y=='2026' else 'after_champions'
        teams = mr['ratings'][y]['snapshots'].get(snap_key, {}).get('teams', {})
        if teams:
            sds.append(np.std([t['overall_rating'] for t in teams.values()]))
    sd_avg = float(np.mean(sds)) if sds else 0.0

    trophy_rks = winner_ranks(mr)

    # Sweep β for best Brier on series-level
    matches_train = matches[matches['date'] < '2026-01-01'].reset_index(drop=True)
    matches_test  = matches[matches['date'] >= '2026-01-01'].reset_index(drop=True)

    results_by_beta = {}
    for b in [0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.25]:
        p_full, y_full = gen(matches, snaps, b, fmt='bo3')
        p_tr, y_tr = gen(matches_train, snaps, b, fmt='bo3')
        p_te, y_te = gen(matches_test, snaps, b, fmt='bo3')
        if len(p_full) < 100: continue
        A, _ = platt(p_full, y_full)
        results_by_beta[b] = {
            'beta': b,
            'brier_full': brier(p_full, y_full),
            'brier_train': brier(p_tr, y_tr),
            'brier_test': brier(p_te, y_te),
            'll_full': logloss(p_full, y_full),
            'ece_full': expected_calibration_error(p_full, y_full),
            'sharp_full': float(np.abs(p_full-0.5).mean()),
            'platt_A_full': A,
            'maxP_full': float(p_full.max()),
            'pct_4555_full': float(((p_full>=0.45)&(p_full<=0.55)).mean()),
            'n_full': int(len(p_full)),
        }

    if not results_by_beta:
        return None
    # Pick β that minimizes train Brier (avoid overfitting to test)
    best_beta = min(results_by_beta, key=lambda b: results_by_beta[b]['brier_train'])
    summary = results_by_beta[best_beta]
    summary.update({
        'cfg': cfg,
        'top_2026': float(top_2026),
        'top_2024': float(top_2024),
        'top_2025': float(top_2025),
        'sd_avg': sd_avg,
        'trophy_rks': trophy_rks,
        'trophy_avg': float(np.mean(trophy_rks)),
        'all_betas': results_by_beta,
    })
    return summary


def main():
    matches = load_match_data()
    print(f"Loaded {len(matches)} historical series")
    print(f"Will sweep RD_POWER × RD_SCALE; for each, sweep β to find best")
    print()

    # Establish baseline with CURRENT config
    print("━━━ BASELINE (current production) ━━━")
    t0 = time.time()
    baseline = evaluate_cfg(matches, RD_TRANSFORM='power', RD_POWER=0.35, RD_SCALE=1.25,
                            HALF_LIFE_WEEKS=6.0, INTL_WIN_MULT=1.0, CHAMPIONS_MULT=2.0,
                            ROSTER_PERSISTENCE=0.3)
    print(f"  Build time: {time.time()-t0:.1f}s")
    print(f"  best β by train Brier: {baseline['beta']}")
    print(f"  top_2024={baseline['top_2024']:+.2f}  top_2025={baseline['top_2025']:+.2f}  top_2026={baseline['top_2026']:+.2f}")
    print(f"  sd_avg={baseline['sd_avg']:.3f}")
    print(f"  Brier (train/test/full): {baseline['brier_train']:.5f} / {baseline['brier_test']:.5f} / {baseline['brier_full']:.5f}")
    print(f"  Platt A: {baseline['platt_A_full']:.3f}    sharpness: {baseline['sharp_full']:.4f}")
    print(f"  trophy ranks: {baseline['trophy_rks']}  avg: {baseline['trophy_avg']:.2f}")
    print(f"  Showing all β results for baseline config:")
    print(f"     {'β':>5}  {'Brier_tr':>9}  {'Brier_te':>9}  {'A':>6}  {'sharp':>6}  {'maxP':>6}")
    for b, r in sorted(baseline['all_betas'].items()):
        print(f"     {b:>5.2f}  {r['brier_train']:.5f}  {r['brier_test']:.5f}  "
              f"{r['platt_A_full']:.3f}  {r['sharp_full']:.4f}  {r['maxP_full']:.3f}")

    # Sweep configs that increase magnitude
    print(f"\n━━━ SWEEP: less-compressive RD configs ━━━")
    configs = []
    # (RD_POWER, RD_SCALE) pairs — note higher power = less compression, larger
    # ratings before scaling. Larger scale = wider spread.
    rd_configs = [
        # Baseline reference
        (0.35, 1.25),
        # Less compressive (power closer to 1.0 = linear)
        (0.5, 1.5),
        (0.5, 2.0),
        (0.5, 2.5),
        (0.6, 1.5),
        (0.6, 2.0),
        (0.7, 1.2),
        (0.7, 1.5),
        (0.7, 2.0),
        (0.8, 1.0),
        (0.8, 1.5),
        (1.0, 0.7),
        (1.0, 1.0),
        (1.0, 1.5),
    ]
    print(f"  Testing {len(rd_configs)} (RD_POWER, RD_SCALE) configs, ~30s each")
    print(f"  Each rebuild + 9-β sweep ≈ 35s. Total ETA ≈ {len(rd_configs)*35/60:.1f} min")
    print()
    print(f"  {'pow':>5}  {'scale':>5}  {'β*':>5}  {'top26':>6}  {'top24':>6}  {'sdAvg':>5}  "
          f"{'Br_tr':>7}  {'Br_te':>7}  {'A':>5}  {'sharp':>5}  {'trophy':>6}")
    results = [baseline]
    for pw, rd in rd_configs:
        if (pw, rd) == (0.35, 1.25):
            r = baseline
        else:
            t0 = time.time()
            r = evaluate_cfg(matches, RD_TRANSFORM='power', RD_POWER=pw, RD_SCALE=rd,
                             HALF_LIFE_WEEKS=6.0, INTL_WIN_MULT=1.0, CHAMPIONS_MULT=2.0,
                             ROSTER_PERSISTENCE=0.3)
            elapsed = time.time()-t0
            if r is None:
                print(f"  {pw:>5}  {rd:>5}  --- failed ---")
                continue
            results.append(r)
        print(f"  {pw:>5}  {rd:>5}  {r['beta']:>5.2f}  {r['top_2026']:>+6.2f}  {r['top_2024']:>+6.2f}  "
              f"{r['sd_avg']:>5.2f}  {r['brier_train']:.5f}  {r['brier_test']:.5f}  "
              f"{r['platt_A_full']:.2f}  {r['sharp_full']:.3f}  {r['trophy_avg']:>6.2f}", flush=True)

    # Save full results
    out_path = '/tmp/magnitude_sweep.json'
    with open(out_path, 'w') as f:
        save = []
        for r in results:
            o = {k:v for k,v in r.items() if k != 'all_betas'}
            o['all_betas'] = {str(k): v for k, v in r['all_betas'].items()}
            save.append(o)
        json.dump(save, f, indent=2)
    print(f"\nSaved to {out_path}")

    # Pareto analysis: which configs strictly dominate baseline?
    print(f"\n━━━ PARETO ANALYSIS vs BASELINE ━━━")
    print(f"  Baseline: Brier_tr={baseline['brier_train']:.5f} top_2026={baseline['top_2026']:+.2f}")
    print(f"  Looking for configs with:")
    print(f"    (a) Brier_tr ≤ baseline (≤ {baseline['brier_train']:.5f})")
    print(f"    (b) top_2026 > baseline (> {baseline['top_2026']:+.2f})")
    print(f"    (c) Platt A in [0.85, 1.15] (better calibration)")
    print(f"    (d) trophy_avg ≤ 2.0 (rankings preserved)")
    print()
    hits = [r for r in results
            if r['brier_train'] <= baseline['brier_train']
            and r['top_2026'] > baseline['top_2026']
            and 0.85 <= r['platt_A_full'] <= 1.15
            and r['trophy_avg'] <= 2.0
            and r is not baseline]
    print(f"  Hits: {len(hits)}")
    if hits:
        hits.sort(key=lambda r: r['brier_train'])
        for r in hits[:10]:
            c = r['cfg']
            print(f"    pow={c.get('RD_POWER')} scale={c.get('RD_SCALE')} β={r['beta']:.2f}  "
                  f"Brier_tr={r['brier_train']:.5f} top_2026={r['top_2026']:+.2f} "
                  f"A={r['platt_A_full']:.2f} trophy={r['trophy_avg']:.2f}")

    # Looser: any config with bigger magnitude + better Platt within Brier tolerance
    print(f"\n  Looser: bigger top_2026 + Brier within 0.5% of baseline:")
    tol = baseline['brier_train'] * 1.005
    loose = [r for r in results
             if r['brier_train'] <= tol
             and r['top_2026'] > baseline['top_2026']
             and r['trophy_avg'] <= 2.5
             and r is not baseline]
    loose.sort(key=lambda r: -r['top_2026'])
    for r in loose[:10]:
        c = r['cfg']
        print(f"    pow={c.get('RD_POWER')} scale={c.get('RD_SCALE')} β={r['beta']:.2f}  "
              f"Brier_tr={r['brier_train']:.5f} top_2026={r['top_2026']:+.2f} "
              f"top_2024={r['top_2024']:+.2f} A={r['platt_A_full']:.2f} "
              f"trophy={r['trophy_avg']:.2f}")

    print(f"\n━━━ Restoring current config ━━━")
    rebuild()
    print("Done.")


if __name__ == '__main__':
    main()
