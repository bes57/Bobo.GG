"""
Phase 2: Rating-parameter co-optimization for prediction Brier.

Method:
  - Save BuildMapRatings.py constants
  - For each candidate, monkey-patch a constant, rebuild map_ratings.json,
    re-run series backtest, record CV Brier
  - One-at-a-time first to find sensitivities
  - Then joint search on top movers

Constants under test:
  HALF_LIFE_WEEKS   (current 5)
  INTL_WIN_MULT     (current 2.0)
  CHAMPIONS_MULT    (current 4.0)
  ROSTER_PERSISTENCE (current 0.3)
  CN_PRIOR          (current -2.0)
  CN_INTL_K         (current 4.0)
  RD_SCALE          (current 2.0)
"""
import os, sys, json, importlib, time
import numpy as np
import pandas as pd
sys.path.insert(0, 'scrapers')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_snapshots():
    with open(os.path.join(ROOT, 'data/map_ratings.json')) as f:
        mr = json.load(f)
    out = []
    for y, yblock in mr['ratings'].items():
        for snap, sdata in yblock['snapshots'].items():
            rd = sdata.get('ref_date')
            if not rd: continue
            d = {t: v.get('overall_rating', 0.0) for t, v in sdata.get('teams', {}).items()}
            out.append((rd, y, snap, d))
    out.sort(key=lambda x: x[0])
    return out


def predict_series(rA, rB, beta=0.17):
    import math
    p = 1 / (1 + math.exp(-beta * (rA - rB)))
    p = max(1e-6, min(1-1e-6, p))
    return p**2 * (3 - 2*p)


def brier(probs, outs):
    return float(np.mean((np.asarray(probs) - np.asarray(outs)) ** 2))


from BacktestSeriesPredictions import load_match_data


def cv_brier(matches, snaps):
    splits = [
        ('2024-01-01', '2024-07-01'),
        ('2024-07-01', '2025-01-01'),
        ('2025-01-01', '2025-07-01'),
        ('2025-07-01', '2026-01-01'),
        ('2026-01-01', '2026-06-01'),
    ]
    total_b, total_n = 0.0, 0
    for lo, hi in splits:
        fold = matches[(matches['date'] >= lo) & (matches['date'] < hi)]
        if len(fold) < 30: continue
        ps, ys = [], []
        for _, m in fold.iterrows():
            best = None
            for rd, _, _, d in snaps:
                if rd < m['date']:
                    best = d
                else:
                    break
            if not best or m['a'] not in best or m['b'] not in best:
                continue
            p = predict_series(best[m['a']], best[m['b']])
            ps.append(p); ys.append(m['a_wins'])
        if not ps: continue
        b = brier(ps, ys)
        total_b += b * len(ps); total_n += len(ps)
    return total_b / max(total_n, 1), total_n


def holdout_2026(matches, snaps):
    test = matches[matches['date'] >= '2026-01-01']
    ps, ys = [], []
    for _, m in test.iterrows():
        best = None
        for rd, _, _, d in snaps:
            if rd < m['date']: best = d
            else: break
        if not best or m['a'] not in best or m['b'] not in best: continue
        p = predict_series(best[m['a']], best[m['b']])
        ps.append(p); ys.append(m['a_wins'])
    return brier(ps, ys) if ps else float('nan'), len(ps)


def evaluate_config(matches, **overrides):
    """Monkey-patch BuildMapRatings constants, rebuild, evaluate."""
    # Save and clobber sys.argv to keep argparse-style code from grabbing our flags
    saved_argv = sys.argv[:]
    sys.argv = ['BuildMapRatings.py']
    try:
        import BuildMapRatings
        importlib.reload(BuildMapRatings)
        for k, v in overrides.items():
            setattr(BuildMapRatings, k, v)
        # Suppress prints
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            BuildMapRatings.main()
    finally:
        sys.argv = saved_argv

    snaps = load_snapshots()
    cvb, n_cv = cv_brier(matches, snaps)
    hob, n_ho = holdout_2026(matches, snaps)
    return cvb, hob, n_cv, n_ho


def main():
    matches = load_match_data()
    print(f'Loaded {len(matches)} matches')

    # ─── Baseline ───
    print('\n━━━ Baseline (current production parameters) ━━━')
    t0 = time.time()
    cvb, hob, n_cv, n_ho = evaluate_config(matches)
    base_cv = cvb; base_ho = hob
    print(f'  CV Brier: {cvb:.5f} (n={n_cv})   2026 Brier: {hob:.5f} (n={n_ho})   '
          f'[{time.time()-t0:.1f}s]')

    # ─── One-at-a-time sensitivity ───
    print('\n━━━ One-at-a-time parameter sweeps ━━━')
    sweeps = [
        ('HALF_LIFE_WEEKS',    [3, 4, 5, 6, 7, 8, 10]),
        ('INTL_WIN_MULT',      [1.0, 1.5, 2.0, 2.5, 3.0]),
        ('CHAMPIONS_MULT',     [2.0, 3.0, 4.0, 5.0, 6.0]),
        ('ROSTER_PERSISTENCE', [0.0, 0.15, 0.3, 0.5, 0.7]),
        ('CN_PRIOR',           [-3.0, -2.5, -2.0, -1.5, -1.0]),
        ('CN_INTL_K',          [2.0, 3.0, 4.0, 6.0, 8.0]),
        ('RD_SCALE',           [1.0, 1.5, 2.0, 2.5, 3.0]),
    ]
    best_per_param = {}
    for name, values in sweeps:
        print(f'\n  {name}:')
        print(f'  {"value":>8}  {"CV Brier":>9}  {"delta":>9}  {"2026":>7}')
        best = (None, 1.0)
        for v in values:
            t0 = time.time()
            cvb, hob, _, _ = evaluate_config(matches, **{name: v})
            d = cvb - base_cv
            marker = '  ←' if cvb < best[1] else ''
            print(f'  {v:>8}  {cvb:>9.5f}  {d:>+9.5f}  {hob:.5f}  [{time.time()-t0:.0f}s]{marker}', flush=True)
            if cvb < best[1]:
                best = (v, cvb)
        best_per_param[name] = best
        print(f'    → best {name} = {best[0]}  (Brier {best[1]:.5f})')

    # ─── Joint with top movers ───
    print('\n━━━ Joint search (apply ALL best per-param simultaneously) ━━━')
    combo = {name: best for name, (best, _) in best_per_param.items()}
    cvb, hob, _, _ = evaluate_config(matches, **combo)
    print(f'  Combined config: {combo}')
    print(f'  CV Brier: {cvb:.5f} (Δ {cvb - base_cv:+.5f})')
    print(f'  2026 Brier: {hob:.5f} (Δ {hob - base_ho:+.5f})')

    # ─── Restore baseline ───
    print('\n━━━ Restoring baseline ━━━')
    evaluate_config(matches)
    print('  Done.')


if __name__ == '__main__':
    main()
