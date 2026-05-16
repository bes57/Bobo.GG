"""
Surgical CN shrinkage fix: add INDIRECT intl evidence.

When a CN team plays intl-tested CN teams a lot, give them partial calibration
credit from those games. The exact formula:

  evidence(team) = direct_intl_w(team) + INDIRECT_WEIGHT * indirect_intl_w(team)

Where:
  direct_intl_w = team's own decayed intl game weight (current behavior)
  indirect_intl_w = sum over team's CN-vs-CN games of (decay * opp_direct_intl_w)

Then c = max(c_min, min(evidence/K, 1.0))  — current floor-based formula
Or:  c = evidence/(evidence + K)             — smooth Bayesian, no floor

This is SURGICAL: tested teams (high direct_intl_w) unchanged. Untested teams
who play tested ones a lot get more credit. Truly isolated teams stay at floor.

Test multiple INDIRECT_WEIGHT values + floor vs no-floor.
"""
import os, sys, math, json, importlib, contextlib, io
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scrapers'))

from BacktestSeriesPredictions import load_match_data, load_snapshots, find_snapshot_for, brier

CN_SET = {'EDG','BLG','TE','DRG','ASE','AG','XLG','WOL','FPX','JDG','NOVA','TEC','TYL','TYLOO'}
BETA = 0.136


def compute_indirect_intl_w(games, intl_w, lambda_decay, ref_date):
    """For each CN team t, sum decay * opp_direct_intl_w over CN-vs-CN games where opp is CN."""
    import BuildMapRatings as B
    indirect = {}
    for g in games:
        if g.get('event_id') in B.INTL_EVENTS: continue
        a, b = g['winner'], g['loser']
        if a not in CN_SET or b not in CN_SET: continue  # only count CN-vs-CN
        eff_a = B._effective_weeks_ago(a, g['date'], ref_date)
        eff_b = B._effective_weeks_ago(b, g['date'], ref_date)
        decay = math.sqrt(math.exp(-lambda_decay * eff_a) * math.exp(-lambda_decay * eff_b))
        indirect[a] = indirect.get(a, 0.0) + decay * intl_w.get(b, 0.0)
        indirect[b] = indirect.get(b, 0.0) + decay * intl_w.get(a, 0.0)
    return indirect


def custom_cn_shrinkage(ratings, intl_w, indirect_w, K, prior, c_min, indirect_weight,
                         use_smooth_bayesian=False):
    """Apply CN shrinkage with optional indirect evidence + optional smooth formula."""
    import BuildMapRatings as B
    out = {}
    for t, r in ratings.items():
        if t in B.CN_TEAMS_SET:
            ev = intl_w.get(t, 0.0) + indirect_weight * indirect_w.get(t, 0.0)
            if use_smooth_bayesian:
                c = ev / (ev + K)
            else:
                c = max(c_min, min(ev / K, 1.0))
            out[t] = c * r + (1 - c) * prior
        else:
            out[t] = r
    return out


def rebuild_with_custom_shrinkage(indirect_weight, c_min=0.5, K=2.0, prior=-4.0,
                                    use_smooth=False):
    saved_argv = sys.argv[:]
    sys.argv = ['BuildMapRatings.py']
    try:
        import BuildMapRatings as B
        importlib.reload(B)

        # Override CN shrinkage
        def new_shrink(ratings, intl_weights, **kwargs):
            # need games + lambda_decay + ref_date — get from a closure or stash
            # Easier: re-derive from stash
            games = B._LAST_SOLVE_GAMES
            lam = B._LAST_SOLVE_LAM
            ref = B._LAST_SOLVE_REF
            indirect = compute_indirect_intl_w(games, intl_weights, lam, ref)
            return custom_cn_shrinkage(ratings, intl_weights, indirect,
                                         K=K, prior=prior, c_min=c_min,
                                         indirect_weight=indirect_weight,
                                         use_smooth_bayesian=use_smooth)

        # Stash games/lam/ref in massey wrapper
        orig_massey = B.massey_ratings
        def massey_wrapper(games, lam, ref, min_games=0):
            B._LAST_SOLVE_GAMES = games
            B._LAST_SOLVE_LAM = lam
            B._LAST_SOLVE_REF = ref
            return orig_massey(games, lam, ref, min_games)
        B.massey_ratings = massey_wrapper
        B._apply_cn_shrinkage = new_shrink

        with contextlib.redirect_stdout(io.StringIO()):
            B.main()

        B.massey_ratings = orig_massey
    finally:
        sys.argv = saved_argv


def predict_series(rA, rB, beta=BETA):
    p = 1.0 / (1.0 + math.exp(-beta * (rA - rB)))
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


def summary(label, matches):
    snaps = load_snapshots()
    with open(f'{ROOT}/data/map_ratings.json') as f:
        mr = json.load(f)
    teams = mr['ratings']['2026']['snapshots']['after_stage1']['teams']
    cn = sorted([(o, t['overall_rating']) for o, t in teams.items() if o in CN_SET], key=lambda x:-x[1])

    yearly = {}
    for yr in ['2024','2025','2026']:
        m = matches[matches['date'].str.startswith(yr)].reset_index(drop=True)
        p, y = gen_series(m, snaps)
        yearly[yr] = float(brier(p, y)) if len(p) >= 30 else None
    p_all, y_all = gen_series(matches, snaps)
    full = float(brier(p_all, y_all))

    # Trophy
    SNAPS_T = [(2024,'after_champions','EDG'),(2025,'after_champions','NRG'),(2026,'after_santiago','NS')]
    ranks = []
    for year, snap_key, winner in SNAPS_T:
        s = mr['ratings'][str(year)]['snapshots'].get(snap_key, {}).get('teams', {})
        items = sorted(s.items(), key=lambda x:-x[1]['overall_rating'])
        rk = next((i+1 for i,(t,_) in enumerate(items) if t==winner), 50)
        ranks.append(rk)

    print(f"\n━━━ {label} ━━━")
    cn_str = ', '.join(f'{o}={r:+.2f}' for o, r in cn)
    print(f"  CN s1: {cn_str}")
    print(f"  Brier: 2024={yearly['2024']:.5f}  2025={yearly['2025']:.5f}  2026={yearly['2026']:.5f}  full={full:.5f}  trophy={sum(ranks)/3:.2f}")


def rebuild_baseline():
    saved_argv = sys.argv[:]
    sys.argv = ['BuildMapRatings.py']
    try:
        import BuildMapRatings as B
        importlib.reload(B)
        with contextlib.redirect_stdout(io.StringIO()):
            B.main()
    finally:
        sys.argv = saved_argv


def main():
    matches = load_match_data()
    print(f"Loaded {len(matches)} series")

    rebuild_baseline()
    summary("DEPLOYED (no indirect, K=2, c_min=0.5)", matches)

    print("\n" + "="*80)
    print("SWEEP: SATURATING formula, NO FLOOR (c_min=0)")
    print("Tested teams unchanged (c=1 once direct >= K). Untested teams escape via indirect.")
    print("="*80)
    for iw in [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]:
        rebuild_with_custom_shrinkage(indirect_weight=iw, c_min=0.0, K=2.0)
        summary(f"NO-FLOOR + sat, indirect_weight={iw}", matches)

    print("\n" + "="*80)
    print("SWEEP: SMOOTH Bayesian formula (also no floor by construction)")
    print("c = ev/(ev+K). Tested teams never fully escape (c < 1 always).")
    print("="*80)
    for iw in [0.20, 0.50, 1.0, 2.0]:
        rebuild_with_custom_shrinkage(indirect_weight=iw, use_smooth=True, K=2.0)
        summary(f"SMOOTH, indirect_weight={iw}", matches)

    print("\n━━━ Restoring deployed ━━━")
    rebuild_baseline()
    print("Done.")


if __name__ == '__main__':
    main()
