"""
Replace post-hoc CN shrinkage with Bayesian-in-Massey + adaptive prior.

Implementation:
  1. Run Massey WITHOUT CN prior → raw ratings
  2. Compute adaptive prior = weighted_mean(raw_CN, weights=intl_w)
     (only counts CN teams with intl_w > min_evidence as "tested")
  3. Re-solve Massey WITH virtual prior games for CN teams:
       λ_i = STRENGTH * K / (K + intl_w_i)    [smooth, no floor]
       M[i,i] += λ_i
       p[i] += λ_i * adaptive_prior

Result:
  - No hardcoded c_min floor
  - Transitivity preserved (NOVA's rating shifts when its opponents shift)
  - Prior adapts to measured CN skill (not stuck at -4)

We test 3 variants:
  (A) DEPLOYED — current post-hoc shrinkage (CN_PRIOR=-4, K=2, c_min=0.5)
  (B) BAYESIAN+ADAPTIVE — full proposal
  (C) BAYESIAN+FIXED prior (-4) — to isolate impact of adaptive vs fixed prior
"""
import os, sys, math, json, importlib, contextlib, io
import numpy as np
import pandas as pd
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scrapers'))

from BacktestSeriesPredictions import (
    load_match_data, load_snapshots, find_snapshot_for, brier
)

BETA = 0.136
STRENGTH = 25.0  # virtual game weight at zero intl exposure
CN_INTL_K = 2.0  # K for the smooth trust function
MIN_EVIDENCE = 0.5  # min intl_w to count as "tested" for adaptive prior


def _massey_with_priors(games, lambda_decay, ref_date, prior_weights, prior_targets,
                         min_games=5):
    """Modified Massey: also accept per-team prior weights and targets.

    For each team t:
      M[t,t] += prior_weights[t]
      p[t]   += prior_weights[t] * prior_targets[t]

    This embeds the prior as virtual games inside the linear solve.
    """
    import BuildMapRatings as B
    # Filter teams with min_games
    counts = {}
    for g in games:
        counts[g['winner']] = counts.get(g['winner'], 0) + 1
        counts[g['loser']]  = counts.get(g['loser'], 0) + 1
    teams = sorted(t for t, c in counts.items() if c >= min_games)
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    if n < 2: return {t: 0.0 for t in teams}

    M = np.zeros((n, n)); p = np.zeros(n)

    for g in games:
        if g['winner'] not in idx or g['loser'] not in idx: continue
        is_intl = g.get('event_id') in B.INTL_EVENTS
        is_champions = 'champions' in g.get('event_id', '')
        eff_w = B._effective_weeks_ago(g['winner'], g['date'], ref_date)
        eff_l = B._effective_weeks_ago(g['loser'],  g['date'], ref_date)
        w_winner = math.exp(-lambda_decay * eff_w)
        w_loser  = math.exp(-lambda_decay * eff_l)
        base_w = math.sqrt(w_winner * w_loser)
        cont_w = B._team_continuity_factor(g['winner'], g['date'], ref_date)
        cont_l = B._team_continuity_factor(g['loser'],  g['date'], ref_date)
        base_w *= math.sqrt(cont_w * cont_l)
        if is_champions:
            win_mult = los_mult = B.CHAMPIONS_MULT
        elif is_intl:
            win_mult = B.INTL_WIN_MULT; los_mult = B.INTL_LOSS_MULT
        else:
            win_mult = los_mult = 1.0
        w_win = base_w * win_mult
        w_los = base_w * los_mult
        w_sym = min(w_win, w_los)
        raw_rd = g['wr'] - g['lr']
        if B.RD_TRANSFORM == 'sqrt':
            rd = math.copysign(math.sqrt(abs(raw_rd)) * B.RD_SCALE, raw_rd)
        elif B.RD_TRANSFORM == 'power':
            rd = math.copysign((abs(raw_rd) ** B.RD_POWER) * B.RD_SCALE, raw_rd)
        else:
            rd = raw_rd
        i, j = idx[g['winner']], idx[g['loser']]
        M[i, i] += w_sym; M[j, j] += w_sym
        M[i, j] -= w_sym; M[j, i] -= w_sym
        p[i] += w_win * rd; p[j] -= w_los * rd

    # Default ridge
    ridge = 0.5
    for i in range(n - 1):
        M[i, i] += ridge

    # NEW: team-specific priors (the Bayesian addition)
    for t, i in idx.items():
        if i == n - 1: continue  # last row is the anchor
        pw = prior_weights.get(t, 0.0)
        pt = prior_targets.get(t, 0.0)
        if pw > 0:
            M[i, i] += pw
            p[i] += pw * pt

    # Anchor: mean(r) = 0
    M[-1, :] = 1.0
    p[-1] = 0.0
    M[-1, :] = 1.0; p[-1] = 0.0

    try:
        r = np.linalg.solve(M, p)
    except np.linalg.LinAlgError:
        r, *_ = np.linalg.lstsq(M, p, rcond=None)
    return {t: float(r[idx[t]]) for t in teams}


def bayesian_cn_ratings(games, lambda_decay, ref_date, *,
                         strength=STRENGTH, intl_k=CN_INTL_K,
                         use_adaptive_prior=True, fixed_prior=-4.0,
                         min_evidence=MIN_EVIDENCE):
    """Two-pass: first pass to compute adaptive prior, second pass with Bayesian Massey."""
    import BuildMapRatings as B

    # Pass 1: vanilla Massey
    raw_pass1 = _massey_with_priors(games, lambda_decay, ref_date,
                                     prior_weights={}, prior_targets={})

    # Compute intl_weights
    intl_w = B._compute_intl_weights(games, lambda_decay, ref_date)

    # Compute adaptive prior (weighted mean of intl-tested CN raw ratings)
    if use_adaptive_prior:
        num = 0.0; den = 0.0
        for t in raw_pass1:
            if t in B.CN_TEAMS_SET and intl_w.get(t, 0) > min_evidence:
                w = intl_w[t]
                num += raw_pass1[t] * w
                den += w
        if den > 0:
            cn_prior = num / den
        else:
            # No tested CN — fall back to fixed prior
            cn_prior = fixed_prior
    else:
        cn_prior = fixed_prior

    # Compute prior weights and targets for CN teams (smooth, no floor)
    prior_weights = {}
    prior_targets = {}
    for t in raw_pass1:
        if t in B.CN_TEAMS_SET:
            iw = intl_w.get(t, 0.0)
            prior_weights[t] = strength * intl_k / (intl_k + iw)
            prior_targets[t] = cn_prior

    # Pass 2: Bayesian Massey with prior virtual games
    final = _massey_with_priors(games, lambda_decay, ref_date,
                                  prior_weights, prior_targets)
    return final, raw_pass1, intl_w, cn_prior


def rebuild_with_bayesian(use_adaptive=True, strength=STRENGTH,
                            intl_k=CN_INTL_K, fixed_prior=-4.0):
    """Rebuild map_ratings.json using Bayesian CN prior approach.
    Monkey-patches build_year_ratings to use Bayesian solver."""
    saved_argv = sys.argv[:]
    sys.argv = ['BuildMapRatings.py']
    try:
        import BuildMapRatings as B
        importlib.reload(B)

        original_massey = B.massey_ratings
        original_apply_cn = B._apply_cn_shrinkage

        # Wrap massey_ratings: do nothing different, but record that we're being called
        # The key change is in build_year_ratings which calls massey + _apply_cn_shrinkage
        # We need to bypass _apply_cn_shrinkage and instead use Bayesian.

        # Approach: monkey-patch _apply_cn_shrinkage to be a no-op, then patch massey_ratings
        # to internally do the Bayesian solve.
        def bayesian_massey(games, lam, ref_date, min_games=0):
            final, raw, iw, prior = bayesian_cn_ratings(
                games, lam, ref_date,
                strength=strength, intl_k=intl_k,
                use_adaptive_prior=use_adaptive, fixed_prior=fixed_prior,
                min_evidence=MIN_EVIDENCE,
            )
            return final

        B.massey_ratings = bayesian_massey
        B._apply_cn_shrinkage = lambda ratings, intl_weights, **kwargs: dict(ratings)  # no-op

        with contextlib.redirect_stdout(io.StringIO()):
            B.main()

        # Restore
        B.massey_ratings = original_massey
        B._apply_cn_shrinkage = original_apply_cn
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


def cn_table(snap_data, label):
    teams = snap_data
    cn = sorted([(o, t['overall_rating']) for o, t in teams.items() if o in CN_SET],
                key=lambda x: -x[1])
    return f"  {label}:\n" + "\n".join(f"     {o:>6}  {r:+.2f}" for o, r in cn)


CN_SET = {'EDG','BLG','TE','DRG','ASE','AG','XLG','WOL','FPX','JDG','NOVA','TEC','TYL','TYLOO'}


def summarize(label, matches):
    """Read current state of map_ratings.json and summarize."""
    snaps = load_snapshots()
    with open(os.path.join(ROOT, 'data/map_ratings.json')) as f:
        mr = json.load(f)
    teams26 = mr['ratings']['2026']['snapshots']['after_stage1']['teams']
    teams26_san = mr['ratings']['2026']['snapshots']['after_santiago']['teams']
    teams24 = mr['ratings']['2024']['snapshots']['after_champions']['teams']

    # Series Brier by year
    yearly = {}
    for yr in ['2024','2025','2026']:
        m = matches[matches['date'].str.startswith(yr)].reset_index(drop=True)
        p, y = gen_series(m, snaps)
        if len(p) >= 30:
            yearly[yr] = float(brier(p, y))
        else:
            yearly[yr] = None
    p_all, y_all = gen_series(matches, snaps)
    full_brier = float(brier(p_all, y_all)) if len(p_all) else None

    # Trophy ranks
    trophy_avg = 0
    for year, snap, winner in [(2024,'after_champions','EDG'),(2025,'after_champions','NRG'),(2026,'after_santiago','NS')]:
        s = mr['ratings'][str(year)]['snapshots'].get(snap, {}).get('teams', {})
        items = sorted(s.items(), key=lambda x:-x[1]['overall_rating'])
        rk = next((i+1 for i,(t,_) in enumerate(items) if t==winner), 50)
        trophy_avg += rk
    trophy_avg /= 3

    # CN ratings 2026 stage1
    cn_s1 = sorted([(o, t['overall_rating']) for o, t in teams26.items() if o in CN_SET], key=lambda x:-x[1])

    print(f"\n━━━ {label} ━━━")
    print(f"  CN after_stage1 (sorted):")
    for o, r in cn_s1:
        print(f"     {o:>6}  {r:+.2f}")
    print(f"  Brier per year: 2024={yearly['2024']:.5f}  2025={yearly['2025']:.5f}  2026={yearly['2026']:.5f}")
    print(f"  Brier full:     {full_brier:.5f}")
    print(f"  Trophy avg:     {trophy_avg:.2f}")


def main():
    matches = load_match_data()
    print(f"Loaded {len(matches)} series\n")

    # Baseline: deploy current production
    sys.argv = ['BuildMapRatings.py']
    import BuildMapRatings
    importlib.reload(BuildMapRatings)
    with contextlib.redirect_stdout(io.StringIO()):
        BuildMapRatings.main()
    summarize("OPTION A: DEPLOYED (post-hoc shrinkage, CN_PRIOR=-4, K=2, c_min=0.5)", matches)

    # Bayesian + Adaptive prior
    rebuild_with_bayesian(use_adaptive=True, strength=STRENGTH, intl_k=CN_INTL_K)
    summarize(f"OPTION B: BAYESIAN + ADAPTIVE prior (STRENGTH={STRENGTH}, K={CN_INTL_K})", matches)

    # Bayesian with fixed prior (to isolate variable)
    rebuild_with_bayesian(use_adaptive=False, fixed_prior=-4.0, strength=STRENGTH, intl_k=CN_INTL_K)
    summarize(f"OPTION C: BAYESIAN with FIXED prior=-4 (no adaptive)", matches)

    # Bayesian + Adaptive with lower strength
    rebuild_with_bayesian(use_adaptive=True, strength=15, intl_k=CN_INTL_K)
    summarize(f"OPTION D: BAYESIAN + ADAPTIVE, STRENGTH=15 (gentler)", matches)

    # Bayesian + Adaptive with higher strength
    rebuild_with_bayesian(use_adaptive=True, strength=40, intl_k=CN_INTL_K)
    summarize(f"OPTION E: BAYESIAN + ADAPTIVE, STRENGTH=40 (stronger)", matches)

    print(f"\n━━━ Restoring deployed config ━━━")
    sys.argv = ['BuildMapRatings.py']
    importlib.reload(BuildMapRatings)
    with contextlib.redirect_stdout(io.StringIO()):
        BuildMapRatings.main()
    print("Done.")


if __name__ == '__main__':
    main()
