"""Phase 2C: Test time-smoothing combinations.

Builds two timeline sets (current ROSTER_PERSISTENCE=0.3 and ROSTER_PERSISTENCE=1.0),
then evaluates 4 configs: (i) neither, (ii) ensemble only, (iii) roster only, (iv) both.
"""
from __future__ import annotations
import json, os, sys, math, shutil, time, tempfile
from datetime import datetime, timedelta, date
from pathlib import Path
import numpy as np

ROOT = Path('/Users/benny_es1/PythonTest')
DATA = ROOT / 'data'
ART  = ROOT / 'artifacts'

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scrapers'))
sys.path.insert(0, str(ART))

import harness as H

# Step 1: rebuild timelines with ROSTER_PERSISTENCE=1.0 into a separate directory
# We monkey-patch BuildMapRatings.ROSTER_PERSISTENCE then re-execute the build.
ALT_DIR = ART / 'alt_timelines_rp1'
ALT_DIR.mkdir(exist_ok=True)


def build_alt_timelines():
    """Build timelines with ROSTER_PERSISTENCE = 1.0 (no roster smoothing)."""
    import BuildMapRatings as bmr
    import BuildRatingTimeline as brt

    # Force ROSTER_PERSISTENCE = 1.0. But ROSTER_PERSISTENCE = 1.0 means:
    #   effective_weeks = raw_weeks * (1 - 1.0 * cont)
    # If cont=1 (fully intact roster) effective_weeks = 0 (freezes time).
    # That's NOT what we want for "no roster smoothing".
    # We want NO roster effect: effective_weeks = raw_weeks * (1 - 0 * cont) = raw_weeks
    # So "no roster smoothing" = ROSTER_PERSISTENCE = 0.0
    #
    # Re-read prompt: "ROSTER_PERSISTENCE (0.7 default) — _effective_weeks_ago discounts
    # time when roster has rotated". So in prompt-speak, higher = more smoothing.
    # But in code, the formula is (1 - ROSTER_PERSISTENCE * cont) — higher
    # ROSTER_PERSISTENCE = more freezing. So "no roster smoothing" = 0.0.
    # The prompt says "ROSTER_PERSISTENCE=1.0 (no roster smoothing)" — that's wrong
    # given the code semantics. The right value is 0.0.
    bmr.ROSTER_PERSISTENCE = 0.0
    # Update brt to use the patched value indirectly (it imports via module ref)
    # Note: BuildRatingTimeline imports massey_ratings and _effective_weeks_ago
    # is module-level in bmr. Since the function reads ROSTER_PERSISTENCE at
    # call time, patching the module variable should suffice.

    # Reload BuildRatingTimeline to ensure it uses patched bmr funcs
    print(f"  Building with ROSTER_PERSISTENCE={bmr.ROSTER_PERSISTENCE}")

    all_games = brt.load_all_games()
    print(f"  Loaded {len(all_games)} games")

    for year in [2023, 2024, 2025, 2026]:
        print(f"  Building {year}...")
        checkpoints, match_events = brt.build_year_timeline(all_games, year, existing=None)
        out = {
            "year":         year,
            "lambda_decay": round(brt.LAMBDA_DECAY, 6),
            "checkpoints":  checkpoints,
            "match_events": match_events,
            "generated":    datetime.now().strftime("%Y-%m-%d"),
        }
        if year == 2026:
            fname = 'rating_timeline.json'
        else:
            fname = f'rating_timeline_{year}.json'
        with open(ALT_DIR / fname, 'w') as f:
            json.dump(out, f, separators=(',', ':'))
        print(f"    Saved {len(checkpoints)} cps, {len(match_events)} events")


# Step 2: load both timeline sets and build a checkpoint-lookup structure
def load_timeline_set(dirpath):
    """Load all rating_timeline files in a directory, return list of dicts."""
    files = [
        dirpath / 'rating_timeline.json',
        dirpath / 'rating_timeline_2023.json',
        dirpath / 'rating_timeline_2024.json',
        dirpath / 'rating_timeline_2025.json',
    ]
    all_cps = []
    all_me  = []
    for fp in files:
        d = json.load(open(fp))
        for cp in d['checkpoints']:
            all_cps.append((date.fromisoformat(cp['date']), cp['ratings']))
        all_me.extend(d.get('match_events', []))
    all_cps.sort(key=lambda x: x[0])
    return all_cps, all_me


def build_lookup(checkpoints):
    """Return (dates_sorted, ratings_list) for binary searching."""
    dates = [c[0] for c in checkpoints]
    ratings = [c[1] for c in checkpoints]
    return dates, ratings


def rating_at_or_before(team, target_date, dates, ratings):
    """Return team rating from most recent checkpoint <= target_date.
    Returns None if no such checkpoint exists or team not in it."""
    # binary search for largest dates[i] <= target_date
    import bisect
    i = bisect.bisect_right(dates, target_date) - 1
    while i >= 0:
        if team in ratings[i]:
            return ratings[i][team]
        i -= 1
    return None


# Step 3: build match list from timeline set with optional ensembling
def build_matches_from_timeline(dirpath, ensemble=False, ensemble_mode='arith'):
    """Mimics H.load_matches but pulls from a custom timeline directory and
    optionally applies ensembling to winner_before/loser_before.

    Ensembling: average winner_before/loser_before across snapshots at D, D-14, D-28.
    We use arithmetic average since ratings are already in logit-ish (round differential) space.
    """
    timeline_files = [
        dirpath / 'rating_timeline.json',
        dirpath / 'rating_timeline_2023.json',
        dirpath / 'rating_timeline_2024.json',
        dirpath / 'rating_timeline_2025.json',
    ]
    # First, load checkpoints for lookup if ensembling
    if ensemble:
        all_cps, _ = load_timeline_set(dirpath)
        cp_dates, cp_ratings = build_lookup(all_cps)
    # Use harness to load matches but with custom files
    matches = H.load_matches(timeline_files=timeline_files)

    if not ensemble:
        return matches

    # Apply ensembling: replace w_before/l_before with avg across {D, D-14, D-28}
    # Note: the *_before fields in match_events are already "pre-match" (the snapshot
    # at the same match day, which is the rating BEFORE adding today's games). To get
    # earlier snapshots, we look up at D-14 and D-28 in the checkpoint history.
    for m in matches:
        md = date.fromisoformat(m['date'])
        w_curr = m['w_before']
        l_curr = m['l_before']

        # D-14 and D-28: look up team ratings in checkpoint history
        # NOTE: those checkpoint ratings are "after that day's games", so they're
        # the team's rating AT END of D-14 / D-28. That's the closest leak-free
        # approximation to "snapshot at D-14 / D-28".
        d14 = md - timedelta(days=14)
        d28 = md - timedelta(days=28)

        w14 = rating_at_or_before(m['winner'], d14, cp_dates, cp_ratings)
        l14 = rating_at_or_before(m['loser'],  d14, cp_dates, cp_ratings)
        w28 = rating_at_or_before(m['winner'], d28, cp_dates, cp_ratings)
        l28 = rating_at_or_before(m['loser'],  d28, cp_dates, cp_ratings)

        # Fill missing snapshots with current
        w_ens_vals = [w_curr]
        l_ens_vals = [l_curr]
        if w14 is not None: w_ens_vals.append(w14)
        if l14 is not None: l_ens_vals.append(l14)
        if w28 is not None: w_ens_vals.append(w28)
        if l28 is not None: l_ens_vals.append(l28)

        # arithmetic average
        w_avg = sum(w_ens_vals) / len(w_ens_vals)
        l_avg = sum(l_ens_vals) / len(l_ens_vals)

        new_delta = w_avg - l_avg
        if new_delta == 0:
            # degenerate; keep original
            continue

        m['w_before'] = w_avg
        m['l_before'] = l_avg
        m['delta_before'] = new_delta
        m['abs_delta'] = abs(new_delta)

        # Re-pick fav/dog based on new delta
        if new_delta > 0:
            m['fav'] = m['winner']; m['dog'] = m['loser']; m['fav_won'] = True
        else:
            m['fav'] = m['loser'];  m['dog'] = m['winner']; m['fav_won'] = False
        m['fav_region'] = H.TEAM_REGIONS.get(m['fav'])
        m['dog_region'] = H.TEAM_REGIONS.get(m['dog'])

    return matches


def eval_config(matches, label):
    """Compute series Brier, Platt slope, return dict and per-match probs."""
    probs, outs = H.predict_series(matches)
    b = H.brier(probs, outs)
    a, slope = H.platt_slope(probs, outs)
    print(f"  {label}: brier={b:.6f} platt_b={slope:.4f} n={len(probs)}")
    return {'brier': b, 'platt_b': slope, 'probs': probs, 'outs': outs, 'n': len(probs)}


def main():
    t0 = time.time()

    # Build alt timelines if they don't exist yet
    needs_build = not all(
        (ALT_DIR / fn).exists()
        for fn in ['rating_timeline.json', 'rating_timeline_2023.json',
                   'rating_timeline_2024.json', 'rating_timeline_2025.json']
    )
    if needs_build:
        print("Building alt timelines with ROSTER_PERSISTENCE=0.0 (no roster smoothing)...")
        build_alt_timelines()
        print(f"  Built in {time.time()-t0:.1f}s")
    else:
        print("Alt timelines already exist, skipping rebuild.")

    # Config (iii) and (iv) use current timelines (ROSTER_PERSISTENCE=0.3)
    # Config (i) and (ii) use alt timelines (ROSTER_PERSISTENCE=0.0)

    print("\nLoading matches for each config...")
    m_i   = build_matches_from_timeline(ALT_DIR, ensemble=False)
    m_ii  = build_matches_from_timeline(ALT_DIR, ensemble=True)
    m_iii = build_matches_from_timeline(DATA,    ensemble=False)
    m_iv  = build_matches_from_timeline(DATA,    ensemble=True)

    print("\nEvaluating...")
    r_i   = eval_config(m_i,   'i_neither       ')
    r_ii  = eval_config(m_ii,  'ii_ensemble_only')
    r_iii = eval_config(m_iii, 'iii_roster_only ')
    r_iv  = eval_config(m_iv,  'iv_both         ')

    # Also test probability-space ensembling: average final series probs across 3 snapshots
    print("\nAlt: probability-space ensembling (avg probs across D, D-14, D-28)...")

    def predict_with_prob_ensemble(dirpath):
        timeline_files = [
            dirpath / 'rating_timeline.json',
            dirpath / 'rating_timeline_2023.json',
            dirpath / 'rating_timeline_2024.json',
            dirpath / 'rating_timeline_2025.json',
        ]
        all_cps, _ = load_timeline_set(dirpath)
        cp_dates, cp_ratings = build_lookup(all_cps)
        matches = H.load_matches(timeline_files=timeline_files)

        # Compute probs for each snapshot offset separately, then average
        snaps_probs = []
        snaps_outs  = []
        for offset in [0, 14, 28]:
            mats = []
            for m in matches:
                m2 = dict(m)
                md = date.fromisoformat(m['date'])
                target = md - timedelta(days=offset)
                w_r = rating_at_or_before(m['winner'], target, cp_dates, cp_ratings)
                l_r = rating_at_or_before(m['loser'],  target, cp_dates, cp_ratings)
                if w_r is None: w_r = m['w_before']
                if l_r is None: l_r = m['l_before']
                d2 = w_r - l_r
                if d2 == 0:
                    d2 = m['delta_before']  # keep original to avoid losing match
                m2['delta_before'] = d2
                m2['abs_delta'] = abs(d2)
                if d2 > 0:
                    m2['fav'] = m['winner']; m2['dog'] = m['loser']; m2['fav_won'] = True
                else:
                    m2['fav'] = m['loser'];  m2['dog'] = m['winner']; m2['fav_won'] = False
                m2['fav_region'] = H.TEAM_REGIONS.get(m2['fav'])
                m2['dog_region'] = H.TEAM_REGIONS.get(m2['dog'])
                mats.append(m2)
            probs, outs = H.predict_series(mats)
            # convert to "winner of original match" perspective
            probs_w = np.array([p if mats[i]['fav'] == matches[i]['winner'] else 1-p
                                for i, p in enumerate(probs)])
            snaps_probs.append(probs_w)
            snaps_outs = np.ones(len(probs_w))  # winner always won
        avg = np.mean(snaps_probs, axis=0)
        return matches, avg, snaps_outs

    # prob-space ensemble on top of roster smoothing (config iv-prob)
    m_iv_pp, p_iv_pp, o_iv_pp = predict_with_prob_ensemble(DATA)
    b_iv_pp = float(np.mean((p_iv_pp - o_iv_pp) ** 2))
    print(f"  iv_both_prob_avg : brier={b_iv_pp:.6f} n={len(p_iv_pp)}")
    m_ii_pp, p_ii_pp, o_ii_pp = predict_with_prob_ensemble(ALT_DIR)
    b_ii_pp = float(np.mean((p_ii_pp - o_ii_pp) ** 2))
    print(f"  ii_ens_prob_avg  : brier={b_ii_pp:.6f} n={len(p_ii_pp)}")

    # Align matches by match_id for paired bootstrap. The configs differ slightly
    # in count (some matches drop with delta_before==0 under different RP).
    # Build dict {match_id: (prob, out)} per config and intersect.
    def to_map(matches, probs, outs):
        return {m['match_id']: (probs[i], outs[i]) for i, m in enumerate(matches)}

    map_i   = to_map(m_i,   r_i['probs'],   r_i['outs'])
    map_ii  = to_map(m_ii,  r_ii['probs'],  r_ii['outs'])
    map_iii = to_map(m_iii, r_iii['probs'], r_iii['outs'])
    map_iv  = to_map(m_iv,  r_iv['probs'],  r_iv['outs'])

    common_ids = set(map_i) & set(map_ii) & set(map_iii) & set(map_iv)
    common_ids = sorted(common_ids)
    print(f"\nCommon match IDs across all configs: {len(common_ids)} (sizes: {len(map_i)},{len(map_ii)},{len(map_iii)},{len(map_iv)})")

    p_i   = np.array([map_i[mid][0]   for mid in common_ids])
    p_ii  = np.array([map_ii[mid][0]  for mid in common_ids])
    p_iii = np.array([map_iii[mid][0] for mid in common_ids])
    p_iv  = np.array([map_iv[mid][0]  for mid in common_ids])
    outs  = np.array([map_i[mid][1]   for mid in common_ids])

    # Identify best single (ii vs iii) on the COMMON set for fair comparison
    brier_ii_common  = H.brier(p_ii,  outs)
    brier_iii_common = H.brier(p_iii, outs)
    best_single = 'ii_ensemble_only' if brier_ii_common < brier_iii_common else 'iii_roster_only'
    best_probs = p_ii if best_single == 'ii_ensemble_only' else p_iii
    best_brier = min(brier_ii_common, brier_iii_common)

    # Paired bootstrap: does iv beat best_single?
    bs = H.paired_bootstrap_brier(best_probs, p_iv, outs, n_boot=2000, seed=42)
    p_value = bs['p_two_sided']
    brier_iv_common = H.brier(p_iv, outs)
    iv_better = brier_iv_common < best_brier

    print(f"\nOn common set ({len(common_ids)} matches):")
    print(f"  i_neither       : brier={H.brier(p_i, outs):.6f}")
    print(f"  ii_ensemble_only: brier={brier_ii_common:.6f}")
    print(f"  iii_roster_only : brier={brier_iii_common:.6f}")
    print(f"  iv_both         : brier={brier_iv_common:.6f}")
    print(f"Best single: {best_single} (brier={best_brier:.6f})")
    print(f"iv brier (common): {brier_iv_common:.6f}")
    print(f"Paired bootstrap iv vs best_single: mean_diff={bs['mean']:.6e} p={p_value:.4f}")
    print(f"iv better? {iv_better}")

    # Interpretation
    if iv_better and p_value < 0.10:
        interp = 'complementary'
        recommended = 'iv_both'
    else:
        interp = 'redundant'
        # Pick simpler of ii/iii
        # iii is "simpler" — no extra per-match snapshot lookups
        # But pick whichever has lower brier as well
        recommended = best_single

    # Also compare each config to baseline (i)
    bs_ii_vs_i  = H.paired_bootstrap_brier(p_i, p_ii,  outs, n_boot=1000, seed=1)
    bs_iii_vs_i = H.paired_bootstrap_brier(p_i, p_iii, outs, n_boot=1000, seed=2)
    bs_iv_vs_i  = H.paired_bootstrap_brier(p_i, p_iv,  outs, n_boot=1000, seed=3)

    out = {
        'configs': {
            'i_neither':        {'brier': r_i['brier'],   'platt_b': r_i['platt_b'],   'n': r_i['n']},
            'ii_ensemble_only': {'brier': r_ii['brier'],  'platt_b': r_ii['platt_b'],  'n': r_ii['n']},
            'iii_roster_only':  {'brier': r_iii['brier'], 'platt_b': r_iii['platt_b'], 'n': r_iii['n']},
            'iv_both':          {'brier': r_iv['brier'],  'platt_b': r_iv['platt_b'],  'n': r_iv['n']},
        },
        'best_single': best_single,
        'iv_vs_max_single_p': p_value,
        'iv_vs_max_single_mean_diff': bs['mean'],
        'iv_better_than_best_single': iv_better,
        'interpretation': interp,
        'recommended_config': recommended,
        'config_vs_baseline_i': {
            'ii_vs_i_mean_diff':  bs_ii_vs_i['mean'],
            'ii_vs_i_p':          bs_ii_vs_i['p_two_sided'],
            'iii_vs_i_mean_diff': bs_iii_vs_i['mean'],
            'iii_vs_i_p':         bs_iii_vs_i['p_two_sided'],
            'iv_vs_i_mean_diff':  bs_iv_vs_i['mean'],
            'iv_vs_i_p':          bs_iv_vs_i['p_two_sided'],
        },
        'note': (
            "Code semantic: effective_weeks = raw_weeks*(1-ROSTER_PERSISTENCE*cont). "
            "Higher value = more time-freezing = more smoothing. "
            "'No roster smoothing' configs (i, ii) use RP=0.0. "
            "'Roster smoothing' configs (iii, iv) use the repo default RP=0.3. "
            "Ensembling: arithmetic average of pre-match rating with team's rating "
            "at most-recent checkpoint <= D-14 and <= D-28 days. Missing snapshots fall back."
        ),
        'elapsed_sec': time.time() - t0,
        'prob_space_ensemble': {
            'iv_both_prob_avg_brier': b_iv_pp,
            'ii_ens_only_prob_avg_brier': b_ii_pp,
            'note': 'Probability-space average of series probs from snapshots at D, D-14, D-28. '
                    'Outs vector here is all-1s (winner perspective), so directly comparable '
                    'across the same match set but not to other configs (different perspective).',
        },
    }

    out_path = ART / 'time_smoothing.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")
    print(json.dumps({k: v for k, v in out.items() if k != 'configs'}, indent=2))
    print(json.dumps(out['configs'], indent=2))


if __name__ == '__main__':
    main()
