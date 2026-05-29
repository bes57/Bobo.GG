"""CN drag diagnostic.

For a set of CN teams, walk through 2025 day-by-day and report:
   raw rating, intl_w, c factor, shrunk rating, prior-pull = (1-c)*(prior - raw)

This proves whether the "CN ratings sag during domestic windows" pathology
is being driven by the time-decay of intl_w → c collapse → shrinkage toward
CN_PRIOR, separately from any actual rating change in the raw Massey solve.
"""

import os, sys, math
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scrapers'))

from BuildMapRatings import (
    load_games, massey_ratings, _compute_intl_weights,
    _compute_indirect_intl_w, _apply_cn_shrinkage,
    _compute_cn_personal_anchors, _compute_long_intl_weights,
    HALF_LIFE_WEEKS,
    CN_PRIOR, CN_INTL_K, CN_C_MIN, CN_INDIRECT_WEIGHT,
    CN_ANCHOR_BLEND_K,
    CN_TEAMS_SET, INTL_EVENTS, TEAM_REGIONS,
)

LAMBDA_DECAY = math.log(2) / HALF_LIFE_WEEKS
MIN_GAMES    = 5

# Probe these teams (top-tier CN in 2025)
PROBE_TEAMS = ['EDG', 'XLG', 'BLG', 'JDG']

# Subsample days — every ~14 days through 2025 + key event boundaries
def main():
    all_games = load_games()
    # Filter to 2024+ for prior connection + 2025 for trajectory
    year_games_25 = [g for g in all_games if g['date'].year == 2025]
    prior_games = [g for g in all_games if g['date'].year < 2025
                    and g.get('event_id') in INTL_EVENTS]

    days = sorted({g['date'].date() for g in year_games_25})
    if not days:
        print('No 2025 games found.')
        return

    # Sample every ~10 days
    probe_days = []
    last = None
    for d in days:
        if last is None or (d - last).days >= 10:
            probe_days.append(d)
            last = d
    # Ensure final day in
    if probe_days[-1] != days[-1]:
        probe_days.append(days[-1])

    print(f"Probing {len(probe_days)} days through 2025 (every ~10 days).\n")
    print(f"{'date':<12} {'team':<5} {'raw':>7} {'intl_w':>7} {'anchor':>7} "
          f"{'eff_a':>7} {'c':>5} {'shrunk':>7}")
    print('-' * 75)

    for d in probe_days:
        d_dt = datetime(d.year, d.month, d.day)
        solve_games = prior_games + [g for g in year_games_25 if g['date'].date() <= d]
        if not solve_games:
            continue

        raw = massey_ratings(solve_games, LAMBDA_DECAY, d_dt, MIN_GAMES)
        intl_w = _compute_intl_weights(solve_games, LAMBDA_DECAY, d_dt)
        indirect = _compute_indirect_intl_w(solve_games, intl_w, LAMBDA_DECAY, d_dt)
        anchors = _compute_cn_personal_anchors(solve_games, d_dt)
        long_iw = _compute_long_intl_weights(solve_games, d_dt)
        shrunk = _apply_cn_shrinkage(raw, intl_w, solve_games, LAMBDA_DECAY, d_dt)

        for t in PROBE_TEAMS:
            if t not in raw:
                continue
            r = raw[t]
            iw = intl_w.get(t, 0.0)
            ind = indirect.get(t, 0.0)
            evidence = iw + CN_INDIRECT_WEIGHT * ind
            c = max(CN_C_MIN, min(evidence / CN_INTL_K, 1.0))
            s = shrunk.get(t, r)
            anchor = anchors.get(t)
            if anchor is None:
                anchor_str = '  N/A  '
                eff_a = CN_PRIOR
            else:
                anchor_str = f"{anchor:>7.2f}"
                liw = long_iw.get(t, 0.0)
                c_anchor = min(liw / CN_ANCHOR_BLEND_K, 1.0)
                eff_a = c_anchor * anchor + (1 - c_anchor) * CN_PRIOR
            print(f"{d.isoformat():<12} {t:<5} {r:>7.2f} {iw:>7.2f} {anchor_str} "
                  f"{eff_a:>+7.2f} {c:>5.2f} {s:>7.2f}")
        print()


if __name__ == '__main__':
    main()
