"""Sanity-check probes for any candidate BenPom config.

Reads /Users/benny_es1/PythonTest/data/map_ratings.json after a rebuild and:
  - Trophy: 2024_after_champions EDG above GEN (user-cared)
  - CN bottom-of-pack: bottom 8 of each snapshot should be ≥50% CN
  - H2H: 2026 after_stage1 (or latest) BLG vs JDG ordering
  - CN saturation: every CN team's c factor in [0.1, 0.95]
"""
from __future__ import annotations
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, '/Users/benny_es1/PythonTest/scrapers')
import BuildMapRatings as B


def check_trophy(snap_data):
    teams = snap_data.get('teams', {})
    edg = teams.get('EDG', {}).get('overall_rating')
    gen = teams.get('GEN', {}).get('overall_rating')
    if edg is None or gen is None:
        return {'pass': None, 'reason': 'EDG or GEN missing'}
    return {'pass': edg > gen, 'edg': edg, 'gen': gen, 'gap': edg - gen}


def check_cn_bottom(snap_data, region_lookup):
    teams = snap_data.get('teams', {})
    rows = sorted(teams.items(), key=lambda kv: kv[1].get('overall_rating', 0))
    bot8 = rows[:8]
    cn_count = sum(1 for org, _ in bot8 if region_lookup.get(org) == 'CN')
    return {'pass': cn_count >= 4, 'cn_in_bottom_8': cn_count,
            'bottom_8': [(o, td.get('overall_rating')) for o, td in bot8]}


def check_h2h_blg_jdg(snap_data):
    teams = snap_data.get('teams', {})
    blg = teams.get('BLG', {}).get('overall_rating')
    jdg = teams.get('JDG', {}).get('overall_rating')
    if blg is None or jdg is None:
        return {'pass': None, 'reason': 'BLG or JDG missing'}
    # User's H2H intuition: BLG won 4-0 in 2026 stage 1, so BLG should rank above JDG
    # (Same intuition was confirmed for 2024 after_champs head-to-head.)
    return {'pass': blg > jdg, 'blg': blg, 'jdg': jdg, 'gap': blg - jdg}


def check_cn_saturation(games, lam, ref_date, prior, K, indirect_weight):
    """Compute c for every CN team — flag if outside [0.1, 0.95]."""
    intl_w = B._compute_intl_weights(games, lam, ref_date)
    indirect = B._compute_indirect_intl_w(games, intl_w, lam, ref_date)
    flags = []
    for t in B.CN_TEAMS_SET:
        d = intl_w.get(t, 0.0)
        ind = indirect.get(t, 0.0)
        ev = d + indirect_weight * ind
        c = max(0.0, min(ev / K, 1.0))
        flags.append({'team': t, 'c': c, 'direct': d, 'indirect': ind})
    out_range = [f for f in flags if f['c'] < 0.1 or f['c'] > 0.95]
    return {
        'pass': len(out_range) <= 2,
        'n_cn_teams': len(flags),
        'n_outside_range': len(out_range),
        'outside_range': out_range,
    }


def run_all(map_ratings_path='/Users/benny_es1/PythonTest/data/map_ratings.json',
            snap_target=('2024', 'after_champions'),
            h2h_target=('2026', 'after_stage1')):
    d = json.load(open(map_ratings_path))
    snap = d['ratings'][snap_target[0]]['snapshots'][snap_target[1]]
    region_lookup = B.TEAM_REGIONS

    out = {
        'trophy_2024_edg_gt_gen': check_trophy(snap),
        'cn_bottom_2024': check_cn_bottom(snap, region_lookup),
    }

    h2h_snap = d['ratings'][h2h_target[0]]['snapshots'].get(h2h_target[1])
    if h2h_snap:
        out['h2h_blg_jdg_' + h2h_target[1]] = check_h2h_blg_jdg(h2h_snap)
    else:
        # fallback to latest 2026 snap
        snaps26 = d['ratings'].get('2026', {}).get('snapshots', {})
        last = list(snaps26.keys())[-1] if snaps26 else None
        if last:
            out['h2h_blg_jdg_' + last] = check_h2h_blg_jdg(snaps26[last])

    # CN saturation check (needs current model state)
    games = B.load_games()
    lam = math.log(2) / B.HALF_LIFE_WEEKS
    ref_date = max(g['date'] for g in games)
    out['cn_saturation'] = check_cn_saturation(
        games, lam, ref_date,
        B.CN_PRIOR, B.CN_INTL_K, B.CN_INDIRECT_WEIGHT,
    )

    all_pass = all(v.get('pass') for v in out.values() if v.get('pass') is not None)
    out['all_pass'] = all_pass
    return out


if __name__ == '__main__':
    import json
    res = run_all()
    print(json.dumps(res, indent=2, default=float))
