"""
CN shrinkage refinement — the c_min=0.5 floor is binding for intl-tested teams
(EDG, XLG with intl_w ≈ 2) because CN_INTL_K=4.0 requires more weight than
they have. Test fixes:

  (A) Lower CN_INTL_K (4 → 1.5 or 2): intl-attendees escape faster
  (B) Lower c_min (0.5 → 0.3): less aggressive floor
  (C) Combined: K=2, c_min=0.4

For each option, measure:
  - EDG rating at each 2026 snapshot (the user's case)
  - All-CN ratings at each snapshot (make sure non-intl teams stay low)
  - Trophy winner ranks (preserved)
  - Brier on full + per-year
  - 2024 and 2025 CN snapshots (does it break historical years?)
"""
import os, sys, math, json, importlib, contextlib, io
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scrapers'))

from BacktestSeriesPredictions import (
    load_match_data, load_snapshots, find_snapshot_for, brier
)

DEPLOYED = dict(
    RD_TRANSFORM='power', RD_POWER=0.5, RD_SCALE=2.5,
    HALF_LIFE_WEEKS=6.0, ROSTER_PERSISTENCE=0.3,
    INTL_WIN_MULT=1.0, INTL_LOSS_MULT=1.0, CHAMPIONS_MULT=2.0,
    CN_PRIOR=-4.0, CN_INTL_K=4.0, CN_C_MIN=0.5,
    REGION_SPILLOVER_ALPHA=0.5, SHRINK_K=5,
)
BETA = 0.136


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


def rebuild(**overrides):
    saved_argv = sys.argv[:]
    sys.argv = ['BuildMapRatings.py']
    try:
        import BuildMapRatings
        importlib.reload(BuildMapRatings)
        cfg = dict(DEPLOYED, **overrides)
        for k, v in cfg.items():
            setattr(BuildMapRatings, k, v)
        # Critical: patch _apply_cn_shrinkage defaults (CN_PRIOR, CN_INTL_K, CN_C_MIN
        # are default args — bound at module load, not call time)
        BuildMapRatings._apply_cn_shrinkage.__defaults__ = (
            cfg['CN_PRIOR'], cfg['CN_INTL_K'], cfg['CN_C_MIN']
        )
        with contextlib.redirect_stdout(io.StringIO()):
            BuildMapRatings.main()
    finally:
        sys.argv = saved_argv


def evaluate(matches, **overrides):
    rebuild(**overrides)
    snaps = load_snapshots()
    with open(os.path.join(ROOT, 'data/map_ratings.json')) as f:
        mr = json.load(f)

    # CN ratings at each interesting snapshot
    CN_ORGS = {'EDG','BLG','TE','DRG','ASE','AG','XLG','WOL','FPX','JDG','NOVA','TEC','TYL','TYLOO'}
    cn_snaps = {}
    for (yr, snap) in [('2024','after_champions'), ('2025','after_champions'),
                       ('2026','after_santiago'), ('2026','after_stage1')]:
        teams = mr['ratings'][yr]['snapshots'].get(snap, {}).get('teams', {})
        cn_ratings = sorted([(o, t['overall_rating']) for o, t in teams.items() if o in CN_ORGS],
                            key=lambda x: -x[1])
        cn_snaps[f'{yr}_{snap}'] = cn_ratings

    # Top/bot of overall
    teams26 = mr['ratings']['2026']['snapshots']['after_santiago']['teams']
    top_26 = max(t['overall_rating'] for t in teams26.values())
    bot_26 = min(t['overall_rating'] for t in teams26.values())

    # Trophy ranks
    SNAPS = [(2024,'after_madrid','SEN'),(2024,'after_shanghai','GEN'),
             (2024,'after_champions','EDG'),(2025,'after_bangkok','T1'),
             (2025,'after_toronto','PRX'),(2025,'after_champions','NRG'),
             (2026,'after_santiago','NS')]
    trophy_rks = []
    for year, snap, winner in SNAPS:
        s = mr['ratings'][str(year)]['snapshots'].get(snap, {}).get('teams', {})
        if not s: trophy_rks.append(50); continue
        items = sorted(s.items(), key=lambda x:-x[1]['overall_rating'])
        rk = next((i+1 for i,(t,_) in enumerate(items) if t==winner), 50)
        trophy_rks.append(rk)

    # Brier per year
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

    return {
        'cn_snaps': cn_snaps, 'top_26': top_26, 'bot_26': bot_26,
        'trophy_rks': trophy_rks, 'trophy_avg': float(np.mean(trophy_rks)),
        'yearly': yearly, 'full_brier': full_brier,
    }


def fmt_cn_2026(r):
    """Show the after_stage1 + after_santiago CN ratings (current focus)."""
    s1 = r['cn_snaps'].get('2026_after_stage1', [])
    sa = r['cn_snaps'].get('2026_after_santiago', [])
    s1_str = ', '.join(f'{org}={rt:+.2f}' for org, rt in s1)
    sa_str = ', '.join(f'{org}={rt:+.2f}' for org, rt in sa)
    return s1_str, sa_str


def main():
    matches = load_match_data()
    print(f"Loaded {len(matches)} series\n")

    print("="*100)
    print("BASELINE (deployed)")
    print("="*100)
    base = evaluate(matches)
    s1_str, sa_str = fmt_cn_2026(base)
    print(f"  CN after_stage1: {s1_str}")
    print(f"  CN after_santiago: {sa_str}")
    print(f"  Trophy ranks: {base['trophy_rks']} (avg {base['trophy_avg']:.2f})")
    print(f"  Brier per year: 2024={base['yearly']['2024']:.5f} "
          f"2025={base['yearly']['2025']:.5f} 2026={base['yearly']['2026']:.5f}")
    print(f"  Brier full: {base['full_brier']:.5f}")

    options = [
        ('K=2.0',  {'CN_INTL_K': 2.0}),
        ('K=1.5',  {'CN_INTL_K': 1.5}),
        ('K=1.0',  {'CN_INTL_K': 1.0}),
        ('K=2.0 + c_min=0.4', {'CN_INTL_K': 2.0, 'CN_C_MIN': 0.4}),
        ('K=1.5 + c_min=0.4', {'CN_INTL_K': 1.5, 'CN_C_MIN': 0.4}),
        ('K=2.0 + c_min=0.3', {'CN_INTL_K': 2.0, 'CN_C_MIN': 0.3}),
        ('CN_PRIOR=-3.0',     {'CN_PRIOR': -3.0}),
        ('K=2.0 + PRIOR=-3.5',{'CN_INTL_K': 2.0, 'CN_PRIOR': -3.5}),
    ]

    print("\n" + "="*100)
    print("OPTIONS")
    print("="*100)
    results = [('DEPLOYED', base)]
    for label, overrides in options:
        r = evaluate(matches, **overrides)
        results.append((label, r))
        s1_str, sa_str = fmt_cn_2026(r)
        # Highlight EDG specifically
        edg_stage1 = next((rt for org, rt in r['cn_snaps'].get('2026_after_stage1', []) if org == 'EDG'), None)
        edg_santiago = next((rt for org, rt in r['cn_snaps'].get('2026_after_santiago', []) if org == 'EDG'), None)
        nova_stage1 = next((rt for org, rt in r['cn_snaps'].get('2026_after_stage1', []) if org == 'NOVA'), None)
        print(f"\n  {label}")
        print(f"     EDG: stage1={edg_stage1:+.2f}  santiago={edg_santiago:+.2f}    NOVA stage1={nova_stage1:+.2f}")
        print(f"     CN after_stage1: {s1_str}")
        print(f"     Brier 2024={r['yearly']['2024']:.5f}  2025={r['yearly']['2025']:.5f}  2026={r['yearly']['2026']:.5f}  "
              f"full={r['full_brier']:.5f}")
        print(f"     Trophy avg: {r['trophy_avg']:.2f}")

    print(f"\n━━━ Restoring deployed config ━━━")
    rebuild()
    print("Done.")


if __name__ == '__main__':
    main()
