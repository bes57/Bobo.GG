"""
Find optimal (indirect_weight, K) for the NO-FLOOR + saturating + indirect formula.

Goals:
  - Minimize Brier across years (focus on 2024-2026 weighted by sample size)
  - Preserve trophy ranks (avg ≤ 1.5)
  - Keep tested CN teams' c saturated at 1.0
"""
import os, sys, math, json, importlib, contextlib, io
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scrapers'))

from BacktestSeriesPredictions import load_match_data, load_snapshots, find_snapshot_for, brier
from CNIndirectEvidence import (rebuild_with_custom_shrinkage, rebuild_baseline,
                                  CN_SET, predict_series, gen_series)


def evaluate(matches, label, **rebuild_kwargs):
    rebuild_with_custom_shrinkage(**rebuild_kwargs)
    snaps = load_snapshots()
    with open(f'{ROOT}/data/map_ratings.json') as f:
        mr = json.load(f)
    teams26 = mr['ratings']['2026']['snapshots']['after_stage1']['teams']

    cn = sorted([(o, t['overall_rating']) for o, t in teams26.items() if o in CN_SET],
                key=lambda x: -x[1])
    # Pick key teams
    pick = {o: r for o, r in cn}
    edg = pick.get('EDG', 0); xlg = pick.get('XLG', 0); ag = pick.get('AG', 0)
    drg = pick.get('DRG', 0); blg = pick.get('BLG', 0); tyl = pick.get('TYL', 0)
    nova = pick.get('NOVA', 0); fpx = pick.get('FPX', 0)

    yearly = {}
    for yr in ['2024','2025','2026']:
        m = matches[matches['date'].str.startswith(yr)].reset_index(drop=True)
        p, y = gen_series(m, snaps)
        yearly[yr] = float(brier(p, y)) if len(p) >= 30 else None
    p_all, y_all = gen_series(matches, snaps)
    full = float(brier(p_all, y_all))

    SNAPS_T = [(2024,'after_champions','EDG'),(2025,'after_champions','NRG'),(2026,'after_santiago','NS')]
    ranks = []
    for year, snap_key, winner in SNAPS_T:
        s = mr['ratings'][str(year)]['snapshots'].get(snap_key, {}).get('teams', {})
        items = sorted(s.items(), key=lambda x:-x[1]['overall_rating'])
        rk = next((i+1 for i,(t,_) in enumerate(items) if t==winner), 50)
        ranks.append(rk)
    trophy_avg = sum(ranks)/3

    return {
        'label': label,
        'edg': edg, 'xlg': xlg, 'ag': ag,
        'drg': drg, 'blg': blg, 'tyl': tyl, 'fpx': fpx, 'nova': nova,
        'b24': yearly['2024'], 'b25': yearly['2025'], 'b26': yearly['2026'],
        'full': full, 'trophy': trophy_avg,
    }


def main():
    matches = load_match_data()
    print(f"Loaded {len(matches)} series\n")

    print("BASELINE deployed:")
    rebuild_baseline()
    snaps = load_snapshots()
    with open(f'{ROOT}/data/map_ratings.json') as f:
        mr = json.load(f)
    p_all, y_all = gen_series(matches, snaps)
    print(f"  Brier full: {brier(p_all, y_all):.5f}")
    print()

    # === Sweep 1: indirect_weight, K=2.0, no floor ===
    print("="*120)
    print("Sweep 1: indirect_weight × K=2.0, no floor")
    print("="*120)
    print(f"  {'iw':>5}  {'K':>4}  {'EDG':>5}  {'XLG':>5}  {'AG':>5}  {'DRG':>5}  {'BLG':>5}  {'TYL':>5}  "
          f"{'FPX':>5}  {'NOVA':>5}  {'B24':>8}  {'B25':>8}  {'B26':>8}  {'Full':>8}  {'Tr':>3}")
    iws = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.70, 1.0, 1.5, 2.0, 3.0]
    results = []
    for iw in iws:
        r = evaluate(matches, f"iw={iw}", indirect_weight=iw, c_min=0.0, K=2.0)
        results.append(r)
        print(f"  {iw:>5.2f}  {2.0:>4}  {r['edg']:>+5.2f}  {r['xlg']:>+5.2f}  {r['ag']:>+5.2f}  "
              f"{r['drg']:>+5.2f}  {r['blg']:>+5.2f}  {r['tyl']:>+5.2f}  {r['fpx']:>+5.2f}  {r['nova']:>+5.2f}  "
              f"{r['b24']:.5f}  {r['b25']:.5f}  {r['b26']:.5f}  {r['full']:.5f}  {r['trophy']:>3.1f}",
              flush=True)

    # === Sweep 2: K variations at best iw range ===
    print("\n" + "="*120)
    print("Sweep 2: K variations at promising iw values, no floor")
    print("="*120)
    print(f"  {'iw':>5}  {'K':>4}  {'EDG':>5}  {'XLG':>5}  {'AG':>5}  {'DRG':>5}  {'BLG':>5}  {'TYL':>5}  "
          f"{'FPX':>5}  {'NOVA':>5}  {'B24':>8}  {'B25':>8}  {'B26':>8}  {'Full':>8}  {'Tr':>3}")
    for K in [1.0, 1.5, 2.0, 2.5, 3.0]:
        for iw in [0.20, 0.50, 1.0]:
            r = evaluate(matches, f"K={K} iw={iw}", indirect_weight=iw, c_min=0.0, K=K)
            print(f"  {iw:>5.2f}  {K:>4}  {r['edg']:>+5.2f}  {r['xlg']:>+5.2f}  {r['ag']:>+5.2f}  "
                  f"{r['drg']:>+5.2f}  {r['blg']:>+5.2f}  {r['tyl']:>+5.2f}  {r['fpx']:>+5.2f}  {r['nova']:>+5.2f}  "
                  f"{r['b24']:.5f}  {r['b25']:.5f}  {r['b26']:.5f}  {r['full']:.5f}  {r['trophy']:>3.1f}",
                  flush=True)

    # === Pareto analysis ===
    print("\n" + "="*120)
    print("PARETO FRONTIER (from sweep 1): trade Brier vs intuition match")
    print("="*120)
    print(f"  {'iw':>5}  {'Brier':>8}  {'Δ deployed':>11}  {'BLG':>5}  {'DRG':>5}  {'NOVA':>5}  {'Verdict':<20}")
    DEPLOYED_BRIER = 0.23163
    for r in results:
        diff = (r['full'] - DEPLOYED_BRIER) * 10000
        # Verdict logic
        if r['blg'] > -2.0 and r['nova'] < -4.0 and r['trophy'] <= 1.5:
            verdict = "✓ matches intuition"
        elif r['blg'] > -1.0:
            verdict = "BLG too high"
        elif r['nova'] > -3.5:
            verdict = "NOVA too high"
        else:
            verdict = ""
        iw = float(r['label'].split('=')[1])
        print(f"  {iw:>5.2f}  {r['full']:.5f}  {diff:>+11.1f}bp  {r['blg']:>+5.2f}  {r['drg']:>+5.2f}  {r['nova']:>+5.2f}  {verdict:<20}")

    # Recommend best
    print("\n" + "="*120)
    print("FINAL RECOMMENDATION")
    print("="*120)
    valid = [r for r in results if r['trophy'] <= 1.5]
    # Best by full Brier among valid
    best = min(valid, key=lambda r: r['full'])
    iw_best = float(best['label'].split('=')[1])
    print(f"  Best Brier (trophy ≤ 1.5): iw={iw_best}, Brier={best['full']:.5f}")
    # Best where BLG rises into [-2, -0.5] AND NOVA stays below -4
    intuition = [r for r in results if -2.0 < r['blg'] < -0.5 and r['nova'] < -4.0]
    if intuition:
        best_int = min(intuition, key=lambda r: r['full'])
        iw_int = float(best_int['label'].split('=')[1])
        print(f"  Best matching intuition (BLG in [-2,-0.5], NOVA<-4): iw={iw_int}, Brier={best_int['full']:.5f}")

    print(f"\n━━━ Restoring deployed ━━━")
    rebuild_baseline()
    print("Done.")


if __name__ == '__main__':
    main()
