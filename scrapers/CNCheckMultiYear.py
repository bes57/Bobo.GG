"""Check CN-team trajectories across 2024 and 2025 from rating_timeline_<year>.json."""

import json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

for year in (2024, 2025):
    path = os.path.join(ROOT, 'data', f'rating_timeline_{year}.json')
    if not os.path.exists(path):
        continue
    with open(path) as f:
        data = json.load(f)

    # Get final checkpoint
    final = data['checkpoints'][-1]
    print(f"\n=== {year} final ({final['date']}) — CN teams ===")
    cn_teams = ['EDG', 'BLG', 'XLG', 'TYL', 'TE', 'JDG', 'DRG', 'WOL', 'AG', 'FPX', 'NOVA', 'TEC']
    for t in sorted(cn_teams, key=lambda x: -final['ratings'].get(x, -99)):
        r = final['ratings'].get(t)
        if r is not None:
            print(f"  {t:<5}  {r:>+6.2f}")

    # Top 10 overall
    sorted_teams = sorted(final['ratings'].items(), key=lambda x: -x[1])
    print(f"\n  Top 10 overall:")
    for t, r in sorted_teams[:10]:
        print(f"  {t:<5}  {r:>+6.2f}")

    # Show CN team trajectory peaks/troughs through year
    print(f"\n  Top CN team trajectory (max - min during {year}):")
    for t in ['EDG', 'BLG', 'XLG']:
        vals = [cp['ratings'].get(t) for cp in data['checkpoints'] if cp['ratings'].get(t) is not None]
        if vals:
            print(f"    {t}: min={min(vals):+.2f}  max={max(vals):+.2f}  range={max(vals)-min(vals):.2f}  final={vals[-1]:+.2f}")
