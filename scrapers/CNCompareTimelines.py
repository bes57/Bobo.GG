"""Compare v7 (CN_PRIOR pull) vs v8 (personal anchor) timelines for CN teams.

Reads /tmp/rating_timeline_2025_v7.json and data/rating_timeline_2025.json,
prints the CN trajectory side-by-side at key dates.
"""

import json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Snapshot v7 first if not present
V7_PATH = '/tmp/rating_timeline_2025_v7.json'
V8_PATH = os.path.join(ROOT, 'data', 'rating_timeline_2025.json')

with open(V8_PATH) as f:
    v8 = json.load(f)

# Use v7 snapshot if available, else compute peak/trough comparison just from v8
try:
    with open(V7_PATH) as f:
        v7 = json.load(f)
    have_v7 = True
except FileNotFoundError:
    have_v7 = False

PROBE = ['EDG', 'BLG', 'XLG', 'TYL', 'TE', 'JDG', 'DRG', 'WOL', 'AG', 'FPX', 'NOVA']

# Key event-window endpoints in 2025
PROBE_DATES = [
    ('2025-02-09', 'after CN Kickoff'),
    ('2025-03-02', 'after Bangkok'),
    ('2025-03-22', 'Stage 1 start'),
    ('2025-05-04', 'after CN Stage 1'),
    ('2025-05-18', 'after Pacific Stage 1'),
    ('2025-06-22', 'after Toronto'),
    ('2025-08-24', 'after CN Stage 2'),
    ('2025-09-21', 'mid-Champions'),
    ('2025-10-05', 'after Champions'),
]

def lookup(timeline, target_date):
    """Find checkpoint on-or-before target_date."""
    best = None
    for cp in timeline['checkpoints']:
        if cp['date'] <= target_date:
            best = cp
    return best

print(f"{'Date':<14} {'Label':<26} {'Team':<5} {'v7':>7} {'v8':>7} {'Δ':>7}")
print('-' * 75)
for date_str, label in PROBE_DATES:
    cp_v8 = lookup(v8, date_str)
    cp_v7 = lookup(v7, date_str) if have_v7 else None
    if cp_v8 is None:
        continue
    for t in PROBE:
        r8 = cp_v8['ratings'].get(t)
        r7 = cp_v7['ratings'].get(t) if cp_v7 else None
        if r8 is None and r7 is None:
            continue
        r7s = f"{r7:>7.2f}" if r7 is not None else '   N/A '
        r8s = f"{r8:>7.2f}" if r8 is not None else '   N/A '
        delta = (r8 - r7) if (r7 is not None and r8 is not None) else None
        ds  = f"{delta:>+7.2f}" if delta is not None else '       '
        print(f"{date_str:<14} {label:<26} {t:<5} {r7s} {r8s} {ds}")
    print()
