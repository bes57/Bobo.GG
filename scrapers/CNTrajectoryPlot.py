"""Plot CN team trajectories from rating_timeline_2025.json.

Just text output — date, team, rating at probe dates.
"""

import json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TL = os.path.join(ROOT, 'data', 'rating_timeline_2025.json')

with open(TL) as f:
    data = json.load(f)

PROBE = ['EDG', 'BLG', 'XLG', 'TYL', 'TE', 'JDG', 'DRG', 'WOL', 'AG', 'FPX', 'NOVA']

# All checkpoint dates, sample every ~10 days
all_dates = [cp['date'] for cp in data['checkpoints']]
sample = []
last = None
for d in all_dates:
    if last is None or (d > last and abs((int(d.split('-')[1])*30 + int(d.split('-')[2])) -
                                          (int(last.split('-')[1])*30 + int(last.split('-')[2]))) >= 10):
        sample.append(d)
        last = d
if all_dates[-1] not in sample:
    sample.append(all_dates[-1])

# Build lookup
by_date = {cp['date']: cp['ratings'] for cp in data['checkpoints']}

# Print one row per probe date, columns = teams
header = f"{'date':<12}" + ''.join(f" {t:>6}" for t in PROBE)
print(header)
print('-' * len(header))
for d in sample:
    row = by_date[d]
    line = f"{d:<12}"
    for t in PROBE:
        r = row.get(t)
        if r is None:
            line += '   N/A '
        else:
            line += f" {r:>+6.2f}"
    print(line)
