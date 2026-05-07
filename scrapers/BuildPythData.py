"""
Build data/pyth_data.json — pre-computed Pythagorean ratings payload.
Run after ScrapeMatchData.py and BuildMatchResults.py.
Usage: python scrapers/BuildPythData.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from MapElo import _compute_pyth_data

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'pyth_data.json')

print("Computing Pythagorean data...")
data = _compute_pyth_data()
with open(OUT, 'w') as f:
    json.dump(data, f)
print(f"Written to {OUT} ({os.path.getsize(OUT) / 1024:.1f} KB)")
