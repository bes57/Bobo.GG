"""Prospective prediction logger — run any time (idempotent). For every
upcoming match the site knows about, log the candidate model's probability
alongside a timestamp, so future model-vs-market evaluation is prospective
(no reconstruction). Appends to out/prediction_log.csv; a match is re-logged
only if its stored probability moved by >1pt (ratings updated).

Suggested automation: run every few hours via cron/launchd, or ask Claude to
schedule it as a routine.
"""
import csv
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from engine import Engine

OUT = os.path.join(HERE, "out")
LOG = os.path.join(OUT, "prediction_log.csv")
DATA = os.path.join(HERE, "..", "data")

up_path = os.path.join(DATA, "upcoming_matches.json")
if not os.path.exists(up_path):
    print("no upcoming_matches.json; nothing to log")
    sys.exit(0)
up = json.load(open(up_path))
matches = up if isinstance(up, list) else up.get("matches", [])
if not matches:
    print("no upcoming matches")
    sys.exit(0)

eng = Engine()
s = eng.series.reset_index(drop=True)
stage_by_mid = dict(zip(s.match_id, s.stage))
g_stage = np.array([stage_by_mid.get(g["match_id"], "groups") for g in eng.games])
PO = np.where(np.isin(g_stage, ("playoffs", "grand_final")), 1.6, 1.0)
out = eng.run({"decay": {"kind": "games", "hl_games": 20.0, "hl_games_loss": 12.0},
               "rd": {"power": 0.75, "scale": 2.5}, "roster_mode": "year",
               "roster_persistence": 0.3, "ridge": 0.5, "champ_mult": 2.0,
               "w_custom": PO, "daily_out": True})
daily = out["daily_r"]
latest = daily[sorted(daily.keys())[-1]]
BETA = 0.103


def p_series(rd, fmt):
    pm = 1 / (1 + np.exp(-BETA * rd))
    if fmt in ("bo5", "bo5_gf"):
        return pm ** 3 * (1 + 3 * (1 - pm) + 6 * (1 - pm) ** 2)
    if fmt == "bo1":
        return pm
    return pm ** 2 * (3 - 2 * pm)


existing = {}
if os.path.exists(LOG):
    with open(LOG) as f:
        for row in csv.DictReader(f):
            existing[(row["team_a"], row["team_b"], row["start"])] = float(row["p_a"])

new_rows = []
now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
for mch in matches:
    a = mch.get("team1") or mch.get("org_a") or mch.get("team_a")
    b = mch.get("team2") or mch.get("org_b") or mch.get("team_b")
    start = str(mch.get("time") or mch.get("start") or mch.get("date") or "")
    fmt = str(mch.get("format") or "bo3").lower()
    if not a or not b or a not in eng.tidx or b not in eng.tidx:
        continue
    rd = float(latest[eng.tidx[a]] - latest[eng.tidx[b]])
    p_a = float(p_series(rd, fmt))
    key = (a, b, start)
    if key in existing and abs(existing[key] - p_a) < 0.01:
        continue
    new_rows.append({"logged_utc": now, "team_a": a, "team_b": b,
                     "start": start, "fmt": fmt, "p_a": round(p_a, 4),
                     "model": "cand_asym_w20l12_stack", "beta": BETA})

if new_rows:
    write_header = not os.path.exists(LOG)
    with open(LOG, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(new_rows[0].keys()))
        if write_header:
            w.writeheader()
        w.writerows(new_rows)
print(f"logged {len(new_rows)} new/updated predictions "
      f"({len(matches)} upcoming known)")
