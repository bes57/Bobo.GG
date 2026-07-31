"""agent:bias-h3 — full-corpus recompute of matches/games_since_change for 5d typing.

Definitions mirror preregister.lineups.md verbatim:
  L(org, match) = set of ProfileURLs with >=1 player-map row for (Org, MatchID)
                  (engine.load_match_lineups grouping), union over maps.
  n_maps = distinct numeric MapNum for the match in its maps CSV.
  date = match_dates.json (day granularity); org sequence sorted (date, match_id);
  history = strictly earlier DATE (same-day matches are never history).
  matches_since_change = # consecutive tail matches (strictly earlier) with L equal
  to the current match's L; games_since_change = sum of n_maps over those matches.

Two-mode verification (preregister pre-run amendment):
  (i) restricted-history mode (lineups.csv event universe) must reproduce the
      lineups agent's values on ALL overlap rows with 0 mismatches;
  (ii) full-corpus mode = the typing input; overlap differences counted+explained.

Writes scratch/bias_h3/lineup_topup.csv + lineup_topup_verify.json. Scratch only.
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
TL = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
V8 = os.path.join(TL, "v8")
ROOT = os.path.dirname(TL)
DATA = os.path.join(ROOT, "data")
sys.path.insert(0, TL)

OUT_CSV = os.path.join(HERE, "lineup_topup.csv")
OUT_VER = os.path.join(HERE, "lineup_topup_verify.json")
if os.path.exists(OUT_CSV) and os.path.exists(OUT_VER):
    print("checkpoint exists, skipping", flush=True)
    sys.exit(0)

from engine import Engine  # noqa: E402

eng = Engine()
match_dates = json.load(open(os.path.join(DATA, "match_dates.json")))

# (org, match_id) -> lineup, from the engine's own loader (the definition source)
lineups = eng.lineups
# n_maps per match: numeric map rows in eng.games (identical to distinct numeric
# MapNum — one game row per map)
n_maps = defaultdict(int)
g_event = {}
g_date = {}
for g in eng.games:
    n_maps[g["match_id"]] += 1
    g_event[g["match_id"]] = g["event_id"]
    g_date[g["match_id"]] = g["date_s"]

# org -> [(date, match_id)] over the full corpus
seq_full = defaultdict(list)
for (org, mid) in lineups:
    if mid in g_date:  # matches present in the engine game list only
        seq_full[org].append((g_date[mid], mid))
for org in seq_full:
    seq_full[org].sort()


def msc_for(org, rows):
    """rows = [(date, mid)] sorted. Returns {mid: (msc, gsc, first_date)}."""
    out = {}
    for i, (d, mid) in enumerate(rows):
        L = lineups.get((org, mid))
        msc = 0
        gsc = 0
        j = i - 1
        while j >= 0:
            dj, mj = rows[j]
            if dj >= d:            # strictly-earlier DATE only
                j -= 1
                continue
            if lineups.get((org, mj)) == L and L is not None:
                msc += 1
                gsc += n_maps[mj]
                j -= 1
            else:
                break
        out[mid] = (msc, gsc)
    return out


def build(rows_by_org):
    recs = []
    for org, rows in rows_by_org.items():
        vals = msc_for(org, rows)
        first = rows[0][0] if rows else None
        for d, mid in rows:
            msc, gsc = vals[mid]
            recs.append({"match_id": mid, "org": org, "date": d,
                         "event_id": g_event[mid], "n_maps": n_maps[mid],
                         "matches_since_change": msc,
                         "games_since_change": gsc,
                         "org_first_date": first})
    return pd.DataFrame(recs).sort_values(["date", "match_id", "org"]).reset_index(drop=True)


full = build(seq_full)

# ── verification mode (i): restrict history to the lineups agent's universe ──
lu = pd.read_csv(os.path.join(V8, "data", "lineups.csv"))
lf = pd.read_csv(os.path.join(V8, "data", "lineup_features.csv"))
their_events = set(lu.event_id)
seq_restr = defaultdict(list)
for org, rows in seq_full.items():
    seq_restr[org] = [(d, m) for (d, m) in rows if g_event[m] in their_events]
restr = build({o: r for o, r in seq_restr.items() if r})

ours = restr.set_index(["match_id", "org"])
theirs = lf.set_index(["match_id", "org"])
common = ours.index.intersection(theirs.index)
mism = []
for k in common:
    a = ours.loc[k]
    b = theirs.loc[k]
    bm = b["matches_since_change"]
    bg = b["games_since_change"]
    # their NaN (org_debut rows have msc=NaN? spec says 0 if no prior) — compare
    # with NaN treated as 0 ONLY if their column is NaN and ours is 0
    am, ag = int(a["matches_since_change"]), int(a["games_since_change"])
    bm0 = 0 if pd.isna(bm) else int(bm)
    bg0 = 0 if pd.isna(bg) else int(bg)
    if am != bm0 or ag != bg0:
        mism.append({"match_id": int(k[0]), "org": k[1],
                     "ours": [am, ag], "theirs": [bm0, bg0],
                     "theirs_raw_nan": bool(pd.isna(bm))})
n_nan_theirs = int(theirs["matches_since_change"].isna().sum())

# ── mode (ii): count overlap rows where full-corpus differs from theirs ──────
full_i = full.set_index(["match_id", "org"])
diff_full = []
for k in common:
    a = full_i.loc[k]
    b = theirs.loc[k]
    bm = b["matches_since_change"]
    bm0 = 0 if pd.isna(bm) else int(bm)
    if int(a["matches_since_change"]) != bm0:
        diff_full.append(k)

ver = {"mode_i_restricted": {
           "n_overlap": int(len(common)),
           "n_mismatch": len(mism),
           "mismatches_first20": mism[:20],
           "theirs_nan_rows": n_nan_theirs,
           "nan_convention": "their NaN (no prior match) compared as 0 per spec "
                             "'0 if ... no prior match'"},
       "mode_ii_full_corpus": {
           "n_overlap_rows_differing": len(diff_full),
           "explanation": "corpus-addition matches interleave org histories "
                          "(expected, preregistered)",
           "examples_first10": [[int(a), b] for a, b in diff_full[:10]]},
       "full_rows": int(len(full)),
       "orgs": int(full.org.nunique())}

with open(OUT_VER, "w") as f:
    json.dump(ver, f, indent=1)
print(json.dumps({k: v for k, v in ver.items() if k != "mode_i_restricted"} |
                 {"mode_i": {kk: vv for kk, vv in ver["mode_i_restricted"].items()
                             if kk != "mismatches_first20"}}, indent=1), flush=True)

if mism:
    print("MODE (i) VERIFICATION FAILED — implementation does not reproduce the "
          "lineups agent's table. STOP (preregistered).", flush=True)
    sys.exit(2)

full.to_csv(OUT_CSV, index=False)
print(f"wrote {OUT_CSV} ({len(full)} rows)", flush=True)
