"""agent:roster step 1 — change episodes + msc over the full corpus.

Implements preregister.roster.md §1 exactly. Verifies msc against
scratch/bias_h3/lineup_topup.csv (0 mismatches required — abort otherwise).
Writes scratch/roster/episodes.csv + team_match_state.csv. Scratch only.
"""
import hashlib
import json
import os
import sys
from collections import defaultdict

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
V8 = os.path.dirname(os.path.dirname(HERE))
TL = os.path.dirname(V8)
sys.path.insert(0, TL)

FRAME = os.path.join(V8, "data", "frame_expanded", "series.csv")
crn = json.load(open(os.path.join(V8, "crn.json")))
sha = hashlib.sha256(open(FRAME, "rb").read()).hexdigest()
assert sha == crn["frame_expanded"]["series_csv_sha256"], f"FRAME SHA MISMATCH {sha}"

from engine import Engine  # noqa: E402

eng = Engine()
lineups = eng.lineups          # (org, mid) -> frozenset(ProfileURL)
g_date, g_event, n_maps = {}, {}, defaultdict(int)
for g in eng.games:
    g_date[g["match_id"]] = g["date_s"]
    g_event[g["match_id"]] = g["event_id"]
    n_maps[g["match_id"]] += 1

seq = defaultdict(list)        # org -> [(date, mid)]
for (org, mid) in lineups:
    if mid in g_date:
        seq[org].append((g_date[mid], mid))
for org in seq:
    seq[org].sort()


def overlap(a, b):
    """engine.py L279 formula."""
    return len(a & b) / max(len(a), len(b), 5)


episodes = []                  # one row per change EVENT (rotation-guarded)
state_rows = []                # one row per (org, match)
for org, rows in seq.items():
    Ls = [lineups.get((org, mid)) for _, mid in rows]
    n = len(rows)
    # msc per the lineups-agent definition (strictly-earlier DATE)
    msc = []
    for i in range(n):
        d, mid = rows[i]
        m = 0
        j = i - 1
        while j >= 0:
            dj, mj = rows[j]
            if dj >= d:
                j -= 1
                continue
            if Ls[j] == Ls[i] and Ls[i] is not None:
                m += 1
                j -= 1
            else:
                break
        msc.append(m)
    # change events + rotation guard (preregister §1)
    ep_of_run = [None] * n     # episode idx whose run this match belongs to
    for i in range(1, n):
        if Ls[i] is None or Ls[i - 1] is None:
            continue
        if Ls[i] == Ls[i - 1]:
            ep_of_run[i] = ep_of_run[i - 1]
            continue
        recent = {Ls[k] for k in range(max(0, i - 3), i - 1)}
        rotation = Ls[i] in recent
        # run length R_e: consecutive matches from i fielding exactly Ls[i]
        R = 1
        while i + R < n and Ls[i + R] == Ls[i]:
            R += 1
        censored = (i + R == n)
        sustained = (R >= 3) or censored
        if rotation:
            ep_of_run[i] = None
            continue
        ov = overlap(Ls[i], Ls[i - 1])
        episodes.append({
            "org": org, "change_match_id": rows[i][1],
            "change_date": rows[i][0], "event_id": g_event[rows[i][1]],
            "prev_match_id": rows[i - 1][1],
            "ov": round(ov, 4), "run_len": R,
            "censored": int(censored), "sustained": int(sustained),
            "n_new": len(Ls[i]), "n_prev": len(Ls[i - 1]),
            "kept": len(Ls[i] & Ls[i - 1]),
            "joined": ";".join(sorted(Ls[i] - Ls[i - 1])),
            "left": ";".join(sorted(Ls[i - 1] - Ls[i])),
        })
        ep_of_run[i] = len(episodes) - 1
    for i in range(n):
        e = ep_of_run[i]
        state_rows.append({
            "org": org, "match_id": rows[i][1], "date": rows[i][0],
            "event_id": g_event[rows[i][1]], "msc": msc[i],
            "episode_idx": -1 if e is None else e,
            "ov": episodes[e]["ov"] if e is not None else float("nan"),
            "sustained": episodes[e]["sustained"] if e is not None else 0,
        })

ep = pd.DataFrame(episodes).sort_values(["change_date", "org"]).reset_index(drop=True)
# episode_idx refers to build order; remap to the sorted frame
order = {tuple(r[["org", "change_match_id"]]): i for i, r in ep.iterrows()}
st = pd.DataFrame(state_rows)
st["episode_idx"] = [
    order.get((episodes[e]["org"], episodes[e]["change_match_id"]), -1) if e >= 0 else -1
    for e in st.episode_idx]
st = st.sort_values(["date", "match_id", "org"]).reset_index(drop=True)

# ── mandatory verification vs bias_h3 topup ──────────────────────────────────
top = pd.read_csv(os.path.join(V8, "scratch", "bias_h3", "lineup_topup.csv"))
mrg = st.merge(top[["match_id", "org", "matches_since_change"]],
               on=["match_id", "org"], how="inner")
bad = mrg[mrg.msc != mrg.matches_since_change]
print(f"msc verify: {len(mrg)} common rows, {len(bad)} mismatches", flush=True)
if len(bad):
    print(bad.head(20).to_string(), flush=True)
    sys.exit("MSC VERIFICATION FAILED — abort (preregistered).")

ep.to_csv(os.path.join(HERE, "episodes.csv"), index=False)
st.to_csv(os.path.join(HERE, "team_match_state.csv"), index=False)
sus = ep[ep.sustained == 1]
print(f"episodes: {len(ep)} total, {len(sus)} sustained "
      f"({int(sus.censored.sum())} censored), orgs={ep.org.nunique()}", flush=True)
print("magnitude classes (sustained): "
      f"keep4={((sus.ov>=0.8)).sum()}, keep3={((sus.ov>=0.6)&(sus.ov<0.8)).sum()}, "
      f"overhaul={(sus.ov<0.6).sum()}", flush=True)
