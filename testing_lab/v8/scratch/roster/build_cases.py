"""agent:roster step 3 — ENVY forensic case + gallery of largest overhauls.

All descriptive: stored v6 probabilities + verified daily-replay ratings.
Holdout rows appear in trajectories/tables and are labeled descriptive (not
scored; no selection). Writes stats/roster_case_envy.json +
stats/roster_case_gallery.json (agent:roster is the sole writer).
"""
import hashlib
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
V8 = os.path.dirname(os.path.dirname(HERE))
TL = os.path.dirname(V8)
STATS = os.path.join(V8, "stats")
sys.path.insert(0, TL)

FRAME = os.path.join(V8, "data", "frame_expanded", "series.csv")
crn = json.load(open(os.path.join(V8, "crn.json")))
sha = hashlib.sha256(open(FRAME, "rb").read()).hexdigest()
assert sha == crn["frame_expanded"]["series_csv_sha256"], f"FRAME SHA MISMATCH {sha}"
frame = pd.read_csv(FRAME)

z = np.load(os.path.join(HERE, "v6_daily.npz"), allow_pickle=True)
R, days, teams = z["R"], list(z["days"]), list(z["teams"])
p_all = z["p_all"]
tidx = {t: i for i, t in enumerate(teams)}
ep = pd.read_csv(os.path.join(HERE, "episodes.csv"))
st = pd.read_csv(os.path.join(HERE, "team_match_state.csv"))

from engine import Engine  # noqa: E402
eng = Engine()
lineups = eng.lineups

DESC = ("DESCRIPTIVE — holdout rows shown are trajectories/case panels, not "
        "scored reads; no model selection performed (preregister.roster.md §2)")


def slug(url):
    m = re.search(r"/player/\d+/([^/;]+)", str(url))
    return m.group(1) if m else str(url)


def slugs(joined):
    return [slug(u) for u in str(joined).split(";")] if joined and str(joined) != "nan" else []


# frame rows per org
frame = frame.reset_index(drop=True)
rows_of = defaultdict(list)
for i, r in enumerate(frame.itertuples(index=False)):
    rows_of[r.winner].append(i)
    rows_of[r.loser].append(i)


def team_matches(org):
    out = []
    for i in rows_of[org]:
        r = frame.iloc[i]
        won = int(r.winner == org)
        p_t = float(p_all[i]) if won else 1.0 - float(p_all[i])
        out.append({"row": i, "match_id": int(r.match_id), "date": r.date,
                    "event_id": r.event_id, "opponent": r.loser if won else r.winner,
                    "won": won, "p_team": round(p_t, 4),
                    "score": f"{int(r.w_maps)}-{int(r.l_maps)}" if won
                             else f"{int(r.l_maps)}-{int(r.w_maps)}",
                    "holdout": bool(r.date > "2024-12-31")})
    out.sort(key=lambda x: (x["date"], x["match_id"]))
    return out


def rating_traj(org, d0, d1):
    if org not in tidx:
        return []
    j = tidx[org]
    return [{"d": d, "r": round(float(R[k, j]), 3)}
            for k, d in enumerate(days) if d0 <= d <= d1]


def rating_on_first_day_after(org, date_s):
    j = tidx.get(org)
    if j is None:
        return None
    for k, d in enumerate(days):
        if d > date_s:
            return float(R[k, j])
    return float(R[-1, j])


def rating_on_last_day_before(org, date_s):
    j = tidx.get(org)
    if j is None:
        return None
    out = None
    for k, d in enumerate(days):
        if d < date_s:
            out = float(R[k, j])
        else:
            break
    return out


def case_panel(org, change_date, n_pre=6, n_post=10):
    """Pre/post match tables, stabilization (preregister frozen def), cost."""
    tm = team_matches(org)
    pre = [m for m in tm if m["date"] < change_date][-n_pre:]
    post = [m for m in tm if m["date"] >= change_date][:n_post]
    r_pre = rating_on_last_day_before(org, change_date)
    # r(m') = rating on first solve day strictly after match m'
    r_after = [rating_on_first_day_after(org, m["date"]) for m in post]
    r_post = r_after[min(len(post), 10) - 1] if post else None
    stab = None
    censored = len(post) < 10
    if r_pre is not None and r_post is not None:
        jump = abs(r_pre - r_post)
        if jump < 0.10:
            stab = "n/a (no jump)"
        else:
            stab = ">10" if not censored else f"censored at {len(post)}"
            for m0 in range(len(post)):
                if all(abs(r_after[k] - r_post) <= 0.25 * jump
                       for k in range(m0, len(post))):
                    stab = m0 + 1
                    break
    cost6 = [m["p_team"] - m["won"] for m in post[:6]]
    return {
        "pre_matches": pre, "post_matches": post,
        "r_pre": None if r_pre is None else round(r_pre, 3),
        "r_post_after_last": None if r_post is None else round(r_post, 3),
        "stabilization_matches": stab, "post_censored": censored,
        "carryover_cost_pp_per_match": round(float(np.mean(cost6)) * 100, 1) if cost6 else None,
        "carryover_n": len(cost6),
        "pre_bias_pp": round(float(np.mean([m["p_team"] - m["won"] for m in pre])) * 100, 1) if pre else None,
    }


# ── ENVY ─────────────────────────────────────────────────────────────────────
org = "ENVY"
eps = ep[ep.org == org].to_dict("records")
tm26 = [m for m in team_matches(org) if m["date"] >= "2026-01-01"]
st_org = st[st.org == org].set_index("match_id")
for m in tm26:
    mid = m["match_id"]
    L = lineups.get((org, mid), frozenset())
    m["five"] = sorted(slug(u) for u in L)
    m["msc"] = int(st_org.loc[mid, "msc"]) if mid in st_org.index else None

# the S1 five (last stage1 match) vs S2 five (first stage2 match)
s1_last = [m for m in tm26 if m["event_id"] == "2026_stage1"][-1]
s2_first = [m for m in tm26 if m["event_id"] == "2026_stage2"][0]
L1 = lineups.get((org, s1_last["match_id"]))
L2 = lineups.get((org, s2_first["match_id"]))
agg = {
    "s1_last_match": {"match_id": s1_last["match_id"], "date": s1_last["date"],
                      "five": sorted(map(slug, L1))},
    "s2_first_match": {"match_id": s2_first["match_id"], "date": s2_first["date"],
                       "five": sorted(map(slug, L2))},
    "kept": sorted(slug(u) for u in (L1 & L2)),
    "left_between_s1_s2": sorted(slug(u) for u in (L1 - L2)),
    "joined_between_s1_s2": sorted(slug(u) for u in (L2 - L1)),
    "overlap_s1_to_s2": round(len(L1 & L2) / max(len(L1), len(L2), 5), 2),
}

chain = []
for e in eps:
    if e["change_date"] >= "2026-01-01":
        chain.append({
            "change_date": e["change_date"], "event_id": e["event_id"],
            "change_match_id": int(e["change_match_id"]),
            "out": slugs(e["left"]), "in": slugs(e["joined"]),
            "ov": e["ov"], "kept": int(e["kept"]), "run_len": int(e["run_len"]),
            "sustained": bool(e["sustained"]), "censored": bool(e["censored"]),
        })

focal = "2026-05-12"   # first change of the S1->S2 chain (EWC qual swap)
s2_change = "2026-07-17"
envy = {
    "written_by": "agent:roster", "label": DESC,
    "case": "ENVY 2026 Stage 1 -> Stage 2 (operator-requested forensic)",
    "matches_2026": [{k: v for k, v in m.items() if k != "row"} for m in tm26],
    "change_chain_2026": chain,
    "s1_to_s2_aggregate": agg,
    "panel_chain_start_2026_05_12": case_panel(org, focal),
    "panel_s2_change_2026_07_17": case_panel(org, s2_change),
    "rating_trajectory_2026": rating_traj(org, "2026-01-01", "2026-12-31"),
    "notes": [
        "v6's only roster mechanism is YEAR-boundary continuity 0.3; every "
        "2026 mid-season change above is invisible to the solve.",
        "S1->S2 is a CHAINED pair of one-out swaps (eggsterr->NightZ at EWC "
        "qual 2026-05-12, p0ppin->Glyph at Stage 2 2026-07-17); the fielded "
        "five at Stage 2 keeps 3/5 of the Stage 1 five.",
        "probabilities are the stored v6 baseline (scratch/bias_h3/"
        "v6_baseline.npz), replayed bit-identically for daily ratings.",
    ],
}
with open(os.path.join(STATS, "roster_case_envy.json"), "w") as f:
    json.dump(envy, f, indent=1)
print("ENVY case written:", json.dumps(agg, indent=1), flush=True)
print("chain:", json.dumps(chain, indent=1), flush=True)
for k in ("panel_chain_start_2026_05_12", "panel_s2_change_2026_07_17"):
    p = envy[k]
    print(k, "stab:", p["stabilization_matches"], "cost/match pp:",
          p["carryover_cost_pp_per_match"], "pre bias pp:", p["pre_bias_pp"], flush=True)

# ── gallery: 4 largest sustained overhauls, org has >=5 pre + >=5 post ──────
sus = ep[(ep.sustained == 1) & (ep.org != "ENVY")].copy()
cands = []
for e in sus.itertuples(index=False):
    tm = team_matches(e.org)
    n_pre = sum(1 for m in tm if m["date"] < e.change_date)
    n_post = sum(1 for m in tm if m["date"] >= e.change_date)
    if n_pre >= 5 and n_post >= 5:
        cands.append((e.ov, e.change_date, e))
cands.sort(key=lambda x: (x[0], x[1]))
gallery = []
for ov, cd, e in cands[:4]:
    pan = case_panel(e.org, e.change_date)
    d0 = (np.datetime64(e.change_date) - np.timedelta64(60, "D")).astype(str)
    d1 = (np.datetime64(e.change_date) + np.timedelta64(120, "D")).astype(str)
    gallery.append({
        "org": e.org, "change_date": e.change_date, "event_id": e.event_id,
        "ov": e.ov, "kept": int(e.kept), "run_len": int(e.run_len),
        "out": slugs(e.left), "in": slugs(e.joined),
        "panel": pan, "rating_trajectory": rating_traj(e.org, d0, d1),
    })
# ── operator-named cases (amendment): verified from lineups data, not memory ─
named = []
for norg, cd, expect in (("LEV", "2025-11-29", "improvement (gained Neon)"),
                         ("ENVY", "2026-02-06", "degradation (lost inspire)")):
    e = ep[(ep.org == norg) & (ep.change_date == cd)]
    if len(e) != 1:
        sys.exit(f"NAMED CASE NOT FOUND IN DATA: {norg} {cd} — abort")
    e = e.iloc[0]
    pan = case_panel(norg, cd)
    d0 = (np.datetime64(cd) - np.timedelta64(60, "D")).astype(str)
    d1 = (np.datetime64(cd) + np.timedelta64(120, "D")).astype(str)
    named.append({
        "org": norg, "change_date": cd, "event_id": e.event_id,
        "operator_expectation": expect,
        "ov": float(e.ov), "kept": int(e.kept), "run_len": int(e.run_len),
        "sustained": bool(e.sustained),
        "out": slugs(e.left), "in": slugs(e.joined),
        "panel": pan, "rating_trajectory": rating_traj(norg, d0, d1),
    })
    print(f"named case {norg} {cd}: ov={e.ov} out={slugs(e.left)} "
          f"in={slugs(e.joined)} cost/match={pan['carryover_cost_pp_per_match']}pp "
          f"stab={pan['stabilization_matches']}", flush=True)

with open(os.path.join(STATS, "roster_case_gallery.json"), "w") as f:
    json.dump({"written_by": "agent:roster", "label": DESC,
               "selection_rule": "4 lowest-ov sustained episodes, org has >=5 "
                                 "pre and >=5 post matches in frame, ENVY "
                                 "excluded (preregister §2); named_cases = "
                                 "operator amendment (LEV/Neon, ENVY/inspire), "
                                 "verified from lineups data",
               "cases": gallery, "named_cases": named}, f, indent=1)
print("gallery:", [(g["org"], g["change_date"], g["ov"],
                    g["panel"]["stabilization_matches"],
                    g["panel"]["carryover_cost_pp_per_match"]) for g in gallery],
      flush=True)
