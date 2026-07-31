"""agent:context step 1 — data prep (no experiments, no holdout contact).

1. Verify frame sha256 against crn.json (abort on mismatch).
2. Lineup top-up for the 335 corpus-addition frame matches, definitions
   verbatim from preregister.lineups.md; cross-check my algorithm against 20
   random covered sides (their history universe) — exact match required.
3. Integrity per (match_id, org) for all frame matches (vct-modal primary).
4. Exposure features per frame row (maps14/30, dso, dsi + diffs).
5. Elim flag + event classes per frame row and per engine game.

Writes ONLY into testing_lab/v8/scratch/context/.
"""
import hashlib
import json
import math
import os
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

V8 = "/Users/benny_es1/PythonTest/testing_lab/v8"
TL = os.path.dirname(V8)
SC = os.path.join(V8, "scratch", "context")
sys.path.insert(0, TL)

# ── 1. frame verify ─────────────────────────────────────────────────────────
crn = json.load(open(os.path.join(V8, "crn.json")))
fp = os.path.join(V8, "data", "frame_expanded", "series.csv")
sha = hashlib.sha256(open(fp, "rb").read()).hexdigest()
want = crn["frame_expanded"]["series_csv_sha256"]
assert sha == want, f"FRAME SHA MISMATCH {sha} != {want} — ABORT"
frame = pd.read_csv(fp).sort_values(["date", "match_id"]).reset_index(drop=True)
assert len(frame) == 2058 and (frame.date > "2024-12-31").sum() == 1217
print("frame verified:", sha[:12], len(frame))

from engine import Engine  # noqa: E402

eng = Engine()
sys.path.insert(0, TL.replace("/testing_lab", ""))
sys.path.insert(0, os.path.join(TL.replace("/testing_lab", ""), "scrapers"))
from MoreTestingMaybeFiles import ALL_EVENTS  # noqa: E402

RATINGS_ONLY = {e["id"] for e in ALL_EVENTS if e.get("ratings_only")}
# lineups-agent event_class: ewc iff ratings_only (their frozen definition)
# my 3d class map: ewc_offseason = ratings_only MINUS {2023_lcq}
EWC_3D = RATINGS_ONLY - {"2023_lcq"}

# ── 2. lineup universe: lineups.csv canonical + engine extraction for rest ──
lu = pd.read_csv(os.path.join(V8, "data", "lineups.csv"))
their_sides = {}
for r in lu.itertuples(index=False):
    their_sides[(int(r.match_id), r.org)] = {
        "date": r.date, "event_id": r.event_id, "event_class": r.event_class,
        "players": frozenset(str(r.players).split(";"))}

# engine-side dates + event per match, and org pairs per match
g_date = {}
g_event = {}
match_orgs = defaultdict(set)
n_maps = defaultdict(int)
for g in eng.games:
    mid = g["match_id"]
    g_date[mid] = g["date_s"]
    g_event[mid] = g["event_id"]
    match_orgs[mid].add(g["winner"])
    match_orgs[mid].add(g["loser"])
    n_maps[mid] += 1

universe = dict(their_sides)  # (mid, org) -> row
added = []
for mid, orgs in match_orgs.items():
    for org in orgs:
        k = (mid, org)
        if k in universe:
            continue
        lin = eng.lineups.get((org, mid))
        if not lin:
            continue
        universe[k] = {"date": g_date[mid], "event_id": g_event[mid],
                       "event_class": "ewc" if g_event[mid] in RATINGS_ONLY else "vct",
                       "players": frozenset(lin)}
        added.append(k)
print(f"universe: {len(their_sides)} lineups.csv sides + {len(added)} engine-extracted")

# per-org sequences sorted by (date, match_id)
org_seq = defaultdict(list)
for (mid, org), row in universe.items():
    org_seq[org].append((row["date"], mid, row))
for org in org_seq:
    org_seq[org].sort(key=lambda t: (t[0], t[1]))


def modal5(basis):
    """basis = [(date, mid, players_frozenset)]; preregister.lineups.md rule:
    rank by (-count, -most_recent_date, ProfileURL asc), top 5."""
    if not basis:
        return None
    cnt = defaultdict(int)
    recent = {}
    for d, mid, pl in basis:
        for p in pl:
            cnt[p] += 1
            if p not in recent or d > recent[p]:
                recent[p] = d
    ranked = sorted(cnt, key=lambda p: (-cnt[p],
                                        tuple(-ord(c) for c in recent[p]), p))
    return set(ranked[:5])


def features_for(org, mid, date, players, hist_events=None):
    """overlap_modal, stand_in_flag, overlap_vct, n_vct_basis, no_prior_30d,
    org_debut. hist_events: optional event-id whitelist (cross-check mode)."""
    d0 = int(np.datetime64(str(date)[:10]).astype("datetime64[D]").astype(int))
    hist = [(d, m, r["players"]) for d, m, r in org_seq[org]
            if d < date and (hist_events is None or r["event_id"] in hist_events)]
    w30 = [t for t in hist
           if 1 <= d0 - int(np.datetime64(str(t[0])[:10])
                            .astype("datetime64[D]").astype(int)) <= 30]
    m30 = modal5(w30)
    if m30 is None:
        ov30, standin = np.nan, np.nan
    else:
        ov30 = len(players & m30) / 5.0
        standin = 1.0 if ov30 < 1.0 else 0.0
    vct_hist = [(d, m, r["players"]) for d, m, r in org_seq[org]
                if d < date and r["event_class"] == "vct"
                and (hist_events is None or r["event_id"] in hist_events)]
    basis = vct_hist[-10:]
    mv = modal5(basis)
    ovv = len(players & mv) / 5.0 if mv is not None else np.nan
    return {"overlap_modal": ov30, "stand_in_flag": standin,
            "no_prior_30d": 0 if w30 else 1,
            "overlap_vct_modal": ovv, "n_vct_basis": len(basis),
            "org_debut": 0 if hist else 1}


# ── cross-check: 20 random covered sides, THEIR history universe ────────────
lf = pd.read_csv(os.path.join(V8, "data", "lineup_features.csv"))
their_events = set(lu.event_id.unique())
rng = np.random.default_rng(20260728)  # crn seed, declared
cand = lf[(lf.n_modal_matches > 0) | (lf.n_vct_basis > 0)]
pick = cand.iloc[rng.choice(len(cand), size=20, replace=False)]
bad = 0
for r in pick.itertuples(index=False):
    key = (int(r.match_id), r.org)
    row = their_sides[key]
    mine = features_for(r.org, int(r.match_id), row["date"], row["players"],
                        hist_events=their_events)
    for col in ("overlap_modal", "overlap_vct_modal"):
        a, b = getattr(r, col), mine[col]
        if (pd.isna(a) != pd.isna(b)) or (not pd.isna(a) and abs(a - b) > 1e-9):
            print(f"  MISMATCH {key} {col}: theirs={a} mine={b}")
            bad += 1
    # lineup extraction check where engine has it
    lin = eng.lineups.get((r.org, int(r.match_id)))
    if lin is not None and frozenset(lin) != row["players"]:
        print(f"  LINEUP MISMATCH {key}")
        bad += 1
assert bad == 0, f"cross-check failed on {bad} fields — STOP AND RECONCILE"
print("cross-check: 20/20 sides reproduce lineups-agent values exactly")

# ── top-up for frame matches missing from lineup_features ───────────────────
have_mid = set(lf.match_id.unique())
need = frame[~frame.match_id.isin(have_mid)]
rows = []
for r in need.itertuples(index=False):
    for org in (r.winner, r.loser):
        key = (int(r.match_id), org)
        u = universe.get(key)
        if u is None:
            rows.append({"match_id": r.match_id, "org": org, "date": r.date,
                         "event_id": r.event_id, "gap": 1})
            continue
        ft = features_for(org, int(r.match_id), u["date"], u["players"])
        rows.append({"match_id": r.match_id, "org": org, "date": u["date"],
                     "event_id": r.event_id, "gap": 0, **ft})
topup = pd.DataFrame(rows)
topup.to_csv(os.path.join(SC, "lineup_topup.csv"), index=False)
n_gap = int(topup.gap.sum())
print(f"top-up: {len(topup)} sides for {need.match_id.nunique()} matches, gaps={n_gap}")

# how many covered sides would change overlap_modal under full history
chg = 0
n_chk = 0
sub = lf[lf.match_id.isin(frame.match_id)].sample(n=400, random_state=20260728)
for r in sub.itertuples(index=False):
    key = (int(r.match_id), r.org)
    row = their_sides.get(key)
    if row is None:
        continue
    n_chk += 1
    mine = features_for(r.org, int(r.match_id), row["date"], row["players"])
    a, b = r.overlap_modal, mine["overlap_modal"]
    if (pd.isna(a) != pd.isna(b)) or (not pd.isna(a) and abs(a - b) > 1e-9):
        chg += 1
print(f"corpus-growth sensitivity: {chg}/{n_chk} sampled covered sides change "
      f"overlap_modal under full history (transparency number)")

# ── 3. integrity per (match_id, org) for ALL frame matches ──────────────────
integ = {}
src = {}
for r in lf.itertuples(index=False):
    key = (int(r.match_id), r.org)
    v = r.overlap_vct_modal
    if pd.isna(v):
        v = r.overlap_modal
        s = "modal30"
    else:
        s = "vct_modal"
    if pd.isna(v):
        v, s = 1.0, "debut_default"
    integ[key], src[key] = float(v), s
for r in topup.itertuples(index=False):
    key = (int(r.match_id), r.org)
    if r.gap == 1:
        integ[key], src[key] = 1.0, "gap_default"
        continue
    v = r.overlap_vct_modal
    s = "vct_modal"
    if pd.isna(v):
        v, s = r.overlap_modal, "modal30"
    if pd.isna(v):
        v, s = 1.0, "debut_default"
    integ[key], src[key] = float(v), s

int_w = np.array([integ.get((m, w), 1.0) for m, w in zip(frame.match_id, frame.winner)])
int_l = np.array([integ.get((m, l), 1.0) for m, l in zip(frame.match_id, frame.loser)])
frame_out = frame.copy()
frame_out["integ_w"], frame_out["integ_l"] = int_w, int_l

# ── 4. exposure features (walk-forward, from engine games) ──────────────────
INTL_RE = re.compile(r"\d{4}_(masters_.*|champions|lock_in)$")
EWC_MAINS = {"2025_ewc", "2026_ewc"}
team_days = defaultdict(list)   # org -> [(datenum, event_id)] per MAP
for g in eng.games:
    dn = int(np.datetime64(g["date_s"], "D").astype(int))
    for org in (g["winner"], g["loser"]):
        team_days[org].append((dn, g["event_id"]))
for org in team_days:
    team_days[org].sort()


def exposure(org, date_s):
    d0 = int(np.datetime64(date_s, "D").astype(int))
    hist = [t for t in team_days[org] if t[0] < d0]
    m14 = sum(1 for dn, _ in hist if d0 - dn <= 14)
    m30 = sum(1 for dn, _ in hist if d0 - dn <= 30)
    dso = d0 - hist[-1][0] if hist else 120
    dso = min(dso, 120)
    intl_d = [dn for dn, ev in hist if INTL_RE.fullmatch(ev)]
    dsi = min(d0 - intl_d[-1], 365) if intl_d else 365
    intl_d2 = [dn for dn, ev in hist if INTL_RE.fullmatch(ev) or ev in EWC_MAINS]
    dsi2 = min(d0 - intl_d2[-1], 365) if intl_d2 else 365
    return m14, m30, dso, dsi, dsi2


cols = {k: [] for k in ("m14_w", "m30_w", "dso_w", "dsi_w", "dsi2_w",
                        "m14_l", "m30_l", "dso_l", "dsi_l", "dsi2_l")}
for r in frame.itertuples(index=False):
    a = exposure(r.winner, r.date)
    b = exposure(r.loser, r.date)
    for i, k in enumerate(("m14", "m30", "dso", "dsi", "dsi2")):
        cols[k + "_w"].append(a[i])
        cols[k + "_l"].append(b[i])
for k, v in cols.items():
    frame_out[k] = v

# ── 5. elim flag + classes ──────────────────────────────────────────────────
mn = frame.match_name.astype(str)
st = frame.stage.astype(str)
has = lambda p: mn.str.contains(p, case=False, regex=False)
elim = (st == "grand_final")
for pat in ("Lower", "Elimination", "Decider", "Knockout", "(0-1)", "(1-1)"):
    elim = elim | has(pat)
ko = has("Quarterfinal") | has("Semifinal") | has("Final") | has("Round of")
elim = elim | ((st == "playoffs") & ko & ~has("Upper"))
frame_out["elim"] = elim.astype(int)


def eclass(ev, stage):
    if re.fullmatch(r"\d{4}_champions", ev):
        return "champions"
    if re.match(r"\d{4}_masters_", ev) or ev == "2023_lock_in":
        return "masters"
    if ev in EWC_3D:
        return "ewc_offseason"
    return "vct_playoffs" if stage in ("playoffs", "grand_final") else "vct_regular"


frame_out["eclass"] = [eclass(e, s_) for e, s_ in zip(frame.event_id, frame.stage)]

# per-game vectors for the engine (aligned to eng.games)
stage_by_mid = dict(zip(frame.match_id, frame.stage))
elim_by_mid = dict(zip(frame.match_id, frame_out.elim))
g_rows = []
for g in eng.games:
    mid = g["match_id"]
    stg = stage_by_mid.get(mid, "groups")
    g_rows.append({
        "match_id": mid, "event_id": g["event_id"],
        "po": 1.6 if stg in ("playoffs", "grand_final") else 1.0,
        "elim": int(elim_by_mid.get(mid, 0)),
        "eclass": eclass(g["event_id"], stg),
        "ewc3d": int(g["event_id"] in EWC_3D),
        "integ_w": integ.get((mid, g["winner"]), 1.0),
        "integ_l": integ.get((mid, g["loser"]), 1.0),
    })
gdf = pd.DataFrame(g_rows)
gdf.to_csv(os.path.join(SC, "game_features.csv"), index=False)
frame_out.to_csv(os.path.join(SC, "frame_features.csv"), index=False)

hold = frame_out.date > "2024-12-31"
print("\nsummary:")
print("  elim share: groups-era %.3f overall, holdout n=%d" %
      (frame_out.elim.mean(), int((frame_out.elim & hold).sum())))
print("  eclass counts (holdout):")
print(frame_out[hold].eclass.value_counts().to_string())
print("  integrity<1 sides among ewc3d frame rows: %.3f (w) %.3f (l)" % (
    (frame_out[frame_out.eclass == "ewc_offseason"].integ_w < 1).mean(),
    (frame_out[frame_out.eclass == "ewc_offseason"].integ_l < 1).mean()))
print("  integrity source mix:", pd.Series(list(src.values())).value_counts().to_dict())
print("  games ewc3d share:", gdf.ewc3d.mean().round(4))
print("step1 DONE")
