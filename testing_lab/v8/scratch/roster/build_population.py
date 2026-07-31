"""agent:roster step 4 — population atlas of mid-season roster changes.

DESCRIPTIVE ONLY (preregister §3): counts + adaptation curves from the stored
v6 baseline probabilities. No model selection; holdout rows included and
labeled. Writes stats/roster_population.json.
"""
import hashlib
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
V8 = os.path.dirname(os.path.dirname(HERE))
STATS = os.path.join(V8, "stats")

FRAME = os.path.join(V8, "data", "frame_expanded", "series.csv")
crn = json.load(open(os.path.join(V8, "crn.json")))
sha = hashlib.sha256(open(FRAME, "rb").read()).hexdigest()
assert sha == crn["frame_expanded"]["series_csv_sha256"], f"FRAME SHA MISMATCH {sha}"
frame = pd.read_csv(FRAME)

z = np.load(os.path.join(HERE, "v6_daily.npz"), allow_pickle=True)
p_all = z["p_all"]
teams = list(z["teams"])
region_idx = z["region_idx"]
REGS = ["Americas", "EMEA", "Pacific", "CN"]
reg_of = {t: (REGS[region_idx[i]] if region_idx[i] >= 0 else "other")
          for i, t in enumerate(teams)}

ep = pd.read_csv(os.path.join(HERE, "episodes.csv"))
st = pd.read_csv(os.path.join(HERE, "team_match_state.csv"))
st_ix = st.set_index(["org", "match_id"])


def mag_class(ov):
    if ov >= 0.8:
        return "keep4"
    if ov >= 0.6:
        return "keep3"
    return "overhaul"


# ── counts ───────────────────────────────────────────────────────────────────
sus = ep[ep.sustained == 1].copy()
tra = ep[ep.sustained == 0].copy()
sus["mag"] = sus.ov.apply(mag_class)
sus["year"] = sus.change_date.str[:4].astype(int)
sus["region"] = sus.org.map(reg_of)
counts = {
    "episodes_total": int(len(ep)),
    "sustained": int(len(sus)),
    "transient": int(len(tra)),
    "sustained_censored": int(sus.censored.sum()),
    "by_magnitude": sus.mag.value_counts().to_dict(),
    "by_year": {str(k): int(v) for k, v in sus.year.value_counts().sort_index().items()},
    "by_region": sus.region.value_counts().to_dict(),
    "by_year_magnitude": {f"{y}|{m}": int(n) for (y, m), n in
                          sus.groupby(["year", "mag"]).size().items()},
    "mean_ov_sustained": round(float(sus.ov.mean()), 3),
}

# ── adaptation curve: bias vs matches-since-change, by magnitude ────────────
# team-observation = (frame row, side). Cell: (mag, min(msc,9)); reference =
# msc >= 10 (roster unchanged 10+ matches). bias = mean(won - p_team) in pp.
cells = defaultdict(list)      # (mag, m) -> [won - p_team]
ref = []
ll_cells = defaultdict(list)
ll_ref = []
n_missing = 0
for i, r in enumerate(frame.itertuples(index=False)):
    loss = -np.log(max(p_all[i], 1e-9))
    for org, won, p_t in ((r.winner, 1, p_all[i]), (r.loser, 0, 1 - p_all[i])):
        k = (org, r.match_id)
        if k not in st_ix.index:
            n_missing += 1
            continue
        row = st_ix.loc[k]
        msc = int(row.msc)
        if msc >= 10:
            ref.append(won - p_t)
            ll_ref.append(loss)
            continue
        if row.episode_idx >= 0 and row.sustained == 1:
            cells[(mag_class(row.ov), msc)].append(won - p_t)
            ll_cells[(mag_class(row.ov), msc)].append(loss)

curve = []
for mag in ("keep4", "keep3", "overhaul"):
    pts = []
    for m in range(10):
        v = cells.get((mag, m), [])
        if len(v) < 5:
            pts.append({"m": m, "n": len(v)})
            continue
        v = np.array(v)
        se = float(v.std(ddof=1) / np.sqrt(len(v)))
        pts.append({"m": m, "n": len(v),
                    "bias_pp": round(float(v.mean()) * 100, 2),
                    "ci_lo": round((float(v.mean()) - 1.96 * se) * 100, 2),
                    "ci_hi": round((float(v.mean()) + 1.96 * se) * 100, 2),
                    "ll": round(float(np.mean(ll_cells[(mag, m)])), 4)})
    curve.append({"magnitude": mag, "points": pts})
rv = np.array(ref)
se = float(rv.std(ddof=1) / np.sqrt(len(rv)))
reference = {"n": len(rv), "bias_pp": round(float(rv.mean()) * 100, 2),
             "ci_lo": round((float(rv.mean()) - 1.96 * se) * 100, 2),
             "ci_hi": round((float(rv.mean()) + 1.96 * se) * 100, 2),
             "ll": round(float(np.mean(ll_ref)), 4)}

# aggregate post-change window summary (m 0-2 pooled) per magnitude
pooled = {}
for mag in ("keep4", "keep3", "overhaul"):
    v = np.concatenate([np.array(cells.get((mag, m), [])) for m in range(3)]) \
        if any(cells.get((mag, m)) for m in range(3)) else np.array([])
    if len(v) >= 5:
        se = float(v.std(ddof=1) / np.sqrt(len(v)))
        pooled[mag] = {"n": int(len(v)), "bias_pp": round(float(v.mean()) * 100, 2),
                       "ci_lo": round((float(v.mean()) - 1.96 * se) * 100, 2),
                       "ci_hi": round((float(v.mean()) + 1.96 * se) * 100, 2)}

pm = json.load(open(os.path.join(STATS, "power_mde.json")))
rb = [b for b in pm["buckets"] if "roster" in b["bucket"]]

out = {
    "written_by": "agent:roster",
    "label": ("DESCRIPTIVE — computed from the stored v6 baseline "
              "probabilities over the full corpus (train+holdout); holdout "
              "rows are not scored reads and no selection was performed "
              "(preregister.roster.md §3)"),
    "change_definition": "preregister.roster.md §1 (rotation-guarded change "
                         "events; sustained = run >= 3 matches or censored; "
                         "overlap = engine lineup formula |A∩B|/max(|A|,|B|,5))",
    "counts": counts,
    "adaptation_curve": {"by_magnitude": curve, "reference_stable": reference,
                         "pooled_first3": pooled,
                         "y": "mean(won − p_v6) in probability points ×100; "
                              ">0 = v6 UNDER-rates the post-change team, "
                              "<0 = v6 over-rates it"},
    "team_observations_missing_lineups": n_missing,
    "power_context": {
        "note": "post-change bucket MDEs from stats/power_mde.json (frozen "
                "n=1007-era holdout) — and even these are exploratory-only "
                "on this frame: THE HOLDOUT IS SPENT (398 recorded looks).",
        "buckets": rb,
        "expanded_frame_floors_milli": json.load(
            open(os.path.join(STATS, "power_mde_expanded.json")))["checkpoint_quote"],
    },
}
with open(os.path.join(STATS, "roster_population.json"), "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps(counts, indent=1))
print("pooled first-3 bias by magnitude:", json.dumps(pooled, indent=1))
print("reference:", json.dumps(reference))
for c in curve:
    print(c["magnitude"], [(p["m"], p.get("bias_pp"), p["n"]) for p in c["points"]])
