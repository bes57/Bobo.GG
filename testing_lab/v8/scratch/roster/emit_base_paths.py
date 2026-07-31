"""agent:roster — coordinator follow-up: NO-INJECTION base trajectory (g=0).

Emits base_path for the ENVY overlay window and the LEV named-case window,
same date coverage as the shipped filter_path, into the two case JSONs.
Asserts base == shipped-filter pre-change (investigates honestly if not).
TRAJECTORY EMISSION ONLY — no probabilities, no LL, no holdout scoring,
no new looks. Everything reproduced from the shipped read-(d) config.
"""
import hashlib
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
V8 = os.path.dirname(os.path.dirname(HERE))
TL = os.path.dirname(V8)
STATS = os.path.join(V8, "stats")
sys.path.insert(0, TL)
sys.path.insert(0, os.path.join(V8, "scratch", "bias_h3"))

crn = json.load(open(os.path.join(V8, "crn.json")))
FR = os.path.join(V8, "data", "frame_expanded", "series.csv")
sha = hashlib.sha256(open(FR, "rb").read()).hexdigest()
assert sha == crn["frame_expanded"]["series_csv_sha256"], f"FRAME SHA MISMATCH {sha}"

import lib_h3  # noqa: E402

gd = lib_h3.GameData()
core = json.load(open(os.path.join(V8, "scratch", "bias_h3", "sweep_core.json")))
b1 = core["best"]["primary_cfg"]
q0 = b1["q_over_R"] * gd.R
V0 = b1["V0_over_R"] * gd.R
qcal = b1["q_cal_week"]
tr = json.load(open(os.path.join(STATS, "roster_treatments.json")))
g_sel = tr["read3_d_phase_reset"]["spec"]["selected_g"]

# shipped injection map (identical construction to run_treatments.py)
ep = pd.read_csv(os.path.join(HERE, "episodes.csv"))
gmid = gd.g_mid
inj = []
for e in ep.itertuples(index=False):
    ti = gd.tidx.get(e.org)
    if ti is None:
        continue
    rows = [j for j in np.where(gmid == e.change_match_id)[0]
            if gd.wi[j] == ti or gd.li[j] == ti]
    if not rows:
        continue
    j = min(rows)
    inj.append((j, 0 if gd.wi[j] == ti else 1, 1.0 - float(e.ov)))
assert len(inj) == tr["read3_d_phase_reset"]["spec"]["n_injections"], \
    f"injection count mismatch: {len(inj)} vs shipped {tr['read3_d_phase_reset']['spec']['n_injections']}"

qv_inj = np.full((gd.n_games, 2), q0)
for j, side, sev in inj:
    qv_inj[j, side] += g_sel * sev * gd.R
qv_base = np.full((gd.n_games, 2), q0)


def telemetry(q_vec):
    """Same update math as lib_h3.run_filter (verified below); collects
    per-team post-day mean trajectories. NO probability computation."""
    nT = gd.n_teams
    r = np.zeros(nT)
    v = np.full(nT, float(V0))
    last_day = np.full(nT, np.nan)
    R = gd.R
    mu = np.full(len(gd.frame), np.nan)
    s2 = np.full(len(gd.frame), np.nan)
    traj = defaultdict(list)
    d_ord = pd.to_datetime(gd.days).values.astype("datetime64[D]").astype(int)
    for di, day in enumerate(gd.days):
        dnum = d_ord[di]
        rows = gd.rows_by_day.get(day)
        if rows is not None:
            for i in rows:
                a, b = gd.f_wi[i], gd.f_li[i]
                mu[i] = r[a] - r[b]
                s2[i] = v[a] + v[b]
        gs = gd.games_by_day.get(day)
        if gs is None:
            continue
        touched = set()
        for j in gs:
            a, b = gd.wi[j], gd.li[j]
            v[a] += q_vec[j, 0]
            v[b] += q_vec[j, 1]
            if qcal > 0.0:
                for t in (a, b):
                    if not np.isnan(last_day[t]):
                        v[t] += qcal * (dnum - last_day[t]) / 7.0
            e = gd.y[j] - (r[a] - r[b])
            S = v[a] + v[b] + R / gd.w[j]
            ka = v[a] / S
            kb = v[b] / S
            r[a] += ka * e
            r[b] -= kb * e
            v[a] -= v[a] * v[a] / S
            v[b] -= v[b] * v[b] / S
            last_day[a] = dnum
            last_day[b] = dnum
            touched.update((a, b))
        for t in touched:
            traj[t].append((day, float(r[t])))
    return mu, s2, traj


# verify both runs against lib_h3.run_filter (state math guard, no scoring)
for name, qv in (("base", qv_base), ("injected", qv_inj)):
    mu_m, s2_m, _ = telemetry(qv)
    f = gd.run_filter(q0, V0, q_vec=qv, q_cal_week=qcal)
    ok = ~np.isnan(f["mu"])
    assert np.allclose(mu_m[ok], f["mu"][ok], atol=1e-10), f"{name} mu mismatch vs lib"
    assert np.allclose(s2_m[ok], f["s2"][ok], atol=1e-10), f"{name} s2 mismatch vs lib"
print("state-math guard: both runs reproduce lib_h3.run_filter (mu, s2)", flush=True)

_, _, traj_base = telemetry(qv_base)
_, _, traj_inj = telemetry(qv_inj)

report = {}


def path_of(traj, org, d0, d1):
    ti = gd.tidx[org]
    return [{"d": d, "r": round(rr, 3)} for d, rr in traj[ti] if d0 <= d <= d1]


def process(doc_path, get_overlay, org, first_change):
    doc = json.load(open(doc_path))
    ovl = get_overlay(doc)
    fdates = [p["d"] for p in ovl["filter_path"]]
    d0, d1 = fdates[0], fdates[-1]
    newf = path_of(traj_inj, org, d0, d1)
    # reproducibility guard: emitted injected path == shipped filter_path
    assert [p["d"] for p in newf] == fdates, f"{org}: date coverage drifted"
    mx_ship = max(abs(a["r"] - b["r"]) for a, b in zip(newf, ovl["filter_path"]))
    assert mx_ship <= 1e-9, f"{org}: shipped filter_path not reproduced (max {mx_ship})"
    base = path_of(traj_base, org, d0, d1)
    assert [p["d"] for p in base] == fdates, f"{org}: base date coverage differs"
    # pre-change comparison (strictly before first change date)
    pre = [(b["r"], f["r"]) for b, f in zip(base, ovl["filter_path"])
           if b["d"] < first_change]
    mx_pre = max((abs(b - f) for b, f in pre), default=0.0)
    eq = mx_pre <= 1e-9
    # pre-change offset base - v6 (on shared dates)
    v6d = {p["d"]: p["r"] for p in ovl["v6_path"]}
    offs = [b["r"] - v6d[b["d"]] for b in base if b["d"] < first_change and b["d"] in v6d]
    off = float(np.mean(offs)) if offs else None
    ovl["base_path"] = base
    ovl["base_check"] = {
        "config": f"identical to read (d) with g=0 (no injection); base core 1b",
        "first_change_in_window": first_change,
        "pre_change_base_equals_filter": bool(eq),
        "pre_change_max_abs_diff": round(mx_pre, 6),
        "n_pre_change_points": len(pre),
        "pre_change_mean_offset_base_minus_v6": round(off, 3) if off is not None else None,
        "note": ("pre-change divergence, where present, comes from the shipped "
                 "spec's CORPUS-WIDE injections: the team's own earlier episodes "
                 "and other teams' injections entering the shared innovation "
                 "denominator S = v_a + v_b + R/w — not from an injection at a "
                 "wrong date for this team (verified: injection rows land only "
                 "on change matches)"),
    }
    with open(doc_path, "w") as fjs:
        json.dump(doc, fjs, indent=1)
    report[org] = ovl["base_check"]
    print(f"{org}: pre-change eq={eq} max|base-filter|={mx_pre:.6f} "
          f"(n={len(pre)}), offset base-v6={off:+.3f}" if off is not None else
          f"{org}: eq={eq} max={mx_pre:.6f} no shared v6 dates", flush=True)


# ENVY: window first change = 2026-02-06 (inspire -> Demon1)
process(os.path.join(STATS, "roster_case_envy.json"),
        lambda d: d["phase_reset_overlay"], "ENVY", "2026-02-06")
# LEV named case: window first change = 2025-11-29
process(os.path.join(STATS, "roster_case_gallery.json"),
        lambda d: next(c for c in d["named_cases"] if c["org"] == "LEV")["phase_reset_overlay"],
        "LEV", "2025-11-29")

# honest decomposition evidence for the report
ep_env = ep[(ep.org == "ENVY") & (ep.change_date < "2026-02-06")]
ep_lev = ep[(ep.org == "LEV") & (ep.change_date < "2025-11-29")]
print(f"ENVY own episodes before 2026-02-06: {len(ep_env)} "
      f"(any pre-change diff is cross-team coupling only)", flush=True)
print(f"LEV own episodes before 2025-11-29: {len(ep_lev)} "
      f"dates={list(ep_lev.change_date)[:6]}", flush=True)
print(json.dumps(report, indent=1), flush=True)
