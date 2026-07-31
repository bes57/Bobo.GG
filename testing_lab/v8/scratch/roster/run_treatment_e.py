"""agent:roster — Read 4 (e): v6 + temporary post-change overreaction.

Operator-directed clean mechanism test ON v6 (preregister ADDENDUM 2, frozen
before this run). ONE holdout read at the train-selected (a, tau); grid
evaluated train-only with holdout numbers scrubbed. Appends
read4_e_v6_overreact to stats/roster_treatments.json, updates
stats/roster_looks.json (401->402), adds v6_overreact_path + pre-change
coupling to the two case JSONs, adds the frozen arm to
stats/roster_integration.json.
"""
import hashlib
import json
import math
import os
import sys
import time
from bisect import bisect_left
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

HERE = os.path.dirname(os.path.abspath(__file__))
V8 = os.path.dirname(os.path.dirname(HERE))
TL = os.path.dirname(V8)
STATS = os.path.join(V8, "stats")
sys.path.insert(0, TL)
sys.path.insert(0, V8)

import referee  # noqa: E402

FRAME = os.path.join(V8, "data", "frame_expanded", "series.csv")
crn = json.load(open(os.path.join(V8, "crn.json")))
sha = hashlib.sha256(open(FRAME, "rb").read()).hexdigest()
assert sha == crn["frame_expanded"]["series_csv_sha256"], f"FRAME SHA MISMATCH {sha}"
frame = pd.read_csv(FRAME).reset_index(drop=True)
N = len(frame)
hold = (frame.date > "2024-12-31").values
fmts = frame.fmt.values
event_ids = frame.event_id.values

v6 = np.load(os.path.join(V8, "scratch", "bias_h3", "v6_baseline.npz"))
p_v6 = v6["p_all"]
valid_v6 = v6["valid"]

ep = pd.read_csv(os.path.join(HERE, "episodes.csv"))
st = pd.read_csv(os.path.join(HERE, "team_match_state.csv"))
st_ix = {(r.org, r.match_id): (int(r.msc), int(r.episode_idx),
                               float(r.ov) if r.ov == r.ov else np.nan,
                               int(r.sustained))
         for r in st.itertuples(index=False)}
org_seq = {o: list(zip(g.sort_values(["date", "match_id"]).date,
                       g.sort_values(["date", "match_id"]).match_id))
           for o, g in st.groupby("org")}
org_dates = {o: [d for d, _ in seq] for o, seq in org_seq.items()}
eps_by_org = defaultdict(list)
for k, e in enumerate(ep.itertuples(index=False)):
    seq = org_seq[e.org]
    pos = next(i for i, (d, m) in enumerate(seq) if m == e.change_match_id)
    d_conf = seq[pos + 2][0] if e.run_len >= 3 and pos + 2 < len(seq) else None
    d_dead = seq[pos + e.run_len][0] if pos + e.run_len < len(seq) else None
    eps_by_org[e.org].append({"d": e.change_date, "ov": float(e.ov),
                              "d_conf": d_conf, "d_dead": d_dead, "row": k})
for o in eps_by_org:
    eps_by_org[o].sort(key=lambda x: x["d"])


def sustained_at(e, D):
    if e["d_conf"] is not None and e["d_conf"] < D:
        return True
    return e["d_dead"] is None or e["d_dead"] >= D


# ── bucket masks (identical construction to run_treatments.py) ───────────────
p_team_of = {}
for i, r in enumerate(frame.itertuples(index=False)):
    p_team_of[(r.winner, r.match_id)] = (1, float(p_v6[i]))
    p_team_of[(r.loser, r.match_id)] = (0, 1.0 - float(p_v6[i]))
imp_flag = {}
for o, eps in eps_by_org.items():
    seq = org_seq[o]
    mid_of = {m: i for i, (d, m) in enumerate(seq)}
    for e in eps:
        epr = ep.iloc[e["row"]]
        pos = mid_of[int(epr.change_match_id)]
        vals = [p_team_of[(o, m)][0] - p_team_of[(o, m)][1]
                for d, m in seq[pos:pos + 3] if (o, m) in p_team_of]
        if vals:
            imp_flag[e["row"]] = bool(np.mean(vals) > 0)

msc_w = np.full(N, np.nan)
msc_l = np.full(N, np.nan)
side_info = []
for i, r in enumerate(frame.itertuples(index=False)):
    inf = []
    for org, arr in ((r.winner, msc_w), (r.loser, msc_l)):
        s = st_ix.get((org, r.match_id))
        if s is not None:
            arr[i] = s[0]
            inf.append((org, s[0], s[1], s[2], s[3]))
        else:
            inf.append((org, None, -1, np.nan, 0))
    side_info.append(inf)
mn = np.fmin(msc_w, msc_l)
post_le3 = ~np.isnan(mn) & (mn <= 3)
post_4_10 = ~np.isnan(mn) & (mn >= 4) & (mn <= 10)
stable = ~np.isnan(mn) & (mn > 10)


def min_side(i):
    a, b = side_info[i]
    if a[1] is None:
        return b
    if b[1] is None:
        return a
    return a if a[1] <= b[1] else b


def magc(ov):
    return "keep4" if ov >= 0.8 else ("keep3" if ov >= 0.6 else "overhaul")


mag_of_row = np.array([
    (magc(min_side(i)[3]) if post_le3[i] and min_side(i)[2] >= 0
     and min_side(i)[4] == 1 else "") for i in range(N)])
gated = np.zeros(N, dtype=bool)
chg_sides = [[] for _ in range(N)]
for i in range(N):
    for (org, msc, epr, ovv, sus) in side_info[i]:
        if msc is not None and msc <= 2 and epr >= 0 and sus == 1:
            chg_sides[i].append((org, epr, ovv))
            if ovv <= 0.6:
                gated[i] = True
improve_rows = np.zeros(N, dtype=bool)
degrade_rows = np.zeros(N, dtype=bool)
for i in range(N):
    if len(chg_sides[i]) == 1 and chg_sides[i][0][1] in imp_flag:
        (improve_rows if imp_flag[chg_sides[i][0][1]] else degrade_rows)[i] = True
BUCKETS = [
    ("post-change <=3 (either team, power def)", post_le3),
    ("post-change <=3 · keep4", post_le3 & (mag_of_row == "keep4")),
    ("post-change <=3 · keep3", post_le3 & (mag_of_row == "keep3")),
    ("post-change <=3 · overhaul", post_le3 & (mag_of_row == "overhaul")),
    ("post-change 4-10", post_4_10),
    ("stable (>10)", stable),
    ("gated (first-3 of sustained ov<=0.6 change)", gated),
    ("improvement cases (retrospective slice)", improve_rows),
    ("degradation cases (retrospective slice)", degrade_rows),
]
MDE = {"within_milli": 1.773, "cross_milli": 5.889,
       "post_le3_within_milli": 2.52, "post_le3_cross_milli": 7.81,
       "sources": ["stats/power_mde_expanded.json checkpoint_quote (n=1217)",
                   "stats/power_mde.json roster<=3 bucket (n=598, frozen-npz holdout)"]}
EXPL = ("EXPLORATORY — spent holdout; operator-directed clean mechanism test "
        "on v6 (preregister ADDENDUM 2); read 4 of 4, tallied in "
        "stats/roster_looks.json; not confirmatory, not promotable.")

# ── engine subclass: v6 + overreact boost ───────────────────────────────────
from engine import Engine  # noqa: E402


class EngineOverreact(Engine):
    def enable_overreact(self, a, tau):
        self._e_a, self._e_tau = a, tau
        self._rows_of = {}
        for t, ti in self.tidx.items():
            self._rows_of[t] = np.where((self.wi == ti) | (self.li == ti))[0]

    def _continuity_vec(self, ref_date_s, mode, persistence):
        cw, cl = super()._continuity_vec(ref_date_s, mode, persistence)
        if getattr(self, "_e_a", None) is None:
            return cw, cl
        D = ref_date_s
        a, tau = self._e_a, self._e_tau
        for org, eps in eps_by_org.items():
            if org not in self.tidx:
                continue
            act = None
            for e in reversed(eps):     # most recent visible sustained episode
                if e["d"] < D and sustained_at(e, D):
                    act = e
                    break
            if act is None or act["ov"] >= 1.0:
                continue
            dd = org_dates[org]
            n_since = bisect_left(dd, D) - bisect_left(dd, act["d"])
            if n_since < 1 or n_since / tau > 12:
                continue
            m = 1.0 + a * (1.0 - act["ov"]) * math.exp(-n_since / tau)
            if m <= 1.0 + 1e-12:
                continue
            rows = self._rows_of[org]
            post = rows[self.g_date[rows] >= act["d"]]
            if len(post) == 0:
                continue
            ti = self.tidx[org]
            ws = self.wi[post] == ti
            cw[post[ws]] *= m * m          # sqrt(cw*cl) => x m per side
            cl[post[~ws]] *= m * m
        return cw, cl


def p_series_closed(beta, rdiff, fm):
    pm = 1.0 / (1.0 + np.exp(-beta * rdiff))
    return np.where(np.isin(fm, ("bo5", "bo5_gf")),
                    pm ** 3 * (1 + 3 * (1 - pm) + 6 * (1 - pm) ** 2),
                    np.where(fm == "bo1", pm, pm ** 2 * (3 - 2 * pm)))


stage_by_mid = dict(zip(frame.match_id, frame.stage))
V6_CFG = {"rd": {"power": 0.75, "scale": 2.5}, "roster_mode": "year",
          "roster_persistence": 0.3, "ridge": 0.5, "champ_mult": 2.0,
          "region_prior_ridge": 1.5,
          "decay": {"kind": "games", "consistency": (20.0, 12.0)}}


def run_cfg(a, tau, daily=False):
    eng = EngineOverreact()
    eng.series = frame.copy().reset_index(drop=True)
    eng.pred_days = sorted(frame.date.unique())
    g_stage = np.array([stage_by_mid.get(g["match_id"], "groups") for g in eng.games])
    cfg = dict(V6_CFG, w_custom=np.where(np.isin(g_stage, ("playoffs", "grand_final")), 1.6, 1.0))
    if daily:
        cfg["daily_out"] = True
    if a is not None:
        eng.enable_overreact(a, tau)
    return eng, eng.run(cfg)


# sanity guard: a=None reproduces the stored v6 baseline exactly
_, out0 = run_cfg(None, None)
rd0 = out0["rdiff"]
p0 = np.full(N, np.nan)
ok0 = ~np.isnan(rd0)
p0[ok0] = p_series_closed(out0["beta"], rd0[ok0], fmts[ok0])
mx = float(np.nanmax(np.abs(p0 - p_v6)))
assert mx <= 1e-12 and abs(out0["beta"] - float(v6["beta"][0])) <= 1e-9, \
    f"v6 reproduction guard failed: max|dp|={mx}"
print(f"v6 reproduction guard OK (max|dp|={mx:.1e})", flush=True)

print("=== Read 4 (e): grid, TRAIN ONLY ===", flush=True)
grid = []
store = {}
for a in (0.5, 1.0, 2.0):
    for tau in (2.0, 5.0):
        _, out = run_cfg(a, tau)
        ll_tr = out["ll_train"]
        store[(a, tau)] = {"rdiff": out["rdiff"], "beta": out["beta"]}
        for k in ("ll_test", "brier_test", "p_test"):
            out.pop(k, None)            # LOOK HYGIENE: scrub before recording
        grid.append({"a": a, "tau": tau, "beta": store[(a, tau)]["beta"],
                     "ll_train": ll_tr})
        print(f"  a={a} tau={tau}: ll_train={ll_tr} (holdout scrubbed)", flush=True)
grid.sort(key=lambda x: x["ll_train"])
sel = grid[0]
a_s, tau_s = sel["a"], sel["tau"]
rd_e = store[(a_s, tau_s)]["rdiff"]
beta_e = store[(a_s, tau_s)]["beta"]
valid_e = ~np.isnan(rd_e)
p_e = np.full(N, np.nan)
p_e[valid_e] = p_series_closed(beta_e, rd_e[valid_e], fmts[valid_e])

# ── the ONE holdout read + judging (same schema as the other reads) ─────────
m = hold & valid_v6 & valid_e
d = referee.delta_vector(p_e[m], p_v6[m])
iid = referee.paired_bootstrap_crn(d, mode="iid")
blk = referee.paired_bootstrap_crn(d, mode="block_event", event_ids=event_ids[m])
roi = referee.expected_roi_of_dll(float(d.mean()), p_v6[m])
buckets = []
for name, mask in BUCKETS:
    mm = m & mask
    nn = int(mm.sum())
    if nn < 10:
        buckets.append({"name": name, "n": nn, "note": "n<10, suppressed"})
        continue
    db = referee.delta_vector(p_e[mm], p_v6[mm])
    bb = referee.paired_bootstrap_crn(db, mode="iid")
    buckets.append({"name": name, "n": nn,
                    "delta_milli": round(float(db.mean()) * 1000, 2),
                    "ci_lo_milli": round(bb["ci_lo"] * 1000, 2),
                    "ci_hi_milli": round(bb["ci_hi"] * 1000, 2),
                    "p_better": bb["p_better"]})
dll = float(d.mean())
sym = referee.expected_roi_of_dll(abs(dll), p_v6[m])
res = {
    "label": f"(e) v6 + post-change overreaction a={a_s} tau={tau_s} (operator-directed)",
    "epistemic_status": EXPL,
    "n_scored": int(m.sum()),
    "ll_holdout": round(float(referee.per_series_ll(p_e[m]).mean()), 5),
    "ll_v6_same_rows": round(float(referee.per_series_ll(p_v6[m]).mean()), 5),
    "delta_milli": round(dll * 1000, 3),
    "iid": iid, "block_event": blk, "expected_roi": roi,
    "expected_roi_both_units": {
        "referee_ladder_at_signed_dll": roi["expected_roi_delta"],
        "note": "ladder defined for improvements; symmetric first-order "
                "reading below (sign flipped when dLL<0)",
        "symmetric_reading_roi_delta": (-sym["expected_roi_delta"] if dll < 0
                                        else sym["expected_roi_delta"]),
        "dll_milli": round(dll * 1000, 3),
        "delta_logit_equiv_abs": sym["delta_logit_equiv"],
        "ladder_source": sym["ladder_source"]},
    "buckets": buckets, "mde_context": MDE,
    "spec": {
        "base": "v6 config EXACTLY (stored-baseline reproduction guard passed "
                "to 1e-12 before the grid)",
        "mechanism": "post-change games of a team with a sustained-at-D "
                     "episode get weight x m, m = 1 + a(1-ov) exp(-n_since/tau); "
                     "pre-change games untouched ((b) adjudicated the discount "
                     "side); per-side m^2 folded into the continuity vector; "
                     "corpus-wide (deployable rule)",
        "grid_train_only": grid, "selected": {"a": a_s, "tau": tau_s},
        "train_tie_lt_0p1m": bool((grid[1]["ll_train"] - grid[0]["ll_train"]) * 1000 < 0.1),
        "beta_refit_train": beta_e,
        "attribution_note": "read (d) confounded base-model replacement "
                            "(state-space base −11.75m vs v6 stored) with the "
                            "mechanism; (e) is the clean attribution ON v6 — "
                            "operator directive, preregister ADDENDUM 2",
    },
}
tr = json.load(open(os.path.join(STATS, "roster_treatments.json")))
tr["read4_e_v6_overreact"] = res
tr["read4_reserve"] = {"used": True,
                       "used_for": "read 4 (e) by OPERATOR DIRECTIVE "
                                   "(preregister ADDENDUM 2), not the tie rule",
                       "rule": tr["read4_reserve"]["rule"],
                       "ties": tr["read4_reserve"]["ties"]}
tr["read3_d_phase_reset"]["spec"]["attribution_caveat"] = (
    "CONFOUNDED TEST of the mechanism: the no-injection state-space base is "
    "itself −11.75m vs v6 (stored), so (d) could not cleanly falsify "
    "phase-reset-on-v6. The clean attribution is read4_e_v6_overreact.")
with open(os.path.join(STATS, "roster_treatments.json"), "w") as f:
    json.dump(tr, f, indent=1)
post = next(b for b in buckets if b["name"].startswith("post-change <=3 ("))
print(f"READ 4 (e): a={a_s} tau={tau_s} delta={res['delta_milli']}m "
      f"iid [{iid['ci_lo']*1000:.2f},{iid['ci_hi']*1000:.2f}] "
      f"blk [{blk['ci_lo']*1000:.2f},{blk['ci_hi']*1000:.2f}] "
      f"p_b {iid['p_better']:.3f}/{blk['p_better']:.3f}; post<=3 "
      f"{post['delta_milli']}m [{post['ci_lo_milli']},{post['ci_hi_milli']}]",
      flush=True)

looks = json.load(open(os.path.join(STATS, "roster_looks.json")))
looks["new_reads"].append({"read": 4, "config": f"v6_overreact_a{a_s}_tau{tau_s}",
                           "ll_holdout": res["ll_holdout"],
                           "delta_milli_vs_v6": res["delta_milli"],
                           "status": "EXPLORATORY", "operator_specified": True,
                           "reserve_consumed_by": "operator directive (ADDENDUM 2)"})
looks["new_primary_looks"] = len(looks["new_reads"])
looks["grid_points_train_only"]["read4_grid"] = [[g["a"], g["tau"]] for g in grid]
looks["grand_total_after"] = looks["prior_grand_total_recorded_holdout_numbers"] + len(looks["new_reads"])
with open(os.path.join(STATS, "roster_looks.json"), "w") as f:
    json.dump(looks, f, indent=1)
print(f"looks: grand total now {looks['grand_total_after']}", flush=True)

# ── daily trajectories at the selected config for the overlays ──────────────
print("=== daily run at selected (a, tau) for overlay paths ===", flush=True)
eng_d, out_d = run_cfg(a_s, tau_s, daily=True)
rd_chk = out_d["rdiff"]
assert np.allclose(rd_chk[~np.isnan(rd_chk)], rd_e[valid_e], atol=1e-12), \
    "daily rerun drifted from the scored run"
days_d = sorted(out_d["daily_r"].keys())
Rd = np.stack([out_d["daily_r"][dd] for dd in days_d])
tidx_d = eng_d.tidx


def emit_overlay(doc_path, get_ov, org, first_change):
    doc = json.load(open(doc_path))
    ovl = get_ov(doc)
    v6d = {p["d"]: p["r"] for p in ovl["v6_path"]}
    dates = [p["d"] for p in ovl["v6_path"]]     # same coverage as v6_path
    j = tidx_d[org]
    path = [{"d": dd, "r": round(float(Rd[k, j]), 3)}
            for k, dd in enumerate(days_d) if dates[0] <= dd <= dates[-1]]
    assert [p["d"] for p in path] == dates, f"{org}: date coverage mismatch vs v6_path"
    pre = [(p["r"], v6d[p["d"]]) for p in path if p["d"] < first_change]
    mx_pre = max((abs(a_ - b_) for a_, b_ in pre), default=0.0)
    off = float(np.mean([a_ - b_ for a_, b_ in pre])) if pre else None
    ovl["v6_overreact_path"] = path
    ovl["v6_overreact_check"] = {
        "config": f"read (e) selected a={a_s}, tau={tau_s}, ON THE V6 BASE — "
                  "chart-comparable to v6_path with no base-model confound",
        "first_change_in_window": first_change,
        "pre_change_equals_v6": bool(mx_pre <= 1e-9),
        "pre_change_max_abs_diff": round(mx_pre, 6),
        "pre_change_mean_offset_vs_v6": round(off, 6) if off is not None else None,
        "n_pre_change_points": len(pre),
        "note": "corpus-wide rule (preregistered): other teams' boosts couple "
                "through the global Massey solve, so small pre-change "
                "deviations from v6_path are expected and reported, not "
                "zeroed — same honesty standard as base_check",
    }
    with open(doc_path, "w") as f:
        json.dump(doc, f, indent=1)
    print(f"{org}: pre-change max|e−v6|={mx_pre:.6f} mean offset="
          f"{off if off is None else round(off, 6)} (n={len(pre)})", flush=True)


emit_overlay(os.path.join(STATS, "roster_case_envy.json"),
             lambda d_: d_["phase_reset_overlay"], "ENVY", "2026-02-06")
emit_overlay(os.path.join(STATS, "roster_case_gallery.json"),
             lambda d_: next(c for c in d_["named_cases"] if c["org"] == "LEV")["phase_reset_overlay"],
             "LEV", "2025-11-29")

integ = json.load(open(os.path.join(STATS, "roster_integration.json")))
integ["prospective_validation_plan"]["arms_frozen"]["F_v6_overreact"] = (
    f"read-4 spec, a={a_s} tau={tau_s} on v6 (operator-directed clean "
    f"mechanism test; exploratory read {res['delta_milli']:+.3f}m)")
integ["exploratory_reads_context"]["read4_e_v6_overreact"] = {
    "delta_milli": res["delta_milli"], "status": "EXPLORATORY"}
with open(os.path.join(STATS, "roster_integration.json"), "w") as f:
    json.dump(integ, f, indent=1)
print("integration arm F_v6_overreact added", flush=True)
