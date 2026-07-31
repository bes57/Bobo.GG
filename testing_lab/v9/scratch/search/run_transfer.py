"""v9 search — the 5 one-shot transfer evaluations + candidates file + ladder.

Preregistered: v9/preregister.search.md sections 1b, 2, 3, 4 (LOCKED before
the grid ran); nominations frozen in logs/search.log BEFORE this script
touches any VAL row. Protocol: stats/v9_transfer_protocol.json VERBATIM.
Every evaluation is appended to stats/v9_looks.json selection_reads at run
time, one entry per candidate, before any downstream use. Overlay adj is
computed via the FULL un-cached OverlaySearch.run_general path (the family
evidence leak assert d_i < D executes on every evidence row).
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import searchlib as sl  # noqa: E402
from searchlib import OverlaySearch  # noqa: E402
import v9lib  # noqa: E402
import runner  # noqa: E402

TL = "/Users/benny_es1/PythonTest/testing_lab"
V8 = os.path.join(TL, "v8")
V9 = os.path.join(TL, "v9")
SPEC = os.path.join(V8, "scratch", "roster", "spec_run")
LOG = os.path.join(V9, "logs", "search.log")
LOOKS = os.path.join(V9, "stats", "v9_looks.json")
t0 = time.time()


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


frame = sl.load_frame_checked()
W = sl.windows(frame)
fmts = frame.fmt.values
dates = frame.date.values
years = frame.year.values
event_ids = frame.event_id.values

corpus = v9lib.load_corpus()
base = v9lib.v6_run()
valid = base["valid"]
rd_v6 = base["rdiff"]

# solve-side arrays (deterministic, the same the autopsy used)
man = json.load(open(os.path.join(SPEC, "stage_manifest.json")))
npz = np.load(os.path.join(SPEC, "stage_runs.npz"))
rd_v6_npz = npz["p1w5c5_a0.0_t2.0_s0.0"]
assert np.array_equal(rd_v6, rd_v6_npz, equal_nan=True), \
    "fresh v6 rdiff != stored npz v6 (nesting broken?)"

# (tau, s) per small a: manifest ll_train argmin (FIT1-legal, preregistered)
solve_points = {}
for a in (0.5, 1.0, 2.0):
    cands = {k: m for k, m in man["configs"].items()
             if k.startswith("p1w5c5_") and m["a"] == a}
    best = min(cands, key=lambda k: cands[k]["ll_train"])
    solve_points[a] = {"key": best, "tau": cands[best]["tau"],
                      "s": cands[best]["s"],
                      "ll_train": cands[best]["ll_train"]}
    log(f"solve-side a={a}: manifest ll_train argmin -> {best} "
        f"(ll_train={cands[best]['ll_train']})")

# window betas for v6 (shared across all paired evaluations)
b_v6_fit1 = sl.fit_beta(rd_v6, W["FIT1"], fmts, valid)
assert abs(b_v6_fit1 - 0.1152) <= 1e-3, "beta fixture failed — abort"
b_v6_fit2 = sl.fit_beta(rd_v6, W["FIT2"], fmts, valid)
log(f"beta(FIT1,v6)={b_v6_fit1:.6f} (fixture PASS) "
    f"beta(FIT2,v6)={b_v6_fit2:.6f} [{time.time()-t0:.0f}s]")

# overlay nominees (frozen in logs/search.log before this run)
plan_w5 = v9lib.build_plan("p1", W=5, corpus=corpus)
ov = OverlaySearch(plan_w5, frame=frame)
grid = json.load(open(os.path.join(V9, "stats", "v9_search_grid.json")))


def grid_row(policy, variant, m, b, tau, gamma):
    for r in grid["configs"]:
        if (r["policy"], r["variant"], r["m"], r["b"], r["tau"],
                r["gamma"]) == (policy, variant, m, b, tau, gamma):
            return r
    raise KeyError("nominee not in grid")


NOMINEES = [
    {"id": "N1_delta2", "family": "delta2",
     "config": {"mechanism": "prediction_layer", "policy": "p1", "W": 5,
                "c": 5, "variant": "delta2", "m": 0, "b": 0.65, "tau": 13.0,
                "gamma": 0.5},
     "fit1_diagnostics": {"drop_top5_milli": -7.29, "top10_row_share": 0.748,
                          "c5_time_folds": "CLEAN (drop-one-fold means all > +3.6m)"}},
    {"id": "N2_hybrid6", "family": "hybrid",
     "config": {"mechanism": "prediction_layer", "policy": "p1", "W": 5,
                "c": 5, "variant": "hybrid", "m": 6, "b": 0.20, "tau": 21.0,
                "gamma": 0.5},
     "fit1_diagnostics": {"drop_top5_milli": -3.763, "top10_row_share": 0.725,
                          "c5_time_folds": "CLEAN (drop-one-fold means all > +2.5m)"}},
]

candidates = []
looks_entries = []


def ledger(entry):
    """Append ONE selection read at run time (protocol disclosure clause)."""
    looks = json.load(open(LOOKS))
    looks["selection_reads"]["entries"].append(entry)
    with open(LOOKS, "w") as f:
        json.dump(looks, f, indent=1)
    looks_entries.append(entry["candidate"])


def clauses_failed(adv):
    return [k for k, v in adv.items() if v is False and k != "ADVANCE"]


# ── solve-side verification points (presumed dead; predicted BLOCK) ─────────
for a, sp in solve_points.items():
    key = sp["key"]
    rd = npz[key]
    assert (~np.isnan(rd) == valid).all(), f"valid-mask drift in {key}"
    label = (f"SOLVE p1w5c5 a={a} t={sp['tau']} s={sp['s']} cap=None "
             f"(manifest argmin; verification point)")
    r = sl.transfer_eval(rd, None, rd_v6, frame, W, valid, label,
                         b_v6_fit1, b_v6_fit2)
    adv = r["advance_rule"]
    ledger({"candidate": label, "date": time.strftime("%Y-%m-%d"),
            "T1": r["T1_fit2324_val2025"]["delta_milli"],
            "T2": r["T2_fit2325_val2026H1"]["delta_milli"],
            "pooled": r["pooled_validation"]["delta_milli"],
            "verdict": "ADVANCE" if adv["ADVANCE"] else "DIE",
            "clauses_failed": clauses_failed(adv),
            "agent": "v9-search",
            "note": "one-shot solve-side verification (preregister 1b)"})
    candidates.append({
        "id": f"S_a{a}", "family": "solve_side",
        "config": {"mechanism": "solve_side", "policy": "p1", "W": 5, "c": 5,
                   "a": a, "tau": sp["tau"], "s": sp["s"], "n_min": 3,
                   "cap": None, "npz_key": key},
        "provenance": {"tau_s_rule": "stage_manifest ll_train argmin at this a "
                                     "(FIT1-legal)",
                       "ll_train": sp["ll_train"],
                       "C5": "not applicable — no inner-CV gate score exists "
                             "for this config (manifest ll_train only); disclosed"},
        "transfer": r, "verdict": "ADVANCE" if adv["ADVANCE"] else "DIE"})
    log(f"{label}: T1 {r['T1_fit2324_val2025']['delta_milli']:+.2f}m "
        f"(bar {r['T1_fit2324_val2025']['se_blk_milli']:.2f}) | "
        f"T2 {r['T2_fit2325_val2026H1']['delta_milli']:+.2f}m "
        f"(bar {r['T2_fit2325_val2026H1']['se_blk_milli']:.2f}) | "
        f"pooled {r['pooled_validation']['delta_milli']:+.2f}m "
        f"blkCI {r['pooled_validation']['blk_ci_milli']} | "
        f"{'ADVANCE' if adv['ADVANCE'] else 'DIE'} "
        f"failed={clauses_failed(adv)} [{time.time()-t0:.0f}s]")

# ── prediction-layer nominees ───────────────────────────────────────────────
for nom in NOMINEES:
    c = nom["config"]
    gr = grid_row("p1w5c5", c["variant"], c["m"], c["b"], c["tau"], c["gamma"])
    out = ov.run_general(base, c["variant"], c["b"], c["tau"],
                         gamma=c["gamma"], m=c["m"])   # full path, asserts hot
    label = (f"OVERLAY {c['variant']}(m={c['m']}) p1w5c5 b={c['b']} "
             f"tau={c['tau']} gamma={c['gamma']}")
    r = sl.transfer_eval(rd_v6, out["adj"], rd_v6, frame, W, valid, label,
                         b_v6_fit1, b_v6_fit2)
    adv = r["advance_rule"]
    ledger({"candidate": label, "date": time.strftime("%Y-%m-%d"),
            "T1": r["T1_fit2324_val2025"]["delta_milli"],
            "T2": r["T2_fit2325_val2026H1"]["delta_milli"],
            "pooled": r["pooled_validation"]["delta_milli"],
            "verdict": "ADVANCE" if adv["ADVANCE"] else "DIE",
            "clauses_failed": clauses_failed(adv),
            "agent": "v9-search",
            "note": f"one-shot; nominated by grid rule ({nom['family']} "
                    f"family top era_min)"})
    candidates.append({
        "id": nom["id"], "family": nom["family"], "config": c,
        "provenance": {"grid_row": gr, "fit1_flag_investigation":
                       nom["fit1_diagnostics"],
                       "beta_ref_evidence": base["beta"],
                       "horizon": v9lib.horizon(c["tau"])},
        "transfer": r, "verdict": "ADVANCE" if adv["ADVANCE"] else "DIE"})
    log(f"{label}: T1 {r['T1_fit2324_val2025']['delta_milli']:+.2f}m "
        f"(bar {r['T1_fit2324_val2025']['se_blk_milli']:.2f}) | "
        f"T2 {r['T2_fit2325_val2026H1']['delta_milli']:+.2f}m "
        f"(bar {r['T2_fit2325_val2026H1']['se_blk_milli']:.2f}) | "
        f"pooled {r['pooled_validation']['delta_milli']:+.2f}m "
        f"blkCI {r['pooled_validation']['blk_ci_milli']} | "
        f"{'ADVANCE' if adv['ADVANCE'] else 'DIE'} "
        f"failed={clauses_failed(adv)} [{time.time()-t0:.0f}s]")

# ── hybrid solve+delta clause (preregister 1c) ──────────────────────────────
solve_adv = [c for c in candidates if c["family"] == "solve_side"
             and c["verdict"] == "ADVANCE"]
pred_adv = [c for c in candidates if c["family"] != "solve_side"
            and c["verdict"] == "ADVANCE"]
hybrid_note = ("SKIPPED — preregister 1c requires BOTH parents to advance; "
               f"solve-side advanced: {len(solve_adv)}, prediction-layer "
               f"advanced: {len(pred_adv)}")
log(f"hybrid solve+delta clause: {hybrid_note}")

# ── candidates file ─────────────────────────────────────────────────────────
survivors = sorted([c for c in candidates if c["verdict"] == "ADVANCE"],
                   key=lambda c: -c["transfer"]["pooled_validation"]["delta_milli"])[:3]
cand_out = {
    "written_by": "agent:v9-search",
    "written": time.strftime("%Y-%m-%d %H:%M:%S"),
    "preregistered": "v9/preregister.search.md (locked before grid); "
                     "nominations frozen in logs/search.log before any VAL contact",
    "protocol": "stats/v9_transfer_protocol.json verbatim; one ledgered "
                "evaluation per candidate (5 total)",
    "delta1_family": "nominated NOTHING — zero eligible configs on FIT1 "
                     "(best dF +0.81m < 1.0m floor, LOEO negative); "
                     "pure evidence-sign carries no diffuse FIT1 gain",
    "p3w5c5": "fully v6-identity in the grid (0 active pricings; no "
              "boundaries under frozen-o at c=5)",
    "hybrid_solve_plus_delta": hybrid_note,
    "candidates": candidates,
    "n_advanced": len(survivors),
    "survivor_ids": [c["id"] for c in survivors],
}
cpath = os.path.join(V9, "stats", "v9_candidates.json")
with open(cpath, "w") as f:
    json.dump(cand_out, f, indent=1)
log(f"wrote {cpath} — {len(survivors)} advanced of {len(candidates)} evaluated")

# ── ladder freeze (protocol section 4; v6 is arm 0 always) ──────────────────
full_mask = (dates <= "2026-07-28")
beta_v6_full = sl.fit_beta(rd_v6, full_mask, fmts, valid)
arms = [{"arm": 0, "id": "v6", "label": "pure v6 (baseline; always arm 0)",
         "mechanism": "v6", "beta_frozen": round(beta_v6_full, 6),
         "beta_window": "2023-01-01..2026-07-28 (n_valid rows "
                        f"{int((full_mask & valid).sum())}), protocol beta method"}]
ordered = sorted(survivors,
                 key=lambda c: c["transfer"]["mean_abs_p_gap_val2"])
for i, c in enumerate(ordered, start=1):
    cfg = dict(c["config"])
    if cfg["mechanism"] == "prediction_layer":
        bfull = beta_v6_full          # same rdiff as v6 by construction
        extra = {"beta_ref_evidence": base["beta"],
                 "horizon_matches": v9lib.horizon(cfg["tau"]),
                 "note": "solve is pure v6; beta_frozen shared with arm 0 "
                         "(identical rdiff, identical objective)"}
    else:
        bfull = sl.fit_beta(npz[cfg["npz_key"]], full_mask, fmts, valid)
        extra = {}
    arms.append({"arm": i, "id": c["id"],
                 "label": c["transfer"]["label"],
                 "mechanism": cfg["mechanism"], "config": cfg,
                 "beta_frozen": round(float(bfull), 6),
                 "beta_window": "2023-01-01..2026-07-28, protocol beta method",
                 "pooled_delta_milli":
                     c["transfer"]["pooled_validation"]["delta_milli"],
                 "mean_abs_p_gap_val2":
                     c["transfer"]["mean_abs_p_gap_val2"], **extra})
ladder = {
    "written_by": "agent:v9-search",
    "written": time.strftime("%Y-%m-%d %H:%M:%S"),
    "status": "FROZEN — consumed verbatim by the prospective evaluator "
              "(stats/v9_prospective_protocol.json). Betas refit once on "
              "2023-01-01..2026-07-28 and never again.",
    "ladder_rule": "survivors ranked by pooled delta; ladder order "
                   "conservative -> aggressive by mean|p_cand - p_v6| on "
                   "VAL2 ascending (protocol survivors clause)",
    "n_arms": len(arms),
    "arms": arms,
    "verdict_sentence": ("the ladder is v6 alone" if len(arms) == 1 else
                         f"v6 + {len(arms)-1} survivor(s), conservative first"),
}
lpath = os.path.join(V9, "stats", "v9_ladder.json")
with open(lpath, "w") as f:
    json.dump(ladder, f, indent=1)
log(f"wrote {lpath}: {ladder['verdict_sentence']} "
    f"(beta_v6_full={beta_v6_full:.6f}) [{time.time()-t0:.0f}s]")
log(f"looks ledger: {len(looks_entries)} selection_reads appended this phase")
