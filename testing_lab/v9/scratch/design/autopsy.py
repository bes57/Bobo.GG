"""v9 gate autopsy — preregister.design.md section 3, run AFTER lock.

Reconstructs, for the four v8 spec-run policies, the three-column table
  "gate said / transfer would have said / holdout said"
using the v9 transfer protocol (section 2) verbatim on the configs the v8
gate actually selected. Sources: stage_runs.npz stored full-corpus rdiff
arrays (cap=None) + ONE fresh deterministic engine run for the shipped
config (cap=1.5). All numbers ledgered in stats/v9_looks.json.
"""
import json
import os
import sys
import time

import numpy as np
from scipy.optimize import minimize_scalar

TL = "/Users/benny_es1/PythonTest/testing_lab"
V8 = os.path.join(TL, "v8")
V9 = os.path.join(TL, "v9")
SPEC = os.path.join(V8, "scratch", "roster", "spec_run")
sys.path.insert(0, SPEC)
sys.path.insert(0, V8)
sys.path.insert(0, TL)

import referee  # noqa: E402
import runner  # noqa: E402
from speclib import SpecPlan, load_corpus  # noqa: E402

t0 = time.time()
OUT_PATH = os.path.join(V9, "stats", "v9_gate_autopsy.json")
LOOKS_PATH = os.path.join(V9, "stats", "v9_looks.json")

frame = runner.load_frame()          # sha-verified inside (aborts on mismatch)
fmts = frame.fmt.values
event_ids = frame.event_id.values
dates = frame.date.values

W = {
    "FIT1": (dates <= "2024-12-31"),
    "VAL1": (dates > "2024-12-31") & (dates <= "2025-12-31"),
    "FIT2": (dates <= "2025-12-31"),
    "VAL2": (dates > "2025-12-31") & (dates <= "2026-07-28"),
}
HOLD = W["VAL1"] | W["VAL2"]
SIG_W = 0.02207
ZPOW = 2.8016


def mde_milli(n):
    return round(ZPOW * SIG_W / np.sqrt(n) * 1000, 3)


gate = json.load(open(os.path.join(SPEC, "gate_decision.json")))
man = json.load(open(os.path.join(SPEC, "stage_manifest.json")))
npz = np.load(os.path.join(SPEC, "stage_runs.npz"))

# ── the shipped config: ONE fresh deterministic run (cap=1.5) ───────────────
print("fresh run: p1w8c5 a=28 t=13 s=0.7 n_min=3 cap=1.5 (the ship)",
      flush=True)
corpus = load_corpus()
plan_w8 = SpecPlan("p1", W=8, c=5, corpus=corpus)
ship = runner.run_config(plan_w8, 28.0, 13.0, 0.7, n_min=3, cap=1.5)
print(f"  engine beta={ship['beta']} ll_train={ship['ll_train']} "
      f"[{time.time()-t0:.0f}s]", flush=True)

CONFIGS = {
    "v6": {"rdiff": npz["p1w5c5_a0.0_t2.0_s0.0"],
           "engine_beta": man["configs"]["p1w5c5_a0.0_t2.0_s0.0"]["beta"],
           "label": "v6 (a=0,s=0; nesting-asserted bit-identical)"},
    "p1w3c5": {"rdiff": npz["p1w3c5_a4.5_t13.0_s1.0"],
               "engine_beta": man["configs"]["p1w3c5_a4.5_t13.0_s1.0"]["beta"],
               "label": "p1w3c5 a=4.5 t=13 s=1.0 cap=None (gate-selected)"},
    "p1w5c5": {"rdiff": npz["p1w5c5_a4.5_t13.0_s0.7"],
               "engine_beta": man["configs"]["p1w5c5_a4.5_t13.0_s0.7"]["beta"],
               "label": "p1w5c5 a=4.5 t=13 s=0.7 cap=None (gate-selected)"},
    "p1w8c5_ship": {"rdiff": ship["rdiff"], "engine_beta": ship["beta"],
                    "label": "p1w8c5 a=28 t=13 s=0.7 n_min=3 cap=1.5 (SHIPPED)"},
    "p1w8c5_capNone": {"rdiff": npz["p1w8c5_a28.0_t13.0_s0.7"],
                       "engine_beta": man["configs"]["p1w8c5_a28.0_t13.0_s0.7"]["beta"],
                       "label": "p1w8c5 a=28 cap=None (stage array; sensitivity)"},
    "EXTRA_p1w8c5_a6": {"rdiff": npz["p1w8c5_a6.0_t13.0_s0.7"],
                        "engine_beta": man["configs"]["p1w8c5_a6.0_t13.0_s0.7"]["beta"],
                        "label": "EXTRA p1w8c5 a=6 t=13 s=0.7 cap=None (v9 cap boundary)"},
}

# valid mask identical across configs (gate_cv asserted for stage arrays;
# re-assert including the fresh run)
valid = ~np.isnan(CONFIGS["v6"]["rdiff"])
for k, c in CONFIGS.items():
    v = ~np.isnan(c["rdiff"])
    assert (v == valid).all(), f"valid-mask drift in {k}"


def fit_beta(rdiff, mask):
    m = mask & valid

    def nll(b):
        p = runner.p_series_closed(b, rdiff[m], fmts[m])
        return -np.mean(np.log(np.clip(p, 1e-9, 1.0)))

    r = minimize_scalar(nll, bounds=(0.001, 1.0), method="bounded",
                        options={"xatol": 1e-6})
    return float(r.x)


def probs(rdiff, beta):
    p = np.full(len(frame), np.nan)
    p[valid] = runner.p_series_closed(beta, rdiff[valid], fmts[valid])
    return p


# fixture: beta(FIT1, v6) must reproduce the engine's train fit 0.1152
b_v6_fit1 = fit_beta(CONFIGS["v6"]["rdiff"], W["FIT1"])
fixture_ok = abs(b_v6_fit1 - 0.1152) <= 1e-3
print(f"fixture beta(FIT1, v6) = {b_v6_fit1:.6f} vs engine 0.1152 "
      f"-> {'PASS' if fixture_ok else 'FAIL'}", flush=True)
assert fixture_ok, "beta fixture failed — abort before any read"
b_v6_fit2 = fit_beta(CONFIGS["v6"]["rdiff"], W["FIT2"])
rd_v6 = CONFIGS["v6"]["rdiff"]


def judged(d, ev):
    iid = referee.paired_bootstrap_crn(d, mode="iid")
    blk = referee.paired_bootstrap_crn(d, mode="block_event", event_ids=ev)
    se_blk = (blk["ci_hi"] - blk["ci_lo"]) / 3.92
    return {"delta_milli": round(float(d.mean()) * 1000, 3),
            "n": int(len(d)),
            "iid_ci_milli": [round(iid["ci_lo"] * 1000, 2),
                             round(iid["ci_hi"] * 1000, 2)],
            "blk_ci_milli": [round(blk["ci_lo"] * 1000, 2),
                             round(blk["ci_hi"] * 1000, 2)],
            "p_better_iid": iid["p_better"], "p_better_blk": blk["p_better"],
            "se_blk_milli": round(se_blk * 1000, 3)}


def fragility(d, ev):
    n = len(d)
    k = int(np.ceil(0.05 * n))
    keep = np.argsort(d)[: n - k]          # drop the k largest contributions
    drop5 = float(d[keep].mean())
    jk = {}
    for e in dict.fromkeys(ev):
        m = ev != e
        jk[str(e)] = round(float(d[m].mean()) * 1000, 3)
    worst_e = min(jk, key=jk.get)
    return {"drop_top5_milli": round(drop5 * 1000, 3), "n_dropped": k,
            "jackknife_min_milli": jk[worst_e], "jackknife_min_event": worst_e,
            "jackknife_all": jk}


results = {}
for key, c in CONFIGS.items():
    if key == "v6":
        continue
    rd = c["rdiff"]
    r = {"label": c["label"]}
    # transfer T1: beta on FIT1, score VAL1
    b1c, b1v = fit_beta(rd, W["FIT1"]), b_v6_fit1
    m1 = W["VAL1"] & valid
    p1c, p1v = probs(rd, b1c)[m1], probs(rd_v6, b1v)[m1]
    d1 = referee.delta_vector(p1c, p1v)
    r["T1_fit2324_val2025"] = dict(judged(d1, event_ids[m1]),
                                   beta_cand=round(b1c, 6),
                                   beta_v6=round(b1v, 6),
                                   mde_milli=mde_milli(int(m1.sum())),
                                   diagnostics=fragility(d1, event_ids[m1]))
    # transfer T2: beta on FIT2, score VAL2
    b2c, b2v = fit_beta(rd, W["FIT2"]), b_v6_fit2
    m2 = W["VAL2"] & valid
    p2c, p2v = probs(rd, b2c)[m2], probs(rd_v6, b2v)[m2]
    d2 = referee.delta_vector(p2c, p2v)
    r["T2_fit2325_val2026H1"] = dict(judged(d2, event_ids[m2]),
                                     beta_cand=round(b2c, 6),
                                     beta_v6=round(b2v, 6),
                                     mde_milli=mde_milli(int(m2.sum())),
                                     diagnostics=fragility(d2, event_ids[m2]))
    # pooled (era-legal concat of the two window d vectors)
    dp = np.concatenate([d1, d2])
    evp = np.concatenate([event_ids[m1], event_ids[m2]])
    r["pooled_validation"] = dict(judged(dp, evp),
                                  mde_milli=mde_milli(len(dp)),
                                  note="concat of era-legal window d vectors")
    frag = fragility(dp, evp)
    r["pooled_fragility"] = frag
    # advance rule A1-A5
    a1 = r["T1_fit2324_val2025"]["delta_milli"] >= \
        1.0 * r["T1_fit2324_val2025"]["se_blk_milli"]
    a2 = r["T2_fit2325_val2026H1"]["delta_milli"] >= \
        1.0 * r["T2_fit2325_val2026H1"]["se_blk_milli"]
    a3 = r["pooled_validation"]["blk_ci_milli"][0] > 0
    a4 = r["pooled_validation"]["delta_milli"] >= 1.773
    a5 = bool(frag["drop_top5_milli"] > 0 and frag["jackknife_min_milli"] > 0) \
        if (a1 and a2 and a3 and a4) else None
    r["advance_rule"] = {"A1_val1_ge_1se": bool(a1), "A2_val2_ge_1se": bool(a2),
                         "A3_pooled_ci_gt0": bool(a3),
                         "A4_pooled_ge_mde": bool(a4),
                         "A5_fragility": a5,
                         "ADVANCE": bool(a1 and a2 and a3 and a4 and (a5 or False))}
    # holdout said (v8 headline method: engine betas, pooled 1217)
    mh = HOLD & valid
    ph_c = probs(rd, c["engine_beta"])[mh]
    ph_v = probs(rd_v6, CONFIGS["v6"]["engine_beta"])[mh]
    dh = referee.delta_vector(ph_c, ph_v)
    r["holdout_v8_method"] = dict(judged(dh, event_ids[mh]),
                                  mde_milli=mde_milli(int(mh.sum())),
                                  note="engine train-fit betas, pooled spent holdout")
    # ROI translation (both units law) on the pooled validation number
    roi = referee.expected_roi_of_dll(r["pooled_validation"]["delta_milli"] / 1000.0,
                                      np.concatenate([p1v, p2v]))
    r["pooled_roi_translation"] = {
        "expected_roi_delta": roi["expected_roi_delta"],
        "delta_logit_equiv": roi["delta_logit_equiv"],
        "ladder_source": roi["ladder_source"]}
    results[key] = r
    print(f"{key}: T1 {r['T1_fit2324_val2025']['delta_milli']:+.2f}m "
          f"(SE {r['T1_fit2324_val2025']['se_blk_milli']:.2f}) | "
          f"T2 {r['T2_fit2325_val2026H1']['delta_milli']:+.2f}m "
          f"(SE {r['T2_fit2325_val2026H1']['se_blk_milli']:.2f}) | "
          f"pooled {r['pooled_validation']['delta_milli']:+.2f}m "
          f"blkCI {r['pooled_validation']['blk_ci_milli']} | "
          f"holdout {r['holdout_v8_method']['delta_milli']:+.2f}m | "
          f"ADVANCE={r['advance_rule']['ADVANCE']} [{time.time()-t0:.0f}s]",
          flush=True)

# reproduction bar for the ship vs the recorded read
rec = json.load(open(os.path.join(V8, "stats", "roster_spec_read.json")))
recorded = rec["headline"]["delta_milli"]
mine = results["p1w8c5_ship"]["holdout_v8_method"]["delta_milli"]
repro_gap = abs(mine - recorded)
print(f"reproduction: mine {mine:+.3f}m vs recorded {recorded:+.3f}m "
      f"gap {repro_gap:.3f}m (bar 0.5m)", flush=True)
assert repro_gap <= 0.5, "REPRODUCTION FAILED — halt for investigation"

# gate-jackknife (pure arithmetic on recorded fold deltas)
gate_jack = {}
for pol, g in gate["per_policy"].items():
    f5 = np.array(g["gate_delta_folds_milli"])
    rows = {}
    for i in range(5):
        rest = np.delete(f5, i)
        mn, se = float(rest.mean()), float(rest.std(ddof=1) / np.sqrt(4))
        rows[f"drop_fold_{i+1}"] = {"mean_milli": round(mn, 3),
                                    "se_milli": round(se, 3),
                                    "would_fire": bool(mn > se)}
    gate_jack[pol] = {"recorded_folds_milli": [float(x) for x in f5],
                      "recorded_fired": g["gate_fired"],
                      "leave_one_fold_out": rows,
                      "concentration_flag_C5": bool(
                          g["gate_fired"] and
                          any(not r["would_fire"] for r in rows.values()))}

# the deliverable table
def verdict_row(pol, key):
    g = gate["per_policy"][pol]
    if key is None:
        return {"gate_said": "did not fire (CV +0.0m; s-profile monotone worse)",
                "transfer_would_have_said": "N/A — nothing advanced; config is v6 identity (a=s=0)",
                "holdout_said": "N/A — no ship, delta==0 by construction"}
    r = results[key]
    adv = r["advance_rule"]
    fails = [k for k, v in adv.items() if v is False and k != "ADVANCE"]
    return {
        "gate_said": (f"FIRED: CV {g['gate_mean_milli']:+.4f}m "
                      f"(SE {g['gate_se_milli']:.4f}m), a_shrunk={g['a_shrunk']}"),
        "transfer_would_have_said": (
            f"{'ADVANCE' if adv['ADVANCE'] else 'BLOCK'} — "
            f"VAL1 {r['T1_fit2324_val2025']['delta_milli']:+.2f}m vs bar "
            f"{r['T1_fit2324_val2025']['se_blk_milli']:.2f}m; "
            f"VAL2 {r['T2_fit2325_val2026H1']['delta_milli']:+.2f}m vs bar "
            f"{r['T2_fit2325_val2026H1']['se_blk_milli']:.2f}m; "
            f"pooled {r['pooled_validation']['delta_milli']:+.2f}m "
            f"blkCI {r['pooled_validation']['blk_ci_milli']}; failed {fails}"),
        "holdout_said": (f"{r['holdout_v8_method']['delta_milli']:+.2f}m "
                         f"blkCI {r['holdout_v8_method']['blk_ci_milli']}"),
    }


table = {"p1w3c5": verdict_row("p1w3c5", "p1w3c5"),
         "p1w5c5": verdict_row("p1w5c5", "p1w5c5"),
         "p1w8c5 (the a=28 ship)": verdict_row("p1w8c5", "p1w8c5_ship"),
         "p3w5": verdict_row("p3w5", None)}

blocked = not results["p1w8c5_ship"]["advance_rule"]["ADVANCE"]
out = {
    "written_by": "agent:v9-design",
    "written": "2026-07-29",
    "preregistered": "v9/preregister.design.md section 3 (locked before this run)",
    "epistemic_status": ("autopsy on SPENT data — selection-grade only, "
                         "adjudicates nothing; commissioned by briefs/design.md "
                         "work item 2 as validator design input"),
    "protocol": "v9_transfer_protocol.json applied verbatim to the v8 gate-selected configs",
    "windows": {k: int((m & valid).sum()) for k, m in W.items()},
    "headline_verdict": {
        "question": "would era-transfer have blocked the a=28 ship?",
        "answer": ("YES — BLOCKED" if blocked else
                   "NO — TRANSFER WOULD ALSO HAVE PASSED IT (law 2 weakened; "
                   "see contingency)"),
        "detail": table["p1w8c5 (the a=28 ship)"]["transfer_would_have_said"]},
    "table_gate_transfer_holdout": table,
    "per_config": results,
    "gate_jackknife_C5": gate_jack,
    "reproduction_fixture": {
        "beta_fit1_v6": round(b_v6_fit1, 6), "engine_beta_v6": 0.1152,
        "ship_holdout_mine_milli": mine, "ship_holdout_recorded_milli": recorded,
        "gap_milli": round(repro_gap, 3), "bar_milli": 0.5, "pass": True},
    "mde_context_milli": {"VAL1": mde_milli(674), "VAL2": mde_milli(543),
                          "pooled": mde_milli(1217),
                          "source": "sigma_within 0.02207, z 2.8016 "
                                    "(power_mde_expanded.json)"},
}
with open(OUT_PATH, "w") as f:
    json.dump(out, f, indent=1)
print(f"wrote {OUT_PATH}", flush=True)

# ledger every read taken
looks = json.load(open(LOOKS_PATH))
ent = looks["autopsy_methodological_reads"]["entries"]
for key, r in results.items():
    ent.append({
        "config": r["label"], "date": "2026-07-29",
        "T1_val2025_milli": r["T1_fit2324_val2025"]["delta_milli"],
        "T2_val2026H1_milli": r["T2_fit2325_val2026H1"]["delta_milli"],
        "pooled_validation_milli": r["pooled_validation"]["delta_milli"],
        "holdout_v8_method_milli": r["holdout_v8_method"]["delta_milli"],
        "justification": "preregistered autopsy read (design input for the validator)"})
looks["autopsy_methodological_reads"]["count"] = len(ent)
with open(LOOKS_PATH, "w") as f:
    json.dump(looks, f, indent=1)
print(f"ledgered {len(ent)} autopsy reads in v9_looks.json "
      f"[{time.time()-t0:.0f}s total]", flush=True)
