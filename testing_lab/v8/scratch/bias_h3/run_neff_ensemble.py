"""agent:bias-h3 — Experiments 3 (n_eff telemetry) + 4 (ensemble-where-it-wins).

Emits stats/h3_neff.json + stats/h3_ensemble.json.
Ensemble search space (train-only selection, preregistered gate list x fitted SS
variants {ss_1b (train-primary), ss_5d_roster}): hard gates n_eff<theta
(theta in 3/5/8/12), roster change-adjacent, ewc_offseason events, soft
n_eff blend (2 params). ONE winner scored on holdout with full judging.
"""
import json
import math
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
TL = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
V8 = os.path.join(TL, "v8")
sys.path.insert(0, HERE)
sys.path.insert(0, TL)
sys.path.insert(0, V8)
from lib_h3 import GameData  # noqa: E402
import referee  # noqa: E402

MDE_WITHIN = 1.773 / 1000.0
MDE_CROSS = 5.889 / 1000.0

gd = GameData()
frame = gd.frame
n = len(frame)
holdout = gd.holdout_mask
train = gd.train_mask
ev_ids = frame.event_id.values

z = np.load(os.path.join(HERE, "model_probs.npz"))
pm_prior = np.load(os.path.join(HERE, "prior_maps.npz"))
prior_min = np.minimum(pm_prior["prior_w"], pm_prior["prior_l"])
p6 = z["p_v6"]
v6_valid = z["v6_valid"]
sweep_core = json.load(open(os.path.join(HERE, "sweep_core.json")))
b1b = sweep_core["best"]["candidates"]["1b"]["beta"]
ck5 = json.load(open(os.path.join(HERE, "sweep_5d.json")))
b5d = ck5["A_roster"]["beta"]
R = gd.R

# ── event classes ───────────────────────────────────────────────────────────
pmx = json.load(open(os.path.join(V8, "stats", "power_mde_expanded.json")))
ewc_events = {e for e, c in pmx["new_events"].items() if c == "ewc_offseason"}
ewc_row = (frame.event_id.astype(str).str.startswith(referee.EWC_CLASS_PREFIXES)
           .values | np.isin(ev_ids, sorted(ewc_events)))
intl_row = frame.intl.values.astype(bool)
ev_class = np.where(intl_row, "intl", np.where(ewc_row, "ewc_offseason",
                                               "vct_domestic"))

# ── Experiment 3: n_eff telemetry (primary = train-selected core ss_1b) ─────


def p_point(beta, mu):
    pm = 1.0 / (1.0 + np.exp(-beta * mu))
    is5 = np.isin(gd.fmts, ("bo5", "bo5_gf"))
    is1 = gd.fmts == "bo1"
    return np.where(is5, pm ** 3 * (1 + 3 * (1 - pm) + 6 * (1 - pm) ** 2),
                    np.where(is1, pm, pm ** 2 * (3 - 2 * pm)))


def telemetry(tag, vw, vl, mu, s2, p_int, beta):
    neff_w = R / vw
    neff_l = R / vl
    neff_min = np.minimum(neff_w, neff_l)
    neff_harm = 2.0 / (1.0 / neff_w + 1.0 / neff_l)
    pp = p_point(beta, mu)
    with np.errstate(divide="ignore", invalid="ignore"):
        shrink = np.where(np.abs(pp - 0.5) < 1e-12, 1.0,
                          np.abs(p_int - 0.5) / np.abs(pp - 0.5))
    return {"neff_w": neff_w, "neff_l": neff_l, "neff_min": neff_min,
            "neff_harm": neff_harm, "p_point": pp, "p_int": p_int,
            "shrink": shrink, "sigma_delta": np.sqrt(s2)}


t1b = telemetry("1b", z["vw_1b"], z["vl_1b"], z["mu_1b"], z["s2_1b"],
                z["p_ss_1b"], b1b)
t5d = telemetry("5d", z["vw_5d"], z["vl_5d"], z["mu_5d"], z["s2_5d"],
                z["p_ss_5d"], b5d)


def qtiles(x, m=None):
    x = x[m] if m is not None else x
    if len(x) == 0:
        return None
    return {k: round(float(np.percentile(x, p_)), 3)
            for k, p_ in (("p5", 5), ("p25", 25), ("p50", 50), ("p75", 75),
                          ("p95", 95))}


ll6 = referee.per_series_ll(np.where(np.isnan(p6), 0.5, p6))
ll1b = referee.per_series_ll(z["p_ss_1b"])
m_ho = holdout & v6_valid
dll_abs = np.abs(ll6 - ll1b)
corr = float(np.corrcoef(t1b["neff_harm"][m_ho], dll_abs[m_ho])[0, 1])

summaries = {"overall": qtiles(t1b["neff_harm"]),
             "holdout": qtiles(t1b["neff_harm"], holdout),
             "by_year": {int(y): qtiles(t1b["neff_harm"], (frame.year.values == y))
                         for y in sorted(frame.year.unique())},
             "by_event_class_holdout": {c: qtiles(t1b["neff_harm"],
                                                  holdout & (ev_class == c))
                                        for c in ("vct_domestic", "ewc_offseason",
                                                  "intl")},
             "cold_rows_holdout(prior_min<10)": qtiles(
                 t1b["neff_harm"], holdout & (prior_min < 10)),
             "shrink_factor_holdout": qtiles(t1b["shrink"], holdout),
             "corr_neff_harm_vs_absdeltaLL_v6_holdout": round(corr, 3)}

rows = []
for i in range(n):
    rows.append({"match_id": int(frame.match_id.values[i]),
                 "date": frame.date.values[i],
                 "holdout": bool(holdout[i]),
                 "winner": frame.winner.values[i], "loser": frame.loser.values[i],
                 "v_w": round(float(z["vw_1b"][i]), 4),
                 "v_l": round(float(z["vl_1b"][i]), 4),
                 "sigma_delta": round(float(t1b["sigma_delta"][i]), 4),
                 "neff_w": round(float(t1b["neff_w"][i]), 2),
                 "neff_l": round(float(t1b["neff_l"][i]), 2),
                 "neff_harm": round(float(t1b["neff_harm"][i]), 2),
                 "p_point": round(float(t1b["p_point"][i]), 4),
                 "p_integrated": round(float(t1b["p_int"][i]), 4),
                 "shrink": round(float(t1b["shrink"][i]), 4)})

neff_out = {
    "written_by": "agent:bias-h3 (experiment 3)",
    "model": "ss_1b_qcal (train-selected core, preregistered source); R="
             f"{round(R,4)} — n_eff = R/v is invariant to the R constant",
    "definition": "pre-match per-team posterior variance v; n_eff_T = R/v_T; "
                  "neff_harm = harmonic pair mean; shrink = |p_int-0.5|/"
                  "|p_point-0.5| (integration pull toward 0.5)",
    "purpose": "telemetry input for the metrics-spec confidence-aware sizing; "
               "no selection rides on this",
    "distribution_summaries": summaries,
    "secondary_model_summaries": {
        "ss_5d_roster": {"holdout": qtiles(t5d["neff_harm"], holdout),
                         "shrink_holdout": qtiles(t5d["shrink"], holdout)}},
    "per_match": rows}
with open(os.path.join(V8, "stats", "h3_neff.json"), "w") as f:
    json.dump(neff_out, f, indent=1)
print("wrote h3_neff.json;", "corr(neff, |dLL|) =", corr, flush=True)

# ── Experiment 4: ensemble-where-it-wins ────────────────────────────────────
tp = pd.read_csv(os.path.join(HERE, "lineup_topup.csv"))
tp = tp.sort_values(["org", "date", "match_id"])
last_msc = {}
change_adj = np.zeros(n, dtype=bool)
by_org = {o: list(zip(g.date.values, g.match_id.values,
                      g.matches_since_change.values))
          for o, g in tp.groupby("org")}
for i in range(n):
    d = frame.date.values[i]
    flag = False
    for team in (frame.winner.values[i], frame.loser.values[i]):
        seq = by_org.get(team, [])
        m_last = None
        for dd, mm, msc in seq:
            if dd < d:
                m_last = msc
            else:
                break
        if m_last is not None and m_last <= 3:
            flag = True
    change_adj[i] = flag

candidates = {}
for ss_name, p_ss, tel in (("ss_1b", z["p_ss_1b"], t1b),
                           ("ss_5d_roster", z["p_ss_5d"], t5d)):
    for th in (3, 5, 8, 12):
        candidates[f"{ss_name}|neff<{th}"] = (
            p_ss, tel["neff_min"] < th, None)
    candidates[f"{ss_name}|change_adjacent"] = (p_ss, change_adj, None)
    candidates[f"{ss_name}|ewc_offseason"] = (p_ss, ewc_row, None)
    candidates[f"{ss_name}|soft_blend_neff"] = (p_ss, None, tel["neff_min"])

m_tr = train & v6_valid


def comp_nll(p_comp, m):
    return float(-np.mean(np.log(np.clip(p_comp[m], 1e-9, 1))))


results = {}
for name, (p_ss, gate, soft_x) in candidates.items():
    if soft_x is None:
        p_comp = np.where(gate, p_ss, p6)
        results[name] = {"train_nll": comp_nll(p_comp, m_tr),
                         "n_gated_train": int((gate & m_tr).sum()),
                         "n_gated_holdout": int((gate & holdout & v6_valid).sum())}
    else:
        x = soft_x

        def nll_ab(ab):
            w = 1.0 / (1.0 + np.exp(-(ab[0] + ab[1] * x)))
            p_comp = w * p_ss + (1 - w) * p6
            return comp_nll(p_comp, m_tr)
        best = None
        for a0, b0 in ((0.0, -0.1), (2.0, -0.3), (-2.0, 0.0), (0.0, 0.0)):
            r = minimize(nll_ab, [a0, b0], method="Nelder-Mead")
            if best is None or r.fun < best.fun:
                best = r
        results[name] = {"train_nll": float(best.fun),
                         "ab": [float(best.x[0]), float(best.x[1])]}

v6_train_nll = comp_nll(p6, m_tr)
for name, r in sorted(results.items(), key=lambda kv: kv[1]["train_nll"]):
    print(f"  {name:<34} train={r['train_nll']:.6f}", flush=True)
print(f"  {'v6 alone':<34} train={v6_train_nll:.6f}", flush=True)

winner = min(results, key=lambda k: results[k]["train_nll"])
beats_v6 = results[winner]["train_nll"] < v6_train_nll
scored = winner if beats_v6 else "ss_1b|neff<5"
label = ("train-selected winner" if beats_v6 else
         "EXPLORATORY fallback (no train composite beat v6; preregistered)")

p_ss, gate, soft_x = candidates[scored]
if soft_x is None:
    p_comp = np.where(gate, p_ss, p6)
else:
    ab = results[scored]["ab"]
    w = 1.0 / (1.0 + np.exp(-(ab[0] + ab[1] * soft_x)))
    p_comp = w * p_ss + (1 - w) * p6

m_ho_v = holdout & v6_valid


def judge(pa, pb, mde, p_ref):
    d = referee.delta_vector(pa[m_ho_v], pb[m_ho_v])
    iid = referee.paired_bootstrap_crn(d, mode="iid")
    blk = referee.paired_bootstrap_crn(d, mode="block_event",
                                       event_ids=ev_ids[m_ho_v])
    dll = iid["mean_delta"]
    roi = referee.expected_roi_of_dll(dll, p_ref[m_ho_v])
    return {"delta_milli": round(dll * 1000, 3),
            "mde_milli": round(mde * 1000, 3),
            "verdict": ("INSIDE NOISE FLOOR" if abs(dll) < mde else
                        ("WIN" if dll > 0 else "LOSS")),
            "iid_p_better": iid["p_better"], "block_p_better": blk["p_better"],
            "iid_ci": [iid["ci_lo"], iid["ci_hi"]],
            "block_ci": [blk["ci_lo"], blk["ci_hi"]],
            "expected_roi_delta": roi["expected_roi_delta"],
            "delta_logit_equiv": roi["delta_logit_equiv"]}


vs_v6 = judge(p_comp, p6, MDE_CROSS, p6)
p_1a = z["p_ss_1a"]
vs_core = judge(p_comp, p_1a, MDE_WITHIN, p_1a)
w_, l_ = frame.winner.values, frame.loser.values
bias_comp = referee.per_team_bias(p_comp, w_, l_, holdout=holdout,
                                  valid=v6_valid & ~np.isnan(p_comp))
bias_v6 = referee.per_team_bias(p6, w_, l_, holdout=holdout, valid=v6_valid)

# gated-subset detail for the scored composite
detail = {}
if soft_x is None:
    g_ho = gate & m_ho_v
    detail = {"n_gated_holdout": int(g_ho.sum()),
              "ll_ss_gated": round(referee.logloss(p_ss[g_ho]), 5) if g_ho.sum() else None,
              "ll_v6_gated": round(referee.logloss(p6[g_ho]), 5) if g_ho.sum() else None}

ens_out = {
    "written_by": "agent:bias-h3 (experiment 4)",
    "preregistered": "gate list + selection rule in preregister.bias_h3.md; "
                     "search space = gates x fitted SS variants {ss_1b, "
                     "ss_5d_roster}, selection purely on TRAIN composite NLL",
    "train_table": {k: {kk: (round(vv, 6) if isinstance(vv, float) else vv)
                        for kk, vv in v.items()} for k, v in results.items()},
    "v6_train_nll": round(v6_train_nll, 6),
    "selected": {"name": scored, "label": label,
                 "beats_v6_on_train": bool(beats_v6)},
    "holdout_judging": {
        "composite_vs_v6 (cross-family MDE)": vs_v6,
        "composite_vs_ss_core_1a (within-family MDE)": vs_core,
        "ll_composite": round(referee.logloss(p_comp[m_ho_v]), 5),
        "ll_v6": round(referee.logloss(p6[m_ho_v]), 5),
        "n": int(m_ho_v.sum()),
        "max_abs_bias": {"composite": bias_comp["max_abs_bias"],
                         "v6": bias_v6["max_abs_bias"]},
        "gated_subset_detail": detail},
    "verdict_shape": "candidate shape per brief: gated ensemble, not full "
                     "replacement, if composite wins where SS-core alone loses",
}
with open(os.path.join(V8, "stats", "h3_ensemble.json"), "w") as f:
    json.dump(ens_out, f, indent=1)
print(f"wrote h3_ensemble.json  selected={scored} ({label})", flush=True)
print(json.dumps(ens_out["holdout_judging"], indent=1)[:1200], flush=True)
