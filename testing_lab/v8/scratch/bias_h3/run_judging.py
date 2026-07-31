"""agent:bias-h3 — full referee judging: SS variants vs v6 on the expanded frame.

Emits stats/h3_statespace.json + stats/h3_bias_caterpillar.json.
All bootstraps via referee.paired_bootstrap_crn (crn.json randomness).
MDE quotes: within-family 1.773m, cross-family 5.889m (power_mde_expanded).
"""
import json
import math
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
TL = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
V8 = os.path.join(TL, "v8")
sys.path.insert(0, HERE)
sys.path.insert(0, TL)
sys.path.insert(0, V8)

from lib_h3 import GameData, implied_half_life  # noqa: E402
import referee  # noqa: E402

MDE_WITHIN = 1.773 / 1000.0
MDE_CROSS = 5.889 / 1000.0
COLD_REF = {"n": 57, "ll": 0.70205, "note": "old-frame reference from the brief"}


def pair_judge(d, ev, label, mde, p_ref_holdout):
    iid = referee.paired_bootstrap_crn(d, mode="iid")
    blk = referee.paired_bootstrap_crn(d, mode="block_event", event_ids=ev)
    dll = iid["mean_delta"]
    verdict = ("INSIDE NOISE FLOOR" if abs(dll) < mde else
               ("WIN" if dll > 0 else "LOSS"))
    roi = referee.expected_roi_of_dll(dll, p_ref_holdout)
    return {"label": label, "delta_milli": round(dll * 1000, 3),
            "mde_milli": round(mde * 1000, 3), "verdict": verdict,
            "iid": {k: iid[k] for k in ("mean_delta", "ci_lo", "ci_hi", "p_better",
                                        "seed", "n_boot", "crn_verify", "n")},
            "block_event": {k: blk[k] for k in ("mean_delta", "ci_lo", "ci_hi",
                                                "p_better", "seed", "n_boot", "n_events")},
            "expected_roi": {k: roi[k] for k in ("dll_milli", "delta_logit_equiv",
                                                 "expected_roi_delta", "roi_at_op",
                                                 "roi_at_op_plus_equiv",
                                                 "roi_ci_at_op_plus_equiv")}}


def main():
    gd = GameData()
    frame = gd.frame
    n = len(frame)
    holdout = gd.holdout_mask
    ev_ids = frame.event_id.values

    # v6 baseline (checkpointed)
    z6 = np.load(os.path.join(HERE, "v6_baseline.npz"))
    p6 = z6["p_all"]
    rd6 = z6["rdiff"]
    v6_valid = ~np.isnan(rd6)
    meta6 = json.load(open(os.path.join(HERE, "v6_baseline_meta.json")))

    # SS configs
    core = json.load(open(os.path.join(HERE, "sweep_core.json")))
    c1a = core["points"]["1a|q/R=0.00564622|V0/R=1.27156"]
    c1b = core["best"]["candidates"]["1b"]
    c1c = core["best"]["candidates"]["1c"]
    ck5 = json.load(open(os.path.join(HERE, "sweep_5d.json")))

    models = {}

    def add_eval(name, res, extra=None):
        models[name] = {"p": res["p"], "beta": res["beta"],
                        "ll_train": res["ll_train"], "ll_holdout": res["ll_holdout"],
                        "filter": res.get("filter"), **(extra or {})}

    r1a = gd.eval_config(q=c1a["q_over_R"] * gd.R, V0=c1a["V0_over_R"] * gd.R)
    add_eval("ss_1a", r1a, {"cfg": c1a})
    r1b = gd.eval_config(q=c1b["q_over_R"] * gd.R, V0=c1b["V0_over_R"] * gd.R,
                         q_cal_week=c1b["q_cal_week"])
    add_eval("ss_1b_qcal", r1b, {"cfg": c1b})
    r1c = gd.eval_config(q=c1c["q_over_R"] * gd.R, V0=c1c["V0_over_R"] * gd.R,
                         debut_region_prior=True)
    add_eval("ss_1c_debutprior", r1c, {"cfg": c1c})

    # 5d axis A model
    from run_5d import build_cells, axis_nll
    cA, cB, _ = build_cells(gd)
    V0 = ck5["meta"]["V0"]
    lqA = np.array(ck5["A_roster"]["logqs"])
    nllA, betaA, fA = axis_nll(gd, cA, lqA, V0)
    scA = gd.score(betaA, fA["mu"], fA["s2"])
    models["ss_5d_roster"] = {"p": scA["p"], "beta": betaA,
                              "ll_train": scA["ll_train"],
                              "ll_holdout": scA["ll_holdout"], "filter": fA,
                              "cfg": {"logqs": list(map(float, lqA)), "V0": V0}}

    # ── pairwise judging (holdout, both-valid rows) ─────────────────────────
    pairs = [("ss_1b_qcal", "v6", MDE_CROSS, "PRIMARY (train-selected core) vs v6"),
             ("ss_1a", "v6", MDE_CROSS, "plain core vs v6 (secondary)"),
             ("ss_1c_debutprior", "v6", MDE_CROSS, "debut-prior core vs v6 (secondary)"),
             ("ss_5d_roster", "v6", MDE_CROSS, "5d roster-typed q vs v6"),
             ("ss_5d_roster", "ss_1a", MDE_WITHIN, "5d roster-typed q vs plain core (within-family)")]
    judged = []
    for a, b, mde, lab in pairs:
        pa = models[a]["p"] if a != "v6" else p6
        pb = models[b]["p"] if b != "v6" else p6
        valid = ~np.isnan(pa) & ~np.isnan(pb)
        m = valid & holdout
        d = referee.delta_vector(pa[m], pb[m])
        judged.append({"pair": f"{a} vs {b}", **pair_judge(d, ev_ids[m], lab, mde, pb[m]),
                       "ll_a": round(referee.logloss(pa[m]), 5),
                       "ll_b": round(referee.logloss(pb[m]), 5),
                       "n": int(m.sum())})
        print(f"{lab}: d={judged[-1]['delta_milli']}m [{judged[-1]['verdict']}] "
              f"p_better iid={judged[-1]['iid']['p_better']:.3f} "
              f"blk={judged[-1]['block_event']['p_better']:.3f}", flush=True)

    # ── buckets (referee panel; cold-start via v6 engine ratings) ───────────
    frame_b = frame.copy()
    frame_b["r_w"] = z6["rat_w"]
    frame_b["r_l"] = z6["rat_l"]
    from engine import Engine
    eng = Engine()
    ef, fs = referee.wr_masks(frame_b, eng.games)
    buckets = {}
    for name in ("ss_1a", "ss_1b_qcal", "ss_5d_roster"):
        pm_ = models[name]["p"]
        valid = ~np.isnan(pm_) & v6_valid
        buckets[name] = referee.bucketed(frame_b, pm_, p_ref=p6, rdiff=rd6,
                                         holdout=holdout, valid=valid,
                                         elite_floor=ef, form_shift=fs)
    # cold-start summary per model. COLD_EPS(|r|<5e-4) is EMPTY on engine
    # ratings (region_prior_ridge pulls every solved team off 0), so the
    # honest equivalent on this frame is prior-map-count at prediction date.
    prior_w = np.zeros(n, dtype=int)
    prior_l = np.zeros(n, dtype=int)
    cnt = np.zeros(gd.n_teams, dtype=int)
    fi = 0
    frows = sorted(range(n), key=lambda i: (frame.date.values[i], frame.match_id.values[i]))
    gj = 0
    order_days = sorted(set(gd.g_date) | set(frame.date.values))
    rows_by_day_f = {}
    for i in range(n):
        rows_by_day_f.setdefault(frame.date.values[i], []).append(i)
    games_by_day = gd.games_by_day
    for day in order_days:
        for i in rows_by_day_f.get(day, []):
            prior_w[i] = cnt[gd.f_wi[i]]
            prior_l[i] = cnt[gd.f_li[i]]
        for j in games_by_day.get(day, []):
            cnt[gd.wi[j]] += 1
            cnt[gd.li[j]] += 1
    pmin = np.minimum(prior_w, prior_l)
    cold_defs = {"debut (either 0 prior maps)": pmin == 0,
                 "cold (either <10 prior maps)": pmin < 10,
                 "thin (either <30 prior maps)": pmin < 30}
    cold_rows = {}
    for cname, cmask in cold_defs.items():
        cold_rows[cname] = {}
        for name in ("ss_1a", "ss_1b_qcal", "ss_5d_roster"):
            pm_ = models[name]["p"]
            m = cmask & holdout & ~np.isnan(pm_) & v6_valid
            if m.sum() == 0:
                cold_rows[cname][name] = {"n": 0}
                continue
            cold_rows[cname][name] = {
                "n": int(m.sum()),
                "ll": round(referee.logloss(pm_[m]), 5),
                "ll_v6": round(referee.logloss(p6[m]), 5),
                "delta_milli": round((referee.logloss(p6[m]) -
                                      referee.logloss(pm_[m])) * 1000, 2)}
    cold_rows["cold_eps_note"] = ("referee COLD_EPS bucket is n=0 on engine "
                                  "ratings (region prior => no exact zeros); "
                                  "prior-map-count definitions used instead")
    np.savez_compressed(os.path.join(HERE, "prior_maps.npz"),
                        prior_w=prior_w, prior_l=prior_l)

    # ── per-team bias caterpillar + mechanism check ─────────────────────────
    w_, l_ = frame.winner.values, frame.loser.values
    bias = {}
    for name, pv in (("v6", p6), ("ss_1a", models["ss_1a"]["p"]),
                     ("ss_1b_qcal", models["ss_1b_qcal"]["p"]),
                     ("ss_5d_roster", models["ss_5d_roster"]["p"])):
        valid = ~np.isnan(pv) & v6_valid
        bias[name] = referee.per_team_bias(pv, w_, l_, holdout=holdout, valid=valid)
    # mechanism: per-team mean pre-match n_eff (ss_1a) + sharpness by model
    fh = models["ss_1a"]["filter"]
    team_rows = {}
    for i in np.where(holdout & v6_valid)[0]:
        for team, vT in ((w_[i], fh["v_w"][i]), (l_[i], fh["v_l"][i])):
            team_rows.setdefault(team, {"neff": [], "sharp_v6": [], "sharp_ss": []})
            team_rows[team]["neff"].append(gd.R / vT)
            pw6 = p6[i] if team == w_[i] else 1 - p6[i]
            pws = models["ss_1a"]["p"][i] if team == w_[i] else 1 - models["ss_1a"]["p"][i]
            team_rows[team]["sharp_v6"].append(abs(pw6 - 0.5))
            team_rows[team]["sharp_ss"].append(abs(pws - 0.5))
    cat = []
    b6map = {r["team"]: r for r in bias["v6"]["teams"]}
    bamap = {r["team"]: r for r in bias["ss_1a"]["teams"]}
    b5map = {r["team"]: r for r in bias["ss_5d_roster"]["teams"]}
    for team, rr in team_rows.items():
        if team not in b6map:
            continue
        cat.append({"team": team, "n": b6map[team]["n"],
                    "bias_v6": b6map[team]["bias"],
                    "bias_ss1a": bamap.get(team, {}).get("bias"),
                    "bias_ss5d": b5map.get(team, {}).get("bias"),
                    "ll_v6": b6map[team]["ll"],
                    "ll_ss1a": bamap.get(team, {}).get("ll"),
                    "mean_neff": round(float(np.mean(rr["neff"])), 2),
                    "mean_sharp_v6": round(float(np.mean(rr["sharp_v6"])), 4),
                    "mean_sharp_ss1a": round(float(np.mean(rr["sharp_ss"])), 4)})
    cat.sort(key=lambda r: r["bias_v6"])
    # mechanism quantification: sharpness reduction vs n_eff (thin vs established)
    neffs = np.array([r["mean_neff"] for r in cat])
    dsharp = np.array([r["mean_sharp_ss1a"] - r["mean_sharp_v6"] for r in cat])
    med = float(np.median(neffs))
    thin = neffs < med
    mech = {"median_team_neff": round(med, 2),
            "mean_sharpness_change_thin_half": round(float(dsharp[thin].mean()), 4),
            "mean_sharpness_change_established_half": round(float(dsharp[~thin].mean()), 4),
            "corr_neff_vs_sharpness_change": round(float(np.corrcoef(neffs, dsharp)[0, 1]), 3),
            "prediction": "thin-data teams shrink toward 0.5 (negative change) "
                          "more than established teams"}

    cat_out = {
        "written_by": "agent:bias-h3", "frame_n_holdout": int(holdout.sum()),
        "min_n": 25, "unit": "probability points (x100 = pp), NOT rating points",
        "summary": {name: {"max_abs_bias": bias[name]["max_abs_bias"],
                           "mean_abs_bias": bias[name]["mean_abs_bias"],
                           "n_teams": bias[name]["n_teams"]} for name in bias},
        "mechanism_check": mech,
        "teams": cat}
    with open(os.path.join(V8, "stats", "h3_bias_caterpillar.json"), "w") as f:
        json.dump(cat_out, f, indent=1)

    # ── main stats JSON ─────────────────────────────────────────────────────
    gh = json.load(open(os.path.join(HERE, "gh_vs_mc.json")))
    out = {
        "written_by": "agent:bias-h3 (Phase 5 H3 experiment 1 + judging)",
        "preregistered": "preregister.bias_h3.md (written before runs)",
        "frame": "v8/data/frame_expanded/series.csv sha256-verified; holdout n=1217",
        "v6_baseline": {"beta": meta6["beta"], "ll_train": meta6["ll_train"],
                        "ll_holdout": meta6["ll_test"],
                        "cfg": "consist(20,12) + PO1.6/champ2.0/region-prior, "
                               "beta refit train on this frame"},
        "model_summaries": {name: {"beta": round(models[name]["beta"], 5),
                                   "ll_train": round(models[name]["ll_train"], 6),
                                   "ll_holdout": round(models[name]["ll_holdout"], 6),
                                   "cfg": {k: v for k, v in models[name]["cfg"].items()
                                           if k not in ("ll_train", "ll_holdout")}}
                            for name in models},
        "train_selection": {"rule": "best TRAIN NLL among {1a,1b,1c} (preregistered)",
                            "selected": "ss_1b_qcal",
                            "note": "holdout numbers stored in sweep_core.json "
                                    "played no role in selection"},
        "pairwise_holdout": judged,
        "buckets_vs_v6": buckets,
        "cold_start_bucket": {"reference_old_frame": COLD_REF,
                              "definition": "prior-map-count at prediction "
                                            "date (strictly earlier days)",
                              "models": cold_rows},
        "gh_validation": gh,
        "mde_context": {"within_family_milli": 1.773, "cross_family_milli": 5.889},
    }
    with open(os.path.join(V8, "stats", "h3_statespace.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("wrote h3_statespace.json + h3_bias_caterpillar.json", flush=True)

    # cache model probs for the ensemble step
    np.savez_compressed(os.path.join(HERE, "model_probs.npz"),
                        p_v6=p6, v6_valid=v6_valid,
                        p_ss_1a=models["ss_1a"]["p"],
                        p_ss_1b=models["ss_1b_qcal"]["p"],
                        p_ss_5d=models["ss_5d_roster"]["p"],
                        vw_1a=models["ss_1a"]["filter"]["v_w"],
                        vl_1a=models["ss_1a"]["filter"]["v_l"],
                        vw_1b=models["ss_1b_qcal"]["filter"]["v_w"],
                        vl_1b=models["ss_1b_qcal"]["filter"]["v_l"],
                        vw_5d=fA["v_w"], vl_5d=fA["v_l"],
                        mu_1b=models["ss_1b_qcal"]["filter"]["mu"],
                        s2_1b=models["ss_1b_qcal"]["filter"]["s2"],
                        mu_5d=fA["mu"], s2_5d=fA["s2"],
                        mu_1a=models["ss_1a"]["filter"]["mu"],
                        s2_1a=models["ss_1a"]["filter"]["s2"])
    print("cached model_probs.npz", flush=True)


if __name__ == "__main__":
    main()
