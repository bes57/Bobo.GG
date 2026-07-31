"""agent:context step 4a — emit stats/context_{seriousness,stakes,exposure,
shrink,adjacency}.json from step2a/step3 outputs (3d waits for step2b).
Adds CRN boots + ROI for the train-selected solve-grid points."""
import json
import os
import sys

import numpy as np
import pandas as pd

V8 = "/Users/benny_es1/PythonTest/testing_lab/v8"
TL = os.path.dirname(V8)
SC = os.path.join(V8, "scratch", "context")
ST = os.path.join(V8, "stats")
sys.path.insert(0, TL)
sys.path.insert(0, V8)
import referee  # noqa: E402

frame = pd.read_csv(os.path.join(SC, "frame_features.csv"))
b0 = np.load(os.path.join(SC, "b0.npz"), allow_pickle=True)
loss_b0 = b0["loss_b0"]
p_b0 = b0["p_test"]
ev_test = b0["event_ids"]
test = b0["test_mask"]
s2a = json.load(open(os.path.join(SC, "step2a_results.json")))
s3 = json.load(open(os.path.join(SC, "step3_results.json")))
MDE_W = 1.773
ECL = frame.eclass.values
ewcfull_t = (ECL == "ewc_offseason")[test]
legacy_t = frame.event_id.astype(str).str.startswith(
    ("2026_ewc", "2026_china_evo")).values[test]

PROV = {"agent": "context", "written": "2026-07-28",
        "frame": "v8/data/frame_expanded/series.csv sha256 ff772d41... (verified vs crn.json)",
        "baseline_B0": {"desc": "v6 champion replayed on expanded frame "
                        "(consist 20/12, PO 1.6, champ x2, beta train-fit)",
                        **s2a["B0"]},
        "mde_milli": {"within": 1.773, "cross": 5.889,
                      "source": "stats/power_mde_expanded.json"},
        "selection_rule": "train NLL only; holdout curves shown are "
                          "transparency, never selection"}


def judge_pt(tag, x):
    p_t = np.load(os.path.join(SC, f"ptest_{tag}_{x:.2f}.npy"))
    loss = -np.log(np.clip(p_t, 1e-9, 1))
    d = loss_b0 - loss
    dll = float(d.mean())
    bi = referee.paired_bootstrap_crn(d, mode="iid")
    bb = referee.paired_bootstrap_crn(d, mode="block_event", event_ids=ev_test)
    roi = referee.expected_roi_of_dll(dll, p_b0)
    verdict = "INSIDE NOISE FLOOR"
    if abs(dll) * 1000 >= MDE_W:
        verdict = ("WIN" if dll > 0 and bi["ci_lo"] > 0 else
                   "DEAD" if dll < 0 and bi["ci_hi"] < 0 else
                   "INSIDE NOISE FLOOR (CI spans 0)")
    return {"dll_milli": round(dll * 1000, 3), "pair_mde_milli": MDE_W,
            "boot_iid_ci_milli": [round(bi["ci_lo"] * 1000, 3),
                                  round(bi["ci_hi"] * 1000, 3)],
            "boot_block_ci_milli": [round(bb["ci_lo"] * 1000, 3),
                                    round(bb["ci_hi"] * 1000, 3)],
            "p_better_iid": bi["p_better"],
            "expected_roi_delta": roi["expected_roi_delta"],
            "roi_note": "ladder clamps at delta_logit 0; negative dll -> 0.0",
            "ewc_full_bucket_dll_milli": round(float(d[ewcfull_t].mean()) * 1000, 2),
            "ewc_legacy_bucket_dll_milli": round(float(d[legacy_t].mean()) * 1000, 2),
            "verdict": verdict}


# ── 3a seriousness ──────────────────────────────────────────────────────────
gi = s2a["grids"]["3a_integrity"]
gb = s2a["grids"]["3a_blanket"]
w0_star = min(gi, key=lambda r: r["ll_train"])["x"]
we_star = min(gb, key=lambda r: r["ll_train"])["x"]
out = {
    "provenance": PROV,
    "preregistered": "preregister.context.md 3a",
    "mechanism": "down-weight EWC-class solve games by lineup integrity "
                 "(vct-modal overlap), f(x)=w0+(1-w0)x per side, w=f_w*f_l",
    "integrity_source": "overlap_vct_modal, fallback overlap_modal, fallback "
                        "1.0; lineups agent table + scratch top-up "
                        "(cross-checked 20/20 exact)",
    "grid_integrity_w0": gi, "grid_blanket_we": gb,
    "train_argmin": {"w0": w0_star, "we": we_star,
                     "we_note": "argmin at grid EDGE 1.2 — train prefers "
                                "UP-weighting EWC games, not down"},
    "result_integrity": {
        "train_selected_w0": w0_star,
        "note": "train argmin is w0=1.0 == B0 exactly: the preregistered "
                "falsifier fired — train NLL rejects lineup-conditioned "
                "down-weighting outright (monotone worse toward w0=0). "
                "No candidate to judge on holdout; holdout curve shown "
                "above is descriptive (its own min, +0.08m at w0=0.8, is "
                "deep inside the 1.773m floor).",
        "verdict": "DEAD (train selects baseline; falsifier fired)"},
    "result_blanket": {"train_selected_we": we_star,
                       **judge_pt("3a_blanket", we_star)},
    "verdict_summary": "3a DEAD both ways: integrity conditioning rejected by "
                       "train; blanket train pick is an UP-weight (1.2) whose "
                       "holdout delta is -0.05m, inside the 1.773m floor. The "
                       "operator's EWC down-weight intuition finds no support "
                       "in the solve objective.",
}
json.dump(out, open(os.path.join(ST, "context_seriousness.json"), "w"), indent=1)

# ── 3c stakes ───────────────────────────────────────────────────────────────
ge = s2a["grids"]["3cA_elim"]
wl_star = min(ge, key=lambda r: r["ll_train"])["x"]
hold = (frame.date > "2024-12-31").values
out = {
    "provenance": PROV,
    "preregistered": "preregister.context.md 3c",
    "elim_flag_def": "stage grand_final OR name has Lower/Elimination/Decider/"
                     "Knockout/(0-1)/(1-1) OR playoffs-stage KO name w/o Upper",
    "elim_counts": {"overall": int(frame.elim.sum()),
                    "holdout": int((frame.elim.values.astype(bool) & hold).sum()),
                    "share": round(float(frame.elim.mean()), 3)},
    "dead_rubbers": "DECLARED UNTESTABLE (preregistered): group membership, "
                    "advancement thresholds and tiebreakers not derivable "
                    "from the corpus; approximation forbidden by brief.",
    "testA_solve_weight": {
        "grid": ge, "train_argmin_w_elim": wl_star,
        "edge_note": "argmin at grid EDGE 0.7 — train prefers DOWN-weighting "
                     "elimination games (they are noisier in-sample), the "
                     "opposite of the 'stakes reveal true strength' story",
        **judge_pt("3cA_elim", wl_star)},
    "testB_variance": s3["3cB_elim_variance"],
    "verdict_summary": "3c DEAD/INSIDE NOISE FLOOR: train wants elim games "
                       "down-weighted (w=0.7 edge) but that pick scores "
                       "-0.54m on holdout (floor 1.773m); the prediction-"
                       "layer shrink fits a_elim=-0.348 on train (elim "
                       "matches less predictable) and lands -0.37m on "
                       "holdout, inside the floor. Stakes carry no usable "
                       "probability information at this n.",
}
json.dump(out, open(os.path.join(ST, "context_stakes.json"), "w"), indent=1)

# ── 3b exposure + decomposition ─────────────────────────────────────────────
out = {
    "provenance": PROV,
    "preregistered": "preregister.context.md 3b",
    "features": "walk-forward from engine games: maps in 14/30d, days since "
                "last official map (cap 120), days since intl LAN (cap 365; "
                "exact-shape masters/champions/lock_in; sensitivity + EWC "
                "mains); winner-referenced diffs dmaps30/10, dlog1p(dso), "
                "dlog1p(dsi)",
    "prediction_term": s3["3ba_exposure_term"],
    "prediction_term_sensitivity_ewcLAN": s3["3ba_exposure_term_ewcLAN"],
    "decomposition": s3["3bb_decomposition"],
    "forest_plot": [
        {"label": "b_form (HL5) alone", "coef": s3["3bb_decomposition"]["form5"]["b_form_alone"]},
        {"label": "b_form (HL5) + exposure", "coef": s3["3bb_decomposition"]["form5"]["b_form_with_exposure"]},
        {"label": "b_form (HL3) alone", "coef": s3["3bb_decomposition"]["form3"]["b_form_alone"]},
        {"label": "b_form (HL3) + exposure", "coef": s3["3bb_decomposition"]["form3"]["b_form_with_exposure"]},
        {"label": "published v7 old-frame HL5", "coef": -0.0242},
        {"label": "published v7 old-frame HL3", "coef": -0.0872}],
    "quotable": "With footage-exposure controls, the v7 form coefficient "
                "collapses from -0.130 to +0.012 at HL5 (109% absorbed; HL3 "
                "-0.118 to -0.049, 58%): at HL5 'form is mean-reverting' was "
                "scouting/exposure in disguise, at HL3 about half of it was "
                "- and neither the form term nor the exposure term survives "
                "the expanded holdout anyway (all INSIDE the 1.773m floor).",
    "verdict_summary": "3b: exposure term train-fits (c_dm30 +0.118, c_dso "
                       "+0.073, c_dsi -0.087) but is INSIDE NOISE FLOOR on "
                       "holdout (-0.28m) and hurts the EWC bucket (-10.9m). "
                       "Decomposition answered as above.",
}
json.dump(out, open(os.path.join(ST, "context_exposure.json"), "w"), indent=1)

# ── 3e shrink ───────────────────────────────────────────────────────────────
out = {
    "provenance": PROV,
    "preregistered": "preregister.context.md 3e",
    "mechanism": "z' = beta*rdiff*exp(-k*X); X1 stand-in load "
                 "(1-integ_w)+(1-integ_l); X2 |dlog1p(dso)|",
    "ewc_bucket_baselines": {
        "published_oldframe_legacy2026": {"ll": 0.69182, "n": 109},
        "B0_expanded_legacy2026": {"ll": 0.68633, "n": 115},
        "B0_expanded_fullclass": {"ll": 0.67224, "n": 291}},
    "fit_X1_standin": s3["3e_shrink_standin"],
    "fit_X2_prepasym": s3["3e_shrink_prepasym"],
    "fit_joint": s3["3e_shrink_joint"],
    "falsifier_class_dummy": s3["3e_classdummy_falsifier"],
    "verdict_summary": "3e INSIDE NOISE FLOOR everywhere. The mechanism sign "
                       "is right (k_standin=+0.347 train; EWC full-class "
                       "bucket +3.46m better) and the class-dummy falsifier "
                       "does NOT beat it in-bucket (dummy k=+0.86 makes the "
                       "EWC bucket -3.76m WORSE while gaining +0.50m overall "
                       "via beta re-inflation on non-EWC rows). So: weak "
                       "mechanism-consistent signal in the right bucket, but "
                       "overall effect -0.40m, inside the 1.773m floor - not "
                       "promotable, kept as a Phase-5 lead.",
}
json.dump(out, open(os.path.join(ST, "context_shrink.json"), "w"), indent=1)

# ── adjacency ───────────────────────────────────────────────────────────────
out = {
    "provenance": PROV,
    "preregistered": "preregister.context.md 3b-adjacency",
    **s3["3b_adjacency"],
    "verdict": "UNTESTABLE AT THIS N (published as such): c_hat +0.021, "
               "95% CIs [-0.081,+0.124] Wald / [-0.088,+0.137] CRN boot, "
               "n=57 series where both orgs attended the previous Masters. "
               "Point sign is OPPOSITE the deep-run-fatigue prediction; the "
               "CI comfortably spans zero.",
}
json.dump(out, open(os.path.join(ST, "context_adjacency.json"), "w"), indent=1)
print("step4a: wrote context_{seriousness,stakes,exposure,shrink,adjacency}.json")
