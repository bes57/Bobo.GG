"""agent:context step 4b — stats/context_weights.json (3d) from step2b
results + CRN boots on the fitted config's holdout delta."""
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
test = b0["test_mask"]
ev_test = b0["event_ids"]
s2a = json.load(open(os.path.join(SC, "step2a_results.json")))
s2b = json.load(open(os.path.join(SC, "step2b_results.json")))
MDE_W = 1.773

p_f = np.load(os.path.join(SC, "ptest_3d_fitted.npy"))
loss_f = -np.log(np.clip(p_f, 1e-9, 1))
d = loss_b0 - loss_f
dll = float(d.mean())
bi = referee.paired_bootstrap_crn(d, mode="iid")
bb = referee.paired_bootstrap_crn(d, mode="block_event", event_ids=ev_test)
roi = referee.expected_roi_of_dll(dll, b0["p_test"])
ECL = frame.eclass.values
ewcfull_t = (ECL == "ewc_offseason")[test]
verdict = ("DEAD" if dll < 0 and abs(dll) * 1000 >= MDE_W and bi["ci_hi"] < 0
           else "WIN" if dll > 0 and dll * 1000 >= MDE_W and bi["ci_lo"] > 0
           else "INSIDE NOISE FLOOR")

out = {
    "provenance": {"agent": "context", "written": "2026-07-28",
                   "preregistered": "preregister.context.md 3d",
                   "baseline_B0": s2a["B0"],
                   "mde_milli": {"within": 1.773, "cross": 5.889}},
    "design": "per-game event-class solve weights replacing v6's hand-set "
              "{stage-playoffs 1.6, champions x2.0}; anchor vct_regular=1.0; "
              "Nelder-Mead in log-space on train NLL (beta refit per eval, "
              "walk-forward inherent), x0=[1.6,2.0,1.3,1.0], 162 evals",
    "classes": "champions=YYYY_champions; masters=YYYY_masters_*+lock_in; "
               "ewc_offseason=ratings_only minus 2023_lcq; vct split "
               "regular/playoffs by stage",
    "fitted_weights": s2b["fitted"],
    "fitted_train_ll": s2b["ll_train_fitted"],
    "B0_train_ll": s2a["B0"]["ll_train"],
    "ci_per_weight": s2b["ci"],
    "ci_recipe": s2b["ci_recipe"],
    "profiles": s2b["profiles"],
    "holdout": {
        "ll_fitted": round(float(loss_f.mean()), 5),
        "ll_B0_handset": round(float(loss_b0.mean()), 5),
        "dll_milli": round(dll * 1000, 3), "pair_mde_milli": MDE_W,
        "boot_iid_ci_milli": [round(bi["ci_lo"] * 1000, 3),
                              round(bi["ci_hi"] * 1000, 3)],
        "boot_block_ci_milli": [round(bb["ci_lo"] * 1000, 3),
                                round(bb["ci_hi"] * 1000, 3)],
        "p_better_iid": bi["p_better"],
        "expected_roi_delta": roi["expected_roi_delta"],
        "roi_note": "ladder clamps at delta_logit 0; negative dll -> 0.0",
        "ewc_full_bucket_dll_milli": round(float(d[ewcfull_t].mean()) * 1000, 2),
        "verdict": verdict},
    "ewc_weight_deliverable": {
        "fitted": s2b["fitted"]["ewc_offseason"],
        "ci95": [s2b["ci"]["ewc_offseason"]["ci_lo"],
                 s2b["ci"]["ewc_offseason"]["ci_hi"]],
        "reading": "the train objective does NOT want EWC-class games "
                   "down-weighted (fitted ~1.0); the operator's seriousness "
                   "discount is unsupported in-sample, matching 3a's "
                   "blanket-grid result",
    },
}
json.dump(out, open(os.path.join(ST, "context_weights.json"), "w"), indent=1)
print("context_weights.json written; holdout dll %+.3fm iidCI[%+.2f,%+.2f]m "
      "verdict=%s; w_ewc=%s CI %s" % (
          dll * 1000, bi["ci_lo"] * 1000, bi["ci_hi"] * 1000, verdict,
          out["ewc_weight_deliverable"]["fitted"],
          out["ewc_weight_deliverable"]["ci95"]))
