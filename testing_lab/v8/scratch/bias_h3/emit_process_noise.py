"""Emit stats/h3_process_noise.json (EARLY deliverable — agent:decay references it)."""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
V8 = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
from lib_h3 import implied_half_life  # noqa: E402

ck = json.load(open(os.path.join(HERE, "sweep_5d.json")))
core = json.load(open(os.path.join(HERE, "sweep_core.json")))
meta = ck["meta"]
R = meta["R"]

CELLS = {
    "A_roster": {
        "axis": "roster stability (matches_since_change at the game's match; "
                "lineups-agent definition, full-corpus scratch top-up, "
                "implementation verified 0/3466 mismatches)",
        "cells": ["change-adjacent (msc<=3)", "settling (4<=msc<=10)",
                  "stable (msc>10)"],
        "prediction": "q(change-adjacent) > q(stable) by >=2x"},
    "B_orgage": {
        "axis": "org age (days since org's first corpus game; LEFT-CENSORED at "
                "corpus start 2023 — pre-2023 history invisible)",
        "cells": ["young (<180d)", "mid (180-540d)", "established (>540d)"],
        "prediction": "q(young) > q(established)"},
    "C_volatility": {
        "axis": "trailing rating volatility (std of last 12 standardized "
                "innovations from the fitted plain core; train terciles; "
                "<6 prior innovations -> middle cell)",
        "cells": ["low vol", "mid vol", "high vol"],
        "prediction": "q(high vol) > q(low vol)"},
}

axes_out = {}
for tag, spec in CELLS.items():
    r = ck[tag]
    qs = r["q"]
    axes_out[tag] = {
        "axis_definition": spec["axis"],
        "preregistered_prediction": spec["prediction"],
        "cells": spec["cells"],
        "n_games_per_cell_side": meta[f"axis{tag[0]}_cells_games"],
        "q_mle": [round(x, 6) for x in qs],
        "q_over_R_mle": [round(x / R, 6) for x in qs],
        "half_life_games_mle": [round(x, 2) for x in r["hl_games"]],
        "half_life_games_ci95_profile": [[round(a, 2), round(b, 2)]
                                         for a, b in r["hl_ci"]],
        "logq_ci95_profile": [[round(a, 3), round(b, 3)] for a, b in r["ci_logq"]],
        "logq_se_curvature": [round(x, 3) for x in r["se_logq"]],
        "partial_pooling_DL": {
            "tau2_logq": round(r["pooling"]["tau2"], 5),
            "Q_heterogeneity": round(r["pooling"]["Q"], 4),
            "pooled_q": [round(x, 6) for x in r["q_pooled"]],
            "pooled_half_life_games": [round(x, 2) for x in r["hl_games_pooled"]],
            "shrink_fraction": r["pooling"]["shrink_fraction"]},
        "beta_refit_train": round(r["beta"], 5),
        "ll_train": r["ll_train_final"],
        "ll_holdout_record": r["ll_holdout"],
    }

core_1a = core["points"]["1a|q/R=0.00564622|V0/R=1.27156"]
out = {
  "written_by": "agent:bias-h3 (Phase 5 H3 / experiment 5d)",
  "preregistered": "preregister.bias_h3.md experiment 2 (written before fits)",
  "frame": "v8/data/frame_expanded/series.csv (sha256 verified vs crn.json)",
  "model": {
    "family": "state-space Kalman analog; per-team (rating, variance); "
              "observation per map y = sign(rd)|rd|^0.75*2.5; R fixed = "
              "Var(y) on train games (identification constant — all reported "
              "quantities depend only on q/R and are invariant to this choice); "
              "game weights champ x2 / playoffs x1.6 scale R_i = R/w_i; "
              "q per own map-game (no calendar accrual, in-family with v6 "
              "games-counted decay); strict-day walk-forward",
    "R": round(R, 4),
    "base_core": {"q": round(meta["q0"], 5), "q_over_R": core_1a["q_over_R"],
                  "V0_over_R": core_1a["V0_over_R"],
                  "half_life_games": core_1a["hl_games"],
                  "ll_train": core_1a["ll_train"],
                  "ll_holdout": core_1a["ll_holdout"]},
    "fit_protocol": "per-axis K=3 free q_k (V0 frozen at core's fit), "
                    "coordinate descent on log q, TRAIN series NLL only, "
                    "beta refit train-only at every evaluation; profile-"
                    "likelihood CIs (Delta total train NLL = 1.92); "
                    "DerSimonian-Laird random-effects pooling on log q "
                    "(no free per-team q anywhere)"},
  "v6_reference": {"ll_train": 0.64823, "ll_holdout": 0.64216,
                   "note": "consist(20,12) engine baseline on the same frame; "
                           "v6 half-lives 20 (consistent) / 12 (anomalous) "
                           "map-games for comparison"},
  "mde_context": {"within_family_milli": 1.773, "cross_family_milli": 5.889,
                  "source": "stats/power_mde_expanded.json"},
  "axes": axes_out,
  "headline": {
    "roster_axis_point_estimates": "change-adjacent HL 7.4 games vs stable "
        "24.8 games (q ratio ~11x, predicted direction, >2x prereg bar); "
        "settling cell hits the q->0 boundary (HL unbounded) — non-monotone",
    "pooling_verdict": "DL tau2 = 0 on ALL axes: between-cell spread is NOT "
        "resolvable against the train profile curvature at n_train=841 series "
        "(2095 train games) — cells fully shrink to the pooled q (~HL 8 "
        "games). Point ordering and the pooled null are BOTH the published "
        "result; the roster axis's holdout record (ll 0.64748 vs core "
        "0.65103, ~+3.6m within-family) is independent evidence the "
        "structure generalizes — final paired-bootstrap adjudication in "
        "stats/h3_statespace.json",
    "org_age_axis": "REVERSED vs prediction (established orgs fit the "
        "LARGEST q, HL ~4 games) and degrades holdout by ~14m — read as "
        "left-censoring artifact + small-cell overfit, prediction falsified",
    "volatility_axis": "flat (HL 8.9-11.0 across terciles) — prediction "
        "falsified at this n"},
  "caveats": [
    "cell-1 (settling) roster q sits on the search boundary (q -> 0); its "
    "profile CI upper end is well below the other cells' MLEs — treat the "
    "329-game HL as 'indistinguishable from no drift', not a measurement",
    "DL pooling treats the three log-q MLEs as independent Gaussians; cells "
    "interact through shared beta refits and opponent variances",
    "org-age axis is left-censored at corpus start (2023) — 'young' cannot "
    "be distinguished from 'newly visible'",
    "R is an identification constant; q/R, half-lives and n_eff are "
    "invariant to its value"],
}

path = os.path.join(V8, "stats", "h3_process_noise.json")
with open(path, "w") as f:
    json.dump(out, f, indent=1)
print("wrote", path)
