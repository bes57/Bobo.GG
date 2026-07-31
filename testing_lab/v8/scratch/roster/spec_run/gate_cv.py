"""SPEC RUN — shrinkage (lambda by inner-CV) + activation gate + policy pick.

TRAIN ONLY. Consumes stage_manifest.json + stage_runs.npz (rdiff per config,
beta per config). No holdout aggregate is computed anywhere here: per-row
NLL is evaluated on train rows exclusively.

Gate (preregistered): Delta_f = meanNLL_f(v6) - meanNLL_f(model at
a_hat_{-f}(lambda_hat)); fires iff mean_f(Delta) > sd(Delta,ddof=1)/sqrt(5)
AND the shrunk parameter > 0. Policy = max mean_f(Delta) among firing;
if none fires the deployed model IS v6 and the documentation config is the
unshrunk train-argmin.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
V8 = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)
import runner  # noqa: E402

LAM_GRID = [0.0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
K = 5

man = json.load(open(os.path.join(HERE, "stage_manifest.json")))
npz = np.load(os.path.join(HERE, "stage_runs.npz"))
frame = runner.load_frame()
fmts = frame.fmt.values
train_rows_all = (frame.date <= "2024-12-31").values
tau_hat = man["tau_hat"]

# valid mask identical across configs (engine skips the same warmup days)
valids = {}
for cid in man["configs"]:
    valids[cid] = ~np.isnan(npz[cid])
ref_valid = next(iter(valids.values()))
for cid, v in valids.items():
    assert (v == ref_valid).all(), f"valid-mask drift in {cid}"
train_m = ref_valid & train_rows_all
tr_idx = np.where(train_m)[0]
folds = np.array_split(np.arange(len(tr_idx)), K)   # contiguous in date order
print(f"train rows n={len(tr_idx)}; folds {[len(f) for f in folds]}")


def nll_train_rows(cid):
    rd = npz[cid][train_m]
    beta = man["configs"][cid]["beta"]
    p = runner.p_series_closed(beta, rd, fmts[train_m])
    return -np.log(np.clip(p, 1e-9, 1.0))


NLL = {cid: nll_train_rows(cid) for cid in man["configs"]}
v6_cid = next(cid for cid, m in man["configs"].items()
              if m["a"] == 0 and m["s"] == 0)
nll_v6 = NLL[v6_cid]


def profile_configs(pol, s_sel):
    """(param_value, cid) along the policy's shrinkage profile."""
    out = {}
    for cid, m in man["configs"].items():
        if m["policy"] != pol:
            continue
        if pol == "p3w5":
            out[m["s"]] = cid            # P3: parameter is s
        else:
            if m["s"] == s_sel and (m["tau"] == tau_hat or m["a"] == 0):
                out[m["a"]] = cid
    if pol != "p3w5" and s_sel > 0 and 0.0 not in out:
        pass
    if pol == "p3w5":
        out.setdefault(0.0, v6_cid)
    return sorted(out.items())


results = {}
for pol in ("p1w3c5", "p1w5c5", "p1w8c5", "p3w5"):
    s_sel = man["s_hat"][pol] if pol != "p3w5" else None
    prof = profile_configs(pol, s_sel if pol != "p3w5" else None)
    params = [p for p, _ in prof]
    assert 0.0 in params, f"{pol}: profile lacks the 0 point"
    mat = np.stack([NLL[cid] for _, cid in prof])     # (n_param, n_train)
    fold_mean = np.stack([mat[:, f].mean(axis=1) for f in folds])  # (K,np)
    fold_n = np.array([len(f) for f in folds], dtype=float)
    full_mean = (fold_mean * fold_n[:, None]).sum(0) / fold_n.sum()
    pen = np.array(params) ** 2

    cv_scores = {}
    for lam in LAM_GRID:
        tot = 0.0
        for fi in range(K):
            w = fold_n.copy(); w[fi] = 0.0
            loo = (fold_mean * w[:, None]).sum(0) / w.sum()
            j = int(np.argmin(loo + lam * pen))
            tot += fold_mean[fi, j] * fold_n[fi]
        cv_scores[lam] = tot / fold_n.sum()
    lam_hat = min(cv_scores, key=cv_scores.get)
    j_hat = int(np.argmin(full_mean + lam_hat * pen))
    a_shrunk = params[j_hat]
    a_raw = params[int(np.argmin(full_mean))]

    deltas = []
    for fi in range(K):
        w = fold_n.copy(); w[fi] = 0.0
        loo = (fold_mean * w[:, None]).sum(0) / w.sum()
        j = int(np.argmin(loo + lam_hat * pen))
        d_f = nll_v6[folds[fi]].mean() - fold_mean[fi, j]
        deltas.append(float(d_f))
    deltas = np.array(deltas)
    mean_d, se_d = float(deltas.mean()), float(deltas.std(ddof=1) / np.sqrt(K))
    fired = bool(mean_d > se_d and a_shrunk > 0)
    results[pol] = {
        "parameter": "s" if pol == "p3w5" else "a",
        "s_hat_trainstage": s_sel, "tau_hat": tau_hat,
        "profile": {str(p): round(float(m), 6)
                    for p, m in zip(params, full_mean)},
        "a_raw_unshrunk": a_raw, "lambda_hat": lam_hat,
        "a_shrunk": a_shrunk,
        "cv_scores_milli_vs_v6": {str(l): round((float(nll_v6.mean()) - v)
                                                * 1000, 4)
                                  for l, v in cv_scores.items()},
        "gate_delta_folds_milli": [round(d * 1000, 4) for d in deltas],
        "gate_mean_milli": round(mean_d * 1000, 4),
        "gate_se_milli": round(se_d * 1000, 4),
        "gate_fired": fired,
    }
    print(f"{pol}: raw={a_raw} lam={lam_hat} shrunk={a_shrunk} "
          f"CVimp={mean_d*1000:+.3f}m SE={se_d*1000:.3f}m fired={fired}",
      flush=True)

firing = [p for p in results if results[p]["gate_fired"]]
TIE_ORDER = ["p3w5", "p1w5c5", "p1w3c5", "p1w8c5"]
if firing:
    best = sorted(firing, key=lambda p: (-results[p]["gate_mean_milli"],
                                         TIE_ORDER.index(p)))[0]
    r = results[best]
    shipped = {"policy": best, "a": r["a_shrunk"] if best != "p3w5" else 0.0,
               "tau": tau_hat,
               "s": r["a_shrunk"] if best == "p3w5" else r["s_hat_trainstage"],
               "n_min": man["n_min"], "deployed_is_v6": False}
else:
    best = min(results, key=lambda p: min(
        results[p]["profile"].values()))
    r = results[best]
    a_doc = r["a_raw_unshrunk"]
    shipped = {"policy": "v6", "deployed_is_v6": True,
               "documentation_policy": best,
               "documentation_a": a_doc if best != "p3w5" else 0.0,
               "documentation_tau": tau_hat,
               "documentation_s": (a_doc if best == "p3w5"
                                   else results[best]["s_hat_trainstage"]),
               "n_min": man["n_min"]}

decision = {"written_by": "agent:roster-g SPEC RUN (train-only)",
            "preregistered": "ADDENDUM 4 gate bar: inner-CV improvement > "
                             "its own inner-CV SE (K=5 contiguous time "
                             "folds), shrinkage lambda*a^2 by inner-CV",
            "v6_train_nll": round(float(nll_v6.mean()), 6),
            "per_policy": results, "gate_fired_any": bool(firing),
            "selection": shipped,
            "p2": "NOT RUN (fixtures: region-prior recursion breaks "
                  "final-timeline identity; root cause proven)"}
with open(os.path.join(HERE, "gate_decision.json"), "w") as f:
    json.dump(decision, f, indent=1)
print(json.dumps(shipped, indent=1))
