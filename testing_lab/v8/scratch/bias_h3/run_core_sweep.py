"""agent:bias-h3 — Experiment 1: SS-core (q, V0) train-only sweep + variants 1b/1c.

Selection criterion: TRAIN mean series NLL, beta refit per config (preregistered).
Holdout numbers are computed and stored for the record but play no role in
selection (they are printed only for the final train-selected primary).
Checkpoints every grid point to sweep_core.json (restart skips finished points).
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lib_h3 import GameData, implied_half_life  # noqa: E402

CKPT = os.path.join(HERE, "sweep_core.json")


def load_ckpt():
    if os.path.exists(CKPT):
        return json.load(open(CKPT))
    return {"points": {}}


def save_ckpt(ck):
    tmp = CKPT + ".tmp"
    with open(tmp, "w") as f:
        json.dump(ck, f, indent=1)
    os.replace(tmp, CKPT)


def key(tag, qr, v0r, extra=""):
    return f"{tag}|q/R={qr:.6g}|V0/R={v0r:.6g}{extra}"


def eval_point(gd, ck, tag, qr, v0r, **kw):
    extra = "".join(f"|{k}={v:.6g}" if isinstance(v, float) else f"|{k}={v}"
                    for k, v in kw.items())
    k = key(tag, qr, v0r, extra)
    if k in ck["points"]:
        return ck["points"][k]
    res = gd.eval_config(q=qr * gd.R, V0=v0r * gd.R, **kw)
    rec = {"q_over_R": qr, "V0_over_R": v0r, **{kk: vv for kk, vv in kw.items()},
           "beta": round(res["beta"], 5),
           "ll_train": round(res["ll_train"], 6),
           "ll_holdout": round(res["ll_holdout"], 6),
           "hl_games": round(implied_half_life(qr * gd.R, gd.R), 2)}
    ck["points"][k] = rec
    save_ckpt(ck)
    return rec


def refine(gd, ck, tag, best_qr, best_v0r, **kw):
    """One local refinement: factor-of-2 window, 7x5 log grid."""
    qs = np.geomspace(best_qr / 2, best_qr * 2, 7)
    vs = np.geomspace(max(best_v0r / 2, 1e-3), best_v0r * 2, 5)
    best = None
    for qr in qs:
        for v0r in vs:
            r = eval_point(gd, ck, tag, float(qr), float(v0r), **kw)
            if best is None or r["ll_train"] < best["ll_train"]:
                best = r
    return best


def main():
    t0 = time.time()
    gd = GameData()
    ck = load_ckpt()
    print(f"GameData ready ({time.time()-t0:.1f}s), R={gd.R:.4f}", flush=True)

    # ── 1a coarse grid ──────────────────────────────────────────────────────
    best = None
    for qr in np.geomspace(1e-4, 3e-2, 13):
        for v0r in np.geomspace(0.05, 3.0, 9):
            r = eval_point(gd, ck, "1a", float(qr), float(v0r))
            if best is None or r["ll_train"] < best["ll_train"]:
                best = r
    print(f"1a coarse best: {best}", flush=True)
    best1a = refine(gd, ck, "1a", best["q_over_R"], best["V0_over_R"])
    print(f"1a refined best: {best1a}", flush=True)

    # ── 1b calendar leak: sweep q_cal at 1a's (q,V0), then local re-refine ──
    best1b = dict(best1a, q_cal_week=0.0)
    for qc in [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]:
        r = eval_point(gd, ck, "1b", best1a["q_over_R"], best1a["V0_over_R"],
                       q_cal_week=float(qc * gd.R))
        if r["ll_train"] < best1b["ll_train"]:
            best1b = r
    if best1b.get("q_cal_week", 0.0) > 0.0:
        b2 = refine(gd, ck, "1b", best1b["q_over_R"], best1b["V0_over_R"],
                    q_cal_week=best1b["q_cal_week"])
        if b2["ll_train"] < best1b["ll_train"]:
            best1b = b2
    print(f"1b best: {best1b}", flush=True)

    # ── 1c debut region prior: local re-refine around 1a ────────────────────
    r1c = eval_point(gd, ck, "1c", best1a["q_over_R"], best1a["V0_over_R"],
                     debut_region_prior=True)
    best1c = refine(gd, ck, "1c", r1c["q_over_R"], r1c["V0_over_R"],
                    debut_region_prior=True)
    print(f"1c best: {best1c}", flush=True)

    cand = {"1a": best1a, "1b": best1b, "1c": best1c}
    primary = min(cand, key=lambda k: cand[k]["ll_train"])
    ck["best"] = {"candidates": cand, "primary": primary,
                  "primary_cfg": cand[primary],
                  "note": "primary = best TRAIN NLL (preregistered); holdout "
                          "stored for the record, not used in selection"}
    save_ckpt(ck)
    print(f"PRIMARY = {primary}: {cand[primary]}", flush=True)
    print(f"total {time.time()-t0:.1f}s, {len(ck['points'])} points", flush=True)


if __name__ == "__main__":
    main()
