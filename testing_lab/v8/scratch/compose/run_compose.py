"""agent:compose — Wave 3 stacks, per preregister.compose.md (written first).

Phases:
  train : verify frame + inputs, build gate, joint train fits (S2/S3),
          machinery smoke test (v6-vs-v6 self gate; reveals nothing),
          checkpoint to train_fits.json.
  score : ONE holdout scoring per stack (sentinel-guarded), promotion_gate +
          CV boots + caterpillar + buckets -> stats/compose_stacks.json,
          stats/compose_gate.json.
Fail loudly everywhere; no silent substitution.
"""
import hashlib
import json
import math
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
V8 = os.path.dirname(os.path.dirname(HERE))
TL = os.path.dirname(V8)
sys.path.insert(0, TL)
sys.path.insert(0, V8)
import referee  # noqa: E402

LOG = os.path.join(V8, "logs", "compose.log")
STATS = os.path.join(V8, "stats")
FRAME = os.path.join(V8, "data", "frame_expanded", "series.csv")
SENTINEL = os.path.join(HERE, "SCORED")
FITS = os.path.join(HERE, "train_fits.json")

MDE_WITHIN = 1.773e-3
MDE_CROSS = 5.889e-3
R_H3 = 11.2933           # h3 identification constant (stats/h3_process_noise.json)
NEFF_TH = 12.0
GATE_COUNTS_PUB = (330, 178)   # h3_ensemble.json ss_5d_roster|neff<12


def jlog(msg):
    line = f"[{datetime.now().strftime('%F %T')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def die(msg):
    jlog("FATAL: " + msg)
    raise SystemExit(1)


def sp(pm, fm):
    return np.where(np.isin(fm, ("bo5", "bo5_gf")),
                    pm ** 3 * (1 + 3 * (1 - pm) + 6 * (1 - pm) ** 2),
                    np.where(fm == "bo1", pm, pm ** 2 * (3 - 2 * pm)))


def load_all():
    sha = hashlib.sha256(open(FRAME, "rb").read()).hexdigest()
    crn = json.load(open(os.path.join(V8, "crn.json")))
    want = crn["frame_expanded"]["series_csv_sha256"]
    if sha != want:
        die(f"frame sha mismatch {sha} != {want}")
    frame = pd.read_csv(FRAME, dtype={"date": str})
    assert len(frame) == 2058
    holdout = (frame.date > "2024-12-31").values
    train = ~holdout
    assert int(holdout.sum()) == 1217 and int(train.sum()) == 841

    z = np.load(os.path.join(V8, "scratch", "bias_h3", "model_probs.npz"))
    p_v6 = z["p_v6"]
    v6_valid = z["v6_valid"].astype(bool)
    p_5d = z["p_ss_5d"]
    vw5, vl5 = z["vw_5d"], z["vl_5d"]

    b0 = np.load(os.path.join(V8, "scratch", "context", "b0.npz"),
                 allow_pickle=True)
    rd_v6 = b0["rdiff"]
    # cross-checks: the two independently-built v6 replays must agree
    tmask = b0["test_mask"].astype(bool)
    dd = np.nanmax(np.abs(b0["p_test"] - p_v6[tmask]))
    if not dd < 1e-4:   # solver jitter between independent replays; both
        die(f"b0 vs model_probs v6 disagree: max|dp|={dd}")
    jlog(f"v6 replay cross-check: max|dp| = {dd:.2e} (b0 vs model_probs; "
         "canonical p_v6 = model_probs, the h3-ensemble/gate-count source)")
    ll6 = referee.logloss(p_v6[holdout & v6_valid])
    if abs(ll6 - 0.64216) > 5e-6:
        die(f"v6 holdout LL {ll6} != 0.64216")
    ll5d = referee.logloss(p_5d[holdout])
    if abs(ll5d - 0.647483) > 5e-6:
        die(f"5d holdout LL {ll5d} != 0.647483")

    ze = np.load(os.path.join(V8, "scratch", "decay", "probs",
                              "eclass_on_v6_m0.8.npz"))
    rd_fade, p_fade = ze["rdiff"], ze["p"]
    beta_fade = float(ze["beta"])
    llf = referee.logloss(p_fade[holdout & ~np.isnan(rd_fade)])
    if abs(llf - 0.64192) > 5e-6:
        die(f"fade holdout LL {llf} != 0.64192")

    ff = pd.read_csv(os.path.join(V8, "scratch", "context",
                                  "frame_features.csv"))
    if not (ff.match_id.values == frame.match_id.values).all():
        die("frame_features not row-aligned to frame")
    X1 = (1 - ff.integ_w.values) + (1 - ff.integ_l.values)
    if np.isnan(X1).any():
        die("X1 has NaN")

    neff_min = np.minimum(R_H3 / vw5, R_H3 / vl5)
    gate = neff_min < NEFF_TH
    n_tr = int((gate & train & v6_valid).sum())
    n_ho = int((gate & holdout & v6_valid).sum())
    if (n_tr, n_ho) != GATE_COUNTS_PUB:
        jlog(f"gate counts ({n_tr},{n_ho}) != published {GATE_COUNTS_PUB} "
             f"with R={R_H3}; recomputing exact R from engine corpus")
        from engine import Engine
        eng = Engine()
        y = np.array([np.sign(g["wr"] - g["lr"]) *
                      abs(g["wr"] - g["lr"]) ** 0.75 * 2.5 for g in eng.games])
        gtr = np.array([g["date_s"] <= "2024-12-31" for g in eng.games])
        R_exact = float(np.var(y[gtr]))
        jlog(f"exact R = {R_exact}")
        neff_min = np.minimum(R_exact / vw5, R_exact / vl5)
        gate = neff_min < NEFF_TH
        n_tr = int((gate & train & v6_valid).sum())
        n_ho = int((gate & holdout & v6_valid).sum())
        if (n_tr, n_ho) != GATE_COUNTS_PUB:
            die(f"gate counts still {(n_tr, n_ho)} != {GATE_COUNTS_PUB}")
    jlog(f"gate ok: {n_tr} train / {n_ho} holdout rows (neff_min<{NEFF_TH})")
    return dict(frame=frame, holdout=holdout, train=train, p_v6=p_v6,
                v6_valid=v6_valid, p_5d=p_5d, rd_v6=rd_v6, rd_fade=rd_fade,
                p_fade=p_fade, beta_fade=beta_fade, X1=X1, gate=gate,
                neff_min=neff_min)


def fit_bk(rd, X1, fmts, mask):
    """3e joint (beta, k) fit: z = b*rd*exp(-k*X1), train ML (Nelder-Mead)."""
    rdm, xm, fm = rd[mask], X1[mask], fmts[mask]

    def nll(prm):
        b, k = prm
        pm = 1 / (1 + np.exp(-(b * rdm * np.exp(-k * xm))))
        return -np.mean(np.log(np.clip(sp(pm, fm), 1e-9, 1)))

    fit = minimize(nll, [0.13, 0.0], method="Nelder-Mead",
                   options={"maxiter": 4000, "xatol": 1e-5, "fatol": 1e-9})
    return float(fit.x[0]), float(fit.x[1]), float(fit.fun)


def surface(rd, X1, fmts, b, k):
    p = np.full(len(rd), np.nan)
    v = ~np.isnan(rd)
    pm = 1 / (1 + np.exp(-(b * rd[v] * np.exp(-k * X1[v]))))
    p[v] = sp(pm, fmts[v])
    return p


def phase_train():
    d = load_all()
    frame, train, holdout = d["frame"], d["train"], d["holdout"]
    fmts = frame.fmt.values
    vfade = ~np.isnan(d["rd_fade"])
    m_tr = vfade & train
    # S2 joint fit on all valid train rows
    b2, k2, nll2 = fit_bk(d["rd_fade"], d["X1"], fmts, m_tr)
    jlog(f"S2 fit: beta={b2:.4f} k={k2:.4f} train_nll={nll2:.5f} "
         f"(n={int(m_tr.sum())})")
    # S3 joint fit on non-gated valid train rows (composite train NLL)
    m_tr3 = m_tr & ~d["gate"]
    b3, k3, nll3 = fit_bk(d["rd_fade"], d["X1"], fmts, m_tr3)
    jlog(f"S3 fit (non-gated train): beta={b3:.4f} k={k3:.4f} "
         f"train_nll={nll3:.5f} (n={int(m_tr3.sum())})")
    # composite train NLLs for the record (train rows, v6-valid aligned)
    p_v6, p_5d, gate = d["p_v6"], d["p_5d"], d["gate"]
    mtv = train & d["v6_valid"]
    p_s1 = np.where(gate, p_5d, p_v6)
    p_s2 = surface(d["rd_fade"], d["X1"], fmts, b2, k2)
    p_s3 = np.where(gate, p_5d, surface(d["rd_fade"], d["X1"], fmts, b3, k3))
    rec = {
        "written": datetime.now().strftime("%F %T"),
        "S2": {"beta": b2, "k_standin": k2, "nll_train_fitmask": nll2,
               "n_train_fit": int(m_tr.sum())},
        "S3": {"beta": b3, "k_standin": k3, "nll_train_fitmask": nll3,
               "n_train_fit": int(m_tr3.sum())},
        "train_nll_composites": {
            "v6": referee.logloss(p_v6[mtv]),
            "S1_gate5d": referee.logloss(p_s1[mtv]),
            "S2_fade_shrink": referee.logloss(p_s2[mtv & ~np.isnan(p_s2)]),
            "S3_full": referee.logloss(p_s3[mtv & ~np.isnan(p_s3)])},
        "gate_counts": GATE_COUNTS_PUB}
    with open(FITS, "w") as f:
        json.dump(rec, f, indent=1)
    jlog("train composites (train rows): " +
         json.dumps(rec["train_nll_composites"]))
    # machinery smoke test: v6 vs v6 self-gate (reveals nothing about stacks)
    from engine import Engine
    eng = Engine()
    g_self = referee.promotion_gate(
        {"label": "v6_copy", "p": p_v6.copy()}, {"label": "v6", "p": p_v6},
        mde=MDE_WITHIN, frame=frame, rdiff_ref=d["rd_v6"], games=eng.games)
    if g_self["verdict"] != "HOLD":
        die("self-gate smoke test did not HOLD")
    jlog(f"smoke test ok: self-gate HOLD, n_scored={g_self['n_scored']}, "
         f"v6 max|bias|={g_self['bias_tables']['v6']['max_abs_bias']}")
    jlog("phase_train DONE")


def cv_judge(p_c, p_ref, rd_ref, mask, ev):
    """raw + CUPED(l_v6) CRN boots, pair-MDEs, both units (decay recipe)."""
    d = referee.delta_vector(p_c[mask], p_ref[mask])
    n = int(mask.sum())
    l6 = referee.per_series_ll(p_ref[mask])
    Xc = np.column_stack([l6]) - l6.mean()
    th = np.linalg.lstsq(Xc, d, rcond=None)[0]
    d_cv = d - Xc @ th
    b_iid = referee.paired_bootstrap_crn(d, mode="iid")
    b_blk = referee.paired_bootstrap_crn(d, mode="block_event",
                                         event_ids=ev[mask])
    c_iid = referee.paired_bootstrap_crn(d_cv, mode="iid")
    c_blk = referee.paired_bootstrap_crn(d_cv, mode="block_event",
                                         event_ids=ev[mask])
    roi = referee.expected_roi_of_dll(float(d.mean()), p_ref[mask])
    keep = ("mean_delta", "ci_lo", "ci_hi", "p_better")
    return {
        "n": n, "delta_milli": round(float(d.mean()) * 1000, 3),
        "pair_mde_raw_milli": round(2.8016 * float(np.std(d, ddof=1))
                                    / math.sqrt(n) * 1000, 3),
        "pair_mde_cv_milli": round(2.8016 * float(np.std(d_cv, ddof=1))
                                   / math.sqrt(n) * 1000, 3),
        "boot_iid": {k: b_iid[k] for k in keep},
        "boot_block": {k: b_blk[k] for k in keep},
        "boot_iid_cv": {k: c_iid[k] for k in keep},
        "boot_block_cv": {k: c_blk[k] for k in keep},
        "expected_roi_delta": roi["expected_roi_delta"],
        "roi_at_op": roi["roi_at_op"],
        "delta_logit_equiv": roi["delta_logit_equiv"]}


def phase_score():
    if os.path.exists(SENTINEL):
        die("SCORED sentinel exists — one holdout scoring per stack, ever")
    if not os.path.exists(FITS):
        die("train fits absent — run phase train first")
    fits = json.load(open(FITS))
    d = load_all()
    frame, holdout, train = d["frame"], d["holdout"], d["train"]
    fmts = frame.fmt.values
    ev = frame.event_id.values
    p_v6, p_5d, gate = d["p_v6"], d["p_5d"], d["gate"]
    rd_v6 = d["rd_v6"]
    from engine import Engine
    eng = Engine()

    b2, k2 = fits["S2"]["beta"], fits["S2"]["k_standin"]
    b3, k3 = fits["S3"]["beta"], fits["S3"]["k_standin"]
    p_s2 = surface(d["rd_fade"], d["X1"], fmts, b2, k2)
    p_s2p = surface(d["rd_fade"], d["X1"], fmts, b3, k3)
    stacks = {
        "S1_gate5d": {"p": np.where(gate, p_5d, p_v6), "mde": MDE_CROSS,
                      "regime": "cross"},
        "S2_fade_shrink": {"p": p_s2, "mde": MDE_WITHIN, "regime": "within"},
        "S3_full": {"p": np.where(gate, p_5d, p_s2p), "mde": MDE_CROSS,
                    "regime": "cross"}}

    # ---- the one-and-only holdout pass ----
    open(SENTINEL, "w").write(datetime.now().strftime("%F %T") + "\n")
    out_stacks, out_gate = {}, {}
    for name, sk in stacks.items():
        p_c = sk["p"]
        gate_res = referee.promotion_gate(
            {"label": name, "p": p_c}, {"label": "v6", "p": p_v6},
            mde=sk["mde"], frame=frame, rdiff_ref=rd_v6, games=eng.games)
        m = holdout & ~np.isnan(p_c) & ~np.isnan(p_v6) & ~np.isnan(rd_v6)
        jr = cv_judge(p_c, p_v6, rd_v6, m, ev)
        fam = sk["mde"] * 1000
        verdict = ("INSIDE NOISE FLOOR" if abs(jr["delta_milli"]) < fam else
                   ("WIN" if jr["delta_milli"] > 0 else "LOSS"))
        # gated / non-gated splits (diagnostics of the same single scoring)
        gm = m & gate
        ngm = m & ~gate
        split = {
            "n_gated": int(gm.sum()),
            "delta_milli_gated": round(float(referee.delta_vector(
                p_c[gm], p_v6[gm]).mean()) * 1000, 3) if gm.sum() else None,
            "delta_milli_nongated": round(float(referee.delta_vector(
                p_c[ngm], p_v6[ngm]).mean()) * 1000, 3) if ngm.sum() else None}
        ewc = frame.event_id.astype(str).str.startswith(
            ("2026_ewc", "2026_china_evo")).values
        ecl_full = pd.read_csv(os.path.join(
            V8, "scratch", "context", "frame_features.csv")).eclass.values
        ewc_full = (ecl_full == "ewc_offseason")
        buck_extra = {
            "ewc_legacy2026": {
                "n": int((m & ewc).sum()),
                "delta_milli": round(float(referee.delta_vector(
                    p_c[m & ewc], p_v6[m & ewc]).mean()) * 1000, 2)},
            "ewc_fullclass": {
                "n": int((m & ewc_full).sum()),
                "delta_milli": round(float(referee.delta_vector(
                    p_c[m & ewc_full], p_v6[m & ewc_full]).mean()) * 1000, 2)}}
        cat = gate_res["bias_tables"]["candidate"]
        resid = {t["team"]: t["bias"] for t in cat["teams"]
                 if t["team"] in ("PRX", "NRG")}
        row = {
            "spec": {
                "S1_gate5d": "p = p_ss_5d where neff_min<12 else p_v6 "
                             "(h3 5d roster-typed SS; all params h3 train-fit)",
                "S2_fade_shrink": f"z = {b2:.4f}*rd_fade*exp(-{k2:.4f}*X1); "
                                  "rd_fade = eclass_on_v6_m0.8",
                "S3_full": f"S1 gate over S2 base with (beta,k)=({b3:.4f},"
                           f"{k3:.4f}) fit on non-gated train"}[name],
            "train": (fits["S2"] if name == "S2_fade_shrink" else
                      fits["S3"] if name == "S3_full" else
                      {"note": "no new free parameters"}),
            "train_nll_composite": fits["train_nll_composites"][
                {"S1_gate5d": "S1_gate5d", "S2_fade_shrink": "S2_fade_shrink",
                 "S3_full": "S3_full"}[name]],
            "ll_holdout": round(referee.logloss(p_c[m]), 5),
            "ll_v6_holdout": round(referee.logloss(p_v6[m]), 5),
            "judging": jr,
            "family_mde_milli": fam, "regime": sk["regime"],
            "verdict_vs_noise_floor": verdict,
            "gate_verdict": gate_res["verdict"],
            "gated_split": split,
            "ewc_buckets": buck_extra,
            "prx_nrg_residual_pp": {k: round(v * 100, 1)
                                    for k, v in resid.items()},
            "caterpillar": gate_res["bias_tables"],
            "buckets": gate_res["buckets"]}
        out_stacks[name] = row
        out_gate[name] = {
            "verdict": gate_res["verdict"],
            "candidate": gate_res["candidate"], "baseline": "v6",
            "n_scored": gate_res["n_scored"],
            "clauses": gate_res["clauses"],
            "expected_roi": gate_res["expected_roi"],
            "bias_summary": {
                "candidate_max_abs": gate_res["bias_tables"]["candidate"]
                ["max_abs_bias"],
                "v6_max_abs": gate_res["bias_tables"]["v6"]["max_abs_bias"]},
            "full_tables_in": "stats/compose_stacks.json"}
        jlog(f"SCORED {name}: dLL={jr['delta_milli']:+.3f}m "
             f"(fam MDE {fam}m, pair raw {jr['pair_mde_raw_milli']}m / cv "
             f"{jr['pair_mde_cv_milli']}m) p_iid={jr['boot_iid']['p_better']:.3f} "
             f"p_blk={jr['boot_block']['p_better']:.3f} -> {verdict}; "
             f"gate={gate_res['verdict']}; gated split {split}; "
             f"ewc_full {buck_extra['ewc_fullclass']['delta_milli']:+.2f}m; "
             f"PRX/NRG {resid}")

    hdr = {
        "written_by": "agent:compose", "written": datetime.now().strftime("%F %T"),
        "preregistered": "testing_lab/v8/preregister.compose.md (BEFORE fits "
                         "and scoring)",
        "frame": "v8/data/frame_expanded/series.csv sha256 verified vs crn.json",
        "one_scoring_rule": "each stack scored on holdout exactly once "
                            "(sentinel scratch/compose/SCORED)",
        "baseline_v6": {"ll_holdout": 0.64216, "beta": 0.1152,
                        "max_abs_bias": None},
        "mde_source": "stats/power_mde_expanded.json (within 1.773m / cross "
                      "5.889m); empirical pair-MDE = 2.8016*SD(d)/sqrt(n), "
                      "raw + CUPED(l_v6) CV per stats/variance_reduction.json",
        "units": "milli-LL + expected ROI via referee.expected_roi_of_dll"}
    hdr["baseline_v6"]["max_abs_bias"] = out_gate["S1_gate5d"][
        "bias_summary"]["v6_max_abs"]
    with open(os.path.join(STATS, "compose_stacks.json"), "w") as f:
        json.dump({"provenance": hdr, "stacks": out_stacks}, f, indent=1,
                  default=float)
    with open(os.path.join(STATS, "compose_gate.json"), "w") as f:
        json.dump({"provenance": hdr, "gates": out_gate}, f, indent=1,
                  default=float)
    jlog("wrote stats/compose_stacks.json + stats/compose_gate.json")


if __name__ == "__main__":
    {"train": phase_train, "score": phase_score}[sys.argv[1]]()
