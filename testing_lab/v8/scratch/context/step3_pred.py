"""agent:context step 3 — prediction-layer experiments on B0's rdiff:
3b-a exposure term, 3b-b form-vs-exposure decomposition, 3c-B stakes
variance, 3e mechanism shrink (+ class-dummy falsifier), 3b-adjacency.
All coefficient fits on train rows only; holdout scored once per
preregistered candidate; CRN bootstrap + ROI translation via referee."""
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar

V8 = "/Users/benny_es1/PythonTest/testing_lab/v8"
TL = os.path.dirname(V8)
SC = os.path.join(V8, "scratch", "context")
sys.path.insert(0, TL)
sys.path.insert(0, V8)
import referee  # noqa: E402
from engine import Engine  # noqa: E402

frame = pd.read_csv(os.path.join(SC, "frame_features.csv"))
b0 = np.load(os.path.join(SC, "b0.npz"), allow_pickle=True)
rd = b0["rdiff"]
BETA0 = float(b0["beta"])
test = b0["test_mask"]
loss_b0 = b0["loss_b0"]
p_b0_test = b0["p_test"]
ev_test = b0["event_ids"]
fmts = frame.fmt.values
train_m = (frame.date <= "2024-12-31").values
v = ~np.isnan(rd)
m_tr = v & train_m
assert int(test.sum()) == 1217

MDE_WITHIN = 1.773


def sp(pm, fm):
    return np.where(np.isin(fm, ("bo5", "bo5_gf")),
                    pm ** 3 * (1 + 3 * (1 - pm) + 6 * (1 - pm) ** 2),
                    np.where(fm == "bo1", pm, pm ** 2 * (3 - 2 * pm)))


ECL = frame.eclass.values
legacy_t = frame.event_id.astype(str).str.startswith(
    ("2026_ewc", "2026_china_evo")).values[test]
ewcfull_t = (ECL == "ewc_offseason")[test]


def judge(name, p_full, extra=None):
    """holdout delta vs B0 + CRN boots (iid+block) + ROI + buckets."""
    p_t = p_full[test]
    loss = -np.log(np.clip(p_t, 1e-9, 1))
    d = loss_b0 - loss
    dll = float(d.mean())
    bi = referee.paired_bootstrap_crn(d, mode="iid")
    bb = referee.paired_bootstrap_crn(d, mode="block_event", event_ids=ev_test)
    roi = referee.expected_roi_of_dll(dll, p_b0_test)
    verdict = "INSIDE NOISE FLOOR"
    if abs(dll) * 1000 >= MDE_WITHIN:
        verdict = ("WIN" if dll > 0 and bi["ci_lo"] > 0 else
                   "DEAD" if dll < 0 and bi["ci_hi"] < 0 else
                   "INSIDE NOISE FLOOR (CI spans 0)")
    out = {
        "name": name, "ll_test": round(float(loss.mean()), 5),
        "ll_test_B0": round(float(loss_b0.mean()), 5),
        "dll_milli": round(dll * 1000, 3), "pair_mde_milli": MDE_WITHIN,
        "mde_family": "within (v6-family variant, power_mde_expanded)",
        "boot_iid": {k: bi[k] for k in ("mean_delta", "ci_lo", "ci_hi", "p_better")},
        "boot_block": {k: bb[k] for k in ("mean_delta", "ci_lo", "ci_hi", "p_better")},
        "expected_roi_delta": roi["expected_roi_delta"],
        "roi_at_op": roi["roi_at_op"],
        "delta_logit_equiv": roi["delta_logit_equiv"],
        "bucket_ewc_legacy2026": {
            "n": int(legacy_t.sum()),
            "ll": round(float(loss[legacy_t].mean()), 5),
            "ll_B0": round(float(loss_b0[legacy_t].mean()), 5),
            "dll_milli": round(float((loss_b0 - loss)[legacy_t].mean()) * 1000, 2)},
        "bucket_ewc_fullclass": {
            "n": int(ewcfull_t.sum()),
            "ll": round(float(loss[ewcfull_t].mean()), 5),
            "ll_B0": round(float(loss_b0[ewcfull_t].mean()), 5),
            "dll_milli": round(float((loss_b0 - loss)[ewcfull_t].mean()) * 1000, 2)},
        "verdict": verdict}
    if extra:
        out.update(extra)
    print(f"{name}: dll={out['dll_milli']:+.3f}m iidCI[{bi['ci_lo']*1000:+.2f},"
          f"{bi['ci_hi']*1000:+.2f}]m roi_d={roi['expected_roi_delta']:+.4f} "
          f"ewc_full={out['bucket_ewc_fullclass']['dll_milli']:+.2f}m -> {verdict}",
          flush=True)
    return out


def fit_terms(feats, mults=None, x0_extra=None):
    """z = beta*rd*(prod of (1+a_j*mult_j)) + sum c_i*feat_i, train ML fit."""
    feats = [np.asarray(f, dtype=float) for f in (feats or [])]
    mults = [np.asarray(g, dtype=float) for g in (mults or [])]
    k_f, k_m = len(feats), len(mults)
    x0 = [0.13] + [0.0] * (k_f + k_m) if x0_extra is None else x0_extra

    def z_of(params, mask):
        b = params[0]
        z = b * rd[mask]
        for j, g in enumerate(mults):
            z = z * (1 + params[1 + k_f + j] * g[mask])
        for i, x in enumerate(feats):
            z = z + params[1 + i] * x[mask]
        return z

    def nll(params, mask):
        pm = 1 / (1 + np.exp(-z_of(params, mask)))
        return -np.mean(np.log(np.clip(sp(pm, fmts[mask]), 1e-9, 1)))

    fit = minimize(nll, x0, args=(m_tr,), method="Nelder-Mead",
                   options={"maxiter": 4000, "xatol": 1e-5, "fatol": 1e-9})
    p_full = np.full(len(frame), np.nan)
    mall = v.copy()
    pm = 1 / (1 + np.exp(-z_of(fit.x, mall)))
    p_full[mall] = sp(pm, fmts[mall])
    return fit.x, float(nll(fit.x, m_tr)), p_full


R = {}

# ── exposure features ───────────────────────────────────────────────────────
dm30 = (frame.m30_w.values - frame.m30_l.values) / 10.0
ddso = np.log1p(frame.dso_w.values) - np.log1p(frame.dso_l.values)
ddsi = np.log1p(frame.dsi_w.values) - np.log1p(frame.dsi_l.values)
ddsi2 = np.log1p(frame.dsi2_w.values) - np.log1p(frame.dsi2_l.values)

# 3b-a prediction term
x, lltr, p = fit_terms([dm30, ddso, ddsi])
R["3ba_exposure_term"] = judge("3b-a exposure term", p, {
    "coef": {"beta": round(x[0], 4), "c_dmaps30/10": round(x[1], 4),
             "c_dlog_dso": round(x[2], 4), "c_dlog_dsi": round(x[3], 4)},
    "ll_train": round(lltr, 5)})
x, lltr, p = fit_terms([dm30, ddso, ddsi2])
R["3ba_exposure_term_ewcLAN"] = judge("3b-a exposure (dsi incl EWC mains)", p, {
    "coef": {"beta": round(x[0], 4), "c_dmaps30/10": round(x[1], 4),
             "c_dlog_dso": round(x[2], 4), "c_dlog_dsi2": round(x[3], 4)},
    "ll_train": round(lltr, 5)})

# ── 3b-b decomposition: form term with/without exposure controls ────────────
eng = Engine()  # data only — no solves


def wr_series(hl):
    lam = math.log(2) / hl
    state = defaultdict(lambda: [0.0, 0.0])
    at = {}
    sdates = sorted(set(frame.date))
    si = 0
    for g in sorted(eng.games, key=lambda g: (g["date_s"], g["match_id"])):
        while si < len(sdates) and sdates[si] <= g["date_s"]:
            for t_, (n_, d_) in state.items():
                if d_ > 3:
                    at[(t_, sdates[si])] = n_ / d_
            si += 1
        for team, won in ((g["winner"], 1.0), (g["loser"], 0.0)):
            st = state[team]
            st[0] = st[0] * math.exp(-lam) + won
            st[1] = st[1] * math.exp(-lam) + 1.0
    while si < len(sdates):
        for t_, (n_, d_) in state.items():
            if d_ > 3:
                at[(t_, sdates[si])] = n_ / d_
        si += 1
    return at


def arr(at):
    w = np.array([at.get((r.winner, r.date), 0.5) for r in frame.itertuples(index=False)])
    l = np.array([at.get((r.loser, r.date), 0.5) for r in frame.itertuples(index=False)])
    return w, l


w16w, w16l = arr(wr_series(16.0))
forms = {}
for hl in (3.0, 5.0):
    wsw, wsl = arr(wr_series(hl))
    forms[int(hl)] = (wsw - w16w) - (wsl - w16l)

decomp = {}
for hl, dform in forms.items():
    xa, lla, pa = fit_terms([dform])
    xb, llb, pb = fit_terms([dform, dm30, ddso, ddsi])
    ja = judge(f"3b-b form{hl} alone", pa, {"coef": {"beta": round(xa[0], 4),
               "b_form": round(xa[1], 4)}, "ll_train": round(lla, 5)})
    jb = judge(f"3b-b form{hl}+exposure", pb, {"coef": {
        "beta": round(xb[0], 4), "b_form": round(xb[1], 4),
        "c_dmaps30/10": round(xb[2], 4), "c_dlog_dso": round(xb[3], 4),
        "c_dlog_dsi": round(xb[4], 4)}, "ll_train": round(llb, 5)})
    shrink = (1 - xb[1] / xa[1]) * 100 if abs(xa[1]) > 1e-9 else np.nan
    decomp[f"form{hl}"] = {
        "b_form_alone": round(float(xa[1]), 4),
        "b_form_with_exposure": round(float(xb[1]), 4),
        "shrink_pct": round(float(shrink), 1),
        "published_v7_oldframe": {"form3": -0.0872, "form5": -0.0242}[f"form{hl}"],
        "judge_alone": ja, "judge_with": jb}
    print(f"  DECOMP form{hl}: b_form {xa[1]:+.4f} -> {xb[1]:+.4f} "
          f"(shrink {shrink:+.1f}%)", flush=True)
tr = m_tr
decomp["correlations_train"] = {
    "corr(dform5,dmaps30)": round(float(np.corrcoef(forms[5][tr], dm30[tr])[0, 1]), 3),
    "corr(dform5,dlog_dso)": round(float(np.corrcoef(forms[5][tr], ddso[tr])[0, 1]), 3),
    "corr(dform5,dlog_dsi)": round(float(np.corrcoef(forms[5][tr], ddsi[tr])[0, 1]), 3)}
R["3bb_decomposition"] = decomp

# ── 3c-B stakes as variance term ────────────────────────────────────────────
elim = frame.elim.values.astype(float)
x, lltr, p = fit_terms(None, mults=[elim])
R["3cB_elim_variance"] = judge("3c-B elim variance", p, {
    "coef": {"beta": round(x[0], 4), "a_elim": round(x[1], 4)},
    "ll_train": round(lltr, 5)})

# ── 3e mechanism shrink (multiplier exp(-k*X)) ──────────────────────────────
integ_w = frame.integ_w.values
integ_l = frame.integ_l.values
X1 = (1 - integ_w) + (1 - integ_l)          # stand-in load
X2 = np.abs(ddso)                            # prep asymmetry magnitude
is_ewc = (ECL == "ewc_offseason").astype(float)


def fit_shrink(Xs, name):
    Xs = [np.asarray(x_, dtype=float) for x_ in Xs]

    def z_of(params, mask):
        b = params[0]
        e = np.zeros(mask.sum())
        for j, x_ in enumerate(Xs):
            e = e + params[1 + j] * x_[mask]
        return b * rd[mask] * np.exp(-e)

    def nll(params, mask):
        pm = 1 / (1 + np.exp(-z_of(params, mask)))
        return -np.mean(np.log(np.clip(sp(pm, fmts[mask]), 1e-9, 1)))

    fit = minimize(nll, [0.13] + [0.0] * len(Xs), args=(m_tr,),
                   method="Nelder-Mead",
                   options={"maxiter": 4000, "xatol": 1e-5, "fatol": 1e-9})
    p_full = np.full(len(frame), np.nan)
    pm = 1 / (1 + np.exp(-z_of(fit.x, v)))
    p_full[v] = sp(pm, fmts[v])
    return fit.x, float(nll(fit.x, m_tr)), p_full


x, lltr, p = fit_shrink([X1], "k1")
R["3e_shrink_standin"] = judge("3e shrink X1=stand-in load", p, {
    "coef": {"beta": round(x[0], 4), "k_standin": round(x[1], 4)},
    "ll_train": round(lltr, 5)})
x, lltr, p = fit_shrink([X2], "k2")
R["3e_shrink_prepasym"] = judge("3e shrink X2=|dlog dso|", p, {
    "coef": {"beta": round(x[0], 4), "k_prep": round(x[1], 4)},
    "ll_train": round(lltr, 5)})
x, lltr, p = fit_shrink([X1, X2], "k1k2")
R["3e_shrink_joint"] = judge("3e shrink X1+X2", p, {
    "coef": {"beta": round(x[0], 4), "k_standin": round(x[1], 4),
             "k_prep": round(x[2], 4)}, "ll_train": round(lltr, 5)})
x, lltr, p = fit_shrink([is_ewc], "kdummy")
R["3e_classdummy_falsifier"] = judge("3e FALSIFIER class dummy", p, {
    "coef": {"beta": round(x[0], 4), "k_ewc_dummy": round(x[1], 4)},
    "ll_train": round(lltr, 5),
    "role": "falsifier control — never a candidate"})

# ── 3b-adjacency ────────────────────────────────────────────────────────────
ADJ = {"2023_champions": "2023_masters_tokyo",
       "2024_masters_shanghai": "2024_masters_madrid",
       "2024_champions": "2024_masters_shanghai",
       "2025_masters_toronto": "2025_masters_bangkok",
       "2025_ewc": "2025_masters_toronto",
       "2026_masters_london": "2026_masters_santiago",
       "2026_ewc": "2026_masters_london"}
n_series = defaultdict(int)
for r in frame.itertuples(index=False):
    n_series[(r.event_id, r.winner)] += 1
    n_series[(r.event_id, r.loser)] += 1
drc = {}
for prev in set(ADJ.values()):
    orgs = [o for (e, o) in n_series if e == prev]
    mu = np.mean([n_series[(prev, o)] for o in orgs])
    for o in orgs:
        drc[(prev, o)] = n_series[(prev, o)] - mu
rows = []
for i, r in enumerate(frame.itertuples(index=False)):
    prev = ADJ.get(r.event_id)
    if prev is None:
        continue
    a, b_ = drc.get((prev, r.winner)), drc.get((prev, r.loser))
    if a is None or b_ is None or np.isnan(rd[i]):
        continue
    rows.append((i, a - b_))
idx_sub = np.array([t[0] for t in rows])
ddr = np.array([t[1] for t in rows])
rd_sub, fm_sub = rd[idx_sub], fmts[idx_sub]
n_sub = len(rows)


def nll_c(c):
    pm = 1 / (1 + np.exp(-(BETA0 * rd_sub + c * ddr)))
    return -np.mean(np.log(np.clip(sp(pm, fm_sub), 1e-9, 1)))


c_hat = float(minimize_scalar(nll_c, bounds=(-1, 1), method="bounded").x)
h = 1e-4
d2 = (nll_c(c_hat + h) - 2 * nll_c(c_hat) + nll_c(c_hat - h)) / h / h
se = float(1 / np.sqrt(max(n_sub * d2, 1e-12)))
crn = json.load(open(os.path.join(V8, "crn.json")))
rng = np.random.default_rng(crn["bootstrap"]["seed"])
bidx = rng.integers(0, n_sub, size=(crn["bootstrap"]["n_boot"], n_sub))
boots = []
for r_ in range(bidx.shape[0]):
    ii = bidx[r_]

    def nb(c):
        pm = 1 / (1 + np.exp(-(BETA0 * rd_sub[ii] + c * ddr[ii])))
        return -np.mean(np.log(np.clip(sp(pm, fm_sub[ii]), 1e-9, 1)))
    boots.append(minimize_scalar(nb, bounds=(-1, 1), method="bounded").x)
boots = np.array(boots)
per_ev = frame.iloc[idx_sub].event_id.value_counts().to_dict()
R["3b_adjacency"] = {
    "pairs": ADJ, "n_series_bothattended": n_sub, "per_next_event_n": per_ev,
    "deep_run_measure": "n series at prev Masters, centered within event",
    "beta0_frozen": BETA0, "c_hat": round(c_hat, 4),
    "wald_se": round(se, 4),
    "wald_ci95": [round(c_hat - 1.96 * se, 4), round(c_hat + 1.96 * se, 4)],
    "crn_boot_ci95": [round(float(np.percentile(boots, 2.5)), 4),
                      round(float(np.percentile(boots, 97.5)), 4)],
    "crn_seed": crn["bootstrap"]["seed"],
    "note": "inference only; nothing promoted from this",
    "dll_equiv_if_used_milli": None}
print(f"adjacency: n={n_sub} c_hat={c_hat:+.4f} wald_ci=[{c_hat-1.96*se:+.4f},"
      f"{c_hat+1.96*se:+.4f}] boot_ci=[{np.percentile(boots,2.5):+.4f},"
      f"{np.percentile(boots,97.5):+.4f}]", flush=True)

with open(os.path.join(SC, "step3_results.json"), "w") as f:
    json.dump(R, f, indent=1, default=float)
print("step3 DONE", flush=True)
