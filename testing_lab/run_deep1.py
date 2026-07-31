"""Deep pass 1: artifact-free calibration, cold-start segmentation, rolling
walk-forward beta recalibration, per-format/playoff recalibration tests.
Writes out/deep1.json."""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import (BETA_LIVE, OUT, brier, calib_slope, intl_attendance_asof,
                     load_series, logloss, paired_bootstrap, predict,
                     reliability, summarize)

df = load_series()
att = intl_attendance_asof(df)
p_prod = predict(df, beta=BETA_LIVE, gating="backend", attendance=att)
df["p_prod"] = p_prod
res = {}

# ── cold-start segmentation ──────────────────────────────────────────────────
# A rating of exactly 0.0 (to 4dp) means no rated games yet (mean-zero anchor
# makes true 0.0 vanishingly unlikely otherwise).
cold_w = df.r_w.abs() < 5e-4
cold_l = df.r_l.abs() < 5e-4
df["cold"] = (cold_w | cold_l)
res["cold_n"] = int(df.cold.sum())
res["cold_both_n"] = int((cold_w & cold_l).sum())
res["cold_metrics"] = summarize(p_prod[df.cold.values], "cold-start matches")
res["warm_metrics"] = summarize(p_prod[~df.cold.values], "warm matches")
print("cold:", res["cold_n"], "both-cold:", res["cold_both_n"])
print("cold:", res["cold_metrics"])
print("warm:", res["warm_metrics"])

warm = df[~df.cold].copy()
p_w = warm.p_prod.values

# tie-free favorite metrics (warm only)
is_tie = np.abs(p_w - 0.5) < 1e-9
p_fav = np.maximum(p_w, 1 - p_w)[~is_tie]
fav_won = (p_w >= 0.5).astype(float)[~is_tie]
res["reliability_fav_warm"] = reliability(p_fav, fav_won, n_bins=10)
print("\nRELIABILITY warm, tie-free:")
for r in res["reliability_fav_warm"]:
    print(f"  [{r['bin_lo']:.2f},{r['bin_hi']:.2f}) pred {r['pred_mean']:.3f} "
          f"emp {r['emp']:.3f} ({r['ci_lo']:.3f},{r['ci_hi']:.3f}) n={r['n']}")

res["calib_slope_warm"] = calib_slope(p_w)
for yr in sorted(warm.year.unique()):
    res[f"calib_slope_warm_{yr}"] = calib_slope(warm[warm.year == yr].p_prod.values)
print("\nslopes warm:", res["calib_slope_warm"],
      {yr: res[f"calib_slope_warm_{yr}"] for yr in sorted(warm.year.unique())})

# link deciles warm-only
gap = np.abs(warm.r_w.values - warm.r_l.values)
qs = np.quantile(gap, np.linspace(0, 1, 11))
link = []
for i in range(10):
    m = (gap >= qs[i]) & (gap < qs[i + 1]) if i < 9 else (gap >= qs[i])
    mm = m & ~is_tie
    if mm.sum():
        link.append({"gap_mid": round(float(gap[mm].mean()), 3), "n": int(mm.sum()),
                     "pred_fav": round(float(np.maximum(p_w[mm], 1 - p_w[mm]).mean()), 4),
                     "emp_fav": round(float((p_w[mm] >= 0.5).mean()), 4)})
res["link_deciles_warm"] = link
print("\nLINK warm:")
for r in link:
    print(f"  gap~{r['gap_mid']:.2f} n={r['n']} pred {r['pred_fav']:.3f} emp {r['emp_fav']:.3f}")

# ── rolling walk-forward beta ────────────────────────────────────────────────
# At each match, beta = argmin logloss over all *prior* series (windowed
# variants). Pure recalibration: deployable as-is, no leakage.
from scipy.optimize import minimize_scalar

d_all = df.sort_values(["date", "match_id"]).reset_index(drop=True)
logit_in = (d_all.r_w - d_all.r_l).values  # rating diff, winner-referenced
fmts = d_all.fmt.values
dates = d_all.date.values


def series_p_vec(beta, rd, fmt_arr):
    p_map = 1 / (1 + np.exp(-beta * rd))
    p = np.where(np.isin(fmt_arr, ("bo5", "bo5_gf")),
                 p_map ** 3 * (1 + 3 * (1 - p_map) + 6 * (1 - p_map) ** 2),
                 np.where(fmt_arr == "bo1", p_map, p_map ** 2 * (3 - 2 * p_map)))
    return p


def fit_beta_on(idx_mask):
    rd, fm = logit_in[idx_mask], fmts[idx_mask]
    if rd.size < 60:
        return BETA_LIVE

    def nll(b):
        return -np.mean(np.log(np.clip(series_p_vec(b, rd, fm), 1e-9, 1)))
    return float(minimize_scalar(nll, bounds=(0.05, 0.35), method="bounded").x)


for window_days, key in [(None, "rolling_all"), (365, "rolling_365"),
                         (240, "rolling_240"), (150, "rolling_150")]:
    p_roll = np.empty(len(d_all))
    beta_path = []
    # refit weekly for speed
    uniq_dates = sorted(d_all.date.unique())
    cur_beta = BETA_LIVE
    last_fit = None
    date_beta = {}
    for ud in uniq_dates:
        if last_fit is None or (pd.Timestamp(ud) - pd.Timestamp(last_fit)).days >= 7:
            prior = dates < ud
            if window_days is not None:
                lo = (pd.Timestamp(ud) - pd.Timedelta(days=window_days)).strftime("%Y-%m-%d")
                prior &= dates >= lo
            cur_beta = fit_beta_on(prior)
            last_fit = ud
        date_beta[ud] = cur_beta
    for i in range(len(d_all)):
        b = date_beta[dates[i]]
        p_roll[i] = series_p_vec(b, logit_in[i:i+1], fmts[i:i+1])[0]
        beta_path.append(b)
    # score only matches after enough history (2024+)
    m24 = (d_all.year >= 2024).values
    res[key] = summarize(p_roll[m24], f"{key} (2024+)")
    res[key + "_boot"] = paired_bootstrap(p_roll[m24], p_prod_sorted[m24]) \
        if (p_prod_sorted := d_all.p_prod.values) is not None else None
    res[key + "_beta_now"] = round(beta_path[-1], 4)
    print(f"\n{key}: {res[key]}  beta_now={res[key+'_beta_now']}")
    print("  boot vs prod:", res[key + "_boot"])

res["prod_2024plus"] = summarize(d_all.p_prod.values[(d_all.year >= 2024).values],
                                 "production (2024+)")
print("\nprod 2024+:", res["prod_2024plus"])

# ── per-format & playoff recalibration (walk-forward yearly) ────────────────
# Test: playoffs get their own beta, fit on all prior playoff series.
stage_arr = d_all.stage.values
p_stage = d_all.p_prod.values.copy()
po_mask = np.isin(stage_arr, ("playoffs", "grand_final"))
uniq_dates = sorted(d_all.date.unique())
cur_bpo, cur_bgr = BETA_LIVE, BETA_LIVE
last_fit = None
for ud in uniq_dates:
    if last_fit is None or (pd.Timestamp(ud) - pd.Timestamp(last_fit)).days >= 14:
        prior = dates < ud
        cur_bpo = fit_beta_on(prior & po_mask)
        cur_bgr = fit_beta_on(prior & ~po_mask)
        last_fit = ud
    today = dates == ud
    for i in np.where(today)[0]:
        b = cur_bpo if po_mask[i] else cur_bgr
        p_stage[i] = series_p_vec(b, logit_in[i:i+1], fmts[i:i+1])[0]
m24 = (d_all.year >= 2024).values
res["stage_beta"] = summarize(p_stage[m24], "per-stage rolling beta (2024+)")
res["stage_beta_boot"] = paired_bootstrap(p_stage[m24], d_all.p_prod.values[m24])
res["stage_beta_po_only"] = summarize(p_stage[m24 & po_mask], "playoffs only")
res["prod_po_only"] = summarize(d_all.p_prod.values[m24 & po_mask], "prod playoffs only")
print("\nper-stage beta:", res["stage_beta"], "\n boot:", res["stage_beta_boot"])
print(" playoffs-only:", res["stage_beta_po_only"], "vs prod", res["prod_po_only"])

with open(os.path.join(OUT, "deep1.json"), "w") as f:
    json.dump(res, f, indent=1, default=str)
print("\nsaved out/deep1.json")
