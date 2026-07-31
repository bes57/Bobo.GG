"""Phase 1 diagnosis: score the production model expansively, test the beta
question, measure calibration. Writes out/diagnosis.json."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import (BETA_LIVE, OUT, brier, calib_slope, intl_attendance_asof,
                     load_series, logloss, paired_bootstrap, predict,
                     reliability, summarize)

df = load_series()
att = intl_attendance_asof(df)
print(f"series: {len(df)}  years: {df.year.min()}-{df.year.max()}")
print(df.groupby("year").size().to_dict())
print("formats:", df.groupby("fmt").size().to_dict())
print("stages:", df.groupby("stage").size().to_dict())
print("intl:", int(df.intl.sum()))

res = {"n": len(df), "by_year_n": df.groupby("year").size().to_dict(),
       "by_fmt_n": df.groupby("fmt").size().to_dict(),
       "by_stage_n": df.groupby("stage").size().to_dict()}

# ── baseline: production backend surface ────────────────────────────────────
p_prod = predict(df, beta=BETA_LIVE, gating="backend", attendance=att)
df["p_prod"] = p_prod
res["baseline"] = summarize(p_prod, "production (beta=0.170, backend gating)")
print("\nBASELINE:", res["baseline"])

# ── beta grid (walk-forward-safe: pure function of frozen ratings) ──────────
grid = []
for b in np.arange(0.06, 0.30, 0.01):
    pb = predict(df, beta=round(float(b), 3), gating="backend", attendance=att)
    grid.append({"beta": round(float(b), 3), "logloss": round(logloss(pb), 5),
                 "brier": round(brier(pb), 5)})
res["beta_grid"] = grid
best_b = min(grid, key=lambda g: g["logloss"])
print("\nbeta grid best:", best_b)

# per-year best beta
res["beta_by_year"] = {}
for yr in sorted(df.year.unique()):
    sub = df[df.year == yr]
    gy = []
    for b in np.arange(0.06, 0.30, 0.005):
        pb = predict(sub, beta=round(float(b), 3), gating="backend", attendance=att)
        gy.append((round(float(b), 3), logloss(pb)))
    bb = min(gy, key=lambda t: t[1])
    res["beta_by_year"][int(yr)] = {"beta": bb[0], "logloss": round(bb[1], 5)}
    print(f"  {yr}: best beta {bb[0]} (ll {bb[1]:.5f})")

# beta=0.136 (the sweep-note value) vs 0.170
p_136 = predict(df, beta=0.136, gating="backend", attendance=att)
res["beta_136"] = summarize(p_136, "beta=0.136")
res["boot_136_vs_prod"] = paired_bootstrap(p_136, p_prod)
print("\nbeta 0.136:", res["beta_136"], "\n boot vs prod:", res["boot_136_vs_prod"])

pb_best = predict(df, beta=best_b["beta"], gating="backend", attendance=att)
res["boot_best_vs_prod"] = paired_bootstrap(pb_best, p_prod)

# ── offset gating variants (F1) ─────────────────────────────────────────────
for g in ("none", "backend", "frontend9"):
    pg = predict(df, beta=BETA_LIVE, gating=g, attendance=att)
    res[f"gating_{g}"] = summarize(pg, f"gating={g}")
    intl9 = df.event_id.isin({"2024_masters_madrid", "2024_masters_shanghai",
                              "2024_champions", "2025_masters_bangkok",
                              "2025_masters_toronto", "2025_champions",
                              "2026_masters_santiago", "2026_masters_london",
                              "2026_champions"}).values
    res[f"gating_{g}_intl9only"] = summarize(pg[intl9], f"gating={g} (9 intl events only)")
print("\nGATING:", {k: v for k, v in res.items() if k.startswith("gating") and "only" in k})

# ── calibration: the 50-50 question ─────────────────────────────────────────
res["calib_slope_all"] = calib_slope(p_prod)
for yr in sorted(df.year.unique()):
    res[f"calib_slope_{yr}"] = calib_slope(df[df.year == yr].p_prod.values)
print("\nCALIB SLOPE (b>1 = too close to 50-50):", res["calib_slope_all"],
      {yr: res[f"calib_slope_{yr}"] for yr in sorted(df.year.unique())})

# favorite-side reliability
p_fav = np.maximum(p_prod, 1 - p_prod)
fav_won = (p_prod >= 0.5).astype(float)  # winner prob >= .5 => favorite won
res["reliability_fav"] = reliability(p_fav, fav_won, n_bins=10)
print("\nRELIABILITY (fav side):")
for r in res["reliability_fav"]:
    print(f"  [{r['bin_lo']:.2f},{r['bin_hi']:.2f}) pred {r['pred_mean']:.3f} "
          f"emp {r['emp']:.3f} ±({r['ci_lo']:.3f},{r['ci_hi']:.3f}) n={r['n']}")

# heavy favorites
res["heavy_fav"] = []
for lo, hi in [(0.5, 0.6), (0.6, 0.7), (0.7, 0.75), (0.75, 0.8), (0.8, 0.85),
               (0.85, 0.9), (0.9, 1.0001)]:
    m = (p_fav >= lo) & (p_fav < hi)
    if m.sum():
        res["heavy_fav"].append({"lo": lo, "hi": hi, "n": int(m.sum()),
                                 "pred": round(float(p_fav[m].mean()), 4),
                                 "emp": round(float(fav_won[m].mean()), 4)})
print("\nHEAVY FAV:", res["heavy_fav"])

# ── buckets on production surface ───────────────────────────────────────────
def bucket(mask, name):
    if mask.sum() == 0:
        return None
    p = p_prod[mask]
    return {"name": name, "n": int(mask.sum()), "logloss": round(logloss(p), 5),
            "brier": round(brier(p), 5),
            "fav_acc": round(float((p >= 0.5).mean()), 4)}

buckets = []
for yr in sorted(df.year.unique()):
    buckets.append(bucket((df.year == yr).values, f"year {yr}"))
for f in ("bo1", "bo3", "bo5", "bo5_gf"):
    buckets.append(bucket((df.fmt == f).values, f"format {f}"))
for s in ("groups", "playoffs", "grand_final", "other"):
    buckets.append(bucket((df.stage == s).values, f"stage {s}"))
buckets.append(bucket(df.intl.values, "international events"))
buckets.append(bucket(~df.intl.values, "domestic events"))
cn_any = ((df.reg_w == "CN") | (df.reg_l == "CN")).values
buckets.append(bucket(cn_any, "CN team involved"))
buckets.append(bucket((df.reg_w != df.reg_l).values, "cross-region"))
same = (df.reg_w == df.reg_l)
for reg in ("Americas", "EMEA", "Pacific", "CN"):
    buckets.append(bucket((same & (df.reg_w == reg)).values, f"domestic {reg}"))
# rating-gap bands
gap = np.abs(df.r_w.values - df.r_l.values)
for lo, hi in [(0, 1), (1, 2), (2, 4), (4, 6), (6, 9), (9, 99)]:
    buckets.append(bucket((gap >= lo) & (gap < hi), f"rating gap [{lo},{hi})"))
res["buckets"] = [b for b in buckets if b]
print("\nBUCKETS:")
for b in res["buckets"]:
    print(f"  {b['name']:<28} n={b['n']:<5} ll={b['logloss']:.4f} "
          f"brier={b['brier']:.4f} fav_acc={b['fav_acc']:.3f}")

# ── link shape: rating-gap deciles, predicted vs empirical ──────────────────
qs = np.quantile(gap, np.linspace(0, 1, 11))
link = []
for i in range(10):
    m = (gap >= qs[i]) & (gap <= qs[i + 1] if i == 9 else gap < qs[i + 1])
    if m.sum():
        link.append({"gap_mid": round(float(gap[m].mean()), 3), "n": int(m.sum()),
                     "pred_fav": round(float(np.maximum(p_prod[m], 1 - p_prod[m]).mean()), 4),
                     "emp_fav": round(float((p_prod[m] >= 0.5).mean()), 4)})
res["link_deciles"] = link
print("\nLINK (gap deciles):")
for r in link:
    print(f"  gap~{r['gap_mid']:.2f} n={r['n']} pred {r['pred_fav']:.3f} emp {r['emp_fav']:.3f}")

df.to_csv(os.path.join(OUT, "series_dataset.csv"), index=False)
with open(os.path.join(OUT, "diagnosis.json"), "w") as f:
    json.dump(res, f, indent=1, default=str)
print(f"\nsaved out/diagnosis.json + series_dataset.csv ({len(df)} rows)")
