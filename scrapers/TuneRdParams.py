"""
TuneRdParams.py
===============
Grid-search the round-diff variance-stabilization parameters
(RD_POWER, RD_SCALE) in BuildMapRatings.py.

For each cell:
  1. Edit BuildMapRatings.py to set RD_POWER, RD_SCALE.
  2. Run BuildMapRatings.py (regenerates data/map_ratings.json).
  3. Delete data/rating_timeline.json so the 2026 file gets a fresh build.
  4. Run BuildRatingTimeline.py (regenerates the 4 yearly timeline files).
  5. Load all match_events; fit BETA via MLE on series-level outcomes
     using closed-form p_series = f(sigmoid(BETA*|delta|), bo3/bo5).
  6. Run AnalyzeProjectionCalibration.py with the MLE beta + intl_bonus 0.22
     + cn_dog_offset 0.47 to score the cell.
  7. Parse the calibration JSON for pool Brier, Platt b, intl/dom Brier,
     and weighted CN-dog Brier from q5_region_buckets ("X vs CN" buckets).
  8. Pull the rating spread (max - min) and top/bottom 5 from
     map_ratings.json at 2025_after_champions.

Writes data/rd_param_tune_grid.json with all cells + best cell + recommendation.

Usage:
    .venv/bin/python scrapers/TuneRdParams.py
    .venv/bin/python scrapers/TuneRdParams.py --quick    # 4 cells smoke test
"""

import json
import math
import os
import re
import subprocess
import sys
import time
from datetime import datetime

import numpy as np
from scipy.optimize import minimize_scalar

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
BMR_PATH = os.path.join(ROOT, "scrapers", "BuildMapRatings.py")
BRT_PATH = os.path.join(ROOT, "scrapers", "BuildRatingTimeline.py")
APC_PATH = os.path.join(ROOT, "scrapers", "AnalyzeProjectionCalibration.py")
PYTHON   = os.path.join(ROOT, ".venv", "bin", "python")
RT_2026  = os.path.join(DATA_DIR, "rating_timeline.json")

GRID_OUT = os.path.join(DATA_DIR, "rd_param_tune_grid.json")

# Canonical values to restore at the end.
CANONICAL_POWER = 0.5
CANONICAL_SCALE = 2.5

# Sweep grid (reduced 12 cells; full 5x4 ≈ 20).
GRID_POWERS = [0.5, 0.6, 0.7, 0.8]
GRID_SCALES = [2.5, 3.0, 3.5]

# Fixed addons that are NOT part of the sweep (we tune RD only).
INTL_BONUS_FIXED   = 0.22
CN_DOG_OFFSET_FIX  = 0.47

# Hold-out / training match-event window for the MLE β fit.
# We mirror what AnalyzeProjectionCalibration uses: 2024-01-01 → latest.
MLE_MIN_DATE = "2024-01-01"


# ── Helper: rewrite RD_POWER and RD_SCALE in BuildMapRatings.py ─────────────
def patch_bmr(power: float, scale: float) -> None:
    """Surgical edit: replace only the leading literal of the RD_POWER and
    RD_SCALE constants. Preserves all trailing comments and surrounding code."""
    with open(BMR_PATH) as f:
        src = f.read()
    new_src = src
    new_src = re.sub(
        r"^(RD_POWER\s*=\s*)[\d.+\-eE]+",
        rf"\g<1>{power}",
        new_src,
        count=1,
        flags=re.MULTILINE,
    )
    new_src = re.sub(
        r"^(RD_SCALE\s*=\s*)[\d.+\-eE]+",
        rf"\g<1>{scale}",
        new_src,
        count=1,
        flags=re.MULTILINE,
    )
    # Verify the regex actually matched (not a no-op due to identical values).
    # We re-extract and compare; if the targeted values don't reflect what we
    # asked for, raise.
    m_pow = re.search(r"^RD_POWER\s*=\s*([\d.+\-eE]+)", new_src, re.MULTILINE)
    m_sc  = re.search(r"^RD_SCALE\s*=\s*([\d.+\-eE]+)", new_src, re.MULTILINE)
    if not m_pow or not m_sc:
        raise RuntimeError("patch_bmr could not locate RD_POWER/RD_SCALE")
    if abs(float(m_pow.group(1)) - power) > 1e-9 or abs(float(m_sc.group(1)) - scale) > 1e-9:
        raise RuntimeError(
            f"patch_bmr post-condition failed: got RD_POWER={m_pow.group(1)} "
            f"(wanted {power}), RD_SCALE={m_sc.group(1)} (wanted {scale})")
    if new_src != src:
        with open(BMR_PATH, "w") as f:
            f.write(new_src)


# ── Subprocess wrappers ──────────────────────────────────────────────────────
def run(cmd, label):
    t0 = time.time()
    proc = subprocess.run(
        cmd, cwd=ROOT, capture_output=True, text=True,
    )
    dt = time.time() - t0
    if proc.returncode != 0:
        print(f"  [{label} FAILED in {dt:.1f}s]")
        print("--- stdout ---")
        print(proc.stdout[-2000:])
        print("--- stderr ---")
        print(proc.stderr[-2000:])
        raise RuntimeError(f"{label} failed with code {proc.returncode}")
    return dt, proc.stdout


def run_build_map_ratings():
    return run([PYTHON, BMR_PATH], "BuildMapRatings")


def run_build_rating_timeline():
    # Force 2026 rebuild by deleting the cached file
    if os.path.exists(RT_2026):
        os.remove(RT_2026)
    return run([PYTHON, BRT_PATH], "BuildRatingTimeline")


def run_analyze(beta, suffix="_rd_trial"):
    cmd = [
        PYTHON, APC_PATH,
        "--beta", f"{beta}",
        "--intl-bonus", f"{INTL_BONUS_FIXED}",
        "--cn-dog-offset", f"{CN_DOG_OFFSET_FIX}",
        "--suffix", suffix,
    ]
    return run(cmd, "AnalyzeProjectionCalibration")


# ── β MLE on series-level outcomes ──────────────────────────────────────────
def sigmoid(x):
    if x >= 0:
        z = math.exp(-x); return 1.0 / (1.0 + z)
    z = math.exp(x); return z / (1.0 + z)


def series_prob_from_map(p, bo):
    if bo == 3:
        return (p ** 2) * (3 - 2 * p)
    if bo == 5:
        return (p ** 3) * (10 - 15 * p + 6 * p * p)
    return p


def infer_bo(series_score, n_maps):
    if not series_score:
        return 3 if n_maps <= 3 else 5
    try:
        a, b = series_score.split("-")
        a, b = int(a), int(b)
        m = max(a, b)
        if m >= 3:
            return 5
        if m == 2:
            return 3
    except Exception:
        pass
    return 3 if n_maps <= 3 else 5


def load_match_events_for_mle():
    """Returns list of (abs_delta, bo, fav_won_series) for matches with model
    opinion (winner_before != loser_before) from 2024+."""
    rows = []
    for fname in ("rating_timeline_2024.json", "rating_timeline_2025.json", "rating_timeline.json"):
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            d = json.load(f)
        for m in d.get("match_events", []):
            wb = float(m.get("winner_before", 0.0))
            lb = float(m.get("loser_before",  0.0))
            if wb == lb:
                continue
            delta = wb - lb
            fav_won = 1 if wb > lb else 0
            n_maps = len(m.get("maps", []))
            bo = infer_bo(m.get("series_score", ""), n_maps)
            rows.append((abs(delta), bo, fav_won))
    return rows


def fit_beta_mle(events):
    """Maximize log-likelihood over β via scipy.minimize_scalar."""
    abs_deltas = np.array([r[0] for r in events])
    bos        = np.array([r[1] for r in events])
    ys         = np.array([r[2] for r in events], dtype=float)

    def neg_ll(beta):
        if beta <= 0:
            return 1e9
        # vectorized p_map
        z = beta * abs_deltas
        z = np.clip(z, -25, 25)
        p_map = 1.0 / (1.0 + np.exp(-z))
        # closed-form per match using bo
        p_series = np.empty_like(p_map)
        m3 = (bos == 3)
        m5 = (bos == 5)
        p_series[m3] = (p_map[m3] ** 2) * (3 - 2 * p_map[m3])
        p_series[m5] = (p_map[m5] ** 3) * (10 - 15 * p_map[m5] + 6 * p_map[m5] ** 2)
        # other bo: fallback to map p
        m_other = ~(m3 | m5)
        p_series[m_other] = p_map[m_other]
        # log loss
        eps = 1e-9
        p_series = np.clip(p_series, eps, 1 - eps)
        ll = ys * np.log(p_series) + (1 - ys) * np.log(1 - p_series)
        return -float(ll.sum())

    res = minimize_scalar(neg_ll, bounds=(0.01, 1.0), method="bounded",
                          options={"xatol": 1e-4})
    return float(res.x), float(-res.fun)


# ── Result extraction ───────────────────────────────────────────────────────
def extract_metrics(cal_json_path):
    with open(cal_json_path) as f:
        d = json.load(f)
    series_brier = d["q1_overall_calibration"]["series"]["brier"]
    map_brier    = d["q1_overall_calibration"]["map"]["brier"]
    platt_ser    = d["q2_favorite_undervaluation"]["platt_series"]
    intl_brier   = d["q3_intl_vs_domestic"]["series"]["intl"]["brier"]
    dom_brier    = d["q3_intl_vs_domestic"]["series"]["domestic"]["brier"]

    # CN-dog (X vs CN) weighted Brier from q5_region_buckets.by_region_pairing
    n_total = 0
    brier_w = 0.0
    for e in d["q5_region_buckets"]["by_region_pairing"]:
        if "vs CN" in e["bucket"] and not e["bucket"].startswith("CN "):
            # "Pacific vs CN", "EMEA vs CN", "Americas vs CN" — favorite non-CN, dog CN
            n_total += int(e["n"])
            brier_w += float(e["brier"]) * int(e["n"])
    cn_dog_brier = brier_w / n_total if n_total > 0 else None
    cn_dog_n     = n_total

    return {
        "pool_brier_series": series_brier,
        "pool_brier_map":    map_brier,
        "platt_b_series":    platt_ser.get("b"),
        "platt_a_series":    platt_ser.get("a"),
        "intl_brier_series": intl_brier,
        "dom_brier_series":  dom_brier,
        "cn_dog_brier":      cn_dog_brier,
        "cn_dog_n":          cn_dog_n,
    }


def extract_rating_spread():
    """Return (spread, top5, bot5) at 2025_after_champions from map_ratings.json."""
    with open(os.path.join(DATA_DIR, "map_ratings.json")) as f:
        d = json.load(f)
    snap = d["ratings"]["2025"]["snapshots"]["after_champions"]
    teams = snap["teams"]
    pairs = [(name, float(t["overall_rating"])) for name, t in teams.items()]
    pairs.sort(key=lambda x: -x[1])
    top5 = pairs[:5]
    bot5 = pairs[-5:]
    spread = pairs[0][1] - pairs[-1][1]
    return {
        "spread": spread,
        "top5": [{"team": t, "rating": round(r, 4)} for t, r in top5],
        "bot5": [{"team": t, "rating": round(r, 4)} for t, r in bot5],
        "max":  pairs[0][1],
        "min":  pairs[-1][1],
    }


# ── Cell runner ─────────────────────────────────────────────────────────────
def run_cell(power, scale, idx, total):
    t_cell = time.time()
    print(f"\n[cell {idx}/{total}]  RD_POWER={power}  RD_SCALE={scale}")
    patch_bmr(power, scale)

    dt1, _ = run_build_map_ratings()
    print(f"  BuildMapRatings:        {dt1:5.1f}s")

    dt2, _ = run_build_rating_timeline()
    print(f"  BuildRatingTimeline:    {dt2:5.1f}s")

    events = load_match_events_for_mle()
    beta_mle, ll = fit_beta_mle(events)
    print(f"  MLE β:                  {beta_mle:.4f}  (n={len(events)}, LL={ll:.2f})")

    suffix = "_rd_trial"
    dt3, _ = run_analyze(beta_mle, suffix=suffix)
    print(f"  AnalyzeCalibration:     {dt3:5.1f}s")

    cal_path = os.path.join(DATA_DIR, f"projection_calibration{suffix}.json")
    metrics = extract_metrics(cal_path)
    spread  = extract_rating_spread()

    total_dt = time.time() - t_cell
    print(f"  pool Brier (series):    {metrics['pool_brier_series']:.4f}")
    print(f"  Platt b (series):       {metrics['platt_b_series']:.4f}")
    print(f"  CN-dog Brier:           {metrics['cn_dog_brier']:.4f}  (n={metrics['cn_dog_n']})")
    print(f"  rating spread:          {spread['spread']:.2f}  "
          f"(top={spread['max']:.2f}  bot={spread['min']:.2f})")
    print(f"  cell time:              {total_dt:5.1f}s")

    return {
        "rd_power":   power,
        "rd_scale":   scale,
        "beta_mle":   round(beta_mle, 5),
        "n_series":   len(events),
        "metrics":    {k: (round(v, 5) if isinstance(v, (int, float)) and v is not None else v)
                       for k, v in metrics.items()},
        "spread":     {
            "spread": round(spread["spread"], 3),
            "max":    round(spread["max"], 3),
            "min":    round(spread["min"], 3),
            "top5":   spread["top5"],
            "bot5":   spread["bot5"],
        },
        "wall_sec": round(total_dt, 1),
    }


def main():
    quick = "--quick" in sys.argv
    if quick:
        powers = [0.5, 0.7]
        scales = [2.5, 3.5]
    else:
        powers = GRID_POWERS
        scales = GRID_SCALES

    cells = [(p, s) for p in powers for s in scales]
    print("=" * 72)
    print(f"TuneRdParams — grid sweep over {len(cells)} cells")
    print(f"  RD_POWERs: {powers}")
    print(f"  RD_SCALEs: {scales}")
    print(f"  fixed: intl_bonus={INTL_BONUS_FIXED}, cn_dog_offset={CN_DOG_OFFSET_FIX}")
    print("=" * 72)

    results = []
    started = datetime.utcnow().isoformat() + "Z"

    try:
        for i, (p, s) in enumerate(cells, 1):
            try:
                r = run_cell(p, s, i, len(cells))
                results.append(r)
                # Save incrementally so we don't lose progress on a crash.
                _save(results, started, cells)
            except Exception as e:
                print(f"  ERROR on cell ({p},{s}): {e}")
                results.append({"rd_power": p, "rd_scale": s, "error": str(e)})
                _save(results, started, cells)
    finally:
        # Always restore canonical params and do one clean rebuild.
        print("\n" + "=" * 72)
        print(f"Restoring canonical RD_POWER={CANONICAL_POWER}, RD_SCALE={CANONICAL_SCALE}")
        print("=" * 72)
        patch_bmr(CANONICAL_POWER, CANONICAL_SCALE)
        try:
            dt, _ = run_build_map_ratings()
            print(f"  BuildMapRatings: {dt:.1f}s")
            dt, _ = run_build_rating_timeline()
            print(f"  BuildRatingTimeline: {dt:.1f}s")
            print("  canonical rebuild complete")
        except Exception as e:
            print(f"  WARNING: canonical rebuild failed: {e}")

    _save(results, started, cells, final=True)
    _print_table(results)


def _save(results, started, cells, final=False):
    # Rank successful cells by pool Brier
    ok = [r for r in results if "error" not in r]
    ok_sorted = sorted(ok, key=lambda r: r["metrics"]["pool_brier_series"])
    best = ok_sorted[0] if ok_sorted else None

    recommendation = _make_recommendation(ok_sorted)

    out = {
        "meta": {
            "started": started,
            "finished": datetime.utcnow().isoformat() + "Z" if final else None,
            "grid_powers": sorted({p for p, _ in cells}),
            "grid_scales": sorted({s for _, s in cells}),
            "intl_bonus_fixed": INTL_BONUS_FIXED,
            "cn_dog_offset_fixed": CN_DOG_OFFSET_FIX,
            "canonical_power": CANONICAL_POWER,
            "canonical_scale": CANONICAL_SCALE,
            "baseline_pool_brier_target": 0.2298,
            "baseline_platt_b_target":    0.994,
            "n_cells_completed": len(ok),
            "n_cells_total":     len(cells),
            "final":             final,
        },
        "cells":           results,
        "cells_sorted_by_brier": ok_sorted,
        "best_cell":       best,
        "recommendation":  recommendation,
    }
    with open(GRID_OUT, "w") as f:
        json.dump(out, f, indent=2, default=str)


def _make_recommendation(cells_sorted):
    """Pick the cell that satisfies the ship criteria the best. Criteria
    (in priority order):
      1. Pool Brier ≤ 0.2298
      2. Platt b in [0.95, 1.05]
      3. Rating spread improves by ≥ 25% vs canonical (baseline ≈ 8.7)
      4. CN-dog Brier doesn't worsen meaningfully (≤ 0.21 + 0.01 slack)
    Among cells passing 1+2+3+4, pick lowest pool Brier.
    If nothing passes, return best pool-Brier cell as a non-ship suggestion.
    """
    if not cells_sorted:
        return {"verdict": "no successful cells", "ship_cell": None}

    canonical = next((c for c in cells_sorted
                      if c["rd_power"] == CANONICAL_POWER
                      and c["rd_scale"] == CANONICAL_SCALE), None)
    canonical_spread = canonical["spread"]["spread"] if canonical else None
    canonical_brier  = canonical["metrics"]["pool_brier_series"] if canonical else 0.2298
    canonical_cn_dog = canonical["metrics"]["cn_dog_brier"]      if canonical else 0.2038

    passing = []
    for c in cells_sorted:
        b  = c["metrics"]["pool_brier_series"]
        pb = c["metrics"]["platt_b_series"]
        sp = c["spread"]["spread"]
        cn = c["metrics"]["cn_dog_brier"]
        ok_brier  = b is not None and b <= canonical_brier + 1e-4
        ok_platt  = pb is not None and 0.95 <= pb <= 1.05
        ok_spread = (canonical_spread is None) or sp >= canonical_spread * 1.25
        ok_cn     = cn is None or canonical_cn_dog is None or cn <= canonical_cn_dog + 0.01
        passes = ok_brier and ok_platt and ok_spread and ok_cn
        passing.append({
            "rd_power":  c["rd_power"], "rd_scale": c["rd_scale"],
            "passes":    passes,
            "ok_brier":  ok_brier, "ok_platt": ok_platt,
            "ok_spread": ok_spread, "ok_cn":    ok_cn,
            "brier":     b, "platt_b": pb, "spread": sp, "cn_dog_brier": cn,
            "beta_mle":  c["beta_mle"],
        })
    shippable = [p for p in passing if p["passes"]]
    if shippable:
        best = shippable[0]  # already sorted by Brier
        return {
            "verdict":     "ship",
            "ship_cell":   {"rd_power": best["rd_power"], "rd_scale": best["rd_scale"],
                            "beta_mle": best["beta_mle"]},
            "metrics":     {"pool_brier": best["brier"], "platt_b": best["platt_b"],
                            "spread": best["spread"], "cn_dog_brier": best["cn_dog_brier"]},
            "all_passing_cells": shippable,
        }
    return {
        "verdict":     "no cell passes all ship criteria",
        "best_brier_cell": {"rd_power": cells_sorted[0]["rd_power"],
                            "rd_scale": cells_sorted[0]["rd_scale"],
                            "beta_mle": cells_sorted[0]["beta_mle"],
                            "metrics":  cells_sorted[0]["metrics"],
                            "spread":   cells_sorted[0]["spread"]["spread"]},
        "criteria_results": passing,
    }


def _print_table(results):
    print("\n" + "=" * 100)
    print("RANKED RESULTS  (lower pool Brier = better)")
    print("=" * 100)
    ok = [r for r in results if "error" not in r]
    ok_sorted = sorted(ok, key=lambda r: r["metrics"]["pool_brier_series"])
    hdr = f"{'pow':>5} {'scale':>6} {'β_mle':>7}   {'brier':>7} {'platt_b':>8} {'intl':>7} {'dom':>7}  {'cn_dog':>7}({'n':>3})  {'spread':>6} {'top':>6} {'bot':>6}"
    print(hdr)
    print("-" * len(hdr))
    for c in ok_sorted:
        m = c["metrics"]
        s = c["spread"]
        print(f"{c['rd_power']:>5.1f} {c['rd_scale']:>6.1f} {c['beta_mle']:>7.4f}   "
              f"{m['pool_brier_series']:>7.4f} {m['platt_b_series']:>8.4f} "
              f"{m['intl_brier_series']:>7.4f} {m['dom_brier_series']:>7.4f}  "
              f"{m['cn_dog_brier']:>7.4f}({m['cn_dog_n']:>3d})  "
              f"{s['spread']:>6.2f} {s['max']:>6.2f} {s['min']:>6.2f}")
    print()


if __name__ == "__main__":
    main()
