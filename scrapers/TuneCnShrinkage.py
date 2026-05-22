"""
TuneCnShrinkage.py
==================

Grid-search the CN shrinkage parameters in BuildMapRatings.py (and the mirror
constants in BuildRatingTimeline.py) so that BenPom's intl predictions for
CN-as-underdog matchups land in the right place WITHOUT needing the
post-prediction CN_DOG_OFFSET band-aid in AnalyzeProjectionCalibration.py.

For each (CN_PRIOR, CN_INTL_K) cell we:
  1. Patch both BuildMapRatings.py and BuildRatingTimeline.py to the trial
     values (in-place edit, restored at the end).
  2. Run BuildMapRatings.py     → data/map_ratings.json
  3. Run BuildRatingTimeline.py → data/rating_timeline*.json
  4. Run AnalyzeProjectionCalibration.py --beta 0.154 --intl-bonus 0.22
        --cn-dog-offset 0 --suffix _trial
     → data/projection_calibration_trial.json
  5. Extract pool_brier (series), intl_brier, dom_brier, Platt b (series),
     plus a CN-as-dog Brier computed by n-weighted averaging the q5
     region buckets where dog_region == CN and fav_region != CN.

Then we restore CN_PRIOR=-4.0, CN_INTL_K=2.0 in both files, rebuild once,
and write the grid + a recommendation to data/cn_rating_tune_grid.json.

Usage:
  .venv/bin/python scrapers/TuneCnShrinkage.py
"""

import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRAPERS = os.path.join(ROOT, "scrapers")
DATA_DIR = os.path.join(ROOT, "data")
PY = os.path.join(ROOT, ".venv/bin/python")

BMR_PATH = os.path.join(SCRAPERS, "BuildMapRatings.py")
BRT_PATH = os.path.join(SCRAPERS, "BuildRatingTimeline.py")
TRIAL_JSON = os.path.join(DATA_DIR, "projection_calibration_trial.json")
GRID_JSON = os.path.join(DATA_DIR, "cn_rating_tune_grid.json")

CANONICAL_PRIOR = -4.0
CANONICAL_K     = 2.0

# Reduced grid to fit the compute budget (~25 min)
GRID_PRIORS = [-4.0, -5.0, -6.0, -7.0, -8.0]
GRID_KS     = [2.0, 3.0, 5.0, 8.0]


# ── File patching ────────────────────────────────────────────────────────────

PRIOR_RE = re.compile(r"^(CN_PRIOR\s*=\s*)([-\d.]+)(.*)$", re.MULTILINE)
K_RE     = re.compile(r"^(CN_INTL_K\s*=\s*)([-\d.]+)(.*)$", re.MULTILINE)


def patch_constants(path, prior, k):
    with open(path) as f:
        txt = f.read()
    new_txt, n_prior = PRIOR_RE.subn(lambda m: f"{m.group(1)}{prior}{m.group(3)}", txt, count=1)
    new_txt, n_k     = K_RE.subn(lambda m: f"{m.group(1)}{k}{m.group(3)}", new_txt, count=1)
    if n_prior != 1 or n_k != 1:
        raise RuntimeError(f"Patch failed for {path}: prior={n_prior} k={n_k}")
    with open(path, "w") as f:
        f.write(new_txt)


def restore_canonical():
    # Only patch BuildMapRatings (BuildRatingTimeline doesn't define them; it
    # imports). But BuildMapRatings is the only one with the constants.
    patch_constants(BMR_PATH, CANONICAL_PRIOR, CANONICAL_K)
    # BuildRatingTimeline.py imports CN_PRIOR/CN_INTL_K from BuildMapRatings,
    # so no separate constants live there. But keep the API in case someone
    # later adds a local override.
    with open(BRT_PATH) as f:
        brt = f.read()
    if PRIOR_RE.search(brt):
        patch_constants(BRT_PATH, CANONICAL_PRIOR, CANONICAL_K)


def patch_both(prior, k):
    patch_constants(BMR_PATH, prior, k)
    with open(BRT_PATH) as f:
        brt = f.read()
    if PRIOR_RE.search(brt):
        patch_constants(BRT_PATH, prior, k)


# ── Subprocess helpers ───────────────────────────────────────────────────────

def run(cmd, label):
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    dt = time.time() - t0
    if proc.returncode != 0:
        print(f"   ✗ {label} failed ({dt:.1f}s)")
        print(proc.stdout[-1500:])
        print(proc.stderr[-1500:])
        raise RuntimeError(f"{label} failed")
    print(f"   ✓ {label} ({dt:.1f}s)")


# ── Metric extraction ────────────────────────────────────────────────────────

def extract_metrics(json_path):
    with open(json_path) as f:
        d = json.load(f)
    q1 = d["q1_overall_calibration"]
    q2 = d["q2_favorite_undervaluation"]
    q3 = d["q3_intl_vs_domestic"]
    q5 = d["q5_region_buckets"]

    pool_brier = q1["series"]["brier"]
    intl_brier = q3["series"]["intl"]["brier"]
    dom_brier  = q3["series"]["domestic"]["brier"]
    platt_b_series = q2["platt_series"]["b"]
    platt_a_series = q2["platt_series"]["a"]
    pool_map_brier = q1["map"]["brier"]

    # CN-as-dog buckets in q5 are labelled "{fav_region} vs CN" where fav_region != CN
    cn_dog_n = 0
    cn_dog_brier_weighted = 0.0
    cn_dog_mean_pred = 0.0
    cn_dog_winrate = 0.0
    cn_fav_n = 0
    cn_fav_brier_weighted = 0.0
    for b in q5["by_region_pairing"]:
        bucket = b["bucket"]
        if bucket == "same-region":
            continue
        # format "fav_region vs dog_region"
        parts = bucket.split(" vs ")
        if len(parts) != 2:
            continue
        fav_r, dog_r = parts
        if fav_r == "CN" and dog_r != "CN":
            cn_fav_n += b["n"]
            cn_fav_brier_weighted += b["n"] * b["brier"]
        elif fav_r != "CN" and dog_r == "CN":
            cn_dog_n += b["n"]
            cn_dog_brier_weighted += b["n"] * b["brier"]
            cn_dog_mean_pred += b["n"] * b["mean_pred"]
            cn_dog_winrate += b["n"] * b["fav_winrate"]

    cn_dog_brier = (cn_dog_brier_weighted / cn_dog_n) if cn_dog_n else None
    cn_fav_brier = (cn_fav_brier_weighted / cn_fav_n) if cn_fav_n else None
    cn_dog_mp    = (cn_dog_mean_pred / cn_dog_n) if cn_dog_n else None
    cn_dog_wr    = (cn_dog_winrate / cn_dog_n) if cn_dog_n else None

    return {
        "pool_brier": pool_brier,
        "pool_map_brier": pool_map_brier,
        "intl_brier": intl_brier,
        "dom_brier": dom_brier,
        "platt_a_series": platt_a_series,
        "platt_b_series": platt_b_series,
        "cn_dog_n": cn_dog_n,
        "cn_dog_brier": cn_dog_brier,
        "cn_dog_mean_pred": cn_dog_mp,
        "cn_dog_winrate": cn_dog_wr,
        "cn_dog_residual": (cn_dog_wr - cn_dog_mp) if cn_dog_n else None,
        "cn_fav_n": cn_fav_n,
        "cn_fav_brier": cn_fav_brier,
    }


# ── Single trial ─────────────────────────────────────────────────────────────

def _wipe_timeline_caches():
    """Delete rating_timeline*.json so BuildRatingTimeline can't short-circuit
    on its incremental cache when CN params change underneath it."""
    for fn in ("rating_timeline.json",
               "rating_timeline_2023.json",
               "rating_timeline_2024.json",
               "rating_timeline_2025.json"):
        p = os.path.join(DATA_DIR, fn)
        if os.path.exists(p):
            os.remove(p)


def run_trial(prior, k, cn_dog_offset, suffix="_trial", label=""):
    patch_both(prior, k)
    print(f"\n── Trial: CN_PRIOR={prior}, CN_INTL_K={k}, offset={cn_dog_offset}  {label}")
    _wipe_timeline_caches()
    run([PY, os.path.join(SCRAPERS, "BuildMapRatings.py")], "BuildMapRatings")
    run([PY, os.path.join(SCRAPERS, "BuildRatingTimeline.py")], "BuildRatingTimeline")
    run([PY, os.path.join(SCRAPERS, "AnalyzeProjectionCalibration.py"),
         "--beta", "0.154", "--intl-bonus", "0.22",
         "--cn-dog-offset", str(cn_dog_offset), "--suffix", suffix],
        "AnalyzeProjectionCalibration")
    metrics = extract_metrics(os.path.join(DATA_DIR, f"projection_calibration{suffix}.json"))
    metrics["cn_prior"] = prior
    metrics["cn_intl_k"] = k
    metrics["cn_dog_offset"] = cn_dog_offset
    return metrics


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"== CN shrinkage tuning grid ==")
    print(f"   Priors: {GRID_PRIORS}")
    print(f"   Ks:     {GRID_KS}")
    print(f"   Total cells: {len(GRID_PRIORS) * len(GRID_KS)}")

    # Baseline #1: canonical params + band-aid (option A)
    baseline_A = run_trial(CANONICAL_PRIOR, CANONICAL_K, 0.47,
                            suffix="_trial", label="(baseline A: band-aid)")
    # Baseline #2: canonical params + no band-aid
    baseline_noband = run_trial(CANONICAL_PRIOR, CANONICAL_K, 0.0,
                                 suffix="_trial", label="(baseline no-bandaid)")

    grid = []
    for prior in GRID_PRIORS:
        for k in GRID_KS:
            if prior == CANONICAL_PRIOR and k == CANONICAL_K:
                # already covered as baseline_noband
                m = dict(baseline_noband)
                grid.append(m)
                continue
            try:
                m = run_trial(prior, k, 0.0, suffix="_trial")
                grid.append(m)
            except Exception as e:
                print(f"!! Trial {prior},{k} failed: {e}")
                grid.append({"cn_prior": prior, "cn_intl_k": k, "error": str(e)})

    # Sort by pool_brier ascending for selection
    valid = [g for g in grid if "pool_brier" in g]
    valid.sort(key=lambda r: r["pool_brier"])

    # Print summary table
    print("\n\n=== GRID RESULTS (sorted by pool_brier asc) ===")
    print(f"{'PRIOR':>6} {'K':>4} | {'pool':>7} {'intl':>7} {'dom':>7} | {'platt_b':>7} | {'cn_dog_br':>9} {'cn_dog_n':>9} {'cn_dog_res':>11}")
    for g in valid:
        print(f"{g['cn_prior']:>6.1f} {g['cn_intl_k']:>4.1f} | "
              f"{g['pool_brier']:>7.4f} {g['intl_brier']:>7.4f} {g['dom_brier']:>7.4f} | "
              f"{g['platt_b_series']:>7.3f} | "
              f"{(g['cn_dog_brier'] or 0):>9.4f} {(g['cn_dog_n'] or 0):>9d} "
              f"{(g['cn_dog_residual'] or 0):>+11.4f}")

    # Pick best cell that doesn't degrade domestic beyond no-band baseline
    dom_cap = baseline_noband["dom_brier"] + 0.0005  # tiny tolerance
    eligible = [g for g in valid if g["dom_brier"] <= dom_cap]
    if eligible:
        best = eligible[0]
    else:
        best = valid[0]
        print(f"\nWARN: no cell satisfied dom_brier <= {dom_cap:.4f}; falling back to overall best.")

    # Option C: combine best rating tune + half-strength band-aid
    optC = run_trial(best["cn_prior"], best["cn_intl_k"], 0.235,
                      suffix="_trial", label="(Option C combo: best tune + half offset)")

    # Restore canonical
    print("\n── Restoring canonical CN_PRIOR=-4.0, CN_INTL_K=2.0 and rebuilding ──")
    patch_both(CANONICAL_PRIOR, CANONICAL_K)
    run([PY, os.path.join(SCRAPERS, "BuildMapRatings.py")], "BuildMapRatings (restore)")
    run([PY, os.path.join(SCRAPERS, "BuildRatingTimeline.py")], "BuildRatingTimeline (restore)")

    # Decide recommendation: B if best beats band-aid pool Brier; else A; C only if it's better than both
    rec_metric = lambda m: m["pool_brier"]
    by_pool = sorted([
        ("A", baseline_A),
        ("B", best),
        ("C", optC),
    ], key=lambda kv: rec_metric(kv[1]))
    rec_letter, _rec_metrics = by_pool[0]

    rationale = (
        f"Best cell (CN_PRIOR={best['cn_prior']}, CN_INTL_K={best['cn_intl_k']}): "
        f"pool={best['pool_brier']:.4f}, intl={best['intl_brier']:.4f}, "
        f"dom={best['dom_brier']:.4f}, Platt b={best['platt_b_series']:.3f}. "
        f"Option A (band-aid) pool={baseline_A['pool_brier']:.4f}. "
        f"Option C (combo) pool={optC['pool_brier']:.4f}. "
        f"Recommendation: Option {rec_letter}."
    )

    out = {
        "generated": datetime.utcnow().isoformat() + "Z",
        "grid_definition": {
            "cn_priors": GRID_PRIORS,
            "cn_intl_ks": GRID_KS,
            "beta": 0.154,
            "intl_bonus": 0.22,
        },
        "grid": grid,
        "grid_sorted_by_pool_brier": valid,
        "baseline_with_bandaid": baseline_A,
        "baseline_no_bandaid": baseline_noband,
        "best_cell": best,
        "option_C_combo": optC,
        "recommendation": rec_letter,
        "rationale": rationale,
    }
    with open(GRID_JSON, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nWrote {GRID_JSON}")
    print(f"\n{rationale}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted — restoring canonical params before exit.")
        patch_both(CANONICAL_PRIOR, CANONICAL_K)
        sys.exit(130)
