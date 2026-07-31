"""Build model_snapshot.json — the BenPom v6 package the trading bot reads.

Since 2026-07-30 the public site and the bot share one model. The site's
refresh pipeline (scrapers/BuildRatingTimeline.py) is the ONLY solver: every
refresh it re-solves the v6 ratings walk-forward and refits the prediction
constants into data/site_model.json. This script is now a thin REPACKAGER —
it composes the bot snapshot from those live artifacts and asserts internal
consistency. There is no second solver to drift (the 2026-07-23 vm-snapshot
incident and the vendored-registry traps all came from second copies).

Chain per refresh:  site scrape → BuildRatingTimeline (solve + site_model)
                    → this script → bot hot-reloads model_snapshot.json.

Run:  python3 trading_model/build_model_snapshot.py
"""
import json
import math
import os
import sys
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "scrapers"))

MODEL_VERSION_PREFIX = "benpom-v6"
SITE_MODEL = os.path.join(ROOT, "data", "site_model.json")
TIMELINE = os.path.join(ROOT, "data", "rating_timeline.json")
ROSTER_FLAGS_SRC = os.path.join(ROOT, "testing_lab", "v8", "stats",
                                "roster_integration.json")


def main():
    with open(SITE_MODEL) as f:
        sm = json.load(f)
    with open(TIMELINE) as f:
        tl = json.load(f)

    cps = tl["checkpoints"]
    last = cps[-1]
    ratings = {t: round(float(v), 4) for t, v in last["ratings"].items()}

    # Region map from the site's canonical table (self-contained: no VCTMM
    # import — its vendored copies have shadowed live modules before).
    from BuildMapRatings import TEAM_REGIONS
    org_regions = {t: TEAM_REGIONS.get(t, "") for t in ratings}

    # ── consistency gates (fail loudly, never ship a broken snapshot) ──
    assert 0.10 <= sm["beta"] <= 0.16, f"beta {sm['beta']} outside sane band"
    assert sm["xregion_offsets"].get("CN") == 0, "CN must be the pinned region"
    assert len(ratings) >= 40, f"suspiciously few rated teams: {len(ratings)}"
    missing = [t for t in ratings if not org_regions.get(t)]
    if missing:
        print(f"  [warn] {len(missing)} org(s) without region: {missing[:8]}")

    snapshot = {
        "model_version": f"{MODEL_VERSION_PREFIX}-{last['date']}",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "ratings_as_of": last["date"],
        "source": {"site_model": sm.get("model_version"),
                   "site_model_generated": sm.get("generated_utc")},
        "beta": sm["beta"],
        "gf_upper_logit": sm["gf_upper_logit"],
        "b_pick": sm["b_pick"],
        "ratings": dict(sorted(ratings.items())),
        "region_priors": sm["region_priors"],
        "xregion_offsets": sm["xregion_offsets"],
        "org_regions": org_regions,
        "config": {
            "decay": "games-counted, consistency-conditioned: HL 20 (consistent) / 12 (anomaly)",
            "margin": "sign(rd)*|rd|^0.75*2.5", "playoff_weight": 1.6,
            "champions_mult": 2.0, "region_prior_ridge": 1.5,
            "roster": "year-boundary continuity 0.3", "ridge": 0.5,
            "beta_policy": "refit each site build on all completed series to date",
        },
        "validation": {
            "holdout_ll_v6": 0.6409, "holdout_ll_old_production": 0.6526,
            "programs": "v7 (18 configs), v8 (adversary-co-signed), v9 "
                        "(3,240-config transfer-gated search) — v6 stands in all three",
            "notes": "see MODEL_EXPLAINED.md §6; per-team bias table is in "
                     "PROBABILITY points (mean predicted P − win rate, ×100)",
        },
    }

    # ── parity self-test: predict.py math on this snapshot reproduces the
    #    closed form exactly (guards against schema/typo regressions) ──
    sys.path.insert(0, HERE)
    import predict
    m = snapshot
    teams = sorted(ratings, key=lambda t: -ratings[t])[:8]
    for a, b in zip(teams[::2], teams[1::2]):
        p = predict.series_probability(m, a, b, "bo3")
        ra, rb = ratings[a], ratings[b]
        adj = 0.0
        if org_regions[a] != org_regions[b]:
            off = m["xregion_offsets"]
            adj = off.get(org_regions[a], 0) - off.get(org_regions[b], 0)
        pm = 1 / (1 + math.exp(-m["beta"] * (ra - rb + adj)))
        want = pm * pm * (3 - 2 * pm)
        assert abs(p - want) < 1e-12, f"parity failed {a}-{b}: {p} vs {want}"
    print(f"  parity self-test OK ({len(teams)//2} pairs)")

    out = os.path.join(HERE, "model_snapshot.json")
    with open(out, "w") as f:
        json.dump(snapshot, f, indent=1)
    print(f"wrote {out} ({len(ratings)} teams, beta {sm['beta']}, "
          f"as of {last['date']})")

    # ── roster_flag sidecar (optional; sizing-only — see ROSTER_FLAG.md) ──
    if os.path.exists(ROSTER_FLAGS_SRC):
        try:
            with open(ROSTER_FLAGS_SRC) as f:
                integ = json.load(f)
            flags = (integ.get("spec_run_roster_flag_extension") or {}).get("teams")
            if flags:
                with open(os.path.join(HERE, "roster_flags.json"), "w") as f:
                    json.dump({"generated_utc": snapshot["generated_utc"],
                               "spec": "ROSTER_FLAG.md", "teams": flags}, f, indent=1)
                print(f"wrote roster_flags.json ({len(flags)} teams)")
        except Exception as e:
            print(f"  [warn] roster_flags sidecar skipped: {e}")


if __name__ == "__main__":
    main()
