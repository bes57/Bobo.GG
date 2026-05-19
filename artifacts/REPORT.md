# BenPom Hour-Pass Optimization Report

**Date**: 2026-05-19
**Branch**: `optimize/hour-pass-20260519`
**Starting commit**: `380f0e7`

## Executive Summary

> **Outcome**: NO CONFIG CHANGES SHIPPED. The model is at a local optimum on these knobs.
>
> Baseline series Brier: **0.23067** (n=1440). Best held-out test improvement found: **none statistically significant**. Optuna's apparent -0.003 in-sample win **degraded test Brier by +0.011 (p=0.021)** — pure overfit. All four parallel Phase 2 experiments either landed within noise or rejected on complexity grounds.

Target was series Brier < 0.215. Not achievable with the knobs swept. The model has been hand-tuned over multiple sessions and is at a meaningful local optimum.

## Final config diff

```diff
# scrapers/BuildMapRatings.py
# (no changes — current values stay)

# scrapers/BuildRatingTimeline.py
# (no changes)

# MapElo.py
# (no changes)
```

## Per-hypothesis verdict

| ID | Hypothesis | Verdict | Test Δ Brier | p | Reason |
|---|---|---|---|---|---|
| H1 | Format-specific BETA | **rejected** | Train -0.0011 / Test +0.0012 | 0.228 | Bo5 n=89 → Brier stderr ±0.046; bo5 optimum at BETA=0.10 is noise-driven |
| H2 | Map-specific BETA | not tested | — | — | Out of scope given H1 result |
| H3 | Margin function | **rejected** | +0.0001 to +0.0009 (in-sample) | 0.49 | Scale-coupling confirmed (BETA spread 0.08 across variants). tanh4 marginally best but no statistical win |
| H4 | Half-life | **rejected** | — | — | Optuna's HL=9.4 was part of the overfit cluster; rejected with whole config |
| H5 | Post-hoc calibration | **rejected (complexity)** | -0.0016 | 0.331 | Isotonic was the only one with directional improvement, but NS and requires JS impl |
| H6 | Huber / IRLS | **rejected** | -0.00006 vs best margin | 0.94 | Within noise |
| H7 | Roster-aware decay | **already in use** | — | — | Current ROSTER_PERSISTENCE=0.3 is the winning config (vs RP=0 baseline: Δ -0.002, p=0.0015) |
| H8 | CN parameter joint opt (D) | **rejected** | — | — | Current split (-4.0 / +0.47) beats both alternatives on full set; differences NS |
| H9 | Snapshot ensembling | **rejected** | — | — | Adds nothing on top of roster smoothing (p=0.26) |
| H10 | Veto-sim ablation | not tested | — | — | Out of scope |

## Interaction findings

- **Scale-coupling confirmed (I1)**: best-BETA spread across margin variants = 0.08 (sqrt:0.16, log1p:0.18, tanh4:0.12, capped_linear_8:0.10, huber4:0.10). Phase 1's discovery of "capped_linear_8 + HL=9 + SHRINK_K=3" was an interaction-driven find; on held-out it didn't generalize.
- **BETA-Platt non-equivalence (I3)**: Phase 2B tested whether a BETA shift = 0.140/1.16 reproduces global Platt's effect. It does NOT — the equivalent-BETA-shift gave test Brier +0.00087 *worse* than uncalibrated, while global Platt gave -0.00019. So Platt is doing more than rescaling. (Doesn't change the recommendation since the overall effect is still NS.)
- **Time-smoothing redundancy (I4)**: snapshot ensembling and roster-aware decay are redundant — stacking them gives p=0.26 vs the best single. Roster decay alone wins clean.
- **CN parameter substitution (I2)**: rating-side-only (CN_PRIOR=-6, offset=0) is actually best on CN-intl-only matches (0.22038 vs current 0.22457) but worse on the full slate. Current split is principled, not band-aid.
- **Compositional collapse (I7)**: not a real concern this pass since no components were accepted. The composed config = baseline.

## Optuna expensive sweep behavior

The 30 trials that completed before the subagent's Python process died (out of 200 planned) clustered around `MARGIN_FN=capped_linear_8` with `HALF_LIFE_WEEKS=9-10`, `SHRINK_K=3-7`, `CN_PRIOR=-2 to -4`. In-sample Brier 0.228 looked great. Held-out test on the top 3 configs (all rebuilt with the function-replacement trick):

| Trial | Config (key dims) | In-sample | Train (70%) | Test (30%) | Test p (vs base) |
|---|---|---|---|---|---|
| 23 | HL=9.4, SK=3.3, CP=-3.17, marg=cl8 | 0.22837 | 0.23307 (+0.0024) | 0.24163 (+0.0109) | 0.021 |
| 26 | HL=9.1, SK=7.5, CP=-3.64, marg=cl8 | 0.22818 | 0.23304 (+0.0024) | 0.24164 (+0.0109) | 0.028 |
| 28 | HL=9.4, SK=2.9, CP=-2.83, marg=cl8 | 0.22810 | 0.23317 (+0.0025) | 0.24222 (+0.0115) | 0.003 |

All three statistically degrade test Brier by ~1bp. The TPE sampler chased noise in the bo5/intl-CN tails.

(Note: there's a ~0.008 gap between the subagent's recorded in-sample Brier and my full-rebuild reproduction — likely an inconsistency in how the patched-massey function got captured between trials. Even granting the subagent's 0.228 is "real," the held-out test number is the disqualifier.)

## Sanity checks on current config

| Check | Status | Detail |
|---|---|---|
| Trophy: EDG > GEN at 2024_after_champions | ✅ pass | EDG +3.43, GEN +3.16, gap 0.27 |
| CN bottom-of-pack: ≥4 CN in bottom 8 (2024_after_champions) | ✅ pass | TYL/NOVA/AG/TEC all bottom 10 |
| H2H: BLG > JDG at latest 2026 snap | ✅ pass | BLG -3.29 > JDG -3.69, gap 0.40 |
| CN c-saturation: ≤2 teams outside [0.1, 0.95] | ⚠ fail | 4 teams (NOVA 0.09, FPX 0.07, ASE ~0, TYLOO 0) — all ancient/decayed at 2026 ref_date |

The c-saturation failure is a property of the LIVE 2026 ref_date: 2024 intl evidence has decayed to weights < 0.005 per game by mid-2026. For each historical snapshot's own ref_date, the c values are healthy. Not a real model bug — flag for future threshold refinement.

## Three concrete follow-ups for next pass

1. **Per-format isotonic calibration on bo5 only** — bo3 is well-calibrated (n=1351, Platt b≈0.92 at BETA=0.16), bo5 is the noisy tail (n=89, Platt b 0.80 at optimal BETA=0.10). A bo5-only isotonic fit on training data, applied to bo5 predictions only, may pull series Brier down a few bp without overfit risk in the bo3 channel. Estimated Δ: -0.002 to -0.005 on test, p-value ambiguous given n=89.

2. **Move from match-level to map-level training** — currently the loss optimized is implicitly series Brier via the bo3/bo5 closed form. Map-level Brier is what the per-map sim drives. Fitting BETA on map-level outcomes directly (with bo5 maps weighted 0.6 to reflect later-map dependence) may produce a more honest scale. Out of scope here but doable in 30 min next pass.

3. **Region pair-specific intercepts (not slopes)** — the per-bucket Platt experiment showed buckets each have similar slope (b ≈ 0.9-1.1) but differing intercepts. A per-bucket intercept-only Platt (5 free params, regularized) is fewer DOF than the full slope+intercept version and may avoid the overfit that sank Phase 2B's per_bucket variant. Estimated Δ: -0.001 to -0.003 on test.

## Phase 1 (Optuna) — what would help next pass

- **Add held-out train/test split inside the objective function**. The current sweep optimizes in-sample Brier directly — Optuna's TPE will always find spurious wins this way. With 14 free parameters, even on 1440 matches a TPE-driven optimum has substantial overfit risk.
- **Constrain MARGIN_FN to {sqrt, tanh4}** — the linear variants need BETA re-tune, which the sweep doesn't do; including them pulls the TPE toward configs where the wrong margin appears competitive because of compensating rating-scale knobs.
- **Increase per-trial budget to 200+ trials with the constrained space** — 30 trials of an effectively unconstrained 14-D search is way under-resourced.

## Git status

```
On branch optimize/hour-pass-20260519
nothing to commit, working tree clean (no model code touched)
artifacts/ contains all experiment scripts, JSON results, and this report
```

No commit on the optimize branch — there are no model changes to ship. Recommend deleting the branch after review, OR keeping it for the `artifacts/` directory as a reference for the next pass.

## Baseline / final metrics

```
n_series:           1440
n_maps:             3667
series_brier:       0.23067   (unchanged)
map_brier:          0.24113
series Platt: a=+0.118, b=0.922  (slightly underconfident)
map    Platt: a=+0.088, b=0.877
series ECE:         0.0355
map    ECE:         0.0193
```

Per-bucket Brier (unchanged):

| Bucket | n | Brier |
|---|---|---|
| bo3 domestic | 1193 | 0.22881 |
| bo3 intl-noCN | 101 | 0.24249 |
| bo3 intl-CN | 57 | 0.22530 |
| bo5 domestic | 75 | 0.24599 |
| bo5 intl-noCN | 11 | 0.25290 |
| bo5 intl-CN | 3 | 0.21071 |
