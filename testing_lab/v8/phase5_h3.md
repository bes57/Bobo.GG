# Phase 5 H3 — rating uncertainty (state-space) + 5d heterogeneous half-lives

agent:bias-h3, 2026-07-28. Preregistered in `preregister.bias_h3.md` (written
before any run; outcomes appended there). Frame: canonical expanded series
frame, sha256-verified vs crn.json; holdout n=1217 (date > 2024-12-31); every
constant fit train-only, β refit per config. Randomness: GH-20 quadrature
(deterministic; ≡ GH-40 to machine epsilon on all 2058 rows; MC cross-check
with crn mc_seeds[0] agrees within 3 MC-SE); bootstraps via
referee.paired_bootstrap_crn (iid + block_event). MDEs quoted from
stats/power_mde_expanded.json: within-family 1.773m, cross-family 5.889m.
v6 baseline re-solved on this frame: β=0.1152, holdout LL **0.64216**.

## Verdicts, plainly

**1. State-space core: LOSS on aggregate; the mechanism is real but lives in
the tails.** Plain core (q/R=0.0056 ⇒ HL 9.2 games, V0/R=1.27):
holdout 0.65103, Δ = **−8.87m** vs v6 (cross-family MDE 5.889m, p_better
iid 0.001 / block 0.003 → LOSS). Train-selected primary (1b, +calendar
process noise) was the best TRAIN config (0.64485) and the WORST holdout:
**−11.75m** — calendar-time uncertainty fit the 2023-24 regime and did not
generalize; the operator should read this as "breaks do not add real drift
once games-counted noise exists." Debut-region-prior variant: −9.09m.
ROI unit: the quote-margin ladder is one-sided (anchored at δ=0), so a
negative ΔLL translates to expected_roi_delta 0.0 with the deficit expressed
as δ_logit ≈ −0.026 (1b) / −0.019 (1a) on the quoting surface.

**2. Where it wins (mechanism prediction confirmed):** cold/thin rows,
exactly as preregistered. Holdout, either team with <10 prior maps (n=39):
5d-model **+71.7m** vs v6 (0.578 vs 0.650); debut rows (n=10) **+58.2m**;
<30 prior maps (n=117) **+28.0m**. Old-frame cold reference was n=57 /
LL 0.70205; the referee COLD_EPS bucket is empty on engine ratings (region
prior ⇒ no exact zeros), so prior-map-count definitions are used and
documented. Everywhere else v6 wins moderately (5d model: 2026 −9.2m,
domestic CN −9.6m, close matchups −8.8m, post-break −11.3m).

**3. 5d heterogeneous half-lives (stats/h3_process_noise.json, published
early):** typed process noise on the plain core, K=3 cells per axis, train
MLE + profile CIs + DerSimonian-Laird pooling on log q. Roster-stability
axis (matches_since_change; lineups-agent definitions reproduced 0/3466
mismatches, full-corpus scratch top-up): **change-adjacent (≤3 matches)
HL 7.4 games vs stable (>10) HL 24.8 games — q ratio ≈ 11× in the predicted
direction** (prereg bar ≥2×); the settling cell (4-10) hit the q→0 boundary
(read as "indistinguishable from no drift", not a measurement). BUT the DL
pooling verdict is τ̂²=0 on every axis: at n_train=841 series the profile
curvature cannot resolve the between-cell spread, and all cells shrink to a
pooled HL ≈ 8 games — published alongside the point ordering, per the
preregistered falsifier language ("heterogeneity unsupported at this n" is
the formal inference; the point table is the operator's quantified
instinct). Independent evidence it is not pure train noise: the roster-typed
model improves holdout over the plain core by **+3.55m** — a within-family
WIN (MDE 1.773m; p_better iid 0.946, block 0.979; ROI +0.29pp on the
ladder), and vs v6 it closes to **−5.32m = INSIDE the cross-family noise
floor** (p_better 0.041/0.050 — leaning worse, not resolved). Org-age axis:
REVERSED (established orgs fit the largest q) and −14m holdout — falsified,
read as corpus-start left-censoring artifact. Volatility axis: flat
(HL 8.9-11.0) — falsified at this n.

**4. Per-team bias caterpillar (stats/h3_bias_caterpillar.json):**
max|bias| improves under uncertainty models (v6 0.1478 → 5d 0.1328,
1b 0.1255) but mean|bias| worsens slightly (0.0473 → 0.0504/0.0522).
Mechanism check: the thin half of teams (by mean pre-match n_eff) loses
sharpness (−0.61pp toward 0.5) while the established half does not (+0.10pp),
corr(n_eff, sharpness change) = +0.40 — the predicted asymmetric shrinkage,
though among min_n≥25 caterpillar teams the n_eff spread is narrow (14.2-15.3;
the truly thin teams live below min_n, in the cold buckets above).

**5. n_eff telemetry (stats/h3_neff.json):** per-match posterior variances,
n_eff (= R/v, invariant to the R identification constant), σ_Δ, point vs
integrated p, shrink factor, for all 2058 rows + distribution summaries by
year/event class/cold rows. corr(pair n_eff, per-row |ΔLL vs v6|) = −0.09:
low-confidence rows are where the models disagree most — usable as the
confidence-aware-sizing input the telemetry spec wants.

**6. Ensemble-where-it-wins (stats/h3_ensemble.json): no deployable
composite found by the preregistered rule.** 14 train-only composites
(hard gates n_eff<θ / change-adjacent / EWC-class / soft n_eff blend ×
{1b, 5d}); several beat v6 on train; the train winner (soft blend on 1b)
LOST holdout by −12.2m — it inherited 1b's train mirage. The honest
candidate shape for a future wave is hard-gate + 5d (its gated rows are the
+28..+72m buckets), but by the one-shot-holdout discipline it was not
selected and is not scored; promotion answer: **HOLD everything**.

## Artifacts
- stats/h3_statespace.json (pairwise judging, buckets, cold buckets, GH check)
- stats/h3_process_noise.json (5d half-life table, CIs, pooling — early)
- stats/h3_neff.json, stats/h3_bias_caterpillar.json, stats/h3_ensemble.json
- scratch/bias_h3/ (checkpointed sweeps, v6 baseline npz, lineup top-up +
  two-mode verification), logs/bias_h3.log
