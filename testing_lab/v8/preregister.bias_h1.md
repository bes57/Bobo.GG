# preregister.bias_h1 — Phase 5 H1: bounded margins censor dominant teams

agent:bias-h1, 2026-07-28. Written BEFORE any experiment run. Frame:
`v8/data/frame_expanded/series.csv` (sha256 ff772d41… verified vs crn.json
frame_expanded block at session start; n=2058, train 841, holdout 1217).
Baseline config ("v6"): run_v7_stage1.py BASE + champion decay —
`rd {power:0.75, scale:2.5}`, decay `games/consistency (20,12)`, roster_mode
year (pers 0.3), ridge 0.5, region_prior_ridge 1.5, champ_mult 2.0, w_custom
= 1.6 on playoffs/grand_final maps (stage from the expanded frame by
match_id), β=None (train-only refit). Engine games verified expanded (EWC/CN
evo events present; 0 frame matches missing). All bootstrap randomness from
crn.json; judging via referee.paired_bootstrap_crn (iid + block_event),
per_team_bias (min_n=25), bucketed, expected_roi_of_dll. MDEs quoted from
stats/power_mde_expanded.json: within-family 1.77m, cross-family 5.89m.

## Data audit (run pre-registration, no outcomes computed)

round_outcomes.csv: 92,311 rounds, 1,716 matches. Coverage vs frame:
train 688/841 (81.8%), holdout 1019/1217 (83.7%). Uncovered = all 26
corpus-addition events at exactly 0% (CN evo family, EWC-2025 chain, LCQ,
shanghai/off-season invitationals) + 2026_ewc_qual partial 53/69. CN-involved
series 70.0% covered, non-CN 88.4%. Sides recorded per round winner
(attack/defense); rounds/map mean 21.2, min 13, max 48.

## Shared machinery (declared before running)

- Massey target scale ("target pts"): t(m) = m^0.75 × 2.5 (v6's transform).
  Cap constant c13 = t(13) = 17.113; OT constant c_ot = t(2) = 4.204.
- Map classes from match_results Score (loser rounds x): REG-EXACT
  (13-x, 2 ≤ margin ≤ 12, x ≤ 11), CAP (13-0, margin 13), OT (winner 14+,
  loser ≥ 12, margin 2).
- Walk-forward rating as-of a game's own date = that day's v6 (or candidate)
  solve, which uses only games dated strictly earlier. Days with no solve
  (engine skips <30 hist games): nearest earlier solved day, else μ=0.
- Parity gate for any self-written solver: must reproduce engine.run's v6
  rdiff vector to max|Δ| < 1e-8 before its candidate variant is trusted.

## E3 (FIRST, cheap) — censoring diagnostic. GATES E1.

Mechanism: if bounded margins censor dominance, elite winners should pile up
at the cap and realized margins should flatten below prediction at the top.
Design (train maps only, date ≤ 2024-12-31, v6 walk-forward ratings):
1. Winner trailing-rating quartile (quartiles over train map-wins of r_winner
   as-of match date) × share of CAP (13-0) results; secondary: near-cap
   (margin ≥ 11), OT share, full loser-round distribution by quartile.
2. Residuals: per-map realized y = t(margin) winner-referenced vs predicted
   ŷ = a·(r_w − r_l), a fit train-only by OLS through origin on all train
   maps. Bin by ŷ decile; report mean(y − ŷ) per decile with iid 95% CI
   (diagnostic only).
- Predicted sign/size: cap-share(Q4) / cap-share(Q2∪Q3) ≈ 2.5–4×; top-ŷ-decile
  mean residual negative, −1 to −3 target pts.
- PREMISE PRONGS: P1 = cap-share(Q4) ≥ 2× cap-share(Q2∪Q3);
  P2 = top-decile mean residual < 0 with 95% CI excluding 0.
- GATE (pre-committed): both prongs fail → E1 (Tobit) is SKIPPED, verdict
  "PREMISE WEAK"; E2 still runs (independent power-play motivation). One or
  both prongs hold → E1 and E2 both run.
- Falsifier of H1's premise: cap-share ratio < 1.5× and flat/positive top
  residuals.

## E1 — Censored/Tobit likelihood (iteratively-reweighted rd_custom; CHOSEN
over a scratch solver — declared here per brief)

Mechanism: CAP games are right-censored (latent ≥ c13); OT games are their
own censoring class, latent ∈ (0, c_ot] (winner won, but the map reaching
12-12 bounds the latent edge at t(2)); REG-EXACT games keep t(m) exactly.
Fit: global EM. σ fixed = std of (t(m)·sign − (r_w − r_l)) over train
REG-EXACT games under the v6 baseline solve (no per-iteration σ update;
sensitivity runs at σ×0.8 and σ×1.25, secondary). Iteration k imputes game
i's target with μ_i from iteration k−1's daily ratings as-of game i's own
date (walk-forward safe; game i never enters its own μ):
  CAP: E[z | z ≥ c13, μ, σ] = μ + σφ(α)/(1−Φ(α)), α=(c13−μ)/σ
  OT:  E[z | 0 < z ≤ c_ot, μ, σ] (two-sided truncated normal mean)
  REG-EXACT: t(m) unchanged.
Iteration 0 = v6 baseline. K = 4 EM iterations (report max|Δtarget| per
iteration; converged if < 0.05 target pts). Each iteration = engine.run with
rd_custom, β refit train-only per config (engine default). Judged config =
iteration K.
- Predicted sign/size: holdout ΔLL vs v6 +0.3 to +1.5 milli (likely INSIDE
  within-family noise floor 1.77m); elite five (T1, PRX, 100T, NRG, TL) mean
  bias moves toward 0 by +1 to +3 prob-pts; TS/JDG/TE/C9 do not inflate
  (their |bias| does not grow by > 1 pt); max|bias| not increased.
- Falsifier: ΔLL ≤ −1.77m (kill), or elite five move AWAY from 0, or the
  mechanism "works" only by inflating the weak quartet.

## E2 — Round-level Bradley-Terry (joint fit with preregistered fallback)

Mechanism: rounds are the uncensored observable — a 13-0 is 13 Bernoulli
wins; dominance keeps accruing evidence past the map result.
Model: P(round win for i vs j | side) = σ(s_i − s_j + h·side_i), h = global
attack advantage, fit per day from round cells only. Covered maps → two
binomial cells (A-attack-vs-B-defense, A-defense-vs-B-attack; OT rounds fall
into their recorded sides). FALLBACK (preregistered): any map WITHOUT round
rows (all 26 uncovered events + gaps) enters the SAME joint fit as one
map-level Bernoulli term P_map(Δs) = race(p̄), p̄ = ½[σ(Δs+h)+σ(Δs−h)],
race(p) = P(Bin(24,p) ≥ 13) + P(Bin(24,p)=12)·p. Granularity: per MAP, not
per match. Exact per-event coverage table published in h1_roundbt.json.
Weights: the SAME per-day per-game weights as v6 (consistency decay, champ
mult, playoff 1.6, year-roster continuity) — extracted by a verbatim copy of
engine.run's weight block, validated by the Massey parity gate above.
Priors: ridge λ toward 0 + region ridge 3λ toward trailing region mean
(v6's 0.5/1.5 ratio preserved); λ fit train-only on a grid
{0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0} (round-logit scale differs from
Massey's) by walk-forward-within-train series LL, β refit per λ.
Solver: damped Newton (Fisher scoring) per day, warm-started from previous
day. Prediction: rdiff := Δs at match day; engine's closed-form series
pipeline; β refit train-only.
- Effective-sample claim (MEASURED, not asserted): at ref date 2025-01-01,
  fit (A) the joint round model and (B) an all-map-level model (same scale,
  same weights/priors' likelihood; SEs from likelihood-only Fisher, gauge
  fixed by pinning the most-played team, 1e-6 jitter): k_eff = median over
  teams (≥ 25 train series) of (SE_B/SE_A)². Check: cluster bootstrap by
  MATCH, B=200 (rng = np.random.default_rng([20260728, 774411]) — derived
  from crn bootstrap seed, documented here; reduced only if runtime > 30 min,
  actual B reported). Also report mean per-map Fisher info ratio.
- Predicted sign/size: holdout ΔLL −2 to +2m (cross-family MDE 5.89m —
  expect INSIDE NOISE FLOOR); elite five toward 0 by +1 to +4 pts; k_eff
  predicted 2.5–4.5 (binomial info ≈ 24·p(1−p) per map vs map-Bernoulli
  ≈ race'²/P(1−P): back-of-envelope ~3-4×, NOT the claimed ~10×).
- Falsifiers: ΔLL ≤ −5.89m (kill); "order of magnitude" claim falsified if
  k_eff < 7; mechanism unsupported if bias table does not move elite teams
  toward 0.
- MDE translation (first-order, labeled): a round-level referee on the
  covered subset would enjoy MDE ≈ series-MDE/√k_eff, stated with the caveat
  that fit-side information ≠ evaluation-side variance reduction.

## Outputs
stats/h1_censor_diag.json, stats/h1_tobit.json, stats/h1_roundbt.json,
stats/h1_bias_caterpillar.json, phase5_h1.md, logs/bias_h1.log,
scratch/bias_h1/. Outcomes appended below AFTER runs, same resolution for
failures as successes.

---
# OUTCOMES (appended after runs, 2026-07-28 22:03–22:14)

## E3 — PREMISE DEAD (both prongs failed)
- P1 FAIL: cap-share Q4 0.19% vs mid 0.19%, ratio **1.00** (predicted 2.5–4×;
  falsifier <1.5 fired). Corpus-wide: 12/5140 maps at 13-0 (0.23%); margin
  density falls ~2×/step into the bound (222/137/68/12 at m=10..13) — no
  boundary pile-up. Near-cap ratio Q4/mid 1.45. n=2057 train maps (34 skipped
  no as-of rating, 0 junk).
- P2 FAIL — and the operationalization was flawed as designed: winner-
  referenced y has a positive floor (t(2)=4.2), so per-decile residuals are
  positive everywhere (top decile +5.19 [+4.71, +5.67]); "flattening below
  the line" cannot appear in this construction. Failure reported at full
  resolution; the P1 density shape carries the premise verdict independently.
- Gate: SKIP_E1 fired. DEVIATION (run-more, logged 22:0x): E1 executed anyway
  as a confirmatory kill — runs cost 2.2 s (prereg assumed expensive) and the
  OT half (11.2% of maps) was untested by P1. Premise verdict unaffected.

## E1 — KILL (prediction wrong on both sign and mechanism)
- Predicted +0.3..+1.5m, elite +1..+3 pts toward 0. Measured: **ΔLL −0.462m**
  (iid [−1.27, +0.37], block [−1.38, +0.42], P(better) 0.147; within-family
  MDE 1.77m → INSIDE NOISE FLOOR, negative sign). ROI translation 0.000.
- Elite five: T1 −2.7→−2.3, PRX −10.2→−10.1, 100T −7.1→−7.4, NRG −9.1→−9.3,
  TL −6.2→−6.6 — no move toward 0. Weak quartet: TS +14.8→+14.9 (max|bias|
  0.1478→0.1492 — falsifier "elite unmoved + max bias up" satisfied).
- σ̂=4.566; sensitivity σ×0.8/×1.25: −0.470/−0.451m (flat). EM converged
  iter 2–3 (max|Δtarget| 0.047/0.003/0.001).

## E2 — TIE on LL; effective-sample claim FALSIFIED
- Coverage exact: 4349/5140 maps covered (84.6%), 791 fallback (all 26
  corpus-addition events + 2026_ewc_qual gaps), 2 score-mismatch demotions
  (<2% abort bar). Parity gate: weight replication rdiff gap 0.0e0.
- DEVIATIONS (logged): (1) fit_beta bounds widened (0.03,40) for the round-
  logit scale — the engine's (0.03,0.6) clipped β at the bound and corrupted
  the first λ grid; refit before any holdout consumption. (2) λ grid extended
  {4,8,16,32} after the preregistered edge (2.0) was the running best; λ=2.0
  then interior-optimal and chosen (train-only throughout).
- Predicted ΔLL −2..+2m: measured **−0.222m** (iid [−2.62, +2.11], P(better)
  0.44; cross-family MDE 5.89m → INSIDE NOISE FLOOR). ROI 0.000. β=1.73,
  h=+0.035. Bias prediction WRONG: elite five slightly worse (PRX −10.2→
  −10.9), max|bias| 0.1515.
- k_eff predicted 2.5–4.5 (claim ~10): measured Fisher-median **1.25**
  (33 teams, 1.00–1.37), per-map info ratio 1.44, **cluster-boot 0.80**
  (B=200 as preregistered, rng [20260728, 774411]). Falsifier k<7 FIRED —
  order-of-magnitude claim dead; even my own 2.5–4.5 prediction was high
  (round correlation within matches erases the nominal binomial info).
  Round-referee MDE (first-order, covered subset): 1.98m within / 6.59m
  cross — no gain.

Verdicts: E3 premise DEAD · E1 DEAD (inside floor, wrong direction, bias
unhelped) · E2 INSIDE NOISE FLOOR as a candidate, and its power-play
rationale is REFUTED by measurement. Full prose: phase5_h1.md.
