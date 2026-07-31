

════════ preregister.autopsy.md ════════

# Pre-registration — agent:autopsy (Phase 7 live P&L autopsy)

Written 2026-07-28, BEFORE any P&L/calibration/markout computation. Inputs read so
far: brief, README, FINDINGS.md, quote_margin.json, trade_sim.json, incident memo,
local VCTMM config.toml + engine/quotes code (mechanics only), local db schema.
No fill rows have been aggregated yet.

## Question
Why is the live bot down money on real (dry_run=0) fills since July 2026:
model, config, execution (adverse selection), fees, or noise?

## Decomposition identity (fixed in advance)

Per event, real fills only (dry_run=0), NO-side prices in cents:

    realized P&L ≡ locked-pair margin + unhedged SETTLED P&L − fees
    total P&L    = realized P&L + open-inventory mark (reported SEPARATELY)

- Locked pairs: NO on both markets of an event pays 100¢/pair. Pairs = FIFO
  match of side-1 and side-2 fill qty per event (cross-check vs lots +
  deploy_index.locked_pairs/locked_profit_dollars; discrepancies REPORTED,
  never smoothed). Margin/pair = 100 − p1 − p2.
- Unhedged settled: unpaired contracts on settled markets; NO pays 100 iff
  the market's team lost. Settlement source: db market status/result, else
  Kalshi public /markets result, else VLR outcome via match_links.
- Open inventory: unsettled markets, marked at last trade (tape; else public
  trades API; else fill price, flagged). Mark P&L is NEVER mixed into realized.
- Fees: per-contract Kalshi fee (schedule verified from public API/docs this
  run; expected: taker ceil(0.07·P·(1−P)) per contract, maker 0 — VERIFY).
  Maker/taker per fill decided by trade_id join to tape.taker_side vs our
  resting side (bot posts limit orders ⇒ expected maker).
- Identity audit: components must sum to independently computed cash P&L
  (Σ settlements + Σ marks − Σ costs − fees). Residual > $1 ⇒ investigate,
  report as its own waterfall line.

## Adverse-selection test (fixed in advance)

1. Fill-conditional calibration: per real fill, model q = P(NO pays) =
   1 − p_model(team wins), at stated vintage. Contract-weighted reliability
   (Wilson CIs) of realized NO-pay rate vs mean q, on settled fills.
2. Unconditional benchmark: same model, ALL VCT Kalshi markets settled in the
   fill window (one obs per market side, and price-band matched view).
3. **Adverse-selection number** = (realized − predicted on fills) −
   (realized − predicted unconditional), in probability points; also stated
   as ¢/contract. Negative ⇒ fills are adversely selected.
4. Markout decomposition per fill from tape/public prints, NO terms
   (NO_mid = 100 − YES_mid): maker P&L = spread capture (mid@fill − price) +
   adverse move (mid@T − mid@fill), T ∈ {+5m, +30m, +2h, start−5m}. Slice by
   side_role, price band, minutes-to-start. Sparse tape ⇒ quantify the gap
   (coverage %), do not interpolate silently.

## Prediction vintage (stated per fill)
Primary: frozen v6 `trading_model/model_snapshot.json` via `predict.py`
(reference math). Fills while the VM served a drifted rebuild (2026-07-23
13:07 UTC → the post-sync re-enable; exact window from VM audit_log
model_rebuild entries) are bracketed: frozen v6 AND price-implied (tape mid
at fill). Any fill whose event can't be priced by v6 (missing org mapping)
is excluded and counted loudly.

## Blame rules (fixed in advance)
- **MODEL**: unconditional v6 calibration in the window is off (CI excludes 0)
  in the same direction as fill losses — the model is wrong everywhere, not
  just where it got filled.
- **CONFIG**: unconditional calibration fine, but losses concentrate in
  pockets research already flagged: fills that a logit +0.5/+0.6 cap would
  have refused, NO quotes on model-p<45% sides (rule confirmed ABSENT from
  live code), fills inside the expiry window, hedge margin not clearing the
  fee stack. Realized ROI inside flat-5¢ sim CI [−9.2%, +22.8%] but below
  logit+0.6 counterfactual ⇒ config gap, not model failure.
- **EXECUTION (adverse selection)**: adverse-selection number significantly
  negative AND markout adverse move exceeds spread capture; unconditional
  calibration fine.
- **FEES**: fee line ≥ 50% of gross loss, or 2¢ hedge margin − fees/pair ≤ 0.
- **NOISE**: bootstrap per-fill P&L under the flat-5¢ sim point edge
  (+6.7% ROI): p = P(cum P&L ≤ observed). p ≥ 0.10 ⇒ not distinguishable
  from noise; p < 0.05 ⇒ real underperformance; else weak evidence.
  Blame is a waterfall, not a single label; dollars per bucket.

## CRN
README rule 3: bootstrap randomness from v8/crn.json. Not yet present at
prereg time (power agent writes it). At variance-check time: re-check; if
still absent, use documented fallback seed 780728 and FLAG the deviation in
autopsy_variance.json + phase7_autopsy.md.

## Outputs
quote_density.json (early, for agent:referee: fill+quote density by NO price
band × side_role, real fills), autopsy_pnl/fees/fill_calib/markouts/
config_gap/variance JSONs, phase7_autopsy.md, snapshot db, log.


════════ preregister.bias_h1.md ════════

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


════════ preregister.bias_h2.md ════════

# preregister.bias_h2 — Phase 5 H2: schedule connectivity

agent:bias-h2, written 2026-07-28 BEFORE any engine run. Frame:
v8/data/frame_expanded/series.csv (sha256 verified against crn.json:
ff772d41…55142, n=2058, 841 train / 1217 holdout, holdout = date > 2024-12-31).
Baseline "v6" engine config (recovered from trading_model/model_snapshot.json +
run_floorfix.py BASE): decay games consistency-conditioned (20,12),
rd |rd|^0.75×2.5, playoff w_custom ×1.6 (stage from the frame; non-frame games
→ 1.0), region_prior_ridge 1.5, roster year/0.3, ridge 0.5, champ_mult 2.0,
β fit on train only (engine bounds 0.03–0.6). `eng._prev_rvec` reset to None
before every run (engine leaves it set across run() calls; reset makes each
config self-contained). MDE quotes: within-family 1.77m, cross-family 5.89m
(stats/power_mde_expanded.json, n=1217). All ΔLL also reported as
referee.expected_roi_of_dll. Holdout scorings: exactly one per published
variant (v6 baseline reference; E2 train-chosen config; E3 hierarchical).
engine.run emits ll_test on every call as a side effect; selection reads ONLY
internal-split numbers (selection code has no access to holdout metrics).

## E1 — Connectivity diagnostic (gates E2/E3)

Mechanism: regional pools are near-disconnected graph components joined mainly
at internationals; cross-pool rating comparisons are weakly identified for
low-connectivity teams, and the fixed region-prior ridge (1.5) pulls
within-region extremes toward the region mean. If H2 is real, per-team holdout
bias should track opponent-graph position.

Features — per team, walk-forward at each holdout month boundary T (first of
month, 2025-01 … 2026-07), from games dated < T in the expanded games list
(engine registry):
- opp_count: distinct opponents over all games < T.
- opp_diversity: distinct opponents / games played (< T).
- eig_centrality: principal eigenvector (np.linalg.eigh, abs, max-normalized)
  of the symmetric opponent graph with edge weight = Σ exp(−ln2·age_weeks/26)
  over games between the pair (calendar HL 26 weeks at T). Teams outside the
  principal component get ≈0 = maximally peripheral (semantically intended).
- xregion_share: decayed-weight share (same HL-26 weights) of games vs
  known-region opponents whose region differs (ORG_REGIONS map; unknown-region
  opponents excluded from numerator and denominator).
Per-team aggregate = mean over holdout months weighted by the team's frame
holdout series count in that month.

Bias: referee.per_team_bias (min_n=25) on the v6 holdout predictions from the
expanded frame (probability points; negative = under-rated).

Statistics, over bias-table teams (expect ≈42; small-N caveat applies):
Pearson r and Spearman rho of (a) signed bias and (b) |bias| against each
feature. CIs: percentile 2.5/97.5 from 4000 team-resamples, numpy PCG64 seeded
with crn.json bootstrap seed 20260728, full index matrix drawn in one call —
documented adaptation: crn's stored matrices resample holdout SERIES; team
resampling reuses the crn seed/generator/n_boot so there is no private seed.
Degenerate resamples (zero variance) dropped and counted.
Secondary (does NOT gate): partial Spearman of bias vs eig_centrality
controlling for mean v6 rating (walk-forward daily ratings at the last solved
day < each month boundary, same month weights) — separates "connectivity per
se" from "strength → qualifies for internationals" proxying.

Predicted signs/sizes:
- Spearman(|bias|, eig_centrality) NEGATIVE, ≈ −0.35.
- Spearman(bias, eig_centrality) NEGATIVE, ≈ −0.30 (insular teams over-rated:
  TS/JDG/TE floor over-rated and domestic-only; traveling elite under-rated).
- |bias| vs opp_count / opp_diversity / xregion_share NEGATIVE, −0.2 … −0.35.

GATE (falsifier of H2): H2 is DEAD unless at least one of
{Spearman(|bias|, eig_centrality), Spearman(bias, eig_centrality)} has a 95%
bootstrap CI excluding 0 AND negative sign. If dead: publish the correlation +
CI in stats/h2_centrality.json + phase5_h2.md and STOP (no E2/E3 model fits).

Corpus-expansion sub-analysis (always published): eig_centrality at
T=2026-07-01 on the full expanded graph vs the pre-expansion graph (games of
the 25 corpus-addition event_ids in stats/power_mde_expanded.json "new_events"
removed); per named team (elite T1 PRX 100T NRG TL; floor TS JDG TE C9):
centrality and centrality-rank before/after, plus mean |Δcentrality| over
bias-table teams. Prediction: additions raise centrality most for EWC/off-season
participants (several elite + CN teams), by ≥ +0.05 normalized units.

## E2 — Ridge ablation (runs only if E1 gate passes)

Mechanism: the fixed region-prior ridge causes elite compression; weakening it
should decompress the elite tail without wrecking thin-data teams.

Grid: region_prior_ridge ∈ {0, 0.75, 1.5} × plain ridge ∈ {0.25, 0.5, 1.0}
(9 configs, all else v6). Internal walk-forward split INSIDE train:
β fit per config on valid rows with date ≤ 2024-06-30, scored on
2024-06-30 < date ≤ 2024-12-31 (internal-val; report n). Selection = lowest
internal-val mean LL; ties within 0.0005 broken toward v6 (fewer changed
constants). Train-chosen config scored ONCE on holdout with engine-native
full-train β refit. If the chosen config IS v6, the baseline scoring is
reused (no extra holdout exposure).

Predicted: internal-val ΔLL across rpr values small (|Δ| < 3m); decompression
story predicts rpr=0 moves elite bias toward 0 by 1–3 prob-pts vs rpr=1.5
(measured on the internal-val bias tables — train-side — for all 9 configs,
and on holdout only for the chosen config).
Falsifier of the compression story: rpr=0 elite bias moves < 1 prob-pt, OR
decompression only at the cost of thin-data blowup (cold-start bucket ΔLL
worse and mean|bias| of 5 ≤ n < 25 teams up by > 2 prob-pts).

Report: 9-config internal table; chosen config holdout LL vs v6 with CRN iid +
block_event CIs, Δmilli vs 1.77m MDE, expected-ROI; per_team_bias caterpillar
v6 vs chosen; max|bias|; named-team movement; cold-start bucket
(referee.bucketed with r_w/r_l columns); thin-data summary (5 ≤ n < 25).

## E3 — Hierarchical partial pooling (runs only if E1 gate passes)

Mechanism: fixed-strength shrinkage over-pulls when true within-region spread
is large; empirical-Bayes region random effects adapt τ per region per day.

Implementation (preregistered choice: engine ridge machinery, subclassed):
scratch/bias_h2/eb_engine.py subclasses testing_lab/engine.Engine; run()
copied verbatim with ONLY the region-prior block replaced (marked EB-BEGIN /
EB-END). Per solve day:
1. Stage 0: solve with plain ridge 0.5, NO region prior → r0; game-noise
   σ̂²_g = Σ w·(rd_t − Δr0)² / Σ w (transformed-margin residual, solve weights).
2. Init per region with ≥4 rated teams (rated = appears in a game < day):
   μ_reg = mean(r0[reg]), τ²_reg = max(var(r0[reg], ddof=1), τ²_floor=1.0).
3. Two EB iterations: per-team λ_t = σ̂²_g / τ²_reg(t) (λ=0 for unknown-region
   or <4-team regions); solve (M + ridge·I + diag(λ)) r = p + λ·μ with the
   engine's sum-to-zero constraint row applied last, as in engine.run;
   posterior var v_t ≈ σ̂²_g / (diag(M) + λ_t); update μ_reg = mean(r[reg]),
   τ²_reg = max(Σ(r−μ)²/(n−1) + mean(v[reg]), 1.0).
4. Final ratings = last solve. σ̂²_g stays at its stage-0 value.
No global constants fit anywhere except the preregistered τ²_floor=1.0,
min-teams=4, iterations=2. β engine-native train-only. Verification before
scoring: with the EB block disabled the subclass must reproduce
Engine.run(rpr=0, ridge=0.5) rdiff bit-exact.

Predicted: holdout ΔLL vs v6 in [−1, +2] milli (likely INSIDE NOISE FLOOR at
1.77m); if H2 real, max|bias| drops ≥ 2 prob-pts and TS bias +16.3 → ≤ +12
(old-frame reference values; expanded-frame v6 table is the operative
baseline). Falsifier: max|bias| within 1 prob-pt of v6, or cold-start bucket
ΔLL worse with no bias gain. Judge: holdout LL (once), bias caterpillar,
max|bias|, cold-start bucket; fitted τ_reg trajectories published.

## Outputs
stats/h2_centrality.json, h2_ridge_ablation.json, h2_hierarchical.json,
h2_bias_caterpillar.json; phase5_h2.md; logs/bias_h2.log; scratch/bias_h2/.
Outcomes appended below AFTER runs, same resolution either direction.

---

## OUTCOMES (appended 2026-07-28, after runs)

### E1 — MEASURED. GATE FAILED → H2 DEAD.
v6 baseline (expanded frame): β=0.1152, holdout LL 0.64216 (n=1217); bias
table 43 teams, max|bias| 0.1478 (TS), mean|bias| 0.0473.
Predicted Spearman(bias, eig) ≈ −0.30: measured −0.304 — sign and size RIGHT,
but CI [−0.566, +0.005] includes 0 → gate clause not met.
Predicted Spearman(|bias|, eig) ≈ −0.35: measured −0.077 [−0.421, +0.281] —
WRONG (null). The error-magnitude prediction, the mechanistic core of H2,
failed cleanly.
Non-gating rows: bias~opp_count −0.314 [−0.575, −0.007], bias~xregion_share
−0.431 [−0.640, −0.157] (CIs exclude 0) — attributed to composition (elite
travel, floor stays home): partial Spearman bias~eig | rating −0.188
[−0.491, +0.161]; per-region mean signed bias ≤ 2.7 pts; PRX (highest named
centrality 0.894) most under-rated (−10.2), TE/JDG central yet over-rated,
C9/FUR over-rated inside the well-connected Americas pool. |bias| rows all
null (opp_count −0.043, diversity −0.010, xshare +0.052).
Expansion sub-analysis: mean |Δeig| 0.099 over bias-table teams at
2026-07-01; prediction "≥ +0.05 for EWC/off-season participants" PARTIALLY
right — CN participants rose (JDG +0.162, TE +0.181) but max-normalization
pushed Western elite DOWN (NRG −0.181, 100T −0.109, T1 −0.077): the additions
re-centered the graph toward the CN/EWC cluster rather than lifting all
participants. Graph moved materially; bias-centrality coupling still absent.

### E2 — NOT RUN (preregistered gate stop). stats/h2_ridge_ablation.json is a
NOT-RUN marker.

### E3 — NOT RUN (preregistered gate stop). stats/h2_hierarchical.json is a
NOT-RUN marker. eb_engine.py never written.

Holdout scorings used: 1 (v6 baseline). Verdict: phase5_h2.md — H2 DEAD.


════════ preregister.bias_h3.md ════════

# Pre-registration — agent:bias-h3 (written 2026-07-28, BEFORE any experiment run)

Scope: Phase 5 H3 state-space rating (Kalman analog, match probability integrates
over rating variance) + 5d heterogeneous half-lives via pooled process noise +
n_eff telemetry + ensemble-where-it-wins. Frame: canonical
`v8/data/frame_expanded/series.csv` (sha256 verified vs crn.json before use:
ff772d41…, n=2058, train 841 = date<=2024-12-31, holdout 1217). All fits train-only;
holdout touched only for final scoring. Randomness: GH quadrature is deterministic;
the single MC cross-check uses crn.json mc_seeds[0]; all bootstraps via
referee.paired_bootstrap_crn. MDEs quoted from stats/power_mde_expanded.json:
within-family 1.773m, cross-family 5.889m.

## Shared machinery (fixed before running)

- **Observation**: per MAP, y = sign(rd)·|rd|^0.75·2.5 (v6's rd transform,
  unchanged), winner-referenced to team A of the pair. One filter update per map,
  in (date_s, match_id) order (engine game order).
- **Filter**: per-team state (r, v). Init r=0, v=V0. Prediction step at each of a
  team's own maps: v += q (games-counted, no calendar accrual — in-family with
  v6's information-replacement decay). Update: e = y − (r_A − r_B),
  S = v_A + v_B + R_i, K_T = v_T/S, r_A += K_A·e, r_B −= K_B·e, v_T −= v_T²/S.
- **Game weights**: R_i = R/w_i with w_i = (2.0 if exact-shape YYYY_champions else
  1.0) × (1.6 if the map's series stage ∈ {playoffs, grand_final} else 1.0) —
  v6's hand-set weights, NOT refit. Stage from the frame; maps of matches not in
  the frame default to weight 1.0 (mirrors run_v7_stage1 stage_by_mid default).
- **Identification**: R fixed := Var(y) over TRAIN games (date<=2024-12-31).
  Fixing R loses no generality (only q/R, V0/R matter up to state scale; β absorbs
  scale). Free params of the core: q, V0. β refit train-only per config
  (bounds 0.03–0.6, minimize_scalar, engine-identical series NLL) — house rule 8.
- **Leak rule**: predictions for day D use the state snapshot after all games with
  date < D (strict day granularity, same as engine m_hist). Same-day games are
  never history for that day's predictions.
- **Match probability**: p_series = ∫ series_wp(sigmoid(β·δ), fmt) N(δ; r_A−r_B,
  v_A+v_B) dδ by 20-node Gauss–Hermite (the SAME δ draw shared across maps of a
  series — rating uncertainty is common across the series, not iid per map).
  bo5_gf uses the plain bo5 closed form (frame carries no bracket side; identical
  treatment for baseline and candidate, so pair deltas are unaffected).
- **Baseline**: v6 = engine cfg {rd 0.75/2.5, roster year/0.3, ridge 0.5,
  champ 2.0, region_prior_ridge 1.5, PO 1.6 w_custom, decay consist(20,12)} run
  on the expanded frame, β refit on train. Judged with referee.py.
- **Fit criterion for (q, V0) and all 5d params**: mean series NLL on TRAIN rows
  only, β refit at every grid point. Log-space grid q/R ∈ [1e-4, 3e-2],
  V0/R ∈ [0.05, 3], coarse 2D grid then one local refinement; every grid point
  journaled to scratch/bias_h3/sweep_*.json (checkpoint; restart skips finished
  points).

## Experiment 1 — state-space core (SS-core)

- Mechanism: point-estimate ratings ignore per-team uncertainty; integrating
  sigmoid over N(Δr, v_A+v_B) shrinks exactly the thin-data predictions toward
  0.5 while leaving established teams sharp — the asymmetry a global β cannot
  express.
- Predicted sign/size: aggregate holdout ΔLL vs v6 positive but small —
  predicted +0 to +3m, i.e. likely INSIDE the cross-family noise floor (5.889m);
  the mechanism should show in the tails: cold-start bucket (old-frame reference
  n=57 / LL 0.70205) improves by ≥10m; per-team bias caterpillar shows reduced
  max|bias| with thin-data teams' bias shrinking and elite teams' bias not
  inflating; favorite-band buckets not degraded.
- Falsifier: cold-start bucket NOT improved, or improved only by shrinking
  everyone (elite/big-gap buckets degrade by more than their bucket noise), or
  max|team bias| increases.
- Variants (each preregistered, β refit, same grid discipline; all train-fit):
  1a. plain core (q, V0) as above.
  1b. + calendar leak: v += q_cal per week elapsed since the team's previous
      game (one extra param; tests whether breaks add real uncertainty).
      Predicted: small train gain, holdout ≈ tie; falsifier: hurts train.
  1c. + debut prior mean: a team's r initialized at its region's trailing mean
      rating of already-rated teams (walk-forward, day before debut) instead
      of 0 (v6 has region_prior_ridge; this is the SS analog). Predicted: helps
      cold-start bucket further; falsifier: no cold-start change.
  Model advanced to judging = best TRAIN NLL among {1a, 1b, 1c}; all three's
  train numbers reported, holdout scored for all three (labelled primary =
  train-selected BEFORE holdout is looked at).
- Judging (referee.py, exactly): holdout LL both units (milli-LL +
  expected_roi_of_dll vs v6 p_ref), paired_bootstrap_crn iid AND block_event,
  cross-family MDE 5.889m quoted, per_team_bias caterpillar (candidate vs v6,
  min_n=25), max|bias|, bucketed() panel incl. cold-start (r_w/r_l attached from
  the v6 engine run; COLD_EPS 5e-4) and EWC-class, favorite bands.
- GH validity check: |GH20 − MC(200k, crn mc_seeds[0])| < 1e-4 on 50 spot
  matches, reported in the stats JSON.

## Experiment 2 — 5d heterogeneous half-lives via pooled process noise

- Mechanism: process noise q IS the memory dial (steady-state gain ⇒ effective
  per-game forgetting (1−K*); HL_games = ln2/−ln(1−K*)). Team types should
  need different q: rebuilt rosters carry more true rating drift than stable
  elites.
- Typing axes (walk-forward observables, all computable at update time; cells
  fixed now):
  - A. roster stability (PRIMARY): at each map of team T, matches_since_change
    (consecutive prior matches with lineup identical to the current one, the
    lineups-agent definition) — cells: change-adjacent msc≤3 / settling
    4≤msc≤10 / stable msc>10. Source: full-corpus recompute of
    lineups/lineup_features into scratch/bias_h3/ (corpus additions lack rows;
    top-up mandated scratch-only). Verification, two modes (pre-run amendment,
    2026-07-28, written before the recompute ran): (i) IMPLEMENTATION check —
    recompute restricted to the lineups agent's own event universe
    (set(lineups.csv.event_id)) must reproduce their matches_since_change /
    games_since_change on ALL overlapping (match, org) rows with 0 mismatches,
    else stop and report; (ii) the full-corpus version used for typing may
    legitimately differ on overlap rows where corpus-addition matches
    interleave an org's history — count and report those rows, never silently.
  - B. org age: days since org's first corpus game at update time — cells
    <180d / 180–540d / >540d. Left-censoring at corpus start (2023) noted as a
    caveat in the output.
  - C. rating volatility: trailing std of the last 12 standardized innovations
    z = e/√S from the FITTED SS-core (train-fit params; walk-forward by
    construction) — cells: train-terciles of the pooled z-std distribution.
- Fit, per axis: q_k free per cell (3 cells), V0 and R and β as in core
  (V0 re-fit jointly? NO — V0 frozen at SS-core's fitted value to keep the
  comparison within-family and the parameter count honest; stated here, before
  fitting). Per-cell MLE q̂_k + curvature/profile SE on TRAIN.
- Partial pooling (the deliverable): random-effects on log q across cells:
  DerSimonian–Laird τ̂² from the K=3 per-axis MLEs and their SEs; pooled
  q̃_k = precision-weighted shrink of log q̂_k toward log q̄. No free-per-team q
  anywhere. Report per cell: q̂, q̃ (pooled), implied HL in games
  (HL = ln2/−ln(1−K*) at R̄=R, w=1), 95% CI from the train profile mapped
  through the HL formula, n_games per cell, τ̂², and the shrink fraction.
- Predicted sign/size: q̂(change-adjacent) > q̂(stable) by ≥2× (HL shorter for
  rebuilt rosters); q̂(young org) > q̂(established); q̂(high vol) > q̂(low vol) —
  the operator's "more recency for some teams" quantified. Holdout ΔLL of
  het-q vs SS-core: predicted +0 to +2m (within-family MDE 1.773m — likely
  INSIDE NOISE FLOOR; the half-life table is the deliverable either way).
- Falsifier: ordering flat or reversed (q̂ ratio < 1.3× between extreme cells),
  or τ̂² ≈ 0 with all cells shrinking to the pooled q (then heterogeneity is
  unsupported at this n — published as such).
- **stats/h3_process_noise.json published as soon as these fits exist** (before
  the rest of the judging suite completes; agent:decay references it).
- Holdout scoring: axis A model (primary), B and C reported as secondary;
  within-family MDE 1.773m quoted on every Δ.

## Experiment 3 — n_eff per prediction

- Emit per frame row (all 2058, holdout flagged): pre-match v_w, v_l,
  σ_Δ = √(v_w+v_l), n_eff_T = R/v_T, harmonic pair n_eff, p_point (β·Δr point
  sigmoid + series form), p_integrated, shrink factor
  (|p_int−0.5|/|p_point−0.5|, 1.0 when p_point=0.5). From the train-selected
  SS-core config. Distribution summaries: quantiles overall / by year / by
  event class (vct, ewc-offseason via EWC_CLASS_PREFIXES + pre-2026 analogs,
  intl) / cold-start rows; correlation of pair n_eff with per-row |ΔLL vs v6|.
  No selection decisions ride on this — telemetry deliverable.

## Experiment 4 — ensemble-where-it-wins

- Question: does SS win only on subsets? Candidate gates (fixed list, chosen on
  TRAIN only): (a) either-team n_eff < θ, θ ∈ {3, 5, 8, 12}; (b) either team
  change-adjacent (msc≤3 at last lineup before the match); (c) event class
  ewc_offseason; (d) soft blend w(x) = σ(a + b·min_neff) fit on train (2 params).
- Selection rule (fixed now): the gate/blend with best TRAIN mean NLL of the
  composite (SS inside gate, v6 outside) wins; ONE composite is then scored
  ONCE on holdout with full referee judging vs v6 AND vs SS-core. Predicted:
  composite beats v6 by more than SS-core does, driven by cold-start/EWC rows.
  Falsifier: no gate's TRAIN composite beats v6's train NLL → "no subset where
  SS pays for itself"; reported plainly, holdout still scored for the
  pre-named primary gate (a, θ=5) for completeness, labelled exploratory.
- MDE context: composite vs v6 is cross-family (5.889m); composite vs SS-core
  within-family (1.773m).

## Outputs (mine alone)

stats/h3_statespace.json, stats/h3_process_noise.json (EARLY),
stats/h3_neff.json, stats/h3_bias_caterpillar.json, stats/h3_ensemble.json,
phase5_h3.md, logs/bias_h3.log, scratch/bias_h3/. Outcomes appended below
AFTER runs, same resolution for failures as successes.

## Not done here

No holdout fitting of anything. No writes outside declared paths. No market
data anywhere in fitting or selection (README rule 9). No network.

---

# OUTCOMES (appended after runs, 2026-07-28; same resolution for failures)

**Exp 1 (SS core).** Predicted +0..+3m vs v6 ⇒ WRONG on aggregate: plain core
−8.87m [LOSS, cross-MDE 5.889m], train-selected primary 1b (+q_cal) −11.75m
[LOSS] — 1b's train win (0.64485, best) was regime overfit; falsifier partially
tripped: cold-start DID improve massively (below) but not "without cost
elsewhere": established buckets degrade 2-12m, and mean|bias| worsens
0.0473→0.052 while max|bias| improves 0.1478→0.1255. Mechanism-in-tails
CONFIRMED: debut +58m, <10 maps +71.7m (n=39), <30 maps +28m (n=117, 5d model);
thin-half sharpness −0.61pp vs established +0.10pp, corr +0.40. GH validation
passed (GH20≡GH40 machine epsilon; MC within 3SE). COLD_EPS bucket empty on
engine ratings (region prior) — prior-map-count definitions substituted,
documented in-artifact.

**Exp 2 (5d).** Roster axis: predicted q(change)>q(stable) ≥2× ⇒ CONFIRMED in
point estimates (11×; HL 7.4 vs 24.8 games; settling cell at q→0 boundary =
non-monotone, not predicted). DL pooling: τ̂²=0 all axes ⇒ the preregistered
"unsupported at this n" clause is the formal verdict; published both. Holdout:
5d vs plain core +3.55m [within-family WIN, MDE 1.773m, p_better 0.946/0.979];
5d vs v6 −5.32m [INSIDE NOISE FLOOR]. Org-age: REVERSED (falsified;
left-censoring caveat as preregistered). Volatility: flat (falsified).
h3_process_noise.json published before the judging suite, as required.

**Exp 3 (n_eff).** Emitted for all 2058 rows from the train-selected core (1b)
+ 5d secondary summaries. corr(pair n_eff, |ΔLL vs v6|) = −0.09.

**Exp 4 (ensemble).** Several composites beat v6 on train (so the "no subset
pays" falsifier did NOT trip); train-selected winner = soft n_eff blend on 1b;
holdout: −12.15m vs v6 [LOSS], −3.28m vs plain core [LOSS] ⇒ the preregistered
selection produced a non-generalizing composite; published as-is, no holdout
shopping for the non-selected hard gates (their subset behavior is already
visible in the preregistered cold buckets). Promotion answer: HOLD.

Deviations from plan: (1) cold-start bucket definition substituted
(COLD_EPS empty on engine ratings — documented in h3_statespace.json);
(2) GH-vs-MC bar restated as GH20-vs-GH40 convergence (original 1e-4 bar was
tighter than MC noise at 200k; both checks pass and are recorded);
(3) exp-4 search space explicitized as gates × {1b, 5d} with selection still
purely on train (documented in h3_ensemble.json). No holdout was used for any
selection anywhere.


════════ preregister.bias_h4.md ════════

# Preregistration — agent:bias-h4 (Phase 5 H4: series aggregation assumes iid maps)

Written 2026-07-28, BEFORE any experiment ran. Frame:
`testing_lab/v8/data/frame_expanded/series.csv`, sha256
`ff772d417b714844aa1b11426d8c04917e1b2848c9c8df811cd9988694d55142` — verified
against crn.json `frame_expanded` at T0 (logs/bias_h4.log). Holdout = date >
2024-12-31 (n=1217); nothing is fit on holdout rows. All resampling from
crn.json: referee.paired_bootstrap_crn (iid + block_event) for holdout deltas;
train-side diagnostic bootstraps use `mc_seeds[0]` (cell dispersion CIs, one
full index matrix per cell in the documented cell order) and `mc_seeds[1]`
(dispersion-parameter refit CIs). No private seeds.

## Scope line (the momentum distinction)
H4 is NOT the rejected intra-series momentum idea (ledger id 27,
`g16_k20_corr07`, sequential map-to-map carryover inside the veto MC, −2.77m,
inside the cross MDE). H4 posits an *exchangeable* series-level random effect
set before map 1 and constant across the series (matchup/day/prep draw): it
changes the marginal p → P(series) aggregation curve via the map-count
dispersion, never feeds one map's outcome into another map's probability, and
is fit on train scores in closed form (quadrature), not hardcoded into an MC.

## v6 reconstruction (shared input to all experiments)
One engine solve, `Engine()` on the expanded game set; `eng.series` replaced by
the frame, `eng.pred_days` from it. Config = the v6 champion exactly as raced
in run_v7_stage1.py: decay `games consistency(20,12)`; rd pow 0.75 scale 2.5;
roster year/0.3; ridge 0.5; champ_mult 2.0 (exact-shape); region_prior_ridge
1.5; w_custom = 1.6 on playoffs/grand_final (stage from the frame, default
groups). β fit on train only (engine default). p_map = sigmoid(β·rdiff),
winner-referenced. Dispersion acts ONLY downstream of this single p (series
aggregation layer); ratings and per-map surfaces are untouched.

## E1 — dispersion diagnostic (TRAIN rows only, fmt ≠ bo1, valid rdiff)
Observable: the final-score distribution under the stopping rule, conditional
on the series winner — insensitive to first-order rating miscalibration.
Implied under iid at winner map-prob q: bo3 P(2-0|win)=1/(3−2q); bo5
P(3-j|win) ∝ {1, 3(1−q), 6(1−q)²}. Primary signed index per cell:
`D_sweep = obs share(l_maps=0) − mean implied P(sweep|win)` (prob points;
positive = over-dispersed, negative = under-dispersed). Secondary:
obs vs implied mean l_maps. CIs: CRN bootstrap (mc_seeds[0], n_boot 4000).
ML dispersion fits on the conditional score likelihood (q frozen from v6):
(a) shared effect u~N(0,σ_u), logit-additive, GH quadrature (31 nodes), σ_u≥0
— over-dispersion arm; (b) pick-spread h≥0: alternating per-map logit offsets
(bo3 [+h,−h,0], bo5 [+h,−h,+h,−h,0]), averaged over both start assignments,
independent maps — under-dispersion arm (veto-structure story; uses no map
identities, no per-map ratings — ledger id 25 stays dead). Parameter CIs via
mc_seeds[1] bootstrap refits (n_boot 2000).
Cells: overall; by format; favorite bands p_fav ∈ [0.5,0.7), [0.7,1.0] (fav =
p≥0.5 side, ties excluded from fav cells); depth terciles (train cutpoints);
depth tercile × strong favorites (p_fav ≥ 0.7) — the H4 cell C*.
Depth feature (walk-forward): favorite team's distinct real maps played in
officials in trailing 90d, intersected with the veto-era pool = distinct real
maps in map_vetos.csv steps (junk map strings filtered; joined to dates via
match_dates.json; only vetos dated < series date) within trailing 60d
(fallback 120d if <5 maps), divided by pool size (`depth_frac`). Map-level
inputs come from the engine's game list (production loader; I verify no
MapNum=="all" aggregate leaks into it and log the check).

**Prediction (mechanism, sign, size):** shared matchup/day effects are real →
mild global OVER-dispersion, D_sweep(bo3, train) ≈ +1 to +3 pp; depth gradient
≈ 0 (depth mostly proxies team quality, not outcome variance); C* NOT
under-dispersed. **H4-premise falsifier (gate G1):** C* (deep-tercile strong
favorites) shows D_sweep < 0 with 95% CI excluding 0 → premise confirmed. If
G1 fails, H4-as-stated (iid understates elite series probability) is reported
weak/refuted.

## E2 — dispersion-parameterized series link (gated deliverable)
Always run (Done-when requires the correlated link scored on holdout): L1
global σ_u. Gates: G2 (any exploitable dispersion: overall or per-format
D_sweep CI excluding 0, or σ̂_u/ĥ CI excluding 0, or tercile-1 vs tercile-3
D_sweep difference CI excluding 0) → also run L2 σ_u(depth) =
softplus(a + b·z_depth), z train-frozen, favorite identity from v6-p (frozen
covariate, walk-forward safe). G1 → also run L3 global h and L4 h(depth) as
the under-dispersion links. If neither gate fires, L1 still runs and its
(expectedly null) holdout score is the published answer; L2–L4 are skipped and
stubbed as such.
Fitting protocol per link: dispersion params by ML on TRAIN conditional score
likelihood (stage A, as in E1); β refit on TRAIN series win/loss LL with
dispersion frozen (stage B; β is scale-bound per config). Nothing touches
holdout. Judge on holdout n=1217: overall ΔLL vs v6 (referee.delta_vector →
paired_bootstrap_crn iid + block_event), referee.bucketed (favorite bands, GF
= bo5_gf/grand_final buckets, all standard buckets), referee.per_team_bias
caterpillar (probability points), mean ΔP(series win) for strong favorites
(p_fav≥0.7) split by format — "does P(bo5 win) for strong favorites rise?" —
answered with its sign and size. Both units everywhere:
milli-LL + referee.expected_roi_of_dll (reporting only). MDE context: these
are probability-layer transforms on an unchanged solve → within-family MDE
**1.77m** (stats/power_mde_expanded.json; ledger id 10 precedent for the
regime call). |Δ| < MDE is published as INSIDE NOISE FLOOR.
**Prediction:** σ̂_u ∈ [0.2, 0.6] (train scores), but after β refit L1 holdout
ΔLL ∈ (−1, +1) milli → INSIDE NOISE FLOOR; L2 likewise; P(bo5) for strong
favorites moves DOWN (σ_u>0 shrinks favorites), i.e. against H4's hoped
direction, unless G1 fired and an h-link carries it up. **Falsifier of my
null:** any link clearing +1.77m with CI excluding 0.

## E3 — interaction guard
Numerical check: every link reduces to the v6 closed form at σ_u=0 / h=0 (max
|ΔP| < 1e-9 on a p×fmt grid, reported). Statement check: link inputs are
exactly (β·rdiff, fmt, z_depth-of-favorite as σ/h covariate) — no per-map
ratings, no pick bonus, no map identities; the ledger's per-map+pick kill
(id 25) stays dead; dispersion acts only between the single p and P(series).

## Outputs
stats/h4_dispersion_diag.json, stats/h4_series_link.json,
stats/h4_bias_caterpillar.json, phase5_h4.md, logs/bias_h4.log,
scratch/bias_h4/. Outcomes appended below after the runs, failures at the
same resolution as successes.

---

## OUTCOMES (appended 2026-07-28 after runs; same resolution for failures)

**E1.** Predicted mild global over-dispersion +1..+3 pp: CONFIRMED in sign,
larger than predicted — D_sweep overall +5.09 pp CI [+1.81, +8.53], bo3
+5.68 [+2.05, +9.17]; σ̂_u = 0.72 [0.38, 1.00]; ĥ = 0 (under-dispersion arm
empty). Predicted depth gradient ≈ 0: tercile σ̂ 1.05/0.63/0.36 trends the
intuited direction but b_depth = −0.44 [−1.79, +0.03] and the T1fav−T3fav
D_sweep gradient CI [−36.9, +43.0] pp do not clear 0 — as predicted, no
significant gradient. C* (deep-pool strong favorites): n=10 train, D −4.5
[−34.0, +24.9] — untestable. **Gate G1 FALSE** (H4 premise falsifier fired);
**G2 TRUE** (overall_D, bo3_D, sigma_overall). Per the preregistered rule:
L1+L2 ran, L3/L4 skipped.

**E2.** Predicted σ̂_u ∈ [0.2,0.6]: actual 0.72 (slightly above range).
Predicted L1/L2 holdout Δ ∈ (−1,+1) milli, INSIDE NOISE FLOOR: CONFIRMED —
L1 −0.048m iid CI [−0.42,+0.32] / block [−0.31,+0.25]; L2 +0.213m
[−0.75,+1.15] / [−0.72,+0.90]; MDE_within 1.773m; expected ROI deltas
0.0000 / +0.0002. Predicted P(bo5|strong fav) falls under σ_u>0: CONFIRMED,
−1.55 pp (n=5 holdout rows; L2 +0.22 pp). β refits 0.1407/0.1401 vs v6
0.1152 — the 0.82 ratio matches the computed iid-equivalent logit rescale of
the σ-link, i.e. v6's β was already absorbing the over-dispersion. My null
was NOT falsified (no link cleared ±1.773m with CI excluding 0).

**E3.** Reduction check passed (max |ΔP| < 5e-16 all formats); guard
statement holds — no map-level inputs anywhere in the links.

**Verdict: H4 REFUTED — maps are over-dispersed, not under-dispersed; the
correlated link is β-degenerate on the deployed surface; both links INSIDE
NOISE FLOOR; iid aggregation + fitted β stands.** Deviation log: none — all
analyses as preregistered; seeds mc_seeds[0]/[1] as declared; L2's b_depth
bootstrap continued the mc_seeds[1] stream after the five cell param boots
(order documented in stats/h4_dispersion_diag.json crn block).


════════ preregister.compose.md ════════

# preregister.compose.md — Wave 3 stacks (agent:compose)

Written 2026-07-28, BEFORE any train fit or holdout scoring by this agent.
Frame: testing_lab/v8/data/frame_expanded/series.csv, sha256 verified against
crn.json `frame_expanded` at every entry point (abort on mismatch). Holdout =
date > 2024-12-31 (n=1217), train n=841. All resampling via referee
paired_bootstrap_crn on crn.json (iid + block_event). Baseline = v6 champion
replayed on the expanded frame (consist 20/12, PO 1.6, champ x2 exact-shape,
ridge 0.5 + region-prior 1.5, rd |.|^0.75*2.5, year continuity), train-fit
β=0.1152, holdout LL 0.64216, from scratch/bias_h3/model_probs.npz `p_v6`
(cross-checked against scratch/context/b0.npz to ≤1e-9 before use).

Components allowed (compose brief): 3e stand-in shrink (context), event-class
fade (decay axis d, on-v6 attachment), n_eff-gated 5d state-space (bias-h3).
Nothing else. No killed component resurrected; 1b (calendar noise) is NOT used.

## Exactly three stacks

### S1 "gate5d" — the named h3 shape, scored for the first time
- p_S1[i] = p_ss_5d[i] if neff_min_5d[i] < 12 else p_v6[i].
- neff_min_5d = min(R/v_w, R/v_l) from the 5d roster-typed SS filter's
  pre-match posterior variances (model_probs.npz vw_5d/vl_5d); R = h3's train
  Var(y) identification constant (11.2933; n_eff invariant to its value).
- Threshold 12 is the ONLY 5d hard gate that beats v6 on TRAIN
  (h3_ensemble.json: 0.646927 vs v6 0.648226, +1.30m train; neff<8/<5/<3 are
  all train-worse). Gate membership verification: gated counts must equal
  h3's published 330 train / 178 holdout; if not, recompute R exactly from
  the engine game corpus (train Var(y)); abort loudly if still unequal.
- Free parameters: none new. β_5d and the 5d q's are h3's train fits; β_v6 is
  the v6 train fit. (No solve constant changes ⇒ no β refit obligation.)
- Mechanism: SS posterior uncertainty helps exactly where data are thin;
  h3's cold buckets (either <10 prior maps +71.7m, <30 +28.0m, holdout,
  recorded as full-5d diagnostics, never as a scoring of this composite).
- Predicted sign +; predicted effect +0.5..+2.5m overall (train +1.30m,
  concentrated on 178/1217 rows). Predicted promotion verdict: HOLD (the
  effect should be real but below the cross-family bar).
- Falsifier: holdout ΔLL ≤ 0 overall, or ΔLL < 0 restricted to the 178 gated
  rows ⇒ the cold-row advantage was corpus-era-specific; component dead for
  stacking and said so.
- G1 MDE regime: cross-family 5.889m (h3 precedent for SS-vs-v6 composites);
  empirical pair-MDE (2.8016·SD(d)/√n, raw + CUPED-CV) quoted alongside.

### S2 "fade+shrink" — the two engine/surface survivors, jointly refit
- Ratings: rdiff_fade from scratch/decay/probs/eclass_on_v6_m0.8.npz — v6
  consist(20,12) with HL multiplier m_ewc=0.8 on ewc_offseason-class games
  (decay axis-d event set, frozen; m=0.8 was the train-selected multiplier in
  the wave-2 grid; the on-v6 attachment is the surviving shape, +0.24m,
  positive in all six subpops). m_ewc is NOT refit here (frozen constant;
  refitting it would require new solver runs outside the surviving artifact).
- Surface: z = β·rdiff_fade·exp(−k·X1), X1 = (1−integ_w)+(1−integ_l) (context
  3e frame_features.csv, 0 NaN), p = house series closed form by fmt.
- Joint train fit: (β, k) by Nelder-Mead (x0=(0.13,0), xatol 1e-5, fatol
  1e-9) on train rows with valid rdiff_fade — the exact 3e procedure, run on
  the fade base. This is the stack's β refit (scale-bound rule satisfied).
- Predicted sign +; predicted k ∈ +0.2..+0.5 (3e fit +0.347 on B0); predicted
  effect −0.2..+0.8m overall (components +0.24m and −0.40m alone; the case
  for the pair is the shared EWC-bucket mechanism: +3.46m and +0.52..+0.65m
  subpop gains), EWC full-class bucket predicted positive. Predicted
  promotion verdict: HOLD.
- Falsifiers: fitted k ≤ 0 on the fade base (mechanism does not survive
  composition — S2 still scored, but the component is declared incoherent);
  holdout ΔLL ≤ −1.773m (stack is worse than v6 beyond within-family noise
  ⇒ pair dead).
- G1 MDE regime: within-family 1.773m (v6-family variant; no SS content).

### S3 "full" — S2 base with the S1 gate on top
- p_S3[i] = p_ss_5d[i] if neff_min_5d[i] < 12 else p_S2'[i], where p_S2' uses
  S3's own (β,k): refit jointly on train to minimize the COMPOSITE train NLL
  (equivalently: the 3e fit restricted to non-gated train rows, since gated
  rows don't depend on (β,k)). Same frozen gate, same frozen m_ewc.
- Predicted sign +; predicted effect ≈ additive: +0.5..+3.0m. Predicted
  promotion verdict: HOLD (below the 5.889m cross bar).
- Falsifier: S3 ΔLL < min(S1, S2) − 0.5m (anti-synergy: stacking hurts), or
  ΔLL ≤ 0 (stack concept dead).
- G1 MDE regime: cross-family 5.889m (contains the SS component).

## Judging (identical for all three; ONE holdout scoring each, ever)
- referee.promotion_gate(candidate, v6, mde = regime MDE above, frame,
  rdiff_ref = v6 rdiff, games = engine corpus for elite/floor + form masks):
  G1 (mean ΔLL ≥ MDE AND p_better ≥ 0.95 in iid AND block CRN), G2 max|team
  bias| strictly < v6's (expected 0.1478, min_n 25), G3 pre-committed bucket
  floors. Verdict published clause by clause, whatever it is.
- Both units everywhere: milli-LL + expected_roi_of_dll on the quoting surface.
- CV: CUPED control variate on centered l_v6 (decay judge recipe, banked in
  stats/variance_reduction.json) — CV boots + CV pair-MDE reported next to
  raw in every CI. Point estimates unchanged by CV.
- Caterpillar (full per-team bias table) + full bucket panel per stack; the
  PRX/NRG unexplained residual (v6: −10.2/−9.1pp) reported per stack, not
  chased; if it persists it is named future work.
- Gated-subset detail for S1/S3 (n=178 rows): ΔLL on gated and non-gated
  splits (diagnostic accompanying the single scoring, not extra scorings).
- No iteration after seeing holdout numbers. Whatever comes out, publishes.
- Multiple-looks tally: every holdout scoring by every agent harvested from
  stats/*.json + logs/*.log, published in stats/compose_looks.json with
  these three additional looks counted, and the family-wise context stated
  next to every p-value quoted in phase_compose.md.

## Program verdict rule (fixed in advance)
If any stack passes G1∧G2∧G3 ⇒ "stack X clears the bar." Else if all three
fall inside their noise floors or below ⇒ "v6 stands because no preregistered
stack of the surviving components beats it at the promotion bar on the
expanded holdout." If results are mixed-sign / unresolvable ⇒ "we could not
tell," in those words.

## Outcomes (appended AFTER the one-shot scorings — same resolution win or lose)

Scored 2026-07-28 22:36 (one pass, sentinel scratch/compose/SCORED). Frame
sha verified; gate counts reproduced h3 exactly (330 train / 178 holdout).

- **S1 gate5d: +1.958m** — INSIDE NOISE FLOOR (cross MDE 5.889m; pair-MDE
  3.08m raw / 3.06m CV; iid CI −0.15..+4.21m, block −1.19..+4.94m; p_better
  0.966/0.888). Predicted +0.5..+2.5m, sign + → **prediction confirmed in
  range**; falsifier did NOT fire (overall > 0 and gated rows +13.39m > 0;
  non-gated identically 0). Promotion gate HOLD: G1 fail (sub-MDE, block
  p<0.95), G2 fail (max|bias| 0.1491 vs 0.1478 — worsened, prediction of a
  clean HOLD was right for the wrong clause count), G3 fail (huge-gap
  −8.86m). ROI +0.16pp at δ_logit +0.0043.
- **S2 fade+shrink: −0.140m** — INSIDE NOISE FLOOR (within MDE 1.773m; iid
  CI −1.89..+1.64m; p 0.43/0.47). Fitted k=+0.352, β=0.1252 → k-range
  prediction confirmed; effect prediction (−0.2..+0.8m) confirmed at the low
  end. EWC full-class +3.15m (predicted positive — confirmed). Neither
  falsifier fired (k>0; ΔLL > −1.773m). Gate HOLD (G1/G2/G3 all fail;
  G2 0.1548; G3 domestic EMEA −4.41m, favorite [0.7,0.8) −7.63m).
- **S3 full: −7.874m** — **LOSS** beyond the cross floor (iid CI
  −14.3..−1.5m, p 0.008/0.043; CV CI −12.9..−2.9m). **Anti-synergy
  falsifier FIRED** (S3 ≪ min(S1,S2) − 0.5m): the declared subset refit
  (non-gated train, n=497) drove k to 1.87 and produced the wave's best
  composite train NLL (0.64469) with the worst holdout (0.65004) — a train
  mirage by the program's own definition; non-gated rows −11.5m. The
  additive-effect prediction (+0.5..+3.0m) was **wrong**; recorded at full
  resolution. Gate HOLD (18 major bucket regressions).
- PRX/NRG residual persists under every stack (−10.1/−9.3, −10.1/−9.3,
  −11.9/−11.4 vs v6 −10.2/−9.1) → named future work.
- Looks tally published (stats/compose_looks.json): 163 primary looks
  program-wide (+227 sweep-checkpoint, +8 EM-iteration = 398 recorded
  holdout numbers); compose added exactly 3.

**Program verdict (preregistered wording): v6 stands because no
preregistered stack of the surviving components beats it at the promotion
bar on the expanded holdout.**


════════ preregister.context.md ════════

# Pre-registration — agent:context (Phase 2: event context, incentives, prep)

Written 2026-07-28, BEFORE any experimental run. Frame verified first:
sha256(testing_lab/v8/data/frame_expanded/series.csv) =
ff772d417b714844aa1b11426d8c04917e1b2848c9c8df811cd9988694d55142 == crn.json
"frame_expanded.series_csv_sha256". n=2058, train 841 (date<=2024-12-31),
holdout 1217.

## Shared machinery (frozen)

- **Baseline B0** = v6 champion replayed on the expanded frame:
  Engine() (expanded registry), `eng.series`/`eng.pred_days` replaced by the
  frame; cfg = {rd power 0.75 scale 2.5, roster year/0.3, ridge 0.5,
  champ_mult 2.0, region_prior_ridge 1.5, w_custom = 1.6 on playoffs/GF-stage
  games, decay games/consistency (20,12)} (run_v7_stage1 BASE + champion
  decay); β refit on train by the engine (bounds 0.03–0.6). Infrastructure
  probe before this file (declared): B0 ll_test 0.64216, β 0.1152,
  legacy-2026 EWC bucket 0.68633 n=115. No experimental config was run
  before this file.
- **Engine hygiene**: `eng._prev_rvec` deleted before every run() call
  (region-prior warm-start otherwise leaks the previous config's final-day
  solve into the next run's first day). One process, one writer; scratch =
  testing_lab/v8/scratch/context/.
- **Prediction layer**: z = β·rdiff (+ terms); p_map = σ(z); series prob via
  the house closed forms (bo1/bo3/bo5). Any added term ⇒ joint (β, c) refit
  on train rows only (Nelder-Mead, x0 β=0.13, c=0). Holdout is never fit.
- **Judging**: d = ll_B0 − ll_cand per holdout series (frame order);
  referee.paired_bootstrap_crn iid + block_event; every ΔLL quoted in
  milli-LL AND referee.expected_roi_of_dll(dll, p_ref=B0 holdout probs);
  pair-MDE quoted per stats/power_mde_expanded.json: within-family 1.773m,
  cross-family 5.889m. |Δ| inside the floor ⇒ verdict INSIDE NOISE FLOOR.
  All my candidates are v6-family variants (same decay family / same rdiff),
  so within-family 1.773m is the applicable floor; block-event CI reported
  alongside as the conservative check.
- **Lineup top-up** (declared data work, not an experiment): 335 frame
  matches (670 sides; all corpus additions) lack lineup_features rows. I
  recompute, for exactly those (match_id, org) sides, the features I use —
  overlap_modal, stand_in_flag, overlap_vct_modal (+ n_vct_basis, debut
  flags) — from eng.lineups (data/maps/*.csv) + engine game dates, using
  preregister.lineups.md definitions verbatim (30d window strictly before D;
  vct modal = last ≤10 vct-class matches; event_class ewc :=
  ratings_only:True in ALL_EVENTS). Written ONLY to
  scratch/context/lineup_topup.csv. Existing lineups-agent rows stay
  canonical where present. Cross-check: on 20 random already-covered sides my
  recomputation must reproduce the lineups agent's overlap_modal /
  overlap_vct_modal exactly; mismatch ⇒ stop and reconcile before use.
- **Integrity(side)** for 3a/3e := overlap_vct_modal if defined, else
  overlap_modal, else 1.0 (org debut / no basis). In [0,1].
- **Event classes** (frozen map, used by 3a/3d/3e reporting):
  champions = ^\d{4}_champions$; masters = ^\d{4}_masters_.*$ ∪ {2023_lock_in};
  ewc_offseason = ALL_EVENTS ratings_only:True MINUS {2023_lcq} (LCQ is a
  Champions qualifier — a stakes-bearing VCT bracket; power's addendum also
  classed it vct_domestic); vct = everything else ∪ {2023_lcq}; vct splits
  per-game into vct_playoffs (frame stage playoffs/grand_final) vs
  vct_regular. EWC-bucket reporting: legacy-2026 definition (referee
  EWC_CLASS_PREFIXES; baseline 0.6918 published old-frame n=109 / 0.68633 B0
  expanded n=115) AND full ewc_offseason holdout bucket (n≈278) — primary =
  full class.

## 3a — Lineup-conditioned EWC solve weighting (stats/context_seriousness.json)

- Mechanism: EWC-class games fielded by non-VCT-modal lineups are weak
  evidence about the org's VCT strength; down-weighting them in the solve
  should improve holdout LL (mostly via later VCT/intl rows).
- Family: per-game weight on ewc_offseason-class games only,
  w = f(integ_w)·f(integ_l), f(x) = w0 + (1−w0)·x. w0=1 ⇒ B0 exactly.
  Multiplies the v6 stage weight (playoffs 1.6 unchanged).
- Blanket comparator: w = w_e flat on all ewc_offseason games.
- Fit: train NLL (β refit per point), grid w0 ∈ {0.0,0.1,…,1.0},
  w_e ∈ {0.4,0.5,…,1.2}. Winner of each family = train argmin; judged on
  holdout vs B0 and vs each other.
- Predicted sign/size: w0* < 1 (point prediction ~0.5); blanket w_e* ~0.8;
  holdout Δ(integrity vs B0) ≈ +0.3 to +1.5 mLL (inside/near the 1.773m
  floor); integrity ≥ blanket.
- Falsifier: train picks w0≈1 (no conditioning), or holdout Δ ≤ 0, or
  blanket ≥ integrity (then the lineup story adds nothing over "EWC games
  are noisier"). Reported at full resolution either way.

## 3b — Footage exposure & prep (stats/context_exposure.json)

Walk-forward per (team, match) from engine games (official = any corpus map;
strictly earlier dates only): maps14, maps30 = maps in trailing 14/30d;
dso = days since last official map (cap 120; debut→120);
dsi = days since last intl-LAN day (intl = exact-shape masters/champions/
lock_in; cap 365; never→365; sensitivity variant adds EWC mains).
Winner-referenced diffs: Δmaps30/10, Δlog1p(dso), Δlog1p(dsi).
- (a) Prediction term: z = β·rdiff + c1·Δmaps30/10 + c2·Δlog1p(dso) +
  c3·Δlog1p(dsi), fit train, scored holdout, CRN boots vs B0.
  Predicted: all c ~ 0, |Δ| inside 1.773m floor. Falsifier of "prep/rust
  matters": tight-CI zeros (then say so plainly).
- (b) THE DECOMPOSITION: reproduce run_v7_stage2 on the expanded frame:
  z = β·rdiff + b_form·Δform, Δform = (wr5−wr16)_w − (wr5−wr16)_l (HL3
  sensitivity), train-fit. Then add the exposure controls (a) jointly.
  Report b_form without vs with controls + Δform↔exposure correlations
  (train rows), forest-plot JSON. Published v7 old-frame reference:
  b_form(form5) = −0.0242, (form3) = −0.0872.
- Predicted: b_form stays negative and shrinks <30% with controls ⇒ form is
  genuine mean-reversion, not scouting. Falsifier (= "scouting in disguise"):
  b_form → 0 (|shrink| ≥ 70% or sign flip) once exposure is controlled —
  if so I say exactly that; if not I say that plainly.

## 3b-adjacency — Deep run → next intl (stats/context_adjacency.json)

- Frozen adjacency pairs (Masters → next intl-class event by date):
  tokyo→2023_champions, madrid→2024_masters_shanghai,
  2024_masters_shanghai→2024_champions, bangkok→2025_masters_toronto,
  toronto→2025_ewc, santiago→2026_masters_london, london→2026_ewc.
- Subset: series of the "next" event where BOTH orgs played ≥1 series at the
  prev Masters. dr(org) = n series played at prev Masters, centered within
  that Masters over attendees. Model: p_map = σ(β0·rdiff + c·(dr_w − dr_l)),
  β0 frozen at B0's train β; c fit by ML on the subset (series closed
  forms). Wald CI + CRN iid bootstrap CI (crn seed machinery, n=subset).
  This is inference, not model selection; nothing is promoted from it.
- Predicted: c < 0 (deep run → underperform next intl), tiny; CI expected to
  span 0 ⇒ published verdict likely "untestable at this n" (acceptable per
  brief). Falsifier: c ≥ 0 or CI spanning 0.

## 3c — Stakes (stats/context_stakes.json)

- Frozen derivable elim flag (series-level, from match_name/stage):
  elim = stage=='grand_final' OR name contains any of {Lower, Elimination,
  Decider, Knockout, "(0-1)", "(1-1)"} OR (stage=='playoffs' AND name
  contains any of {Quarterfinal, Semifinal, Final, Round of} AND NOT
  contains Upper). Swiss "Round 3" without a record suffix stays non-elim
  (declared limitation). Stage buckets groups/playoffs/GF come free with B0.
- Dead rubbers: DECLARED UNTESTABLE now — group membership, advancement
  thresholds and map-diff tiebreakers are not in the corpus; standings
  reconstruction would be approximation, which the brief forbids. Published
  as untestable, no estimate.
- Test A (solve weight): extra per-game multiplier w_elim on elim-series
  games (on top of v6 stage weights), grid {0.7,0.85,1.0,1.15,1.3,1.5},
  train argmin, holdout vs B0. Predicted w_elim* ∈ {1.0,1.15}, holdout Δ
  inside 1.773m floor. Falsifier: argmin 1.0 / holdout worse.
- Test B (prediction variance): z = β·rdiff·(1+a·elim), (β,a) train-fit,
  holdout vs B0. Predicted |a| ≤ 0.1, inside floor. Falsifier: a≈0 with
  tight CI. (GF logit offset of the live config is a different, existing
  term; not retested.)

## 3d — Learned event-class solve weights (stats/context_weights.json)

- Replace champ_mult + stage-1.6 with per-class per-game weights, anchor
  vct_regular=1.0; free (w_vct_po, w_champ, w_masters, w_ewc) fit in
  log-space by Nelder-Mead on train NLL (β refit inside every eval; x0 =
  v6-equivalents [ln1.6, ln2.0, ln1.3, ln1.0], maxiter 200, fatol 1e-5).
  Walk-forward is inherent (run() solves strictly-past games per day).
- CI per weight: 1-D profile grid {0.25,0.4,0.6,0.8,1.0,1.3,1.6,2.0,2.6,3.4}
  (others at fitted), per-train-row NLL vectors, CRN iid resample of train
  rows (crn bootstrap seed 20260728, full-matrix recipe, n=841), argmin per
  resample → percentile CI. Declared approximation: profile CI ignores
  cross-weight covariance; granularity = grid.
- Holdout: fitted config vs B0 (hand-set), CRN boots, both units, within
  1.773m floor. The fitted w_ewc ± CI is the operator-intuition deliverable
  and is published regardless of the holdout verdict.
- Predicted: w_ewc* < 1 (point ~0.6, CI likely wide, may include 1);
  w_champ* ∈ [1.3,2.6]; holdout Δ vs B0 inside the floor (hand-set weights
  are probably not the binding constraint). Falsifier of "EWC games
  mislead": w_ewc CI ⊇ 1.0.

## 3e — Context-conditional confidence, mechanism version (stats/context_shrink.json)

- ONE global shrink: z = β·rdiff·exp(−k·X), k ≥ 0, (β,k) train-fit.
  X1 = stand-in load = (1−integ_w) + (1−integ_l) (all events, from 3a
  integrity); X2 = prep asymmetry = |Δlog1p(dso)| (from 3b). Fits: k1 alone,
  k2 alone, (k1,k2) jointly — max 2 coefficients, no class dummies.
- Report per fit: holdout LL vs B0 (CRN boots, both units, 1.773m floor),
  EWC bucket before/after — legacy-2026 def (0.6918 published / 0.68633 B0
  expanded) AND full ewc_offseason holdout bucket.
- Falsifier control (preregistered): z = β·rdiff·exp(−k_ewc·1[ewc_class]) —
  the free class-dummy shrink. If the dummy improves and no observable-X
  does, published verdict = "no mechanism found" (dummy is the falsifier of
  the mechanism story, not a candidate).
- Predicted: k1 > 0 small; EWC-bucket gain 0–3 mLL; overall Δ inside the
  floor. Honest prior: X2 does nothing.

## Outcomes

Appended below AFTER the runs, same resolution for failures as successes.
All ΔLL vs B0 on the 1217-row holdout; within-family pair-MDE 1.773m
(cross 5.889m); ROI unit via referee.expected_roi_of_dll (ladder clamps to
0.0 for negative shifts). B0 measured: ll_test 0.64216, β 0.1152; EWC
buckets ll 0.68633 (legacy-2026, n=115) / 0.67224 (full class, n=291).
Data note: lineup top-up 670 sides / 335 matches, 0 gaps, definition
cross-check 20/20 exact; 7/400 sampled covered sides would change
overlap_modal under full-corpus history (their rows kept canonical).

### 3a — DEAD (falsifier fired at the train stage)
Predicted w0*~0.5; measured train argmin w0* = 1.0 — train NLL is monotone
WORSE from w0=1 toward w0=0 (0.64823→0.65052). No candidate to judge; the
descriptive holdout curve's own min is +0.08m at w0=0.8, deep inside the
floor. Blanket: predicted w_e*~0.8; measured train argmin at the 1.2 grid
EDGE (train prefers UP-weighting EWC games); holdout at 1.2: −0.051m, iid CI
[−0.74,+0.65]m, INSIDE NOISE FLOOR (EWC-full bucket +0.76m). Prediction
wrong in direction; operator's EWC-down-weight intuition unsupported.

### 3b-a — INSIDE NOISE FLOOR (as predicted)
Fitted train coefs: c_dmaps30/10 = +0.118, c_dlogdso = +0.073, c_dlogdsi =
−0.087 (β 0.1033). Holdout −0.280m, iid CI [−5.17,+4.46]m; EWC-full bucket
−10.91m (the term actively hurts off-season rows). Sensitivity (dsi incl.
EWC mains): −0.542m. Predicted c≈0/floor: correct on the floor verdict, but
the coefficients are NOT tight zeros — they train-fit and fail to transfer.

### 3b-b — DECOMPOSITION ANSWERED
Measured (train fits, expanded frame): b_form(HL5) alone −0.1300 → +0.0120
with exposure controls (109% absorbed, sign flip to ~0); b_form(HL3) alone
−0.1176 → −0.0492 (58% absorbed). Predicted <30% shrink — WRONG: exposure
absorbs the HL5 form penalty entirely and half of HL3's. Quotable: at HL5
v7's "form is mean-reverting" was scouting/footage-exposure in disguise; at
HL3 about half was. Neither survives holdout anyway (form5 alone −0.31m,
form3 alone −0.66m, both + exposure ~−0.2/−0.6m — all INSIDE NOISE FLOOR;
the v7 old-frame +1.3m form gain does NOT replicate on the expanded frame,
where train b_form inflates 5× vs published −0.0242).
corr(dform5, {dm30, dlogdso, dlogdsi}) = {+0.114, −0.142, +0.169}.

### 3b-adjacency — UNTESTABLE AT THIS N (as anticipated)
n=57 both-attended series across the 7 frozen adjacencies. c_hat = +0.0213,
Wald 95% [−0.081,+0.124], CRN boot [−0.088,+0.137]. Sign OPPOSITE the
fatigue prediction; CI spans 0. Published as untestable, nothing promoted.

### 3c — DEAD / INSIDE NOISE FLOOR
Dead rubbers: untestable as declared. Test A: predicted w_elim* ∈
{1.0,1.15}; measured train argmin at the 0.7 grid EDGE (train wants elim
games DOWN-weighted — stakes make games noisier in-sample, opposite of the
reveal-true-strength story); its holdout −0.537m, iid CI [−2.08,+0.93]m,
INSIDE NOISE FLOOR (and negative). Test B: a_elim = −0.348 (train shrink on
elim matches, consistent with A); holdout −0.371m, INSIDE NOISE FLOOR.
Direction prediction wrong; no usable stakes signal.

### 3e — INSIDE NOISE FLOOR everywhere; mechanism lead retained
k_standin = +0.347 (predicted >0 small ✓); holdout −0.402m overall (floor),
EWC-full bucket +3.46m BETTER (0.67224→0.66878, n=291); legacy-2026 bucket
+1.26m (0.68633→0.68506, n=115). k_prep = −0.007, nothing (X2 dead as
predicted). Joint: −0.503m overall, EWC-full +4.13m, legacy +1.75m.
FALSIFIER class dummy k_ewc = +0.860: overall +0.500m (floor); split
buckets — legacy-2026 +3.34m BETTER but EWC-full-class −3.76m WORSE. On the
preregistered PRIMARY bucket (full class) the observable mechanism beats
the dummy where it matters; on the legacy 2026-only slice it does not.
Verdict: no promotable effect (all overall Δ inside the 1.773m floor); the
stand-in-shrink bucket signal is a legitimate Phase-5 lead, flagged with
the bucket-definition sensitivity above.

### 3d — DEAD on holdout; EWC-weight deliverable published
Fitted (NM, 162 evals, train NLL 0.64823→0.64405): {vct_playoffs 0.734,
champions 0.001, masters 1.880, ewc_offseason 1.018}. Holdout: 0.64790 vs
B0 0.64216 ⇒ Δ = −5.737m, iid CI [−10.00,−1.21]m, block CI similar sign —
DEAD (|Δ| > within-MDE 1.773m, ≈ cross-MDE 5.889m; −5.7m ≈ 0.48 ROI pts on
the quoting surface). v6's hand-set weights WIN decisively; the joint fit
is in-sample regime memorization (the champions collapse to ~0 is even
"identified" in-sample — 86% of CRN argmin-bootstrap resamples pick the
0.25 grid floor — and still anti-validates: 2023–24 Champions upsets are
noise the walk-forward future did not repeat). Profile CIs (declared
approximation): vct_po [0.25,3.4], champions [0.25,3.4], masters [0.8,3.4],
ewc [0.4,3.4] — weights weakly identified at n_train=827.
Predicted w_ewc ~0.6 — WRONG: fitted 1.018, CI [0.4,3.4] ⊇ 1.0, so the
preregistered falsifier of "EWC games mislead the solve" FIRED. Labeled
sensitivity (others at v6 hand-set, same grid+CRN recipe): train NLL
monotone improving to the 3.4 grid edge, argmin-boot CI [0.6,3.4] — mass
rejects down-weights below 0.6 (scale-vs-ridge caveat logged). Predicted
holdout "inside floor" also WRONG — fitting the weights actively hurts.


════════ preregister.corpus.md ════════

# Pre-registration — agent:corpus (Phase 1: corpus expansion)

Written 2026-07-28, BEFORE any VLR fetch. Brief: testing_lab/v8/briefs/corpus.md.

## Enumeration methods (VLR only, never memory)
1. **Archive sweep**: https://www.vlr.gg/events/?page=N (completed section),
   walked sequentially from page 1 until an entire page's events end before
   2021-01-01 (plus one confirmation page). Every event row captured:
   vlr_event_id, name, dates, prize, region flag, status.
2. **Year hub cross-check**: the VCT circuit hub/series pages for
   2021, 2022, 2023, 2024, 2025, 2026 (vlr.gg/vct-<year> and the event-series
   listings they link). Any official-circuit event that the archive sweep
   missed gets added to the candidate list from here.
3. **Targeted event search**: vlr.gg event search for name families the brief
   flags: "Esports World Cup", "Evolution Series", "Ludwig", "Tarik",
   "OFF SEASON", "Home Ground". Search results only add candidates; they never
   remove any.
The candidate list is the union of all three. Enumeration output is written to
testing_lab/v8/stats/corpus_diff.json before any match scraping starts.

## Inclusion criteria for "tier-1" (decided per event, recorded in corpus_diff.json)
A candidate event is ADDED to the ratings corpus (2023-2026) iff:
- **C1 — VCT franchised circuit**: an official VCT franchise-era event
  (Kickoff / Stage league / Masters / Champions / China splits, incl. LOCK//IN
  class). These should already be registered; any hole found is backfilled.
- **C2 — EWC class**: Esports World Cup main events, their regional
  qualifiers, and China Evolution Series acts that form the EWC-CN qualifying
  chain. Treatment mirrors the existing 2026 entries: ratings_only + vct_only,
  NOT International-tagged; the regional qualifiers of one year are merged
  into ONE multi-region entry (2026_ewc_qual precedent).
- **C3 — Off-season / one-off with franchised participation**: competitive
  (non-showmatch) bracket events where >= 4 franchised VCT orgs participated,
  so that vct_only filtering yields a meaningful number of both-sides-
  franchised series. Ludwig x Tarik-class invitationals qualify under C3 if
  they meet the >= 4-org bar. Events failing the bar are EXCLUDED with reason.
- Anything else (tier-2 Challengers/Ascension, Game Changers, showmatches,
  collegiate, watch parties) is EXCLUDED with reason.
Pre-franchising 2021-2022: VCT Champions / Masters / Challengers main events
and tier-1 equivalents go ONLY to testing_lab/v8/data/prefranchise/ (separate
registry.json, never ALL_EVENTS, never data/). Priority if volume is huge:
Champions > Masters > regional main events; deferrals reported with counts.

## What counts as verification passing
- **Mechanical, every backfilled series**: winner org + map score parsed from
  the SAME HTML my scraper fetched (cached to disk at scrape time) must equal
  the winner org + score row that scrapers/BuildMatchResults.py (independent
  re-fetch + parse) wrote into data/match_results.csv — for the series row and
  every per-map row. AND the match date (data-utc-ts, ET calendar day) must
  land inside [event start - 1 day, event end + 1 day] of the registry entry.
- **Sample re-fetch**: per backfilled event, 10 series (or all, if fewer)
  drawn WITHOUT replacement using seed 20260728 (crn.json does not exist yet
  at pre-registration time — the power agent has not written it; this seed is
  fixed here instead, and this deviation is disclosed in the report), each
  re-fetched live from VLR and re-parsed; winner + score must match
  match_results.csv. Report N/N per event.
- Any mismatch = verification FAILURE for that event; reported loudly, never
  silently dropped or "fixed" by hand-editing CSVs.

## Falsifier
If VLR's own listings show no 2025 EWC regional qualifiers (or no 2025
Esports World Cup Valorant event, or no 2025 China Evolution Series act), then
the brief's premise about the 2025 EWC chain is wrong, and I report exactly
that — I do not substitute a different event to fill the slot. Same logic for
any expected event: absence on VLR is reported as absence.

## Fetch discipline
All fetching by my own scripts goes through scrapers/enriched/vlr_client.fetch
(sequential, >= 0.75 s between match pages, >= 1.0 s between listing pages).
The standard builders (BuildMatchResults) use their built-in RefreshLiveData
_fetch stack, run only after `pgrep -f RefreshLiveData.py` is empty.
Match dates/times for new MatchIDs are parsed from my cached match HTML with
RefreshLiveData step-5 semantics (data-utc-ts -> ET day in match_dates.json,
_et_walltime_to_utc -> UTC in match_times.json) — zero extra VLR load.
Forbidden and untouched: BuildRatingTimeline, BuildMapRatings, RefreshLiveData,
VCTMM, anything in testing_lab/ outside v8/, rewriting existing event CSVs.


════════ preregister.decay.md ════════

# Pre-registration — agent:decay (Phase 4: recency & asymmetry) — written 2026-07-28, BEFORE any experiment run

Frame: testing_lab/v8/data/frame_expanded/series.csv, sha256 verified
ff772d417b714844aa1b11426d8c04917e1b2848c9c8df811cd9988694d55142 == crn.json
frame_expanded (checked in scratch/decay/audit.py and re-checked at run start;
abort on mismatch). n=2058, train=841 (date<=2024-12-31), holdout=1217.
Engine game corpus at audit: 5140 maps, 73 orgs, 637 pred days, all frame
events covered. All randomness: crn.json (iid seed 20260728, block seed
20260729, n_boot 4000). No fitting on holdout anywhere; all constants and grid
selections on train only.

## Machinery (fixed before runs)

- Runner: scratch/decay/decay_lib.py — a copy of engine.Engine.run's solve loop
  with per-game half-life and per-(game,day) weight hooks. VALIDATION GATE: with
  v6 settings it must reproduce eng.run({v6}) rdiff to atol 1e-9 and identical
  train beta on the expanded frame; if not, stop and fix before any variant run.
- BASE constants (stage-1 replica, never varied here): rd |rd|^0.75 x 2.5,
  ridge 0.5, champ_mult 2.0 (exact-shape YYYY_champions), region_prior_ridge
  1.5, playoff w_custom 1.6 (stage from the expanded frame by match_id, default
  groups), roster_mode year for games-decay (except axis 5b-a which REPLACES
  year continuity with lineup continuity — that substitution is the axis).
  eng._prev_rvec reset to None before every config (removes the stage-1
  cross-config first-day prior contamination; stage-1 v6 semantics preserved).
- beta per config: train-only, bounded (0.03, 0.6), engine's closed-form series
  likelihood. 5c fits (beta, b_form) jointly, Nelder-Mead x0=[0.13, 0]
  (run_v7_stage2 replica).
- Judging: referee.delta_vector (d = l_v6 − l_cand, >0 = candidate better),
  paired_bootstrap_crn iid + block_event on the 1217 holdout rows (valid mask =
  both rdiffs finite; expect all 1217 valid).
- Control variate (5a + headline pairs): CUPED. x1 = l_v6 (primary);
  multivariate X = [l_v6, |rd_v6|, p_fav_v6] (secondary). theta = OLS of d on
  centered X over the full holdout; d_cv = d − (X−mean X)·theta. Same CRN index
  matrix for raw and CV bootstraps. Point estimate unchanged by construction;
  only CI/MDE shrink. (stats/variance_reduction.json predicts ~1.30x CI
  shrink for cross-family pairs, ~1.00-1.03x within-family.)
- MDE quoting: Phase-0 composition-adjusted family MDE (within 1.773m, cross
  5.889m at n=1217) next to every delta, plus the pair's own empirical
  MDE80 = 2.8016·SD(d)/sqrt(n) (raw and CV-adjusted). VERDICT RULE
  (pre-committed): WIN/KILL requires |mean d| >= the pair's raw empirical MDE80
  AND p_better >= 0.95 (<= 0.05 for KILL) in BOTH iid and block modes;
  otherwise INSIDE NOISE FLOOR — published in those words.
- Both units: every headline delta also as expected ROI via
  referee.expected_roi_of_dll(mean_d, p_v6_holdout) (reporting unit only).
- Continuity (not mixed into comparisons): expanded-frame v6 holdout LL
  restricted to the 1007 frozen-npz rows reported next to the published
  0.64095 — corpus additions shift ratings, so an offset is expected; report,
  don't reconcile.

## 5a Re-race the near-ties

Mechanism: old-frame verdicts (v6 vs consist_16_10 −0.39m, vs sym_20 −1.65m)
were inside their pair MDEs. Expanded frame adds ~8% resolution + CV.
Configs: v6 consist(20,12) [champion], consist_16_10 [within-family, family
MDE 1.773m], sym_20, sym_24 [cross-family, 5.889m].
Predicted signs/sizes (holdout mean d vs v6): consist_16_10 in [−1.5, +0.5]m;
sym_20 in [−4, 0]m; sym_24 in [−4, +1]m — i.e. I predict all three remain
below their MDE: "ties" is the expected published answer.
Falsifier: any candidate meeting the WIN rule dethrones v6; v6 meeting the
KILL rule against a candidate resolves that near-tie for real.

## 5b New conditioning axes (each: train-grid select, ONE holdout verdict per axis)

Grid selection rule (all axes): run the declared grid, pick argmin ll_train
(walk-forward predictions inside train; beta refit per config), publish that
config's holdout numbers as THE axis verdict; all grid members' holdout LLs
reported for transparency but carry no verdict weight. Every axis gets CRN
boots vs v6 (and vs its symmetric-base control where stated) + the 5e panel.

a. LINEUP CONTINUITY (outcome-symmetric; the operator's requested axis).
   w_side(g,D) = exp(−ln2/HL · games_ago_side) · Lfac_side;
   Lfac = max(|L_cur ∩ L_then| / max(|L_cur|,|L_then|,5), 0.04)^gamma
   (engine lineup-mode formula). L tables: v8/data/lineups.csv topped up for
   the 335 corpus-addition matches from the engine's maps-CSV lineups
   (identical grouping per preregister.lineups.md; top-up written ONLY to
   scratch/decay/). L_cur(org,D) = lineup of org's latest match with date < D.
   Year continuity NOT applied (replaced by the axis). Grid: HL {16,20,24} x
   gamma {0.5,1,2,4}. Controls: gamma=0 equivalents are 5a's sym runs; sym_16
   run as an extra control config.
   Mechanism: games played by a different five are weaker evidence about the
   current five. Predicted: beats its own-HL sym control by +0.5..+3m; vs v6
   in [−2, +2]m. Falsifier: train grid prefers gamma<=0.5, or holdout delta
   vs own-HL sym control <= 0.
b. OPPONENT QUALITY OF THE ANOMALY (builds on v6 consistency; outcome-
   dependent by construction — stated exception).
   v6 consist flags unchanged (trailing HL16 map winrate, walk-forward).
   Opponent quality at the game's date: daily v6 ratings from the 5a v6 run
   (daily_out), latest pred-day <= game date; active teams = >=1 prior map;
   elite = top quartile of active ratings that day, floor = bottom quartile;
   no rating that day -> mid. HL_anom_eff = 12·m (opp elite), 12/m (opp
   floor), 12 (mid); consistent HL stays 20. Grid m {1.33, 1.67, 2.0}.
   Mechanism: an anomaly against an elite is more likely real signal; against
   a floor team, more likely noise. Predicted: within-family, +0.0..+1.5m,
   most likely INSIDE NOISE FLOOR. Falsifier: holdout delta < 0, or best
   train gain < 0.3m (axis dead on arrival).
c. ANOMALY MARGIN (same family as b). HL_anom_eff = 12 · clip((|rd|/5)^k,
   0.5, 2.0), consistent HL 20 unchanged; grid k {0.25, 0.5, 1.0}.
   Mechanism: blowout anomalies are informative, squeaker anomalies are coin
   flips. Predicted: within-family, +0.0..+1.5m, likely INSIDE NOISE FLOOR.
   Falsifier: holdout delta < 0 or flat train grid (<0.3m spread).
d. EVENT CLASS OF THE RESULT (outcome-symmetric primary). HL_eff = HL_base ·
   m_class(event of the game); m=1 for vct/intl; m_ewc grid {0.4, 0.6, 0.8};
   base sym HL {16,20,24} (3x3 grid). PLUS one on-top-of-v6 config (both
   consist HLs scaled by train-best m_ewc). ewc_offseason set (pre-committed):
   events starting 2026_ewc / 2026_china_evo + the 22 ewc_offseason-classed
   corpus additions in stats/power_mde_expanded.json new_events (full list
   echoed in decay_axes.json). DECAY-side only — agent:context 3a owns the
   solve-WEIGHT version; mechanisms differ on RECENT offseason games (weight
   hits them immediately; decay only as they age).
   Predicted: vs own-HL sym control +0.5..+2m; vs v6 [−2, +2]m. Falsifier:
   train picks m_ewc=0.8 with <0.3m gain, or holdout delta vs sym control <=0.
e. PATCH / MAP-POOL BOUNDARY (outcome-symmetric primary). Rotation dates
   derived mechanically from the games list (pre-committed): per map (excl
   'TBD'), boundaries = first-game date; re-entry date after a >=60d same-map
   gap; day after the last date preceding a >=60d gap (corpus end is not an
   exit). Pool boundaries > 2023-03-15, sort, greedy-cluster within 14d of
   cluster min; rotation date = cluster min. Derived list published in
   decay_axes.json. Weight multiplier gamma_p^(# rotations in (g_date, D]);
   grid gamma_p {0.85, 0.7, 0.55} x sym HL {16,20,24}; plus one on-top-of-v6
   at train-best gamma_p. Known noise source (accepted, mechanical rule):
   offseason events on stale pools create spurious windows.
   Predicted: vs own-HL sym control +0..+2m (partially redundant with games-
   age); vs v6 [−3, +2]m. Falsifier: holdout delta vs sym control <= 0.

Symmetric-vs-asymmetric verdict (pre-committed wording): among the SYMMETRIC
axes (a, d-sym, e-sym), if any meets the WIN rule vs v6 -> "the operator's
objection is vindicated: a symmetric axis beats consistency conditioning."
If any has mean d >= 0 with p_better >= 0.75 both modes -> "vindicated at
preponderance: symmetric matches/edges v6, below the 80%-power bar." If all
symmetric axes sit inside the noise floor with d < 0 -> "unresolved either
way at n=1217: asymmetry is not demonstrably needed, nor demonstrably
better." If v6 meets the WIN rule against every symmetric axis ->
"asymmetry survives its strongest symmetric challengers."

## 5c Performance-based form

All on the v6 rdiff (within-family, prob-layer terms; family MDE 1.773m).
Reference: old-frame b_form(wr, HL3) = −0.0872, delta −0.25m (n.s.).
1. Continuity replication: wr-form (HL16 long; short 3/5/8), expanded frame.
   Predicted: b_form negative (mean-reversion), HL3 magnitude 0.03..0.15;
   holdout delta vs v6-alone inside noise floor.
2. rd-form (PRIMARY): per-map signed transformed margin from team perspective
   m_t = sign(rd)·|rd|^0.75·2.5 (house transform); per-team exp-decayed mean
   at HL_short {3,5,8} vs HL16 (games-counted, denominator > 3 as wr
   machinery); dform_rd = (short−long)_w − (short−long)_l. Fit (beta, b_rd)
   train, score holdout. Predicted: b_rd negative (same reversion mechanism,
   margin flavor), delta inside noise floor. Falsifier of "performance form
   helps": delta <= 0 or inside floor -> published as such.
3. side-form: from data/enriched/round_outcomes.csv. Per round, winner_org won
   on winner_side; the opponent simultaneously lost on the opposite side ->
   per (team, map): atk/def rounds won/played. Exp-decayed per-side round
   winrates (games-counted), HL_short {5} vs HL16, denominators > 12 rounds
   at BOTH horizons else that team's side-form = 0 (neutral) and the row
   counts as uncovered. COVERAGE AUDIT FIRST (audit found: 1707/2058 frame
   matches have round rows; all 25 corpus-addition events 0%): report frame +
   holdout coverage and the share of holdout rows where both teams' forms are
   defined. dform_side = [(d_atk + d_def)_w − (d_atk + d_def)_l]. Fit (beta,
   b_side) train. Predicted: b_side negative, small, inside noise floor.
4. player-form (optional leg, will run): per-player exp-decayed mean R2.0
   (data/maps player rows, maps-counted, HL_short 5 vs HL16, denom > 3 maps
   both horizons else neutral); team form = mean over the match's fielded
   lineup; dform_p analogous. Predicted: negative, inside noise floor.
5. Combined: (beta, b_rd, b_side) joint fit, train; scored holdout.
Non-overlap: agent:context owns exposure-CONTROLLED b_form; I own
performance-DEFINED form. No exposure controls here.

## 5e Subpopulation panel (EVERY config with a holdout verdict)

Masks (pre-committed):
- S1 post-roster-change: either org matches_since_change <= 3, recomputed on
  the expanded (topped-up) lineup table with the lineups-agent rule verbatim
  (walk back while lineup equal); agreement rate with the published column on
  covered rows reported.
- S2 post-patch: series date within 21d after any 5b-e rotation date.
- S3 post-break: referee rest_days — either team's rest > 45d (both teams
  must have a prior series; referee.bucketed definition).
- S4 within-event day 2+: series date > the event_id's first series date in
  the frame.
- S5 quoted band 20-55c: v6 favorite-side prob <= 0.80 (equivalently the
  underdog side priced in [20,55)c under v6 — referee fallback-band
  definition applied to v6 predicted p).
Per config x subpop: n, ll_v6, ll_cand, delta_milli, bucket MDE =
family_MDE_pair · sqrt(1217/n_bucket) (same-sigma scaling, pre-committed),
tag WIN / INSIDE NOISE FLOOR / WORSE by the same verdict rule at bucket MDE
(p_better clause waived in buckets — CIs not run per bucket; tags are
MDE-vs-|delta| only). No aggregate-only verdicts anywhere.

## Outputs (mine alone)
stats/decay_rerace.json, stats/decay_axes.json, stats/decay_form.json,
stats/decay_subpops.json, stats/decay_curves.json (w(g) overlays: v6 pair,
sym_20, lineup-conditioned at overlap {1.0, 0.8, 0.6, 0.4}, best new axis),
phase4_decay.md, logs/decay.log, scratch/decay/*. One writer each: me.

## Outcomes (appended AFTER runs, 2026-07-28 22:15, same resolution for failures)

Validation gate: PASS exact (max |Δrdiff| = 0.0 vs eng.run(v6); β identical
0.1152; holdout LL identical 0.64216). Frame sha re-verified at every run.
v6 continuity on the 1007 frozen rows: 0.64085 vs published 0.64095.

5a — predictions HELD. consist_16_10 −0.53m (predicted [−1.5,+0.5]);
sym_20 −2.17m (predicted [−4,0]); sym_24 −2.42m (predicted [−4,+1]). All
INSIDE NOISE FLOOR at pair MDEs (1.67 / 3.90→3.39 CV / 4.31→3.61 CV).
"Ties" is the published answer. No falsifier fired. CV shrink as predicted
by variance_reduction.json (~1.15x cross, ~1.0 within). sym_16 control:
−2.33m, bare-significant for v6 both modes (p .047/.015) but sub-MDE.

5b outcomes per axis (predicted sign → measured):
- a lineup continuity: predicted +0.5..+3 vs own control → measured +1.68m
  vs sym_24_nc (INSIDE floor, p .685); predicted [−2,+2] vs v6 → measured
  −2.82m (INSIDE floor; slightly below the predicted band). Fitted γ=2 on
  train; falsifier did not fire; axis real-but-redundant with year
  continuity (−0.40m vs sym_24 with year continuity).
- b opponent quality: predicted +0..+1.5 → measured −0.85m (SIGN WRONG,
  inside floor). Train gain +0.40m did not transfer. Axis dead at this power.
- c anomaly margin: predicted +0..+1.5 → measured −0.55m (SIGN WRONG, inside
  floor). Train gain +0.96m did not transfer. Dead at this power.
- d event class: predicted +0.5..+2 vs sym control → measured +0.27m (sign
  right, size under prediction, inside floor). On-v6 addon +0.24m, positive
  in all 6 subpops — only such config; still inside floor everywhere.
- e patch boundary: predicted +0..+2 vs sym control → measured −3.09m
  (SIGN WRONG); FALSIFIER FIRED (holdout delta vs sym control <= 0). Largest
  train-holdout reversal of the wave (best train LL of any config → −5.5m
  holdout, p .022/.001). Axis killed at γ<=0.7.
Symmetric-vs-asymmetric: pre-committed wording case 3 — "unresolved either
way at n=1217: asymmetry is not demonstrably needed, nor demonstrably
better." No symmetric axis reached preponderance (all mean d < 0 vs v6).

5c — sign predictions HELD everywhere (all b_form negative on train:
wr3 −0.118 vs old-frame −0.0872 reference; rd3 −0.0059; side −0.566;
player −2.681); "inside noise floor" size prediction held for wr/rd/side/
combined (−0.27..−0.98m); player R2.0 landed −5.24m (worse than predicted
band, bare-significant negative .007/.004, sub-MDE 6.04) — the overfit case.
Published verdict: performance-defined form adds nothing; v7's
mean-reversion finding replicates under performance definitions.
DEVIATION LOGGED: first fit pass used default Nelder-Mead tolerances and
stalled at the x0 vertex for side/player (b_form −0.0237 both, artifact);
refit with xatol 1e-8 + train-gain identifiability metric before any verdict
was published. No holdout contact in the re-fit decision (train-side
diagnosis only).

5e — panel published for all 16 verdict-carrying configs (S1 731, S2 730,
S3 130, S4 1118, S5 1189 holdout rows). No subpop WIN anywhere. S1 recompute
validated 100% (165/165) on orgs untouched by corpus additions; 77% overall
= interleaving effect, expanded values used as preregistered. Instrument
note recorded: family-MDE bucket tags overstate resolution for
rating-perturbing within-family addons (rot_on_v6, form_player5; empirical
pair σ cross-family-sized) — headline verdicts unaffected.

CLARIFICATION (pre-run ambiguity, resolved before 5b ran, logged): axis-a
"own-HL sym control" was preregistered as the 5a sym runs; the exact γ=0
control additionally requires year_cont off, so BOTH controls (sym_HL_nc and
sym_HL) were run and reported. No selection depended on the choice.


════════ preregister.lineups.md ════════

# Pre-registration — agent:lineups (written 2026-07-28, BEFORE building)

Scope: per-match fielded lineups + walk-forward lineup features, LOCAL data only.
Sources: data/maps/<event_id>.csv (canonical, mirrors engine.load_match_lineups),
data/match_dates.json, data/enriched/vlr/<mid>.json (validation + player_id),
MoreTestingMaybeFiles.ALL_EVENTS (imported once at start; re-checked at end),
BuildMapRatings.EVENT_DATES, engine game list via testing_lab/engine.load_games_real_dates().

## Keys, dates, ordering (fixed before any computation)

- **Player key**: `ProfileURL` verbatim from the maps CSV. `player_id` := the numeric
  segment of ProfileURL (`/player/(\d+)/`), which is VLR's player id (verified:
  koldamenta URL id 339 == enriched player_id "339"). The enriched JSONs are used as
  an independent *validation* join — (casefold(player name), org) within the match —
  and their join rate + id-mismatch count are reported in the coverage JSON. This is
  declared now because ProfileURL definitionally embeds the id; pretending the id is
  "unknown" without the enriched file would be a fake coverage number.
- **Fielded lineup** L(org, match) := set of ProfileURLs with ≥1 player-map row for
  that (Org, MatchID) — identical grouping to engine.load_match_lineups(). Union over
  maps; n_players > 5 ⇒ mid-series substitution, flagged, never collapsed.
- **Match date** D (day granularity), resolution order, with `date_source` recorded:
  1. `match_dates.json[str(mid)]` (`match_dates`)
  2. engine-interpolated date from load_games_real_dates() (`engine_interp`)
  3. `EVENT_DATES[event_id][0]` (`event_window`) — last resort, flagged.
- **Org sequence**: org's matches sorted ascending by (D, match_id).
- **Walk-forward rule (binding)**: every historical aggregate for a match at date D
  uses only the org's matches with **date strictly < D**. Same-day matches are never
  history for each other (day-granularity, matches v8 rule 1). The current match's
  own lineup is the row's *subject*, never part of its own history.
- **Event class**: `ewc` if the ALL_EVENTS entry has `ratings_only: True`, else `vct`.

## Artifact 1 — testing_lab/v8/data/lineups.csv

One row per (match_id, org): `match_id, org, date, date_source, event_id,
event_class, players` (';'-joined sorted ProfileURLs), `player_ids` (';'-joined,
aligned to players order), `n_players, n_maps` (distinct MapNum), `multi_lineup_flag`
(n_players > 5), `short_lineup_flag` (n_players < 5), `source` (= maps_csv).

## Artifact 2 — testing_lab/v8/data/lineup_features.csv

One row per (match_id, org). All history strictly earlier per the rule above.
W30 := org matches with 1 ≤ (D − date)days ≤ 30.

- **modal5_30d**: per player p over W30, c(p) = # matches whose L contains p;
  rank by (−c(p), −most-recent-appearance-date(p), ProfileURL asc); take top 5
  (fewer if <5 distinct players appeared; then `modal5_short=1`). Deterministic.
  If |W30| = 0: modal undefined → modal5_30d empty, `no_prior_30d=1`.
- **overlap_modal** = |L ∩ modal5_30d| / 5; NaN if modal undefined.
- **n_modal_matches** = |W30|.
- **overlap_prev** = min(1.0, |L ∩ L_prev| / 5); L_prev = lineup of the latest match
  with date < D (tie on date → larger match_id). NaN if no prior match.
- **stand_in_flag** = 1 if modal defined and overlap_modal < 1.0; 0 if modal defined
  and overlap_modal = 1.0; NaN if modal undefined. **n_standins** = |L \ modal5_30d|
  (NaN if modal undefined).
- **matches_since_change** = # consecutive matches at the tail of the strictly-earlier
  org sequence whose L equals this match's L (walk back while equal; 0 if the
  immediately previous lineup differs or no prior match).
  **games_since_change** = sum of n_maps over those counted matches.
- **first_after_break_45d** = 1 iff org has ≥1 prior match ever AND none with
  1 ≤ (D − date)days ≤ 45. `org_debut` = 1 iff no prior match ever (then
  first_after_break_45d = 0).
- **offseason_absence_flag** = 1 iff stand_in_flag = 1 AND event_class = ewc
  (brief-exact definition).
- **Robust supplement (declared now, because EWC follows ≥30 quiet days and the 30d
  modal is expected to be undefined there — the brief-exact flag alone would NaN out
  exactly where Phase 2 looks):**
  - **modal5_vct** := modal five (same count+tie-break rule) over the org's last ≤10
    matches in `vct`-class events with date < D; undefined if none; basis size
    `n_vct_basis`.
  - **overlap_vct_modal** = |L ∩ modal5_vct| / 5; **stand_in_vs_vct_flag** analogous
    to stand_in_flag; **offseason_absence_vs_vct** = stand_in_vs_vct_flag=1 AND ewc.
  Both the brief-exact and the supplement numbers are reported side by side; neither
  is tuned on anything.

## Artifact 3 — testing_lab/v8/data/modal5_by_org_date.csv

One row per (event_id, org) for every org appearing in that event in lineups.csv,
computed at S = EVENT_DATES[event_id].start using only matches with date < S:
`event_id, org, event_start, modal5_30d_pre` (30d window ending S−1), `n_30d,
modal5_last10_vct, n_vct_basis`. Same modal + tie-break rules.

## Artifact 4 — testing_lab/v8/stats/lineups_coverage.json

Audit against the engine's own game list (load_games_real_dates()): unique
(match_id, org) sides from engine games vs lineups.csv. Report: total sides, covered,
pct (overall + per event), engine game/match counts per event, named gap list
[{match_id, org, event_id, reason}], n_players>5 and <5 counts + lists, enriched
file-presence rate and player-level enriched join rate + id mismatches, stand-in
rates by event class (brief-exact and vs-VCT-modal, plus no_prior_30d share by
class), registry snapshot start/end + any top-up.

## Coverage bar (declared)

Complete = **≥99.0%** of engine (match_id, org) sides present in lineups.csv with
n_players ≥ 5. Actual number reported regardless; every miss named.

## Not done here

No network. No writes outside the six declared paths. No tuning of any constant
against anything — this is a data/feature deliverable; no holdout contact.


════════ preregister.page.md ════════

# preregister.page — agent:page (Wave 3 reporter), written BEFORE the build

2026-07-28. Scope per briefs/page.md. This file is trivial by design: the
reporter runs NO experiments and makes NO statistical claims of its own.

## What I will NOT compute
- No model fits, no holdout scorings, no bootstraps, no CIs on any model
  comparison, no bucket verdicts, no p-values. Every verdict, Δ, CI, MDE,
  and p on the page is read from an existing stats/*.json (or from
  scratch/adversary/*.json, copied verbatim into a stats/page_*.json so the
  page serves it). The ADVERSARY-AMENDED verdicts are the published verdicts.
- No new looks at the holdout: I never score a model on anything.

## What I WILL compute (mechanical reshapes only, one writer: me)
1. stats/page_v7_ladder.json — v7_reclass.json rows + a derived boolean
   `amended_loss` = (|Δ| ≥ 5.889m cross floor AND sig_block), asserted to
   equal exactly the adversary's four named configs (sym_6, sym_8,
   surprise_12_20, boxexp_c3_hl8); build FAILS LOUDLY if not.
2. stats/page_adversary_fragility.json — verbatim copy of the machine
   numbers in scratch/adversary/recompute_{eclass_cold,3e,compose}.json
   (drop-top-5%, jackknife, floors). No arithmetic beyond copying.
3. stats/page_prereg_scatter.json — predicted-band vs realized rows curated
   from preregister.*.md outcome sections. Each row carries source_file +
   source_quote; the deriver asserts the quote appears verbatim (whitespace-
   normalized) in the file, else the build fails.
4. stats/page_reliability.json — DESCRIPTIVE reliability bins (favorite
   frame: p_fav = max(p, 1−p), y = favorite won) for v6 / ss_1b / ss_5d from
   scratch/bias_h3/model_probs.npz on holdout rows (date > 2024-12-31).
   Per bin: n, mean predicted, empirical rate, Wilson 95% interval (display
   furniture required by the brief's visual conventions — not a test).
   Bins with n<15 merged into neighbors per the house convention.
5. stats/page_mde_curve.json — MDE₈₀(n) = 2.8016·σ_adj/√n evaluated on a
   grid of n from the σ_adj values already stored in
   power_mde_expanded.json, plus n_for_2m = (2.8016·σ_adj/0.002)² (the
   phase-0 "≈10× the holdout" arithmetic, restated from stored σ).
6. stats/page_slate.json — v6 snapshot prices for data/upcoming_matches.json
   via trading_model/predict.py (frozen benpom-v6-2026-07-22 snapshot).
   Mechanical model evaluation of the production surface; no judging.
7. stats/page_timeline.json — session timeline: first/last timestamp parsed
   from logs/<agent>.log plus a one-line outcome per agent (prose; any
   number in it must also exist in a stats JSON).
8. stats/ledger_v8_updates.json — §10 deliverable: do-not-retest additions
   (each pointing at its kill-evidence stats file + the killing numbers
   copied from it), the UNRESOLVED re-openable list copied from
   ledger_reclass.json, the amended headline verdict set, and the tripwire
   thresholds/windows (editorial policy constants, owned by this file so
   the HTML hardcodes nothing).

## Rendering rules I bind myself to
- gen_v8_report.py renders the page from testing_lab/v8/stats/*.json ONLY;
  no numeric literal that represents a measurement appears in the HTML
  template source.
- adversary_report.md publishes verbatim (HTML-escaped only) in its own
  clearly-marked section; no edits, no elisions, no commentary inside it.
- Every chart carries a download link to the exact JSON it renders from.
- "THE HOLDOUT IS SPENT" renders as a banner, not body text.
- The six existing pages' nav strips are string-patched (v7 Lab anchor →
  + v8 Lab anchor); the old pages are NOT regenerated.

## Failure mode
Any assertion failure (sha-style quote checks, ladder-flag mismatch,
missing JSON key) aborts the build with a loud error; nothing is silently
substituted (README rule 6).


════════ preregister.power.md ════════

# Pre-registration — agent:power (Phase 0: power analysis)

Written 2026-07-28, BEFORE computing any result below. Inputs are frozen:
per-series probabilities from `out/v7_probs.npz` / `out/v7_probs2.npz` (v7 era,
1695 rows, holdout n=1007), aligned to `harness.load_series()` rows 0..1694 —
alignment already verified by reproducing all 24 published `ll_test` values to
5 dp and the two motivating seed-7 bootstraps to machine precision
(logs/power.log T1). No model constant is retuned anywhere in this phase; no
new model is fit. Estimating the NOISE of the test statistic on the holdout is
measurement of the instrument, not holdout tuning — the holdout is never used
to select a model here.

## 1. MDE estimator (overall)

Per candidate pair (candidate c vs champion v6), per-series loss difference
`d_i = l_v6,i − l_c,i` with `l = −log(p_winner)`, on the 1007 valid holdout
series. σ_d = sample SD (ddof=1).

- **Analytic MDE** (two-sided α=0.05, power 80%):
  `MDE_80 = (z_.975 + z_.80)·σ_d/√n = 2.8016·σ_d/√n`.
- **Two regimes, reported separately** (σ_d differs by construction):
  - *within-family* — variants sharing the v6 consistency core:
    v5_asym_W20L12, consist_16_10, consist_14_8, consist_12_8,
    v6_consist_20_12+form{3,5,8}.
  - *cross-family* — structurally different decay/shape:
    sym_{6,8,10,12,14,16,20,24}, surprise_{12_20,16_24}, boxexp_{c3_hl8,c5_hl10},
    power_t6_a15, sym_20+form{3,5,8}.
  Regime headline = median σ_d over the regime's pairs; full spread reported.
- **Simulation verification** on two representative pairs (consist_16_10 =
  within; sym_20 = cross): recenter d (subtract its mean), inject shift
  μ ∈ {0.5, 1.0, 1.5}×MDE_analytic, resample n=1007 iid, run the actual
  paired-bootstrap decision (percentile 95% CI excludes 0 ⇔ p_better outside
  [.025,.975]); 400 simulations × 2000 resamples, seeds from crn.json
  mc_seeds. Acceptance: simulated power at μ=MDE_analytic in [0.74, 0.86]
  (binomial noise at 400 sims). If outside, the simulated MDE (interpolated μ
  at 80% rejection) supersedes the analytic number and the discrepancy is
  reported.

## 2. MDE per bucket

Same estimator on bucket subsets of the holdout; bucket n as observed. Bucket
list (fixed now): format bo1/bo3/bo5/bo5_gf; stage groups/playoffs/
grand_final; international/domestic; EWC-class events; CN involved;
cross-region; domestic Americas/EMEA/Pacific/CN; favorite band [.5,.6),
[.6,.7), [.7,.8), [.8,.9), [.9,1] (band of max(p,1−p) under frozen v6 probs);
rating-gap bands <1.5, [1.5,4), [4,7), 7+ (|rd__v6_consist_20_12|);
post-break (either team's days since previous series > 45, computed over the
full series history; target = reproduce v6_profile n=135, definition recovery
documented if a variant is needed); elite-vs-floor and form-shift (frozen
arrays in the npz); roster-change recency from
`testing_lab/v8/data/lineup_features.csv` — if absent at finish time, that
bucket is reported PENDING DATA, not approximated.
EWC-class = holdout event_ids matching ewc/evolution-series (target =
reproduce v6_profile n=109; actual id list documented).
Per bucket: σ_d median within each regime → MDE per regime.

## 3. v7 ladder re-adjudication (all 24 configs)

For every config vs v6_consist_20_12: mean Δ (milli-LL/series), iid bootstrap
CI + P(better) under (a) legacy seed 7 (reproduction of record) and (b)
crn.json canonical seeds; block-by-event bootstrap CI (crn block seed, unit =
event_id, 18 events). **Verdict rule (fixed now): DISTINGUISHABLE iff
|mean Δ| ≥ MDE_80 of that pair; else INSIDE NOISE FLOOR.** Secondary flags:
iid CI excludes 0; block CI excludes 0. sym_20 and consist_16_10 reported
explicitly. v6 appears as its own reference row.

## 4. Ledger reclassification (published ledger; generator source is authoritative)

Note recorded now: the authoritative source `gen_final_model.py` REJECTED list
and the published HTML both contain **32** entries, not the 33 the brief
states; classified as-is and the discrepancy reported.

Status semantics (fixed now, applied to the IDEA):
- **REFUTED** — idea stays dead: (i) mechanism-rejected (bug, double-count,
  leak, structural fit-to-noise) — flagged `mechanism`, immune to magnitude; or
  (ii) recovered effect worse than champion by ≥ its MDE at time of test.
- **UNRESOLVED** — no mechanism ground and recovered |Δ| < MDE (verdict was
  inside the noise floor), or magnitude UNRECOVERABLE from out/*.json.
  UNRECOVERABLE is marked explicitly, never silently dropped.
- **CONFIRMED** — re-analysis shows the idea distinguishably BETTER (≥ MDE)
  than its era champion, i.e. the rejection was itself wrong. Expected rare.
MDE at time of test = 2.8016·σ_regime/√n_at_test, with σ_regime from §1
measured on the current frozen probs (within-family σ for solve-tweak ideas,
cross-family σ for structural ideas) and n_at_test recovered from the source
JSON where present (else current 1007, flagged `n_assumed`). Tie-magnitude
recovery searched across `testing_lab/out/*.json`; the source file for each
recovered number is recorded in the row.

## 5. Variance reduction (each reported as effective-n multiplier)

(a) **Pairing**: (var(l_c)+var(l_v6))/var(d) per pair (unpaired two-sample vs
paired variance of the mean difference at equal n).
(b) **Control variate**: x = l_v6 − mean(l_v6); OLS slope β̂ on the holdout;
variance ratio var(d)/var(d−β̂x) = 1/(1−ρ²). Caveat reported: for
candidate-vs-v6 the covariate is a component of d — the ratio is the design
number for FUTURE candidate-vs-candidate tests and CUPED-style adjustment; the
point estimate of mean Δ is unchanged in-sample.
(c) **Blocking by event**: DEFF = var_block(mean d)/var_iid(mean d) from the
two canonical bootstraps. DEFF > 1 means iid CIs understated σ (reported as an
inflation, not a reduction — this is the honest direction).
(d) **Multivariate CV**: residualize d on [l_v6, |rd_v6|, p_fav_v6];
incremental ratio over (b). No other designs.

## 6. What would change the program's conclusions (fixed before results)

- Cross-family MDE(overall) > 2.0 milli at n=1007 ⇒ every past near-tie
  rejection with |Δ| < 2 milli was a coin flip dressed as a verdict; the v7
  "no promotion" outcome stands only as "not distinguishable", not "worse".
- Within-family MDE < 0.5 milli ⇒ tightly-coupled variant verdicts
  (consist_16_10-class) were adequately powered and stand as stated.
- ≥ 1/3 of ledger entries flip UNRESOLVED ⇒ the do-not-retest ledger cannot
  gate future work without power annotations; the v8 page must carry them.
- Block-vs-iid DEFF > 1.3 ⇒ all historical iid CIs were too narrow; borderline
  verdicts weaken further and block CIs become the program standard.
- CV multiplier ≥ 1.5× ⇒ the CV-adjusted paired test becomes the recommended
  standard for referee.py (Phase 6).

## Randomness

All new numbers use crn.json (iid seed 20260728, block seed 20260729, mc_seeds
pool); legacy seed 7 used only to reproduce published records. n_boot = 4000
for CIs; simulations as in §1.

---

## POST-CORPUS ADDENDUM (2026-07-28, after agent:corpus landed; written BEFORE computing)

Production rating files are NOT rebuilt (`data/rating_timeline*.json` untouched);
per-series probabilities remain the frozen v7 arrays. Two expanded-corpus MDE
estimators, reported side by side:

**E1 — Mechanical scaling.** I count scoreable holdout series myself from the raw
corpus (`data/match_results.csv` per-map rows + org pairs from `data/maps/<event>.csv`
player rows + `data/match_dates.json`), mirroring harness rules exactly: both orgs in
ORG_REGIONS (junk-org filter), series score from per-map winners, winner maps wm > lm
and wm in {1,2,3} (forfeit/incomplete filter), de-dup MatchID, holdout = date >
2024-12-31. Prefranchise (v8/data/prefranchise/) is excluded; I verify no 2021-22
dates exist in match_dates.json. MDE_naive(regime) = 2.8016 * sigma_regime_old /
sqrt(n_new), i.e. the Phase-0 regime MDEs scaled by sqrt(1007/n_new). Calibration
targets the estimator must reproduce before being trusted: 2,068 total raw series,
~1,223 raw 2025+ series, ~+335 raw new series (corpus agent's counts).

**E2 — Composition-adjusted.** Partition (fixed now): intl = event_id contains
masters/champions/lock_in (harness rule); ewc_offseason = EWC/Evolution families plus
off-season invitationals (rbhg, ten_*, radiant_*, convergence, fgc, acl and any new
registry event of that archetype); vct_domestic = everything else (official-circuit
events incl. LCQ/champions quals/league). sigma_class per regime = median over the
regime's frozen v7 pairs of sd(d) on that class's OLD-holdout rows (computed from the
frozen npz probs only). sigma_adj^2 = sum_class w_class * sigma_class^2 with w_class =
class shares of the EXPANDED scoreable holdout. MDE_adj(regime) = 2.8016 * sigma_adj /
sqrt(n_new). Assumptions stated with the result: class sigmas transfer from old to new
series of the same class (new off-season events are mapped to the measured EWC-class
sigma — if anything an understatement of their noise); d-distributions stationary.

**Decision fixed now: the checkpoint quotes the composition-adjusted number** (the
conservative one); mechanical scaling is reported alongside as the optimistic bound.
No resampling is planned (point estimates from frozen vectors); if any resampling
becomes necessary it uses crn.json seeds. Failure accounting: every raw new series
that fails scoreability is counted with its reason (org not in ORG_REGIONS / no map
data / invalid series score / no date). n_holdout_old = 1007 (frozen npz era);
organic post-npz growth (timeline rows after 2026-07-23) is reported separately from
corpus additions, identified by event: corpus additions = events absent from the old
timeline frame.


════════ preregister.referee.md ════════

# Pre-registration — agent:referee (Phase 6 metric suite)

Written 2026-07-28, BEFORE referee.py is coded. Everything below is frozen:
formulas, acceptance bars, and promotion-gate clauses. Any later deviation gets
written down as a deviation.

## 0. Conventions

- All probabilities are **winner-referenced** unless stated: `p[i]` = the
  probability the model assigned to the eventual winner of series i, so
  per-series loss is `-log(p[i])`. Matches harness/engine/v7 npz convention.
- Paired deltas follow the harness sign convention: `d = loss_B - loss_A`,
  positive = A better.
- "Holdout" = series date > 2024-12-31 (BETA_TRAIN_END), valid = non-NaN
  rating diff.
- House reliability-bin shape everywhere:
  `{bin_lo, bin_hi, pred_mean, emp, n, ci_lo, ci_hi}` with 95% Wilson CI
  (z=1.96), identical math to harness.reliability.

## 1. Canonical baseline (reconciliation, pre-committed)

- **CANONICAL: v6 holdout LL 0.64095, n_test=1007** — artifact
  `out/v7_stage1.json` → `results.v6_consist_20_12.ll_test`, per-series probs
  in `out/v7_probs.npz` (key `v6_consist_20_12`, mask `~isnan(rd__…) & test_v`).
  Jul-23 native data frame (1695 valid), β=0.1294 refit on ≤2024.
- 0.6409 (final_model report) = same champion on the **Jul-22 frame**
  (`out/rd_v6_native.npy`, β=0.12556 from `out/v6_native_beta.json`): holdout
  n=999, pooled LL 0.64085 → displayed "0.6409". This frame is what
  `out/v6_profile.json` (bias table, 23 buckets, 4 bands) was computed on.
- 0.64126 (favorites report) = prose-hardcoded number from the mid-session
  v6-correction build (pre-final frame). No artifact; superseded.
- Frame-prefix condition: both historical frames must be exact prefixes of
  today's `harness.load_series()` ordering (sort: date, match_id). This is
  *verified, not assumed*, by the self-test (bias table 42/42 + 23 buckets +
  bands + cold-start all exact). If a future data rebuild edits pre-2026-07-22
  rows, the self-test fails loudly — by design.

## 2. Metric formulas (exact)

### 2.1 per_series_ll
`per_series_ll(p) = -log(clip(p, 1e-9, 1))`, vector. Aggregate = mean.
`delta_vector(p_a, p_b) = per_series_ll(p_b) - per_series_ll(p_a)` (>0 ⇒ A
better).

### 2.2 paired_bootstrap_crn(d, mode)
- Reads `testing_lab/v8/crn.json` **at call time**; absent ⇒ raise
  (RuntimeError). Never a private seed. Schema per briefs/power.md §6.
- `mode='iid'`: rng = `np.random.Generator(PCG64(crn.bootstrap.seed))`,
  `n_boot = crn.bootstrap.n_boot`; resample b: indices
  `rng.integers(0, n, n)` drawn n_boot times in sequence;
  stat_b = mean(d[idx_b]).
- `mode='block_event'`: requires aligned `event_ids`; unit must equal
  `crn.block_bootstrap.unit == "event_id"`; rng seeded from
  `crn.block_bootstrap.seed`; each resample draws `n_events` events with
  replacement (`rng.integers(0, n_events, n_events)` over the sorted unique
  event list) and concatenates their member rows; stat_b = mean over
  concatenated rows.
- Returns `{mean_delta, ci_lo, ci_hi (percentile 2.5/97.5 of resampled means),
  p_better = mean(stat_b > 0), n, n_boot, seed, generator, mode,
  crn_file, crn_verify}` — full provenance in every result.
- When `len(d) == len(crn.holdout_order)` and the verify block exists, the
  first-100 iid index draws are hashed (sha256 of the concatenated int64
  bytes) and compared; mismatch is reported in `crn_verify` (and treated as a
  blocker in any result that carries it).

### 2.3 bucketed(frame, p, p_ref=None)
Bucket definitions frozen to the recovered v6_profile/final_check set
(evaluated on holdout∧valid rows unless stated; bucket emitted when n ≥ 15):
- year ∈ {2025, 2026}; format ∈ {bo1, bo3, bo5, bo5_gf}; stage ∈ {groups,
  playoffs, grand_final}.
- international = `_is_intl_event(event_id)` (harness: masters/champions/
  lock_in substring); domestic = ¬international.
- CN involved = reg_w=CN ∨ reg_l=CN; cross-region = reg_w≠reg_l;
  domestic {Americas, EMEA, Pacific, CN} = same-region pairs.
- gap bands on |rdiff|: `<1.5`, `[1.5,4)`, `[4,7)`, `≥7`.
- favorite bands on max(p,1−p): `[0.5,0.6) [0.6,0.7) [0.7,0.8) [0.8,0.9)
  [0.9,1]`, excluding exact ties (|p−0.5| < 1e-9).
- EWC-class = event_id startswith ("2026_ewc", "2026_china_evo").
- post-break = days since each team's previous series in the dataset; BOTH
  teams must have a prior series (debuts excluded); bucket = either rest > 45
  (strict).
- cold-start = |r_w| < 5e-4 ∨ |r_l| < 5e-4 (rating exactly 0 ⇒ unrated);
  evaluated over the full frame (deep1 convention), not holdout-only.
- elite vs floor = decayed win-rate machinery from run_v7_stage1 (per-game
  exp decay, HL = 16 team-games, denominator > 3, as-of series date):
  max(wr16_w, wr16_l) ≥ 0.60 ∧ min ≤ 0.40. form-shift = |wr5 − wr16| ≥ 0.15
  for either team (same machinery, HL=5).
- roster-recency hook: when `v8/data/lineup_features.csv` exists, join on
  match_id and bucket by days-since-lineup-change: `≤14d, (14,45], >45d,
  unknown`. Column autodetected from a declared candidate list; absent file ⇒
  buckets reported as PENDING DATA, never silently approximated.
Output per bucket: `{name, n, ll, fav_acc}` (+ `ll_ref, delta_milli` when
p_ref given). delta_milli = (ll_ref − ll)×1000, positive = p better.

### 2.4 per_team_bias(p, winners, losers, min_n=25)
Recovered definition (gen_final_model.py L298-300, verified 42/42 exact):
for each team T over holdout∧valid series it played,
`p_T[i] = p[i]` if T won series i else `1 − p[i]`; `won_T[i] ∈ {1, 0}`;
**`bias(T) = mean(p_T) − mean(won_T)`** — probability points (rendered ×100
as "pts"; negative = model under-rates T). Team LL = `-mean(log(p on T's
realized outcomes))` = plain series LL restricted to T's matches. Teams with
n ≥ 25 only. Returns full sorted table + `max_abs_bias`, `mean_abs_bias`,
`n_teams`. This is a PROMOTION-GATE input.

### 2.5 pnl_weighted_ll(p, won, price)
- price in cents (values ≤ 1 auto-scaled ×100); per-row loss
  `-log(p if won else 1−p)`.
- Weight of row = density mass of the price band containing price, from
  `v8/stats/quote_density.json` (accepted shapes: `{bands:[{lo_cents,
  hi_cents, density}]}`, `{"lo-hi": density}` dict, or a bare list of band
  objects). Weights normalized to mean 1 over scored rows.
- **FALLBACK (until agent:autopsy delivers the file): uniform density over
  the 20–55¢ quoted band, zero outside** — every output carries
  `weights_source: "FALLBACK_uniform_20_55c"` vs `"quote_density.json"`.
- Returns `{ll_weighted, ll_unweighted, n, n_weighted_nonzero,
  weights_source, band_table}`.

### 2.6 expected-ROI translation `expected_roi_of_dll(dll, p_ref)`
Uses `out/quote_margin.json` (read-only) as the ROI ladder:
δ→ROI points = `(0.0, flat_0c.roi)` plus `(δ, logit_δ.roi)` for
δ ∈ {0.1,…,0.6}; linear interpolation, CIs interpolated alongside.
ΔLL→δ equivalence (first-order, pre-committed): a correctly-signed logit
shift δ on the winner side changes per-series log score by ≈ δ·(1−p), so
**δ_equiv = dll / mean(1 − p_ref)** with p_ref = the reference model's
winner-side holdout probs. Headline translation = ROI(δ_op + δ_equiv) −
ROI(δ_op) at operating point δ_op (default 0). Every output echoes the
formula name, δ_equiv, ladder points used, and the ladder CIs. This is a
*translation for quoting, not a claim of realized P&L*.

### 2.7 fill_conditional_calibration(fills_df, …)
Interface agreed to autopsy's brief (step 4): input a DataFrame of FILLED
markets with model prob (`p_col`, default "p_model", prob of the side the
fill references) and settlement (`won_col`, default "won"), plus optional
unconditional arrays `(p_uncond, won_uncond)` for the same window; optional
slice columns (`side_role`, price band, minutes-to-start band) sliced when
present. Output: house-shape reliability bins over [0,1] (default 10 bins)
for filled and unconditional, per-slice bins, and a summary with
`{n, ll, brier, mean_p, emp, gap = emp − mean_p}` each side plus
`adverse_selection = gap_filled − gap_uncond`. Wilson CIs throughout.

### 2.8 margin_mse(pred_margin, realized_margin)
`mse = mean((pred − realized)²)`, `corr` = Pearson, n. House constructions
provided: realized series avg transformed margin/map =
`mean_maps(sign(wr−lr)·|wr−lr|^0.75·2.5)` winner-referenced
(run_margin2.py L51); predicted = `a_slope · rdiff` with a_slope from
`out/margin_link.json` (0.59290, train-fit).

### 2.9 reliability emitters
`reliability_emit(p, y, n_bins, lo, hi)` — harness Wilson math, house JSON
shape, chart-ready. `favorite_reliability(p_win)` = deep1 semantics
(favorite-side, tie-free, bins [0.5,1.0]).

## 3. Promotion gate (code-to-be, clauses frozen)

`promotion_gate(candidate, v6, mde, …)` evaluates, each clause returning
pass/fail + numbers:

- **G1 support**: paired ΔLL (v6→candidate) with `paired_bootstrap_crn`:
  `mean_delta ≥ mde` (mde supplied from Phase-0 power output — never
  invented here) AND `p_better ≥ 0.95` in **iid** mode AND `p_better ≥ 0.95`
  in **block_event** mode.
- **G2 team bias**: `max_abs_bias(candidate) < max_abs_bias(v6)` on the same
  holdout, min_n=25, strict.
- **G3 no major bucket regression** ("major" pre-committed numerically):
  over the frozen bucket set (§2.3, buckets with n ≥ 30, excluding
  roster-recency while PENDING DATA), a major regression =
  `delta_milli(candidate vs v6) ≤ −4.0` for buckets with n ≥ 100, or
  `≤ −8.0` for buckets with 30 ≤ n < 100. G3 passes iff zero major
  regressions. (−4 milli mirrors the historical run_final_check "worse"
  flag; the looser small-n bar acknowledges bucket noise, cf. Phase-0 MDEs.)
- **Verdict**: PROMOTE only if G1 ∧ G2 ∧ G3. Output object carries every
  clause's inputs, numbers, pass/fail, and the CRN provenance.

## 4. Self-test acceptance bars (frozen)

Targets are published artifact values; "exact" = equality after rounding to
the artifact's printed precision; LL bars additionally satisfy |got−target|
≤ 1e-4. ALL must pass or the summary reports the failure as a blocker.

| # | Reproduction | Target | Bar |
|---|---|---|---|
| 1 | npz v6_consist_20_12 holdout LL, n | 0.64095, 1007 | ≤1e-5, n exact |
| 2 | npz v5_asym / consist_16_10 / sym_16 | 0.64145 / 0.64135 / 0.64262 | ≤1e-5 |
| 3 | native year buckets 2025 / 2026 | 0.63036 n=498 / 0.65127 n=501 | ≤1e-5, n exact |
| 4 | native pooled holdout | (498·0.63036+501·0.65127)/999 | ≤1e-4 |
| 5 | per-team bias table | 42/42 rows of v6_profile.teams | bias ≤5e-5, ll ≤5e-5, n exact |
| 6 | max abs bias | 0.1634 (TS) | ≤5e-5 |
| 7 | all 23 v6_profile buckets | each (n, v6 LL) | LL ≤1e-5, n exact |
| 8 | 4 favorite bands | n 466/335/152/43 + pred/emp | n exact, pred/emp ≤5e-4 |
| 9 | cold-start (deep1) | n=57, LL 0.70205 | n exact, ≤1e-5 |
| 10 | reliability emitter vs deep1.reliability_fav_warm | all bins | every field to artifact rounding |
| 11 | margin_mse vs margin2.marginMSE_v5 | 43.095 / 0.148 / 999 | mse ≤1e-3, corr ≤1e-3, n exact |
| 12 | CRN: absent file raises | RuntimeError | must raise |
| 13 | CRN (if crn.json exists): determinism + sha verify | identical repeat call; verify ok | exact |
| 14 | pnl_weighted_ll fallback labeling + toy identity | FALLBACK label; uniform-band = plain mean | exact |
| 15 | expected-ROI ladder endpoints | ROI(0)=0.0021, ROI(0.6)=0.2911 | ≤1e-4 |
| 16 | promotion_gate logic on synthetic vectors | fail-on-self, pass-on-dominating | logic |

Items whose upstream dependency is absent (crn.json present-path, real
quote_density.json, lineup_features.csv) are reported PENDING/SKIPPED with
the reason — skips are named, never silent.

## 5. Non-goals / forbidden

No rating solves; no modification of harness.py, gen_report.py, out/*;
no writes outside referee-owned paths; no network; no private randomness.

## 6. Deviations (recorded after the runs, per the header rule)

**D1 (2026-07-28) — item 10 split into 10a/10b.** Item 10 assumed
`out/deep1.json`'s reliability bins were reproducible from current data
files. They are not: deep1 ran 2026-07-22 01:43 on a data state that
pre-dates the last data commit (dd0d81f, Jul 17 — everything since lives
uncommitted in the working tree) and has since been mutated in place by
production scrapes (some 2025–26 pre-match ratings re-solved). The state
exists nowhere on disk or in git. Resolution: 10a (acceptance, referee-owned)
= emitter identical to harness.reliability on identical input — PASS exact;
10b (reported as an upstream_data_drift finding, not a referee defect) =
deep1-bin comparison with the drift quantified (warm n 1521 and the 57-row
cold set reproduce exactly; 6/10 bins differ, max |Δn| = 3, max |Δemp| =
0.016). Nothing about the metric formulas changed.

**D2 (2026-07-28) — games source for wr/margin machinery.** The plan
implicitly assumed `engine.Engine().games` as the game source. Discovered:
today's `BuildRatingTimeline.load_all_games()` is missing every EWC-class
map (the site registry moved those events to per-region files after Jul 23,
uncommitted), which would have silently broken elite-floor masks and margin
reproduction. Resolution: `referee.load_timeline_games()` reads maps
directly from data/rating_timeline*.json (the artifact the series frame
itself comes from). Verified exact: npz elite_floor/form_shift masks
reproduce, marginMSE_v5 43.095/0.148/n=999 reproduces. Formulas unchanged;
only the data plumbing differs, and the finding is flagged for other agents
in referee_selftest.json.
