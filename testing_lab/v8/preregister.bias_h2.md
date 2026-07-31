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
