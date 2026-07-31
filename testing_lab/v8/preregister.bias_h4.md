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
