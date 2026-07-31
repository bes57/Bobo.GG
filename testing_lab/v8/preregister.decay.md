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
