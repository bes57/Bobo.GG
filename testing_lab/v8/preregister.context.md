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
