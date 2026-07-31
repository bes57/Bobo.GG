# preregister.family — v9 family mechanics + conformance fixtures

Agent: v9-family. LOCKED 2026-07-29 11:32, BEFORE any implementation
(logs/family.log carries the ordering). Scope: mechanics only — no fitting,
no scoring, no holdout contact. Nothing here computes a log-loss, Brier, or
any aggregate on any outcome, train or holdout. Fixtures compare model
OUTPUTS (ratings, probabilities) for identity/behavior, never quality.

Reuse: speclib.py (v8/scratch/roster/spec_run) is imported as-is for the
causal classifier (P1/P3, date-strict version_asof, chain merge, census
rules). runner.py's EngineSpec/run_config are imported for the solve path
(holdout metrics popped unseen there). fixtures.py executes on import, so
its synthetic helpers (mini_corpus, mini_solve, max_abs_diff) are
copied-with-attribution into the v9 fixture module, unmodified.

## 1. Solve-side member (spec-run machinery, capped)

Classifier: SpecPlan P1, W ∈ {3, 5, 8}, c = 5 FIXED (family definition;
census exposed c=5 as shippable). P3(W=5) available as the frozen-o
variant. P2 remains NOT RUN (v8 spec-run P2_DECISION: region-prior
recursion residue; do not run it wrong).

Weights: per-side game-weight multiplier via EngineSpec._continuity_vec —
boost 1 + a(1−k/5)e^(−n/τ) on post-boundary games (per-game n, 0 at the
boundary match), thin-phase floor (phase_size < n_min ⇒ boost capped at
`cap`), sub down-weight ×[1 − s(1−o)]. Params (policy, W, a, τ, s, n_min,
cap).

THE a ≤ 6.0 LAW (from the a=28 failure): enforced in code, no widening
path. (i) SpecPlanV9.multipliers asserts 0 ≤ a ≤ 6.0 unconditionally —
no flag, kwarg, or env var can relax it; (ii) solve_side_run asserts the
same before touching the engine and requires a SpecPlanV9 instance, so
every v9 solve path passes through the capped multiplier. a > 6.0 must
raise AssertionError (fixture V6). s ∈ [0, 1] asserted likewise.

## 2. Prediction-layer member (NEW; zero coupling by construction)

The solve is PURE v6 for every team — run_config(plan=None), untouched.
Base run yields per-row walk-forward rdiff (ratings for day D solved from
games dated < D only) and β_ref = the engine's standard train-only β fit
(dates ≤ 2024-12-31; the frame holdout never enters it).

Classifier state for pricing team T's row at date D: v = version_asof(T, D)
(date-strict: matches dated < D only; same-day never visible). ACTIVE
phase ⇔ v exists and has ≥ 1 boundary. j* = last boundary's index (current
open phase), k = that boundary's (chain-merged) k, n = v.nvis − j* ≥ 1 =
the count of T's post-change matches knowable at D (boundary match
included). Detection lag applies naturally: before det_date the boundary is
not in v, so δ = 0 — no peeking.

Adjustment, in MAP-LOGIT units, applied before the closed-form series
step and only to rows where a side has an active phase:
  ℓ = β_ref·rdiff (winner-oriented);  ℓ' = ℓ + δ_Wside − δ_Lside;
  pm' = σ(ℓ');  series closed form unchanged (bo1/bo3/bo5 shapes as in
  runner.p_series_closed). Rows where neither side is active are NOT
  touched — the implementation modifies only affected row indices, so
  identity elsewhere is by construction, then asserted anyway.
Magnitude g = b·(1 − k/5)·e^(−n/τ), b ≥ 0 asserted.

- δ2 "atlas intercept": δ_side = g, unconditionally positive toward the
  changed team (population-atlas early-outperformance prior).
- δ1 "evidence-amplifier": δ_side = g·sign(E(T, D)), sign(0) = 0.

### δ1 evidence definition — FROZEN before implementation
E(T, D) = Σ_m [ maps_T(m) − played(m)·p̂_T(m) ]  over evidence matches m:
- m ranges over T's classifier positions i ∈ [j*, v.nvis) — every
  post-change match of T knowable at D (all dated < D by version_asof
  construction; date-strict, so same-day matches are NEVER evidence),
  mapped by match_id to the canonical frame row where T is winner or
  loser. Corpus matches absent from the frame, and rows whose base rdiff
  is NaN, contribute nothing (disclosed skip; no substitution).
- maps_T(m) = w_maps if T is the row's winner else l_maps;
  played(m) = w_maps + l_maps.
- p̂_T(m) = σ(β_ref·rdiff_m) oriented to T (1 − σ(·) when T is the row's
  loser), where rdiff_m is the PURE-v6 walk-forward diff for row m —
  solved from games dated < date(m). Early in a phase this expectation is
  dominated by pre-change games: it IS the pre-change expectation, and it
  converges to the new-roster level exactly as v6 does.
- Leak-proofness: every input to E(T, D) is a function of matches dated
  strictly before D — positions from version_asof (date-strict), scores
  of past matches only, walk-forward rdiff of past rows, and β_ref (a
  train-only constant). No same-day, no future, anywhere. Fixture V7
  flips a FUTURE match's score and asserts the price at D is
  bit-identical; a same-day flip variant is included when the corpus
  offers one.
- sign_evidence = sign(E) ∈ {−1, 0, +1}. No evidence rows, or E = 0,
  ⇒ δ = 0: δ1 pays nothing without evidence — the deliberate philosophical
  contrast with δ2, which pays full g from the first priced match after
  detection. Transfer validation (not this phase) adjudicates.

## 3. Hybrid hook
overlay(base, plan, δ-params) is generic in the base run: hybrid =
solve-side run (a ≤ 6.0 law applies) + δ overlay computed on the hybrid
base's own rdiff/β_ref, same plan instance. Mechanics only; no hybrid
configs are run in this phase beyond the hook existing and being covered
by the nesting fixture (a=s=b=0 ⇒ v6).

## 4. Conformance fixtures (definitions frozen; predicted outcome: PASS
on every assertion; any FAIL voids downstream numbers and is reported at
identical resolution)

- V1 nesting: (i) solve-side with plan attached and a=s=0 vs pure v6 —
  sha256(rdiff bytes) equal AND sha256(stacked daily ratings) equal;
  (ii) hybrid hook at a=s=b=0 ⇒ p_all sha equal to v6; (iii) prediction
  layer b=0 ⇒ p_all bit-identical to v6, rdiff same object/sha.
- V2 SEN zero-boundary (real corpus, window ≥ 2026-05-01, family c=5):
  P1 W∈{3,5,8} — no boundary at the marved match, none in the window;
  P1w5 marved game o = 0.8; prediction-layer δ2 probe (b=1, τ=5, P1w5c5):
  every SEN row in the window bit-identical to v6.
- V3 synthetic revert (mini corpus, change_round=5, non-sustained):
  P1 all W — zero boundaries; P3w5 frozen-o flags exactly the one sub
  game; mini-solve boost-on == boost-off with max|dR| exactly 0.0.
- V4 synthetic sustained: exactly one detection, j=5, k=3, lag-correct
  det at match index {W3: 6, W5: 7, W8: 8} (2nd/3rd/4th new-five match,
  never earlier); one final boundary (5, 3); new-phase o ≡ 1; per-game
  n = 0.. from the boundary; mini-solve identical for ALL teams through
  the detection date, featured team moves after.
- V5 prediction-layer pre-change identity + zero coupling (real frame,
  δ2 b=1 τ=5 P1w5c5; repeated for δ1): for EVERY org T with ≥1 detection
  event, overlay restricted to T: (a) all rows not involving T
  bit-identical to v6 (zero cross-team coupling, per team); (b) all rows
  involving T dated ≤ T's first det_date bit-identical (exact pre-change
  identity, per team, all dates); full overlay: (c) every row where
  neither side has an active phase at its date bit-identical; (d) rdiff/
  ratings sha unchanged — the prediction layer never re-solves.
- V6 the a-cap law: a=6.0 accepted; a=6.000001 and a<0 raise
  AssertionError at both entry points; overlay b<0 raises; grep-level
  check that no bypass parameter exists (the assert is unconditional).
- V7 δ1 leak-proofness: flip a later-dated match's score for an active
  team in a copied frame ⇒ priced p at D bit-identical; same-day flip
  (where available) likewise.

## 5. Cost model (stats/v9_cost.json)
Wall-clock on this machine, no metrics recorded: engine load (once),
pure v6 run, one solve-side walk-forward run (probe a=2, τ=5, s=0.3,
n_min=3, cap=1.5, P1w5c5 — output discarded), SpecPlan build per policy
(P1w3/5/8 c5, P3w5), δ1 and δ2 overlay passes over the full frame, and
the marginal per-config costs the search agent should budget
(solve-side ≈ one engine run per config; prediction layer ≈ one overlay
per config on a shared base run). Machine descriptor included.

## Outputs (this agent only)
scratch/family/v9lib.py, scratch/family/v9fixtures.py,
stats/v9_fixtures.json, stats/v9_cost.json, preregister.family.md (this
file; outcomes appended AFTER runs), logs/family.log.

## ADDENDUM 1 — LOCKED 2026-07-29 11:40, still BEFORE implementation and
before any fixture run. Motivated by two classifier-only reads (logged;
no scoring, no probability computed): SEN P1w5c5 boundary list and the
population of n-since-last-boundary at corpus end.

(a) Active-phase pricing horizon (definition gap): e^(−n/τ) never reaches
float zero, so without a horizon every ever-changed team stays "active"
forever, δ carries 1e−18 tails, and the identity fixtures would fail on
last-bit noise while the mechanics pretend a 2023 change still prices in
2026. FROZEN: the prediction layer prices a phase only while
n ≤ N_max = ceil(5·τ) (five time constants; e^−5 < 0.7% of the phase-entry
magnitude, below any quoting tick). Beyond it δ = 0 EXACTLY (state
reported inactive). N_max is derived from τ — it is NOT a new free
parameter and the search may not tune it independently.

(b) V2 overlay clause corrected (the preregistered clause was wrong):
SEN has a REAL confirmed boundary 2026-04-19 (k=3; n=2 at the window
start), so the marved-case window overlaps a genuine active phase and
"every SEN window row bit-identical to v6" would assert the subsystem
must IGNORE a real roster change — not a conformance property. What F1
actually establishes is that the marved ONE-OFF creates no boundary.
V2's overlay clause is REPLACED BY the counterfactual identity: overlay
on the real corpus vs overlay on the F1 counterfactual corpus (the
marved match's lineup replaced by the reverted johnqt five) —
p_all bit-identical on EVERY row, for BOTH δ1 and δ2 (the one-off sub
contributes nothing to the prediction layer: no boundary either way, and
δ1's evidence reads scores/rdiff, not lineups). Classifier asserts from
F1 are unchanged and rerun at c=5: no boundary at the marved match, none
dated ≥ 2026-05-01, marved game o = 0.8 under P1w5. The SEN April
boundary state (real, active, n=2) is recorded in the fixture output as
expected subsystem behavior, not a failure.

## OUTCOMES — appended 2026-07-29 11:52, after the runs

Fixtures: 67/67 PASSED (stats/v9_fixtures.json). One failure occurred on
the FIRST run and it was in fixture code, not mechanics: the V6
"no-bypass" source scan tested `"environ" not in src`, which matched the
word "environment" in v9lib's own docstring prose describing the law.
Narrowed to `os.environ`/`getenv` (the actual bypass mechanisms);
v9lib.py untouched; rerun clean. Reported at the same resolution as the
passes per the covenant.

Observed conformance facts worth recording:
- V5: 53 orgs carry P1w5c5 detections; per-team overlays touched 2938
  (org,row) pricings; full overlays changed 1741 rows (δ2) vs 1757 (δ1)
  of 2058. δ2 < δ1 because δ2's symmetric positive adjustments cancel
  EXACTLY on both-sides-active rows with equal (n,k), while δ1's
  evidence-signed directions rarely cancel. Both-construction guarantees
  (zero coupling, pre-change identity, rdiff sha unchanged) held bitwise.
- V2: SEN's real 2026-04-19 boundary is active in the marved window
  (n=2, k=3) as ADDENDUM 1b predicted; the marved one-off produced
  bit-identical overlay p_all under the counterfactual corpus for both
  variants.
- V7: leak case org=2G — flipping an Aug-16 score left every row dated
  ≤ Aug-16 bit-identical while evidence priced Aug-17 moved; the in-code
  date-strict assert ran on every evidence row of every δ1 overlay.

Cost (stats/v9_cost.json, Apple Silicon, single process): engine load
0.61 s once; pure v6 run 2.19 s; solve-side run 2.56 s per config
(~1400/h); SpecPlan build ≤ 0.08 s per policy; overlay 0.00–0.02 s per
(b, τ, variant) config on a shared base (~10⁵/h) — the prediction layer
is ~100× cheaper to search than the solve side, before the era-transfer
split multiplier.
