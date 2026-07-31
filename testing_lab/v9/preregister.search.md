# preregister.search — Phase 2: transfer-gated optimization + the frozen ladder

Agent: v9-search. LOCKED before any grid run, any transfer evaluation, and
any VAL-row aggregation (logs/search.log carries the ordering). Governing
law: stats/v9_transfer_protocol.json executed VERBATIM; v9/README.md three
laws; wave2_common.md. Hyperparameter discipline: every data-driven choice
below uses rows dated <= 2024-12-31 (FIT1) ONLY; candidates are frozen
before their single transfer evaluation; the exploratory read budget (0/3)
belongs to the freeze stage and is not touched here.

## 1. Search spaces

### 1a. Prediction layer (the main spend; ~0.02 s/config on a shared v6 base)
Base: ONE pure-v6 run (runner.run_config(None,...)); rdiff untouched;
β_ref = the base run's train-only engine β (FIT1-legal; fixture ≈ 0.1152).
Generalized magnitude (search extension, nests the frozen family):
  g = b · (1 − k/5)^γ · e^(−n/τ),  active only while n ≤ ceil(5τ)
  (horizon derived from τ per family ADDENDUM 1a — not tuned independently).
Direction by variant (E, ne = the FROZEN δ1 walk-forward evidence and its
row count, computed by the family's Overlay.evidence VERBATIM — the
date-strict leak assert executes on every evidence row of every state
build; no reimplementation):
  - delta2: +1 always (atlas prior).
  - delta1(m): sign(E) if ne ≥ m else 0.  m=0 is the frozen pure δ1.
  - hybrid(m): +1 while ne < m (prior), then sign(E) (evidence takes over).
    sign(E)=0 with ne ≥ m ⇒ no adjustment.
Grid (full factorial, FIT1 scoring only):
  policy ∈ {p1w3c5, p1w5c5, p1w8c5, p3w5c5}   (4)
  variant ∈ {delta2, delta1(0), delta1(3), hybrid(3), hybrid(6)}   (5)
  b ∈ {0.05, 0.10, 0.15, 0.20, 0.30, 0.45, 0.65, 0.90, 1.20}   (9)
  τ ∈ {2, 3, 5, 8, 13, 21}   (6)
  γ ∈ {0.5, 1.0, 2.0}   (3)
  = 3240 configs. c=5 fixed (family law). The a ≤ 6.0 law is inherited
  (no solve-side parameter exists in this member).
Caching disclosure: per (policy, τ) the per-(org,row) state (n, k, E, ne)
is computed ONCE through the real active_state/evidence code path (asserts
hot), then configs are arithmetic on that state. Fixture S2 proves the
cached path bit-identical to the full un-cached Overlay path; every
LEDGERED transfer evaluation and the ladder freeze use the full un-cached
path regardless.

### 1b. Solve-side (presumed dead — cheap verification, not excavation)
Prior: gate autopsy — every tested a transfers negative (a=6 pooled −5.38m,
a=4.5 −3.8/−4.6m, a=28 blocked everywhere). Verification points, straight
to protocol transfer evaluation (that IS the cheap verify; 3 ledger slots):
  a ∈ {0.5, 1.0, 2.0} at W=5 (family workhorse; the only W with stored
  small-a τ-profiles), (τ, s) per a = argmin of stage_manifest ll_train
  (train-only, FIT1-legal) over the stored p1w5c5 configs at that a;
  n_min=3, cap=None; rdiff from stage_runs.npz (deterministic, same arrays
  the autopsy used). Chosen (τ, s) values recorded at run time.
No new solve-side engine runs, no widened grids, no a > 2 re-tests.

### 1c. Hybrid (solve+δ)
Run ONLY if ≥1 solve-side point AND ≥1 prediction-layer candidate BOTH
advance (independent transfer signal in both parents). Expected: skipped
(solve-side presumed dead). If triggered, a single hybrid candidate
(advancing solve config + advancing δ config, both frozen as-is) would be
preregistered by amendment BEFORE its evaluation.

## 2. Grid scoring + candidate-selection rule (frozen)

For every grid config, paired per-series d vs the shared v6 base
(referee.delta_vector; both sides at β_ref; overlay never re-solves) on
FIT1 valid rows. Recorded per config (milli-LL):
  dF (mean, all FIT1), d23 (year 2023 rows), d24 (year 2024 rows),
  era_min = min(d23, d24), loeo_min (leave-one-FIT1-event-out min of the
  recomputed mean, 25 events), n_affected_FIT1.
ELIGIBLE iff: dF ≥ +1.0m AND era_min > 0 AND loeo_min > 0.
  (Justification: era_min/loeo replicate the transfer/fragility shape
  inside legal data — Law 2; the +1.0m floor ≈ half the FIT1 pair-MDE
  (2.13m at n≈841): below it the pooled A4 bar (1.773m) is implausible
  and a one-shot ledger slot would be wasted.)
NOMINATION (≤3 prediction-layer candidates): the top ELIGIBLE config by
era_min (descending) in EACH of the three variant families —
  {delta2}, {delta1(0), delta1(3)}, {hybrid(3), hybrid(6)} —
one candidate per family, preserving the philosophical contrast the
transfer referee is meant to adjudicate. A family with no eligible config
nominates nothing (fewer candidates, stated plainly). Ties: higher dF,
then smaller b, then smaller τ, then γ nearer 1.0, then policy order
w5, w3, w8, p3w5.
C5 report per nominee: 5 contiguous time folds over FIT1 valid rows
(equal row counts); flag CONCENTRATED if dropping any single fold takes
the remaining mean d ≤ 0. Flag, not kill — carried in the nomination blob.
Solve-side points carry no inner-CV gate score (manifest ll_train only);
C5 reported not-applicable for them, disclosed.

## 3. Transfer evaluation (the law, executed once per candidate)

Exactly v9_transfer_protocol.json: frame sha asserted vs crn.json AND the
protocol blob; T1 β on FIT1 → paired d vs v6 on VAL1; T2 β on FIT2 → VAL2;
β by minimize_scalar bounded (0.001, 1.0) xatol 1e-6 on window series-NLL
of p_series_closed(β, rdiff, fmt); v6 refit identically (paired); fixture
β(FIT1, v6) = 0.1152 ± 1e-3 or abort. Overlay candidates: rdiff is v6's,
so β_cand(window) = β_v6(window); the frozen adj (map-logit units;
δ1 evidence uses β_ref per the family preregister — a train-only constant)
rides on top: p = closed-form(σ(β_win·rdiff + adj)). Judge:
referee.paired_bootstrap_crn iid + block_event (primary), crn.json seeds,
SE_blk = blk CI width/3.92. Clauses A1–A5 verbatim (A5 only if A1–A4
pass); pooled = concat of the two window d vectors; MDE context and the
expected-ROI translation quoted on every number. ONE evaluation per
candidate, appended to stats/v9_looks.json selection_reads AT RUN TIME
with {candidate, date, T1, T2, pooled, verdict, clauses_failed}. NO
retuning after seeing transfer scores: this phase nominates NO informed
candidates — if all die, the ladder is v6 alone, period.
Total ledger spend this phase: ≤ 6 entries (3 solve-side + ≤3 overlay)
+ 1 amendment-gated hybrid (expected unused).

## 4. Ladder freeze

stats/v9_ladder.json: arm 0 = v6 always; + survivors (≤3) ranked by pooled
Δ, ladder-ordered conservative → aggressive by mean|p_cand − p_v6| on VAL2
ascending (computed from the evaluation's own vectors; no new read). β per
arm refit ONCE on all rows dated ≤ 2026-07-28 by the protocol's β method
and FROZEN (overlay arms share v6's rdiff hence its full-window β; recorded
per arm anyway). Each arm carries its complete frozen config (mechanism,
policy/W/c, variant, b, τ, γ, m, horizon rule, β_ref provenance) so the
prospective evaluator consumes the file verbatim. Zero survivors ⇒ the
ladder is v6 alone — a publishable outcome stated plainly.

## 5. Predictions (before any run; falsifiers explicit)

P1 Solve-side a ∈ {0.5, 1, 2}: ALL THREE BLOCK. Predicted pooled Δ ∈
   (−2.5, 0)m (autopsy interpolation: monotone-ish toward 0 as a→0), VAL2
   the damage window. FALSIFIER: any ADVANCE ⇒ presumption of death is
   wrong; the survivor enters the pool and the hybrid clause may trigger.
P2 delta2 (atlas prior): positive on FIT1 but modest — predicted winner
   dF ∈ [+0.5, +3]m, era_min ∈ (0, +2]m, at small b (0.05–0.20),
   τ ∈ [3, 8], γ ≈ 1, policy w3 or w5. ENVY-type overpriced chains are its
   known failure; predicted transfer: uncertain, P(ADVANCE) ≈ 0.15.
P3 delta1 (evidence-conditional): the philosophically robust member —
   predicted winner dF ∈ [+0.5, +2.5]m, b larger than δ2's (evidence
   gating cuts exposure), P(ADVANCE) ≈ 0.2. If exactly one of δ1/δ2
   advances, that adjudicates the direction-encoding philosophy.
P4 hybrid-δ: between the parents; P(ADVANCE) ≈ 0.15.
P5 Honest program-level prior: ZERO survivors is the single most likely
   outcome (P ≈ 0.5–0.6); "the ladder is v6 alone" is the publishable
   sentence. A grid winner with dF > +5m is a red flag (bug or overfit),
   to be investigated before nomination, not celebrated.
P6 Fixtures S1–S3 (below): PASS. Any FAIL voids downstream numbers.

## 6. Fixtures (run before the grid; predicted PASS)

S1 nesting of the search extension: OverlaySearch at (γ=1, m=0) — δ1 and
   δ2 — p_all bit-identical to the frozen family Overlay.run on the same
   base (b=0.3, τ=5, p1w5c5).
S2 cache equivalence: cached-state scoring of one config per variant
   family == full un-cached OverlaySearch run, p_all bitwise, on FIT1 d
   to the last bit.
S3 β fixture: β(FIT1, v6) via the protocol's minimize_scalar = 0.1152
   ± 1e-3 (realized 0.115199 expected); abort on fail.
S4 p3w5c5 sanity: if the p3 classifier yields zero active pricings, its
   grid rows are v6-identity (d ≡ 0) and are reported plainly, not
   silently dropped.

## Outputs (this agent only)
stats/v9_search_grid.json, stats/v9_candidates.json, stats/v9_ladder.json,
phase_search.md, this file (+outcomes appended AFTER), logs/search.log,
scratch/search/*. Ledger appends to stats/v9_looks.json selection_reads
per the protocol's disclosure clause (the sanctioned write path).

## OUTCOMES — appended after the runs (same resolution for failures)

Fixtures (P6): ALL PASS on first run — S1 nesting bitwise (δ1+δ2), S2
cache equivalence bitwise (3 variants), S3 β(FIT1, v6) realized 0.115199,
S4 p3w5c5 = exact v6 identity (0 active pricings; no boundaries under
frozen-o at c=5) — reported plainly, its 810 grid rows are d ≡ 0.

Grid: 3240/3240 configs scored on FIT1 (827 valid rows; 327×2023,
500×2024; MDE 2.15m). 231 eligible: delta2 147, hybrid(m=6) 84,
delta1 ZERO (best dF +0.81m < 1.0m floor, LOEO negative).

- P1 CONFIRMED: all three solve-side points DIE. Pooled −0.67 / −0.91 /
  −1.70m (a = 0.5 / 1 / 2), inside the predicted (−2.5, 0), monotone
  worsening in a, VAL2 the damage window. The presumption of death held;
  total spend 3 ledger slots, no engine runs (stored npz arrays).
- P2 HALF-WRONG: δ2 topped FIT1 as expected but at b=0.65, τ=13, γ=0.5 —
  NOT the predicted small-b/τ∈[3,8]/γ≈1 region. Transfer: DIE, and worse
  than "uncertain" — pooled −5.38m with block CI [−9.29, −1.03] entirely
  negative; T1 −6.98m says the atlas prior actively mispriced 2025.
  Realized P(ADVANCE) = 0 vs forecast 0.15.
- P3 WRONG: δ1 (evidence-sign) was predicted the robust member; it never
  cleared eligibility at any of its 1296 grid points. Direction-encoding
  verdict on THIS corpus: neither philosophy transfers; the prior-shaped
  ones at least fit in-era, the evidence-shaped one does not even do that.
- P4 CONSISTENT: hybrid(6) sat between the parents and DIED (pooled
  −1.98m, CI [−5.21, +1.12]).
- P5 CONFIRMED, including the red flag: the δ2 winner's dF +5.47m > +5m
  fired the flag; investigation found 74.8% of its FIT1 gain in 10 rows
  and FIT1 drop-top-5% = −7.29m (hybrid: 72.5%, −3.76m), time-folds C5
  clean. The preregistered rule proceeded unchanged and transfer executed
  the sentence the tails foreshadowed: pooled fragility for N1 was
  drop-top-5% −16.75m, jackknife min −5.95m. The era-transfer referee
  blocked a +5.5m train-gain config — the (b)/(e)/spec-run failure mode,
  caught by construction, second demonstration after the autopsy.

Verdict: ZERO of 5 candidates ADVANCE (every one failed A1, A2, A3, A4;
A5 never reached). Hybrid solve+δ clause not triggered (no advancing
parents). THE LADDER IS V6 ALONE — stats/v9_ladder.json arm 0 only,
β frozen 0.128512 (protocol method, 2044 valid rows ≤ 2026-07-28).
Ledger: exactly 5 selection_reads appended; exploratory budget untouched
(0/3); prospective reads 0/3. ROI translations of negative pooled deltas
clamp to 0.0 (the quote-margin ladder spans positive logit shifts only) —
disclosed, no positive-ROI claim exists anywhere in this phase.
