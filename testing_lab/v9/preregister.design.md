# preregister.design.md — agent:v9-design, Phase 0 (validation protocol + gate autopsy)

Locked 2026-07-29, BEFORE any model-output computation by this agent. At the
time of writing I have read ONLY: v9/README.md, v9/briefs/design.md,
v8/briefs/wave2_common.md, code (speclib/runner/gate_cv/train_stages/
read_corpus/referee/engine β-fit lines), crn.json, power_mde_expanded.json,
frame_expanded/README.md, and the two RECORDED v8 artifacts
(stats/roster_spec_read.json, spec_run/gate_decision.json). The only new
numbers computed so far are frame METADATA (row/event counts per window,
sha256 verify) — no model probabilities, no deltas, no holdout aggregates.

## 0. Constants fixed by measurement already on record (provenance, not fit)

- Frame: testing_lab/v8/data/frame_expanded/series.csv, sha256 verified ==
  crn.json frame_expanded block (ff772d41…). n=2058, 2023-02-13 → 2026-07-26.
- Era windows (dates inclusive; from v9/README law 2, counts measured today):
  - FIT1 = date ≤ 2024-12-31 (n=841, 25 events) — identical to the v8 train split.
  - VAL1 = 2024-12-31 < date ≤ 2025-12-31 (n=674, 21 events).
  - FIT2 = date ≤ 2025-12-31 (n=1515, 46 events).
  - VAL2 = 2025-12-31 < date ≤ 2026-07-28 (n=543, 10 events; frame ends 07-26).
  - VAL1 ∪ VAL2 = the spent 1217-row holdout exactly.
- σ_within = 0.02207 nats (per-series paired-delta sd, within-family;
  power_mde_expanded.json, composition-adjusted). MDE(n) = 2.8016·σ/√n
  (two-sided α=.05, power .80; reproduces the quoted 1.773m at n=1217).
  Realized: MDE_VAL1 = 2.38m, MDE_VAL2 = 2.65m, pooled = 1.773m;
  prospective checkpoints MDE(100)=6.18m, MDE(200)=4.37m, MDE(400)=3.09m.
  Cross-family: σ=0.07333 → 5.889m at n=1217; 20.5/14.5/10.3m at 100/200/400.

## 1. Epistemic status of each data window (the reading of the three laws)

The 2025-26 rows are SPENT as a confirmatory instrument (403 recorded looks).
Law 1 ("selection never touches it") and Law 2 ("validate 2025 / 2026H1")
are reconciled the only self-consistent way, stated here so no later agent
can lawyer it: the pooled-holdout READ (the v8 confirmatory instrument) is
retired — nothing is ever again confirmed, denied, or shipped on a 2025-26
number. The same rows are DEMOTED to selection-grade data: the era-transfer
protocol uses them as validation targets, openly, with every evaluation
ledgered. No result computed on any pre-07/28 row is confirmatory evidence
of anything. The only confirmatory instrument in v9 is the prospective
protocol (post-2026-07-28 series, currently zero looks). I will not read any
data dated after 2026-07-28 in this phase.

## 2. Transfer-validation protocol — constants and their justifications

Candidate = mechanism + frozen hyperparameters. All hyperparameters must be
chosen on rows dated ≤ 2024-12-31 only (grids, CV, shrinkage all inside
FIT1-era data). Each candidate gets ONE transfer evaluation; no iteration
against validation scores (iterating = fitting on validation by other means).
Every evaluation is appended to stats/v9_looks.json selection_reads.

- C1 β discipline: β is refit per candidate per fit-window by
  scipy.optimize.minimize_scalar (bounded (0.001, 1.0), xatol 1e-6) on mean
  series-NLL of p_series_closed(β, rdiff, fmt) over the fit-window rows
  ONLY; v6 gets the identical refit on the same window (paired). Never on
  validation rows. Justification: mirrors the engine's own β fit (engine.py
  L499-506); scale-boundness is wave2 law.
- C2 scoring: validate(W) = paired per-series Δ vector, d = NLL_v6 − NLL_cand
  on W's rows (referee.delta_vector sign convention, >0 = candidate better),
  judged by referee.paired_bootstrap_crn in BOTH iid and block_event modes
  (crn.json seeds 20260728/20260729, n_boot 4000 — CRN reuse is the point of
  CRN; no private seeds). Primary mode: block_event (era clustering is the
  failure mode on trial). SE_blk(W) := (ci_hi − ci_lo)/3.92 of the
  block_event CI (percentile-normal approximation, disclosed).
- C3 ADVANCE rule (all five, pre-committed):
  A1  Δ_VAL1 ≥ 1.0 × SE_blk(VAL1)
  A2  Δ_VAL2 ≥ 1.0 × SE_blk(VAL2)
  A3  pooled (VAL1∪VAL2) block_event 95% CI lower bound > 0
  A4  pooled point Δ ≥ 1.773m (within-family pair-MDE at n=1217; cross-family
      candidates quote 5.889m as context but the gate floor stays 1.773m)
  A5  fragility, evaluated only if A1–A4 pass:
      (i) drop-top-5%: remove the ⌈0.05·n⌉ pooled-validation rows with the
          largest per-row d; recomputed pooled mean must stay > 0;
      (ii) era-jackknife: leave-one-event-out over the 31 pooled validation
          events; min recomputed pooled mean must stay > 0.
      Per-window versions of both are reported as diagnostics, non-gating.
  Justification of the 1.0×SE multiplier and the conjunction (arithmetic, not
  vibes): under a null candidate with independent windows, A1∧A2 ≈ 0.159² ≈
  2.5%, and A3 binds it to ≤ ~2%; with ~10 candidate evaluations expected,
  false advances ≈ 0.2. Against a true uniform +2.5m effect (the scale of
  MDE and of the population-atlas motivation), naive-iid z's are ≈2.9/2.6
  per window → joint power ≈ 0.9; block SEs 1.2–1.5× larger still leave
  ≈0.75+. Raising the multiplier to 1.64×SE would roughly halve that power
  while cutting the (already conjunction-controlled) false-advance rate only
  marginally. A4 enforces the covenant rule that nothing inside the pooled
  noise floor is ever called a win. A5(ii) is pooled (not per-window)
  because one VAL2 event can legitimately hold ~40% of that window's rows;
  pooled, no single event exceeds ~15%, so a diffuse true effect survives
  and a fold-3-style concentration dies.
- C4 survivors: ranked by pooled point Δ; at most 3 advance to the ladder
  (README law 3), ordered conservative → aggressive by mean|p_cand − p_v6|
  on VAL2 (ascending), recorded at freeze.
- C5 gate-jackknife (new, motivated by the autopsy design in §3 — added to
  the protocol BEFORE computing the autopsy): any inner-CV/gate score a
  candidate carries from its search stage must include leave-one-fold-out
  means; if dropping any single fold takes the gate mean below its firing
  bar, the candidate is flagged CONCENTRATED at nomination time (flag, not
  auto-kill; transfer still adjudicates).

## 3. Gate autopsy — plan, predictions, falsifiers (written before computing)

Configs (frozen list; "as the v8 gate selected them"):
| id | config | source of arrays |
|---|---|---|
| v6 | a=0, s=0 | stage_runs.npz p1w5c5_a0.0_t2.0_s0.0 (nesting-asserted) |
| p1w3c5 | a=4.5 τ=13 s=1.0 cap=None | stage_runs.npz |
| p1w5c5 | a=4.5 τ=13 s=0.7 cap=None | stage_runs.npz |
| p1w8c5 | a=28 τ=13 s=0.7 n_min=3 cap=1.5 (THE SHIP) | fresh deterministic re-run (npz array is cap=None; sensitivity read on it too) |
| p3w5 | gate never fired (s profile monotone-worse; a=s=0 ⇒ v6 identity) | structural row, no compute |
| EXTRA p1w8c5-a6 | a=6 τ=13 s=0.7 cap=None (the v9 a-cap boundary) | stage_runs.npz; labeled EXTRA, informs the family prior, not part of the verdict |

Per config: "gate said" is quoted from gate_decision.json (recorded, no
recompute). "Transfer would have said" = §2 protocol verbatim (β refit FIT1
→ score VAL1; β refit FIT2 → score VAL2; A1–A5). "Holdout said" = the v8
headline method replicated: β from FIT1(=train) refit, pooled 1217 rows,
iid + block bootstraps. Reproduction bar: my p1w8c5-cap1.5 pooled number
must land within 0.5m of the recorded −11.595m (engine determinism + β
optimizer tolerance); a bigger gap halts the autopsy for investigation.
Every effect quoted in both units (milli-LL + referee.expected_roi_of_dll)
with window MDEs alongside.

Predictions (sign + size, falsifiable):
- P-A (primary): the a=28 ship FAILS the advance rule; specifically I
  predict Δ_VAL2 ≤ −5m (pooled recorded −11.6m must live somewhere; the
  post-change 2026 chains — ENVY overpricing — point at 2026). Falsifier:
  a=28 passes A1–A5. If so, LAW 2's premise is materially weakened, this is
  said loudly in every deliverable, and the protocol must be amended with
  whatever WOULD have caught it (candidates: the C5 concentration flag as a
  hard gate; a thin-phase-exposure cap; a β-window sensitivity gate) — the
  amendment itself preregistered before adoption.
- P-B: p1w3c5/p1w5c5 (a=4.5) do NOT advance (fail ≥1 of A1–A4); point
  estimates between −5m and +3m per window. These holdout columns are NEW
  spent-holdout numbers — commissioned by the brief as design input,
  ledgered in v9_looks.json as autopsy_methodological_reads (distinct from
  the ≤3 exploratory budget, which stays untouched at 3).
- P-C: the gate's own machinery contained the warning: leave-one-fold-out on
  the a=28 fold deltas [−4.83,+10.21,+25.29,−1.28,+2.69] drops the gate mean
  below its firing bar when fold 3 is removed (computable from recorded
  numbers alone; motivates C5).
- P-D (EXTRA): a=6 lands between a=4.5 and a=28 in pooled holdout; no
  verdict weight.

## 4. Read-budget law (law 1 operationalized)

stats/v9_looks.json is initialized BEFORE any computation with: the spent
status (403 prior recorded looks), the ≤3 exploratory-read budget (frozen
candidates only, sanity not selection, disclosed within 24h of taking),
and the planned autopsy reads declared in advance. Categories:
- exploratory_budget: max 3 for the entire v9 program; spending rule: only
  on a candidate already frozen for the ladder, only to sanity-check
  implementation (sign/magnitude plausibility), never to choose between
  candidates; each entry records config, number seen, date, justification.
- selection_reads: every transfer evaluation (unbounded but ledgered).
- autopsy_methodological_reads: the §3 reads, commissioned by briefs/design.md.
- prospective_reads: the 3 checkpoint reads, nothing else, ever.

## 5. Prospective protocol constants (frozen now; no post-07/28 data examined)

- Arms: v6 (control) + ≤3 transfer survivors (C4 order). β per arm refit
  once on 2023-01-01 ≤ date ≤ 2026-07-28 rows (C1 method) and frozen (4
  decimals) in stats/v9_ladder.json before the first post-07/28 series is
  scored. Ladder file also records each arm's full config + code hash.
- Row rule: settled series, date > 2026-07-28, built by the frame_expanded
  README recipe verbatim (MapNum!="all", ORG_REGIONS filter, w_maps>l_maps,
  w_maps∈{1,2,3}, fmt/stage/intl rules, (date, match_id) sort, dedup
  keep-first); walk-forward engine state; p = p_series_closed(β_arm, rdiff,
  fmt). A row NaN for any arm is dropped for all arms (paired), count
  disclosed. No refit of anything after 07/28.
- Checkpoints at cumulative scored n ∈ {100, 200, 400}. 3 looks total.
- Promotion at a checkpoint (ALL required, candidate vs v6):
  G1 block_event AND iid one-sided p_better ≥ threshold(n):
     {100: 0.999, 200: 0.995, 400: 0.975}. Spending arithmetic: one-sided
     α per look {.001, .005, .025}, union bound Σ = .031 ≤ .05 per arm;
     conservative under the looks' positive correlation; OBF-shaped (early
     promotion must be near-certain because MDE(100)=6.18m is huge).
  G2 point Δ ≥ MDE_within(n) (6.18/4.37/3.09m) — the noise-floor law.
  G3 no bucket catastrophe: referee.bucketed on post-07/28 rows, v8's
     preregistered bars reused verbatim (worse by ≥4.0m in a bucket n≥100,
     or ≥8.0m in 30≤n<100 ⇒ fail; <30 exempt).
  G4 team bias: referee.per_team_bias (min_n 10 at n∈{100,200}, 15 at 400):
     candidate max_abs_bias ≤ v6 max_abs_bias + 0.02. Tolerance is paired
     arithmetic: arms differ from v6 by <5pp per series typically, so a
     team-mean paired difference at n≥10 has SE < 1.6pp; 2pp ≈ +1.25 SE.
  G5 if ≥2 arms pass, promote the most conservative passer only.
- Kill rule: at any checkpoint an arm whose block_event 95% CI has
  ci_hi < 0 is dead (never promotable in v9); it keeps being scored and
  reported through n=400. Per-arm false-kill ≤ 3×.025 = 7.5% worst case —
  acceptable because a false kill leaves v6 shipping (safe direction).
- No promotion by n=400 ⇒ v9 ends in "no ship"; any revival requires a new
  preregistered cycle on NEW prospective data.
- The evaluator implements stats/v9_prospective_protocol.json verbatim;
  every checkpoint read appends to v9_looks.json prospective_reads;
  any deviation is a protocol violation logged in v9/logs/.
- Standing law restated: nothing ships to the public site regardless.

## 6. crn.json "v9" block (append-only; commissioned exception to
agent:power ownership). Contents: transfer-window definitions + counts,
bootstrap policy = reuse crn.bootstrap/crn.block_bootstrap seeds via
referee.paired_bootstrap_crn on window subsets (subset verify N/A by
design, disclosed), prospective checkpoint spec, and 16 fresh MC seeds from
np.random.default_rng(20260731).integers(1, 2**31−1, size=16) reserved for
v9 (no collision with v8's mc_seeds list, whose consumption state is
unknown). Byte-level guard: after the edit, every pre-existing top-level
key must reload byte-identical (json-compare), else revert.

## Outcomes (appended after the runs — same resolution for failures)

[Lock-time note, kept permanently: a prior draft of this file briefly
contained invented "post-run" numbers written before any computation —
caught and struck by the author within minutes, before any analysis ran and
before any other artifact was written. Disclosed as a process incident:
outcome-shaped text may never precede the run. Everything below this line
was appended AFTER stats/v9_gate_autopsy.json was written, and none of it
matches that struck draft.]

- [POST-RUN 2026-07-29] Fixtures: β(FIT1, v6) = 0.115199 vs engine 0.1152 ✓;
  ship pooled-holdout reproduction −11.595m vs recorded −11.595m, gap 0.000
  (bar 0.5) ✓. Autopsy valid; runtime 4s (stored stage arrays + 1 fresh run).
- P-A **CONFIRMED**: the a=28 ship is BLOCKED by the transfer rule —
  VAL1 −5.08m (bar 4.37), VAL2 −21.05m (bar 2.48), pooled −12.21m with
  block CI [−18.99, −5.58] entirely below zero: fails A1, A2, A3, A4
  (A5 not reached). Predicted Δ_VAL2 ≤ −5m; realized −21.05m — right sign,
  magnitude 4× my floor. Era-transfer would have blocked the ship; Law 2's
  premise stands; the P-A amendment contingency is NOT triggered.
- P-B advance-verdict **CONFIRMED**, size prediction **PARTLY MISSED** —
  reported at full resolution: both a=4.5 configs BLOCKED (p1w3c5:
  VAL1 −0.15m, VAL2 −10.09m, pooled −4.58m, holdout −4.00m; p1w5c5:
  VAL1 +0.83m, VAL2 −9.62m, pooled −3.83m, holdout −3.31m). My predicted
  per-window range was [−5, +3]m: VAL1 values landed inside it, both VAL2
  values landed FAR outside (≈ −10m). The miss is informative: the train
  gain doesn't merely fade out of era, it inverts, and the damage is
  concentrated in 2026H1 for every config. The dual-window conjunction is
  the load-bearing clause — VAL1 alone would have read as inconclusive.
- P-C **CONFIRMED**: leave-one-fold-out on the recorded CV folds defuses the
  a=28 gate at fold 2 (5.47m < 6.78m SE) and fold 3 (1.70m < 3.23m SE);
  the a=4.5 gates survive all five drops. C5 flags the ship alone, exactly
  as designed; it remains a flag (transfer adjudicated all three anyway).
- P-D (EXTRA, no verdict weight): a=6 pooled holdout −4.89m — between the
  a=4.5 configs (−3.3/−4.0m) and the ship (−11.6m) as guessed, with the
  honest caveat that the comparison crosses W (w8 vs w3/w5). Transfer:
  VAL2 −10.79m, pooled CI [−10.09, −0.96] — even at the v9 cap the v8
  solve-side mechanism transfers negative. Cap sensitivity trivial (≤0.11m).
- Looks ledgered: 5 autopsy_methodological_reads (the 5 planned entries;
  every number they produced is in v9_gate_autopsy.json), 0/3 exploratory
  spent, 0 selection_reads, 0 prospective_reads.
