# phase_search — transfer-gated optimization (agent:v9-search)

**THE ANSWER: ZERO candidates advanced. THE LADDER IS V6 ALONE.**
stats/v9_ladder.json is arm 0 only — pure v6, β frozen at 0.128512
(protocol method, refit once on 2023-01-01..2026-07-28, 2044 valid rows).
The prospective referee (v9_prospective_protocol.json) therefore has one
arm and nothing to promote: v6 remains the model, and the roster-subsystem
question is settled for this corpus era pending genuinely new prospective
data. That sentence is the deliverable.

## What was searched (train ground only, FIT1 = ≤2024-12-31)
- Prediction layer: 3240-config full factorial — 4 policies (p1w3/5/8c5,
  p3w5c5) × 5 direction variants (δ2; δ1(m=0,3); hybrid(m=3,6)) × 9 b ×
  6 τ × 3 γ, magnitude b·(1−k/5)^γ·e^(−n/τ), horizon ceil(5τ). Scored
  paired vs the shared v6 base on 827 FIT1 valid rows (MDE 2.15m).
  Fixtures S1–S4 all passed first run (nesting bitwise; cache path
  bitwise; β fixture realized 0.115199; p3w5c5 is exact v6 identity —
  zero boundaries under frozen-o, its 810 rows reported plainly).
- 231/3240 eligible (dF ≥ 1.0m, era_min > 0, LOEO min > 0): δ2 147,
  hybrid(6) 84, **δ1 zero** — pure evidence-sign never fit even in-era
  (best dF +0.81m, LOEO negative). δ1 nominated nothing.

## The five one-shot transfer evaluations (all ledgered, all DIE)
| candidate | T1/VAL1 (bar) | T2/VAL2 (bar) | pooled [blk 95% CI] | verdict |
|---|---|---|---|---|
| solve a=0.5 t13 s0.7 | +0.21m (1.91) | −1.75m (2.93) | −0.67m [−3.71, +2.48] | DIE |
| solve a=1.0 t13 s0.7 | +0.76m (2.13) | −2.98m (3.05) | −0.91m [−4.46, +2.66] | DIE |
| solve a=2.0 t13 s0.7 | +1.17m (2.47) | −5.25m (3.32) | −1.70m [−5.96, +2.64] | DIE |
| δ2 p1w5c5 b.65 τ13 γ.5 | −6.98m (3.09) | −3.40m (2.81) | −5.38m [−9.29, −1.03] | DIE |
| hybrid(6) b.20 τ21 γ.5 | −2.20m (2.00) | −1.70m (2.84) | −1.98m [−5.21, +1.12] | DIE |

Every candidate failed A1–A4 (A5 never reached). Pooled MDE context
1.773m; ROI translation of negative deltas clamps to 0.0 (the
quote-margin ladder spans positive shifts only) — no positive-ROI claim
exists in this phase.

## What the deaths mean
- **Solve-side: presumption of death VERIFIED cheaply.** Three stored-array
  points, no engine runs, monotone worsening with a (−0.67 → −1.70m),
  damage in VAL2 as the autopsy pattern predicted. With the autopsy's
  a ∈ {4.5, 6, 28} all negative, the solve-side family is now measured
  dead from a=0.5 to a=28. Closed.
- **Prediction layer: train gain that dies out of era, again.** The δ2
  winner carried +5.47m on FIT1 (both years positive, all-25-events LOEO
  positive, time-folds clean) and still transferred to −5.38m pooled with
  the block CI entirely below zero — T1 −6.98m means the atlas prior
  actively mispriced 2025, not merely failed to help. The preregistered
  +5m red-flag investigation had already found the tell: 74.8% of its
  FIT1 gain lived in 10 rows (FIT1 drop-top-5% −7.29m); pooled fragility
  confirmed (drop-top-5% −16.75m). The win-BOTH-eras conjunction did
  exactly what the gate autopsy said it would — this is the second
  demonstration, now on a fresh mechanism.
- **Philosophy verdict (δ1 vs δ2):** neither direction-encoding survives.
  The prior-shaped members (δ2, hybrid) at least fit in-era; the
  evidence-shaped member (δ1) does not even do that on this corpus.
- Hybrid solve+δ clause: not triggered (no advancing parents), skipped as
  preregistered.

## Ledger state (stats/v9_looks.json)
selection_reads: exactly 5 entries (one per candidate, appended at run
time with clauses_failed). Exploratory budget UNTOUCHED (0/3 — reserved
for the freeze stage). Prospective reads 0/3. No pooled-holdout read of
any kind occurred; the grid stage aggregated FIT1 rows only.

## Artifacts
- stats/v9_search_grid.json — 3240 rows, chart-ready, FIT1-only
- stats/v9_candidates.json — 5 candidates, clause-by-clause A1–A5 + C5
- stats/v9_ladder.json — **v6 alone**, β 0.128512 frozen
- preregister.search.md (+outcomes: P1 confirmed, P2 half-wrong,
  P3 wrong, P4 consistent, P5 confirmed incl. red flag, P6 pass)
- logs/search.log, scratch/search/{searchlib,run_grid,run_transfer}.py
