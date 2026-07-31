# v9 DESIGN — the adjudication machinery (Phase 0, agent:v9-design)

Frozen 2026-07-29. Preregister: preregister.design.md (locked before the
autopsy ran; includes a disclosed process incident). Machine-readable law:
stats/v9_transfer_protocol.json + stats/v9_prospective_protocol.json — the
evaluator implements those files verbatim; this document is the prose and
the evidence.

## 1. Epistemic map (the three laws, operationalized)

One sentence per window, so no later agent can lawyer it:

- **2023–2024 (FIT1)**: free fitting ground. All hyperparameter search lives
  here (≤ 2024-12-31).
- **2025 (VAL1) and 2026H1 (VAL2)**: the SPENT holdout, demoted to
  selection-grade validation targets. Scoring here ranks candidates; it
  confirms nothing, ever. Every evaluation is ledgered in stats/v9_looks.json.
- **post-2026-07-28**: virgin. The only confirmatory instrument. Three
  checkpoint reads, ever, at n ∈ {100, 200, 400} scored series.

The retired instrument is the *pooled-holdout confirmatory read*, not the
rows. Law 1 and Law 2 are consistent under exactly this reading and no other.

## 2. The internal referee: era-transfer selection

A candidate = mechanism + hyperparameters frozen on ≤ 2024-12-31 data. One
transfer evaluation per candidate, one-shot, ledgered (a retuned config
after seeing its scores is a NEW candidate, disclosed as informed).

- T1: β refit on FIT1 (n=841) → paired ΔLL vs v6 on VAL1 (n=674, 21 events).
- T2: β refit on FIT2 (n=1515) → paired ΔLL vs v6 on VAL2 (n=543, 10 events).
- β method mirrors the engine (minimize_scalar on window series-NLL);
  fixture: β(FIT1, v6) = 0.1152 ± 1e-3 (realized 0.115199). v6 is refit
  identically per window — the comparison is paired all the way down.
- Judge: referee.paired_bootstrap_crn, block_event primary, crn seeds.

**ADVANCE rule** (win BOTH transfers, margins from measured variance):
A1 Δ_VAL1 ≥ 1.0×SE_blk(VAL1); A2 Δ_VAL2 ≥ 1.0×SE_blk(VAL2);
A3 pooled block 95% CI > 0; A4 pooled Δ ≥ 1.773m (pair-MDE at n=1217);
A5 fragility: pooled drop-top-5% > 0 AND pooled leave-one-event-out min > 0.
Null false-advance ≈ 2%; power ≈ 0.75–0.9 against a true uniform +2.5m
(arithmetic in the preregister, C3). Realized bar heights on real configs:
SE_blk 2.9–4.4m (VAL1), 2.1–4.8m (VAL2) — noisy candidates face higher bars,
which is the point. Survivors (≤3) rank by pooled Δ and enter the ladder
conservative→aggressive by mean|p_cand − p_v6| on VAL2.

## 3. Gate autopsy — would era-transfer have blocked the a=28 ship?

**YES — BLOCKED, on every clause it touched.** Full numbers:
stats/v9_gate_autopsy.json; reproduction fixture: my pooled-holdout number
for the ship equals the recorded read to the third decimal (−11.595m,
gap 0.000; β fixture passed).

| config (as gate-selected) | gate said (train CV) | transfer would have said | holdout said |
|---|---|---|---|
| p1w3c5 a=4.5 | FIRED +4.97m (SE 2.89) | **BLOCK** — VAL1 −0.15m (bar 3.17), VAL2 −10.09m (bar 4.75), pooled −4.58m CI [−10.8, +1.3] | −4.00m |
| p1w5c5 a=4.5 | FIRED +4.88m (SE 2.49) | **BLOCK** — VAL1 +0.83m (bar 2.93), VAL2 −9.62m (bar 3.69), pooled −3.83m CI [−9.1, +1.6] | −3.31m |
| p1w8c5 a=28 cap1.5 (SHIP) | FIRED +6.42m (SE 5.34) | **BLOCK** — VAL1 −5.08m, VAL2 −21.05m, pooled −12.21m CI [−19.0, −5.6] (fails A1, A2, A3, A4) | −11.60m (recorded −11.595) |
| p3w5 | did not fire | N/A — nothing advanced (v6 identity) | N/A — no ship |

Sensitivity: cap=None vs cap=1.5 moves the ship ≤ 0.11m (the cap was never
the story). EXTRA (no verdict weight): a=6 — the v9 family cap — still
transfers negative (VAL2 −10.79m, pooled −5.38m CI [−10.1, −1.0]): even the
capped solve-side mechanism as v8 specced it dies out of era, a real prior
for the family agent.

**The structure of the failure, and why the conjunction is load-bearing:**
every config's damage is concentrated in VAL2/2026H1 (−9.6 to −21m) while
VAL1/2025 is ≈ flat (−0.2 to +0.8m for a=4.5). A single pooled test — or any
validation window dominated by 2025 — would have looked merely inconclusive
for the a=4.5 configs. The win-BOTH-eras requirement is what turns "meh"
into BLOCK. This is the recurring failure mode ((b), (e), spec run) caught
by construction.

**The gate's own records contained the warning (C5).** Leave-one-fold-out on
the recorded CV folds: the a=28 gate stops firing if fold 2 OR fold 3 is
dropped (5.47m < 6.78m SE; 1.70m < 3.23m SE); the a=4.5 gates survive all
five drops. The concentration flag (C5, preregistered before this was
computed) fires for the ship alone. It stays a *flag*, not a kill — transfer
adjudicates — but nominations must carry it.

Law 2's premise is CONFIRMED, not weakened: the P-A contingency (design an
additional catch if transfer had passed a=28) is not triggered.

## 4. The prospective referee (the real one)

Frozen in stats/v9_prospective_protocol.json before any post-07/28 row was
examined. Skeleton: arms = v6 + ≤3 survivors; β per arm refit once on
2023-01-01..2026-07-28 and frozen in stats/v9_ladder.json; scoring rows =
settled series > 2026-07-28 built by the frame recipe verbatim, walk-forward,
paired NaN-drop; checkpoints at scored n ∈ {100, 200, 400} (the only 3 reads).

Promotion at a checkpoint = ALL of: G1 one-sided p_better ≥ {0.999, 0.995,
0.975} in block AND iid (α-spend {.001, .005, .025}, union 0.031 ≤ 0.05,
OBF-shaped); G2 point Δ ≥ MDE(n) = {6.18, 4.37, 3.09}m; G3 no bucket
catastrophe (v8 bars verbatim: −4m@n≥100 / −8m@30–99); G4 max|team-bias| ≤
v6 + 2pp; G5 most conservative passer wins. Kill: block 95% CI ci_hi < 0 at
any checkpoint (false-kill ≤ 7.5%, safe direction). No promotion by n=400 ⇒
NO SHIP; revival only on new prospective data. Nothing touches the public
site regardless; VCTMM stays hands-off.

## 5. Read budget and ledger

stats/v9_looks.json: exploratory budget 3 (spent 0) — frozen candidates
only, sanity not selection; selection_reads (every transfer evaluation);
autopsy_methodological_reads (5, all preregistered, taken 2026-07-29);
prospective_reads (max 3, none yet). The spent holdout got 5 further
methodological reads today; that is acceptable *because* it adjudicates
nothing forever.
