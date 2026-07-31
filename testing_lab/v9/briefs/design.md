# agent:v9-design — Phase 0: the validation protocol (the part that failed before)

Read testing_lab/v9/README.md (the three laws), then testing_lab/v8/briefs/
wave2_common.md for inherited mechanics. Scope: design and freeze the entire
adjudication machinery BEFORE any search runs, so selection cannot touch the
spent holdout and cannot repeat the gate-transfer failure.

## Work
1. **Transfer-validation protocol** (the internal referee). Define exactly:
   era splits (fit 2023-24 → validate 2025; fit 2023-25 → validate 2026H1
   ≤ 2026-07-28), what "validate" scores (series LL vs the v6 baseline fit
   on the same fit-window, paired, with CIs from crn machinery), the advance
   rule (win BOTH transfers with a pre-committed margin — set it from the
   variance you measure, not vibes), and fragility gates for survivors
   (drop-top-5% of contributing series, era-jackknife). Justify every
   constant in the preregister. β discipline: refit per candidate per
   fit-window, never on validation rows.
2. **Autopsy the spec-run gate failure quantitatively** (it's the design
   input): from stats/roster_spec_read.json's per-policy gate records +
   scratch/roster/spec_run artifacts, reconstruct what the inner-CV saw vs
   what the era-transfer would have said for the same configs. Deliverable:
   the table "gate said / transfer would have said / holdout said" for
   p1w3c5, p1w5c5, p1w8c5, p3w5 — establishing (or refuting) that
   transfer-validation would have blocked the a=28 ship. If transfer would
   ALSO have passed it, say so loudly — that changes how much to trust law 2,
   and the protocol must add whatever WOULD have caught it.
3. **Read-budget law + looks ledger**: stats/v9_looks.json initialized;
   the ≤3 exploratory-read budget written into the preregister with the
   rule for spending them (frozen candidates only, sanity not selection).
4. **Prospective decision rules, frozen now**: for the eventual ladder
   (v6 + ≤3 candidates): scoring cadence (every settled series, evaluated at
   checkpoints n ∈ {100, 200, 400} post-07/28 series), the promotion bar at
   each checkpoint (paired ΔLL with CI clear of 0 AND MDE-at-n printed, no
   bucket catastrophe, max|team-bias| not worse), sequential-looks handling
   (3 checkpoints = 3 looks — pick and justify a correction), and the
   kill rule (drop an arm early if CI excludes +0 at any checkpoint).
   Write stats/v9_prospective_protocol.json — the evaluator (later agent)
   implements it verbatim.
5. **crn extension**: v9 seeds/resample spec appended to testing_lab/v8/
   crn.json under "v9" (don't touch existing blocks).

## Outputs (yours alone)
- testing_lab/v9/DESIGN.md (the protocol, prose + constants)
- testing_lab/v9/stats/v9_transfer_protocol.json, v9_prospective_protocol.json,
  v9_gate_autopsy.json, v9_looks.json
- testing_lab/v9/preregister.design.md, logs/design.log, scratch/design/
- crn.json "v9" block (only addition, nothing else modified)

Return ≤400 words: the advance rule + margins chosen, the gate-autopsy
verdict (would transfer have blocked a=28?), prospective checkpoints + bars,
artifact paths.
