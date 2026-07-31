# preregister.finish — agent:v9-finish (Phase 3: evaluator + report)

Locked 2026-07-29, before score_prospective.py or gen_v9_report.py were
written. Trivial by design: this agent RESHAPES recorded artifacts and
IMPLEMENTS frozen protocol; it computes no new model results, fits nothing,
and takes no selection or confirmatory read of any kind.

## What this phase does
1. **score_prospective.py** — the standing prospective evaluator.
   Implements stats/v9_prospective_protocol.json VERBATIM: frozen arms only
   (v9_ladder.json = v6 alone, beta 0.128512, used verbatim; plus the
   pre-frozen v8 reference arms in v8/stats/roster_integration.json
   arms_frozen, scored under the same machinery, clearly separated from the
   empty candidate set); scoring population = settled series dated
   > 2026-07-28 built by the frame_expanded README recipe verbatim from the
   live data files; checkpoints at scored n in {100, 200, 400} with the
   frozen G1-G5 / alpha-spend / kill rules; before n=100 it reports
   "accumulating (n=X)" and evaluates nothing. Idempotent, read-only over
   data/, refits nothing (reference-arm betas are frozen ONCE on the
   protocol freeze window 2023-01-01..2026-07-28, protocol beta method,
   stored in the scoreboard and reused verbatim on every later run).
2. **gen_v9_report.py** — /testing/report/v9_lab, house pattern; every
   number read from a stats JSON with a download link via a new
   /testing/v9/stats/ route (TestingLab.py, same auth gate as v8's).
3. **Nav** — "v9 Lab" tab added to hub + all existing report pages via
   assert-exactly-once string patches; older pages are never regenerated.
4. **v9_report.md** — answer-first mirror of the page.

## Predictions (bookkeeping, not science)
- P1: the live post-2026-07-28 settled-series population is n = 0 today
  (match_dates.json max date <= 2026-07-28); first run writes an
  ACCUMULATING scoreboard skeleton and no aggregate of any kind.
- P2: no checkpoint fires; stats/v9_looks.json prospective_reads stays
  empty (0/3); the exploratory budget stays 0/3.

## Falsifier / violation rule
Any aggregate over post-07/28 rows computed or displayed outside the three
protocol checkpoints is a protocol violation and gets logged as such in
v9/logs/. Any refit after the one-time freeze is a violation.

## Disclosures
- D_phase_reset (reference arm) cannot be scored on the live corpus: its
  base is the h3-1b round-level state-space filter (v8/scratch/bias_h3
  stack) whose round enrichment does not exist for CN events or for the
  live post-freeze corpus. It is registered in the scoreboard with its
  frozen spec pointer and status NOT_SCOREABLE_LIVE + this reason, rather
  than silently substituted. All other reference arms (B, C, E, F, H) are
  scoreable walk-forward from live data and are wired.
- The evaluator ports three frozen mechanisms verbatim with attribution
  (EngineSpec hook / EngineOverreact hook / EngineChange hook + the
  build_changes.py episode definitions); ports are code moves, not new
  modeling decisions.

## Outcomes (appended after the run)
- P1 CONFIRMED: n_settled_post0728 = 0; scoreboard initialized ACCUMULATING
  (n=0); per-series store empty; no aggregate computed.
- P2 CONFIRMED: no checkpoint fired; prospective_reads 0/3; exploratory 0/3.
- D_phase_reset registered NOT_SCOREABLE_LIVE as disclosed above.
