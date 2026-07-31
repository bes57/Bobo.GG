# preregister.page — agent:page (Wave 3 reporter), written BEFORE the build

2026-07-28. Scope per briefs/page.md. This file is trivial by design: the
reporter runs NO experiments and makes NO statistical claims of its own.

## What I will NOT compute
- No model fits, no holdout scorings, no bootstraps, no CIs on any model
  comparison, no bucket verdicts, no p-values. Every verdict, Δ, CI, MDE,
  and p on the page is read from an existing stats/*.json (or from
  scratch/adversary/*.json, copied verbatim into a stats/page_*.json so the
  page serves it). The ADVERSARY-AMENDED verdicts are the published verdicts.
- No new looks at the holdout: I never score a model on anything.

## What I WILL compute (mechanical reshapes only, one writer: me)
1. stats/page_v7_ladder.json — v7_reclass.json rows + a derived boolean
   `amended_loss` = (|Δ| ≥ 5.889m cross floor AND sig_block), asserted to
   equal exactly the adversary's four named configs (sym_6, sym_8,
   surprise_12_20, boxexp_c3_hl8); build FAILS LOUDLY if not.
2. stats/page_adversary_fragility.json — verbatim copy of the machine
   numbers in scratch/adversary/recompute_{eclass_cold,3e,compose}.json
   (drop-top-5%, jackknife, floors). No arithmetic beyond copying.
3. stats/page_prereg_scatter.json — predicted-band vs realized rows curated
   from preregister.*.md outcome sections. Each row carries source_file +
   source_quote; the deriver asserts the quote appears verbatim (whitespace-
   normalized) in the file, else the build fails.
4. stats/page_reliability.json — DESCRIPTIVE reliability bins (favorite
   frame: p_fav = max(p, 1−p), y = favorite won) for v6 / ss_1b / ss_5d from
   scratch/bias_h3/model_probs.npz on holdout rows (date > 2024-12-31).
   Per bin: n, mean predicted, empirical rate, Wilson 95% interval (display
   furniture required by the brief's visual conventions — not a test).
   Bins with n<15 merged into neighbors per the house convention.
5. stats/page_mde_curve.json — MDE₈₀(n) = 2.8016·σ_adj/√n evaluated on a
   grid of n from the σ_adj values already stored in
   power_mde_expanded.json, plus n_for_2m = (2.8016·σ_adj/0.002)² (the
   phase-0 "≈10× the holdout" arithmetic, restated from stored σ).
6. stats/page_slate.json — v6 snapshot prices for data/upcoming_matches.json
   via trading_model/predict.py (frozen benpom-v6-2026-07-22 snapshot).
   Mechanical model evaluation of the production surface; no judging.
7. stats/page_timeline.json — session timeline: first/last timestamp parsed
   from logs/<agent>.log plus a one-line outcome per agent (prose; any
   number in it must also exist in a stats JSON).
8. stats/ledger_v8_updates.json — §10 deliverable: do-not-retest additions
   (each pointing at its kill-evidence stats file + the killing numbers
   copied from it), the UNRESOLVED re-openable list copied from
   ledger_reclass.json, the amended headline verdict set, and the tripwire
   thresholds/windows (editorial policy constants, owned by this file so
   the HTML hardcodes nothing).

## Rendering rules I bind myself to
- gen_v8_report.py renders the page from testing_lab/v8/stats/*.json ONLY;
  no numeric literal that represents a measurement appears in the HTML
  template source.
- adversary_report.md publishes verbatim (HTML-escaped only) in its own
  clearly-marked section; no edits, no elisions, no commentary inside it.
- Every chart carries a download link to the exact JSON it renders from.
- "THE HOLDOUT IS SPENT" renders as a banner, not body text.
- The six existing pages' nav strips are string-patched (v7 Lab anchor →
  + v8 Lab anchor); the old pages are NOT regenerated.

## Failure mode
Any assertion failure (sha-style quote checks, ladder-flag mismatch,
missing JSON key) aborts the build with a loud error; nothing is silently
substituted (README rule 6).
