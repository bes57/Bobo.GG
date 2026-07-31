# agent:roster — Roster-change adaptation report (operator-requested)

Read briefs/wave2_common.md first; it is law (frame, CRN, referee, walk-
forward, train-only fitting, fail loudly, both units). Scope: how should the
model adapt to mid-season roster changes — detection, continuity, and
magnitude ("how much of the roster changed") — anchored on the ENVY 2026
Stage 1 → Stage 2 case. Deliverable is a REPORT (house-style page + md), not
a promotion.

## Epistemic frame (non-negotiable)
The adversary established THE HOLDOUT IS SPENT (398 recorded looks). Your
structure is therefore:
  (a) case-level forensics (the house method for operator anomaly reports),
  (b) population diagnostics computed on TRAIN or descriptively (no scoring),
  (c) a SMALL, preregistered set of exploratory holdout reads — clearly
      labeled EXPLORATORY on page and in JSON metadata, counted and appended
      to stats/compose_looks.json's successor (write stats/roster_looks.json),
      never called confirmatory, never "promotable".
Power context to print alongside: post-roster-change bucket (≤3 matches)
n=598, MDE 2.5m within-family / 7.8m cross (stats/power_mde.json) — and even
that is now exploratory-only on this frame.

## What already exists (read, reuse, do not recompute blindly)
- v6 roster treatment: YEAR-BOUNDARY continuity 0.3 only (engine.py
  _build_roster_structures/year mode; FINDINGS.md). Mid-season changes are
  invisible to the solve. The engine also has an unused 'lineup' roster mode.
- Per-match lineups + features: v8/data/lineups.csv, lineup_features.csv,
  modal5_by_org_date.csv (walk-forward; corpus-addition top-ups exist in
  scratch/context/ and scratch/bias_h3/ — reuse their definitions).
- Ledger history: "lineup/roster reweighting" and "roster-instability
  shrink" were REJECTED pre-v8 on coarse year-boundary heuristics —
  reclassified UNRESOLVED by Phase 0 (stats/ledger_reclass.json). Per-match
  lineups are a new data source = legitimate re-open. Say this explicitly.
- Wave 2 results to integrate, with the adversary's demotions respected:
  decay 5b-a lineup-continuity (+1.68m over no-continuity, redundant with
  year mode, floors); H3 5d change-adjacent HL 7.4 vs stable 24.8 games
  (SUGGESTIVE — CIs overlap); H3 cold-start <10 maps +71.7m (the one
  adversary-robust lead); compose S1 gated shape (HOLD).

## Work
1. **The ENVY case, forensically.** From lineups.csv + maps CSVs: ENVY's
   exact fives by match through 2026; identify the S1→S2 change (who left,
   who joined, overlap k/5, date); v6's rating trajectory through the change
   (engine daily solve or timeline JSON); per-match predicted p vs outcome
   before/after; how many matches until the rating stabilized; what the
   change-blind carryover cost in probability terms per match (descriptive).
   Also pull 3-5 comparable historical mid-season changes (biggest lineup
   deltas in the corpus, e.g. 2/5+ overhauls) as a case gallery with the
   same panels. All descriptive/train-side — label any holdout rows used in
   trajectories as descriptive, not scored.
2. **Population atlas (descriptive).** Every sustained mid-season roster
   change in the corpus 2023-2026: definition preregistered (new five
   persists ≥N matches, distinguish from one-off stand-ins), count by
   overlap magnitude (4/5 kept, 3/5, ≤2/5), by org/region/year. For each
   magnitude class: post-change performance vs pre-change rating expectation
   over match 1..10 after the change (the adaptation curve the operator is
   asking about). This is the report's centerpiece chart: error-vs-matches-
   since-change, by overlap magnitude, with CIs.
3. **Treatment comparison (exploratory, preregistered, ≤4 holdout reads).**
   Candidates — all walk-forward, params train-fit:
   a. v6 baseline (reference, no new read needed — reuse stored).
   b. Graded event-boundary continuity: continuity factor = f(overlap) at
      the CHANGE POINT rather than year boundary — f monotone, e.g.
      (k/5)^gamma, gamma train-fit. This directly encodes "how much changed".
   c. Change-triggered partial cold start: on a sustained change with
      overlap ≤ threshold, blend the team's rating toward its region prior
      proportionally to (1 - k/5) — the H3-consistent "uncertainty spike"
      in point-estimate form (threshold+blend train-fit).
   d. (Optional, only if cheap from h3 scratch) the n_eff-gated state-space
      scoped to change-adjacent rows.
   Report each vs v6 with CRN CIs, EXPLORATORY label, per-bucket panel
   (post-change buckets by magnitude), both units, and the roster-bucket MDE
   printed in the header.
4. **Bot integration paths (design section, no code changes).** Two tiers:
   (i) SIZING-ONLY: snapshot carries per-team roster_flag {overlap vs modal
   five at last match, matches_since_change, n_eff analog} so VCTMM can
   quarter-size or widen quotes on fresh rosters (fits the existing
   Playbook size-down rules + telemetry spec's roster_obs; zero fair-value
   risk); (ii) FAIR-VALUE: treatment (b)/(c) inside build_model_snapshot
   with β refit — spell out exactly which constants change, the vendor-sync
   steps, and that it requires prospective validation before deployment
   (holdout spent). Recommend a tier honestly based on your evidence.
5. **Prospective validation plan.** Preregister NOW (in the report) the
   exact test to run when new data accumulates: metric, bucket, MDE at
   projected n after Stage 2 + Champions settle, and the decision rule.
   This is what makes the exploratory reads eventually adjudicable.

## Outputs (yours alone)
- stats/roster_case_envy.json, roster_case_gallery.json,
  roster_population.json (adaptation curves), roster_treatments.json,
  roster_integration.json, roster_looks.json
- House-style page installed as testing_lab/out/reports/roster_adaptation.html
  via a gen script testing_lab/gen_roster_report.py (same pattern as
  gen_v8_report.py; auto-listed by the lab's reports glob; add the shared
  tab strip linking the other pages + v8 Lab; EXPLORATORY badges wherever
  applicable; every number from JSON with download links via
  /testing/v8/stats/).
- testing_lab/v8/roster_adaptation.md mirror.
- preregister.roster.md (BEFORE any computation: change definition,
  treatment specs, the ≤4 exploratory reads, falsifiers),
  logs/roster.log, scratch/roster/.

Done when: ENVY case fully reconstructed with named players and dates; the
adaptation-curve atlas built; ≤4 exploratory treatment reads reported with
labels and MDE context; both integration tiers specified; the prospective
plan preregistered; page live and mirroring md delivered.

Return ≤500 words: the ENVY numbers (overlap, cost per match, stabilization
time), the population headline (adaptation curve shape by magnitude), each
treatment's exploratory read, your recommended integration tier, artifact
paths.
