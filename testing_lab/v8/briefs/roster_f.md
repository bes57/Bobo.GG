# agent:roster-f — Treatment (f): per-team post-solve overlay (operator spec)

Read testing_lab/v8/briefs/wave2_common.md first (frame, CRN, referee, MDE
context, train-only, fail loudly), then this file. You continue the roster
program (prior agent's context is gone — everything you need is on disk:
preregister.roster.md incl. ADDENDA 1-2 and outcomes, phase §2's frozen
sustained-change episode definition, logs/roster.log, scratch/roster/ with
the v6 baseline machinery and episode tables, stats/roster_*.json).

## Why (f) exists — the operator's architectural requirement
Treatments (d) (state-space) and (e) (in-solve weight boost) both produced
pre-change rating divergence for uninvolved dates/teams, because anything
inside the shared Massey solve couples corpus-wide. The operator rejected
that twice, in these words: "the whole point is it's the same model, that
then has a new system after a mid-season roster change" / "WHY WOULD RATINGS
BE DIFFERENT PRE-ROSTER CHANGE." The requirement: PRE-CHANGE IDENTITY EXACT
BY CONSTRUCTION. Therefore the system is a PER-TEAM POST-SOLVE OVERLAY —
v6's solve untouched for everyone always; only a changed team's own rating
is adjusted, only after its change.

## Spec
- Reuse stored v6 daily ratings (scratch/roster/ has the replayed v6
  trajectory machinery from reads (b)/(e); do NOT re-run the engine solve).
- For team T with a sustained episode at date c (overlap k/5, the FROZEN §2
  definition — reuse the episode table on disk): anchor A = T's v6 rating
  immediately pre-c. While active:
      r_adj(T, t) = A + G(n)·(r_v6(T, t) − A),  G(n) = 1 + a·(1−k/5)·exp(−n/τ)
  n = T's post-change matches before t. Chained episodes re-anchor at each
  new episode (anchor = r_v6 just before that episode; document).
- All other teams, and T pre-change: r_adj ≡ r_v6 BIT-EXACT. Assert
  programmatically (max |r_adj − r_v6| over all pre-change team-dates must
  be exactly 0.0) and publish the assertion result.
- Prediction: series probs from r_adj on both sides; β refit on TRAIN with
  the overlay active (β is scale-bound; per config).
- Fit (a, τ) TRAIN-only, grid a ∈ {0.5, 1, 2}, τ ∈ {2, 5}; (e) selected the
  a=2 edge, so if the train argmin lands on an edge again, extend that axis
  once (pre-justified here, document the extension). ONE exploratory holdout
  read. Preregister ADDENDUM 3 in preregister.roster.md BEFORE scoring:
  operator's predicted sign (positive), size, falsifier (frozen same shape
  as (e)'s: post-change-bucket ci_hi<0 or overall ≤ −1.77m). Disclose the
  read-budget extension (operator-directed; roster_looks.json 402→403).

## Deliverables
- stats/roster_treatments.json: append read5_f_v6_overlay (same schema as
  read4: delta_milli, iid+block CIs vs v6, post-change ≤3 bucket,
  improvement slice, keep4 slice, both units, MDE context, PLUS
  identity_assertion: {prechange_max_abs_diff, pass}).
- roster_case_envy.json + roster_case_gallery.json (LEV named case): add
  "v6_overlay_path" [{d, r}] over the same overlay windows (daily dates fine
  — the page steps them at match dates).
- stats/roster_integration.json: freeze prospective arm G_v6_overlay.
- preregister.roster.md ADDENDUM 3 + outcome; logs/roster.log entries;
  scratch code under scratch/roster/ (new files, don't overwrite prior).
- No scrapes, no network, no engine solve re-runs, no writes elsewhere.

## Return (≤250 words)
Identity assertion result (must be 0.0), train-selected (a, τ) + whether the
grid extended, the read: overall Δm + CIs, post-change bucket, improvement
slice, β, looks tally, JSON paths.
