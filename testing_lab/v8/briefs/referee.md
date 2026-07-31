# agent:referee — Phase 6: the upgraded referee

## Scope (one question)
Build the metric suite every v8 comparison will be judged by, and prove it
reproduces the v6 baseline numbers before anyone uses it.

## Context
- Rules: testing_lab/v8/README.md. Repo: /Users/benny_es1/PythonTest.
- Existing referee: testing_lab/harness.py (logloss, brier, paired_bootstrap
  4000/seed7 iid, reliability with Wilson CIs, calib_slope) and the bucket
  machinery in testing_lab/gen_report.py + out/v6_profile.json.
- Baseline numbers your self-test must reproduce from artifacts on disk (no
  re-solving): v6 holdout LL 0.64095 (out/v7_stage1.json, n_test=1007; the
  0.6409/0.64126 variants elsewhere are earlier builds — reconcile and state
  which artifact is canonical for the self-test), the per-team bias table
  (T1 −7.6, PRX −7.3, 100T −6.9, NRG −6.9, TL −6.5 / TS +16.3, JDG +12.6,
  TE +10.4, C9 +9.2, FUR +8.6 — recover the exact definition from
  gen_report.py/v6_profile.json; it is a rating-point residual, get its
  formula from the source, don't guess), and the bucket LLs (EWC-class
  0.6918, cold-start n=57 ≈ 0.70).
- CRN: testing_lab/v8/crn.json (agent:power writes it, possibly while you
  run). Design your bootstrap to READ crn.json at call time; if absent, raise
  — never substitute a private seed.
- P&L weighting: density by price band comes from
  testing_lab/v8/stats/quote_density.json (agent:autopsy writes it). Ship a
  documented fallback (uniform over the 20–55¢ quoted band, zero outside)
  clearly labeled FALLBACK in output metadata until the real density exists.

## Pre-register first
testing_lab/v8/preregister.referee.md: each metric's exact formula, the
self-test acceptance bars (e.g. LL match to 1e-4; bias table match within
rounding), and the promotion-gate logic as code-to-be.

## Work — testing_lab/v8/referee.py (importable module) with:
1. per_series_ll(p) and paired Δ machinery operating on aligned vectors.
2. paired_bootstrap_crn(d, mode='iid'|'block_event') reading crn.json;
   returns mean, CI, P(better), and n_boot/seed provenance in the result.
3. bucketed(d_or_p, buckets) reproducing gen_report's bucket definitions
   (format, event class, region pair, favorite band, stage) + hooks for
   roster-recency buckets from v8/data/lineup_features.csv when present.
4. per_team_bias(...) — exact v6_profile definition; returns the full table
   + max|bias|; this is a PROMOTION GATE input, not a footnote.
5. pnl_weighted_ll(p, won, price) — weights from quote_density.json (price
   band × density), fallback documented; also expected-ROI translation using
   the quote_margin machinery (out/quote_margin.json shape) so every ΔLL can
   be quoted in both units the operator demanded.
6. fill_conditional_calibration(fills_df, p_model) — reliability on the
   filled subset vs unconditional, Wilson CIs (autopsy will call this too;
   agree the interface via its brief, don't import its code).
7. margin_mse + reliability emitters in the house JSON shape (bin_lo/bin_hi/
   pred_mean/emp/n/ci_lo/ci_hi) chart-ready.
8. promotion_gate(candidate_result, v6_result, mde) implementing the v8 bar:
   paired-bootstrap support at Phase-0 power, max|team bias| reduced, no
   major bucket regression (define "major" numerically in the preregister,
   pre-committed), returns a verdict object with every clause's pass/fail.
9. Self-test runner writing testing_lab/v8/stats/referee_selftest.json:
   every reproduction above with target, got, pass/fail. ALL must pass or
   the module says so loudly at import? No — selftest is a function, and the
   summary reports any failure as a blocker.

## Outputs (yours alone)
- testing_lab/v8/referee.py, testing_lab/v8/metrics_spec.md
- testing_lab/v8/stats/referee_selftest.json
- testing_lab/v8/preregister.referee.md, logs/referee.log

## Forbidden
Modifying harness.py/gen_report.py/out/. Solving ratings. Network. Writing
outside your paths. Private seeds.

## Done criteria
Self-test green on LL + bias table + buckets (or failures explained as
blockers); every metric documented in metrics_spec.md; promotion gate encoded.

## Return format
≤500 words: self-test results, the canonical-baseline reconciliation
(0.64095 vs 0.64126 — which and why), the exact bias-metric definition you
recovered, interfaces other agents should call, artifact paths.
