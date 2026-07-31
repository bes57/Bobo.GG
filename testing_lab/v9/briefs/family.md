# agent:v9-family — Phase 1: the model family, implemented and fixtured

Read testing_lab/v9/README.md (the three laws), then testing_lab/v8/briefs/
wave2_common.md. Scope: implement the FULL v9 family as a clean library +
rerun the conformance fixtures. No fitting, no scoring, no holdout contact —
mechanics only. agent:v9-design is running in parallel; do not touch its
outputs.

## Work
1. **Library** testing_lab/v9/scratch/family/v9lib.py, building on
   scratch/roster/spec_run/speclib.py (reuse the causal classifier P1/P3,
   episode tables, census code — import or copy-with-attribution, don't
   reimplement):
   a. SOLVE-SIDE member: the spec-run boost/down-weight machinery with the
      a ≤ 6.0 HARD CAP (assert; no widening path in the code at all),
      params (a, τ, s, W, n_min, cap), c=5 fixed per the census rule.
   b. PREDICTION-LAYER member (new): solve = pure v6 untouched; for a team
      with an active phase (same classifier), adjust its series logit by
      δ(n, k) before the closed-form series step. Implement two variants:
      δ1 "evidence-amplifier": δ = b·(1−k/5)·e^(−n/τ)·sign_evidence where
      sign_evidence is the team's post-change net map differential vs
      pre-change expectation, computed walk-forward (define precisely;
      leak-proof — only matches before the one being priced);
      δ2 "atlas intercept": δ = b·(1−k/5)·e^(−n/τ) unconditionally positive
      toward the changed team (the population atlas's early-outperformance
      skew as a prior; direction-encoding is a deliberate contrast to δ1 —
      the transfer validation will adjudicate which philosophy holds).
   c. HYBRID hook: solve-side with small a + prediction-layer δ on top.
2. **Conformance fixtures, rerun against the library** (reuse spec-run
   fixture code): nesting (a=s=b=0 ⇒ bit-identical v6, sha256 on rdiff +
   daily), SEN zero-boundary, synthetic revert/sustained with lag-correct
   detection, and NEW: prediction-layer pre-change identity (exact, per
   team, all dates — trivial by construction, assert anyway) and
   zero-coupling (no other team's rating or logit moves, ever — assert).
   Publish stats/v9_fixtures.json with pass counts.
3. **Cost model**: time one solve-side walk-forward run and one
   prediction-layer fit on this machine; publish stats/v9_cost.json so the
   search agent can budget its grid honestly.

## Outputs (yours alone)
- testing_lab/v9/scratch/family/v9lib.py (+ tests)
- testing_lab/v9/stats/v9_fixtures.json, v9_cost.json
- testing_lab/v9/preregister.family.md (mechanics + fixture definitions,
  written before implementing), logs/family.log

Return ≤300 words: fixture pass counts, the δ1 evidence definition you
froze, cost numbers, artifact paths.
