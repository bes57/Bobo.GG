# Adversary report — BenPom v8, Wave 3 (publishes verbatim)

Independent hostile review. Everything below was recomputed with my own code
(`scratch/adversary/recompute_{core,3e,eclass_cold,compose}.py`): my own engine
runs from raw inputs, my own loss math, my own re-implementation of both CRN
bootstrap recipes from the `crn.json` text. I imported nothing from
`referee.py`. Machine-readable findings: `stats/adversary_report.json`.

## What survived my attacks (and I attest to)

- **Integrity.** Frame sha256 matches crn.json; all 1695 frozen rows survive in
  the expanded frame with 0 column mismatches; holdout_order hash and the 1007
  npz-era ids verify. The B0 baseline (β 0.1152, 0.64823 train / 0.64216
  holdout, n 827/1217) reproduces **bit-for-bit** from raw inputs, and the
  stored baselines of bias_h1/h2/h3/h4, context and compose all match my run to
  0.0. My independent bootstrap implementation reproduces every CI and p I
  checked to the last digit. No leakage found on any path I traced.
- **Honest gates.** All five decay axes selected the train argmin while the
  holdout argmin was a *different* config in 5/5 cases — selection demonstrably
  did not peek. h3's train-selected ensemble then lost holdout by −12.2m;
  context's train-fitted class weights lost by −5.7m. Falsifiers carry numeric
  bars written before runs, and fired falsifiers (h4 premise, decay axis-e,
  compose S3 anti-synergy) are documented at full resolution.
- **Every Wave-2 kill.** h1, h2, h4, the decay axes and form redefinitions,
  context 3a–3d: reproduced or audited, all sound.
- **All three compose HOLDs.** S1 +1.958m, S2 −0.14m, S3 −7.874m reproduce
  exactly under my independent reconstruction (own gate build, own (β,k)
  refits). "v6 stands" is correct.
- **H3 cold(<10 prior maps): +71.7m at n=39** — the one positive cell that
  survives everything I threw at it (bucket floor 32.9m; drop-top-5% → +53.1m;
  event jackknife min +66.0m). Caveats: 4 events, and the parent model loses
  overall (−5.3m).
- **compose_looks.json is honest**: 163 primary / 398 recorded holdout numbers,
  *larger* than my independent tally (~130 configs; 301 holdout-LL quotes in
  stats/*.json).

## What I would NOT sign

1. **"0/23 v7 configs distinguishable" (Phase 0).** Under the program's own
   Wave-2 floor (5.889m cross) and its own block bootstraps, sym_6 (−10.3m,
   p .0032), surprise_12_20 (−7.4m), sym_8 (−6.2m) and boxexp_c3 (−6.0m) are
   distinguishable-as-worse. Phase 0 quietly used per-pair MDE80 (up to 12.4m,
   inflated by each pair's own sd) — a different, laxer standard than the one
   every later agent was held to. The claim as published is wrong by the
   program's own rules.
2. **3e stand-in shrink, "EWC bucket +3.46m" (n=291).** The delivered bucket is
   not the preregistered one (prereg froze "full ewc_offseason holdout bucket
   (n≈278)"; the shipped n=291 folds in Shanghai Masters and Super Champions
   Cup, which agent:power's frozen class map calls *intl*). No bucket-level MDE
   is quoted; by the program's own scaling rule the floor at n=291 is 3.63m —
   **the +3.46m is inside its own noise floor**, and at the preregistered 278
   rows it is 3.80m vs a 3.71m floor: knife-edge either way. Drop the top 5% of
   contributing series (15 of 291) and it flips to −2.96m. The falsifier dummy
   *beats* the mechanism on the legacy-2026 bucket (+3.34m vs +1.26m). And
   stand-in load is not even concentrated in the bucket (X1 mean 0.357 inside
   vs 0.349 outside), so the mechanism story has no compositional support.
   Noise-compatible; not a lead I would carry.
3. **Event-class fade "+0.24m, positive in all subpops".** The subpops overlap
   90–98% of the holdout (S5 alone covers 97.7%) — five positive signs are one
   observation wearing five hats. The effect is 13% of the within-family floor,
   flips under drop-top-5% (−1.24m) and under single-event jackknife, and is
   **negative on its own target population** (−0.06m on the 115 EWC-class rows;
   the entire gain comes from non-EWC rows). The "all-subpop positivity"
   framing overstates a null.
4. **The 5d half-life table (7.4 vs 24.8 games).** The program's own profile
   CIs overlap enormously ([3.9, 18.5] vs [5.3, 300.2]) and its own
   DerSimonian–Laird pooling returns τ²=0 — all cells shrink to one pooled HL
   of 8 games. The middle cell sits non-monotonically at the q→0 boundary. The
   supporting "WIN" (5d vs core, +3.55m) has an iid CI that includes zero
   (p_better .946 < .95), flips to −6.34m under drop-top-5%, and is one of 163
   primary holdout looks — it survives no family-wise correction. Point
   ordering is a hypothesis to test on new data, not a finding.
5. **Any residue of compose S1 as "a real effect below the bar".** Its active
   surface was pre-screened on holdout: the 5d component and its n_eff gate
   family were chosen after Wave 2 published 5d's holdout cold-row wins, and 68
   of the 178 gated rows are those same thin rows. The nongated delta is
   exactly 0.000; the gated +13.39m (n=178) is inside its own 15.4m bucket
   floor; overall +1.958m flips to −3.96m under drop-top-5% and already fails
   block bootstrap (p .888). HOLD is not just the gate's verdict — it is the
   evidence's verdict.
6. **Debut/thin cold buckets.** debut (n=10, +58m) is inside its own 65m floor
   and halves on removing one series; thin (n=117, +28m) falls below its floor
   under drop-top-5%. Only cold(<10) survives (see above).

## Structural notes

- **Corpus verification is fetch-independent but parser-dependent**: the sample
  check re-fetches live pages yet parses them with the same parser that built
  the corpus. A systematic selector bug would pass both layers. No error found;
  the limit should be stated wherever the corpus is called "verified".
- **The holdout is spent.** 398 recorded holdout numbers exist on disk,
  including per-grid-point holdout LLs published for entire sweeps. This wave's
  selections were verifiably train-only, but any future "train-only" claim on
  this frame is unfalsifiable with those menus published. Confirmatory power
  now requires fresh series.
- Minor: h1's cluster bootstrap used a preregistered *derived* seed stream
  rather than crn mc_seeds; h2/context reused the iid seed for non-holdout
  designs. Documented, low impact.

## Bottom line

The program's negative results are excellent: reproducible to the last digit,
honestly gated, honestly preregistered. Its positive residue is weaker than
published — of every surviving positive claim, only H3's cold(<10) cell
withstands its own program's floors plus my fragility attacks. I would sign
exactly this sentence: **"v6 stands; nothing is promotable; cold-start
handling is the one lead that has earned a test on new data."**

— agent:adversary
