# Phase compose — Wave 3: stacking what survived Wave 2

agent:compose, 2026-07-28. Preregistered in `preregister.compose.md` (three
stacks, composition + fitting + predictions + falsifiers written before any
fit or scoring; outcomes appended there). Frame: canonical expanded series
frame, sha256-verified; holdout n=1217; CRN referee throughout; one holdout
scoring per stack, sentinel-enforced (`scratch/compose/SCORED`). MDEs:
within 1.773m / cross 5.889m (stats/power_mde_expanded.json).

## Verdict, plainly

**v6 stands because no preregistered stack of the surviving components beats
it at the promotion bar on the expanded holdout.** All three stacks scored
HOLD; every gate clause (G1 CRN support, G2 bias, G3 buckets) failed on every
stack. The best stack (S1) is a real-looking but sub-MDE +1.96m whose entire
gain sits exactly where preregistered — and it still worsens max|team bias|
and one bucket floor, so it is not promotable even at face value.

## The three one-shot scorings

**S1 gate5d** — v6 everywhere, h3's 5d roster-typed state-space only on
hard-gated thin-data rows (n_eff_min < 12; 178/1217 holdout rows; zero new
fitted constants): **+1.958m** [iid CI −0.15..+4.21; block −1.19..+4.94; CV
−0.18..+4.20], p_better 0.966 iid / 0.888 block → INSIDE NOISE FLOOR (cross
MDE 5.889m; even the empirical pair-MDE is 3.08m raw / 3.06m CV). Gated rows
**+13.39m**, non-gated 0.0 — the mechanism delivered precisely where
predicted, at ~1/3 the density needed to clear the bar. Gate: G1 ✗ (below
MDE, block p<0.95), G2 ✗ (max|bias| 0.1491 vs v6 0.1478), G3 ✗ (huge-gap
bucket −8.86m). ROI unit: +0.16pp at δ_logit +0.0043.

**S2 fade+shrink** — event-class fade ratings (eclass_on_v6_m0.8) + stand-in
shrink surface, (β,k) jointly train-refit (β=0.1252, k=+0.352 — inside the
preregistered +0.2..+0.5): **−0.140m** [iid −1.89..+1.64], p 0.43/0.47 →
INSIDE NOISE FLOOR (within MDE 1.773m). EWC full-class bucket +3.15m
(predicted positive — the one place the pair works). Gate: G1 ✗, G2 ✗
(0.1548), G3 ✗ (domestic EMEA −4.41m, favorite [0.7,0.8) −7.63m). HOLD.

**S3 full** — S1 gate over the S2 base with (β,k) refit on non-gated train
rows: **−7.874m** → LOSS beyond the cross floor (iid CI −14.3..−1.5, p
0.008/0.043; CV CI −12.9..−2.9). The preregistered anti-synergy falsifier
fired: fitting k on the 497 non-gated train rows drove k to **1.87** (vs
S2's 0.35) — the best composite TRAIN NLL of the wave (0.64469) and the
worst holdout (0.65004), a textbook train mirage; non-gated rows −11.5m
while the gated rows still gave +13.39m. G3 wipeout (18 major bucket
regressions, huge-gap −55.9m). HOLD; the S3 fitting shape is dead.

## Multiple-looks accounting (stats/compose_looks.json)

Every recorded holdout number, harvested from stats/*.json + logs + scratch
checkpoints: **163 primary candidate looks** (decay 56, context 88, h1 4,
h3 8, h4 4, compose 3; h2 gate-stopped at 0), plus 227 h3 core-sweep
checkpoint numbers and 8 h1 EM-iteration diagnostics = **398 recorded
holdout numbers program-wide**. Family-wise context: Bonferroni α at K=163
is 3.1e-4; S1's best nominal p (iid ≈0.034 one-sided) is two orders of
magnitude short, and the promotion bar (≥MDE and p≥0.95 in both CRN modes)
was never near. No look in the program cleared the gate — the tally is the
adversary's answer, in the open.

## The unexplained residual (reported, not chased)

PRX/NRG under-rating persists under every stack: v6 −10.2/−9.1pp → S1
−10.1/−9.3, S2 −10.1/−9.3, S3 −11.9/−11.4. No surviving component touches
it. **Future work**, explicitly: whatever drives the PRX/NRG residual is not
stand-in load, not event-class recency, not rating uncertainty.

## Forward lead (not a promotion)

The S1 shape is the program's one live thread: zero new parameters, gain
concentrated on preregistered thin-data rows, +13.4m there. It cannot clear
a full-holdout bar at 15% gated density; the honest deployment question for
the operator is a *scoped* one (quote-sizing/telemetry on low-n_eff rows,
stats/h3_neff.json), not a model swap. Exact reproducible spec:
stats/compose_spec.json.

## Artifacts
- stats/compose_stacks.json (specs, train fits, one-shot judging, CV boots,
  caterpillars, buckets, gated splits, EWC buckets, PRX/NRG)
- stats/compose_gate.json (promotion_gate verdicts, clause by clause)
- stats/compose_spec.json (S1 exact spec) · stats/compose_looks.json (tally)
- preregister.compose.md (+ outcomes appendix) · logs/compose.log ·
  scratch/compose/ (runner, fits, SCORED sentinel)
