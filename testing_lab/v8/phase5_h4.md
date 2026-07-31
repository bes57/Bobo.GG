# Phase 5 H4 — series aggregation assumes iid maps (agent:bias-h4)

H4 is not the rejected intra-series momentum idea, and the line matters. The
momentum entry (ledger id 27, `g16_k20_corr07`, −2.77m, inside its MDE) made
map t+1's probability depend on the *outcome* of map t — a sequential
carryover hardcoded (σ=0.7) inside the per-map veto MC. H4 instead posits an
*exchangeable* series-level effect: a latent matchup/day term u drawn before
map 1, constant across the series, integrated out in closed form (GH-31
quadrature). No map outcome ever feeds another map's probability; nothing is
path-dependent. What changes is only the marginal p → P(series) curve — the
aggregation layer over the single p — and its dispersion parameter is *fit on
train scores*, not asserted. Preregistered in preregister.bias_h4.md before
any run; frame sha verified; β refit on train per config.

## E1 — dispersion diagnostic (train, n=794, bo1 excluded)
Conditional-on-winner score distributions vs the iid closed form at v6's p
(reconstructed on the expanded frame: β=0.1152, holdout LL 0.64216, n=1217).
Maps are **over-dispersed, not under-dispersed**: sweep excess D_sweep =
+5.1 pp, CI [+1.8, +8.5] (bo3 +5.7 [+2.1, +9.2]); shared-effect fit σ̂_u =
0.72 [0.38, 1.00]; the under-dispersion arm (pick-spread h, sign-balanced map
heterogeneity) fits ĥ = 0. Depth tercile σ̂ falls 1.05 → 0.63 → 0.36 (the
intuited direction) but the train depth coefficient b = −0.44 [−1.79, +0.03]
does not clear 0, and the H4 cell itself — deep-pool strong favorites — has
n=10 train (5 holdout bo5), CI [−34, +25] pp: **untestable at this n**. Gate
G1 (under-dispersion for deep-pool favorites) fails; H4's premise —
"iid understates elite series probability" — is refuted in sign.

## E2 — correlated series link on holdout (n=1217, MDE_within 1.773m)
- L1 global σ_u=0.72, β refit 0.1407: ΔLL **−0.048m** iid CI [−0.42, +0.32],
  block [−0.31, +0.25] — INSIDE NOISE FLOOR. Expected ROI delta ~0.0000.
- L2 σ_u(depth) softplus(0.009 − 0.435·z), β 0.1401: **+0.213m**
  [−0.75, +1.15] — INSIDE NOISE FLOOR. Expected ROI delta +0.0002.
- L3/L4 (h links): correctly skipped, G1 false.
P(bo5 win) for strong favorites does **not** rise — it falls 1.6 pp under L1
(n=5 rows). Per-team bias unchanged (max|bias| 0.148 → 0.147/0.142); GF
bucket −2.0m/−1.4m at n=35 (deep inside bucket noise).

Mechanism, quantified: the σ-link is nearly a per-format logit rescale
(iid-equivalent logit ratio ≈0.82 bo3 / 0.76 bo5 at σ=0.72), and v6's
train-fit β already absorbs it — β_iid/β_σ = 0.115/0.141 = 0.82. The closed
form is genuinely misspecified (real over-dispersion, worth σ≈0.7), but the
deployed surface has been implicitly paying that correction through β all
along; the residual identifiable content is the bo5-vs-bo3 contrast, too
small at 88 holdout bo5s. Verdict: **H4 REFUTED (premise inverted); both
links INSIDE NOISE FLOOR; v6 aggregation stands.**

## E3 — interaction guard
Links consume only (β·rdiff, fmt, z_depth as σ covariate); no per-map
ratings, no pick features, no map identities — the ledger id 25 per-map+pick
kill stays dead; dispersion acts solely between the single p and P(series).
Reduction at σ=0/h=0 exact to <5e-16.

Artifacts: stats/h4_dispersion_diag.json, stats/h4_series_link.json,
stats/h4_bias_caterpillar.json; journal logs/bias_h4.log; scratch/bias_h4/.
