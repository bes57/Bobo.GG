# Phase 5 H1 — bounded margins censor dominant teams: DEAD

agent:bias-h1 · 2026-07-28 · preregistered in `preregister.bias_h1.md` (gates
+ deviations logged there and in `logs/bias_h1.log`) · frame:
`v8/data/frame_expanded/series.csv` (sha256-verified; holdout n=1217) ·
baseline v6 = consist(20,12) champion re-run on the expanded frame (β=0.1152,
holdout LL 0.64216; caterpillar in `stats/h1_bias_caterpillar.json`) ·
numbers: `stats/h1_censor_diag.json`, `h1_tobit.json`, `h1_roundbt.json`.
MDEs (stats/power_mde_expanded.json): within-family 1.77m, cross 5.89m.

## E3 diagnostic — the premise is empirically false

The corpus barely touches the cap: **12 of 5,140 maps (0.23%) are 13-0**, and
the margin density *falls ~2× per step* approaching the bound (m=10: 222,
11: 137, 12: 68, 13: 12) — a censored variable shows a pile-up spike at the
boundary; this is the opposite. Cap share by winner trailing-rating quartile:
Q1 0.0%, Q2 0.0%, Q3 0.39%, Q4 0.19% — **Q4/mid ratio 1.00** (preregistered
premise prong needed ≥2; the <1.5 falsifier fired). Near-cap (margin ≥ 11):
Q4 4.7% vs mid 3.2% — mild, nowhere near censoring-scale. Elite teams do not
pile up at the bound. (Residual prong P2 also failed, though its
winner-referenced design turned out flawed — documented in the prereg
outcomes; the density shape above is the decisive evidence.) Gate verdict:
premise weak; Tobit skippable. Both fits were still run — engine runs cost
2.2 s, and OT-pinning (11.2% of maps) was untested by P1 (deviation, run-more
direction, logged).

## E1 Tobit (CAP right-censored at 17.11; OT interval-censored (0, 4.20]) — KILL

EM converged in 2–3 iterations. Holdout **ΔLL −0.462m** vs v6 (iid CI
[−1.27, +0.37], block [−1.38, +0.42], P(better) 0.15) — INSIDE NOISE FLOOR
(within-family 1.77m), sign negative; expected-ROI translation 0.000 (below
ladder resolution). σ-sensitivity ×0.8/×1.25: −0.470/−0.451m — flat. Bias:
elite five DID NOT move toward 0 (T1 −2.7→−2.3, PRX −10.2→−10.1,
100T −7.1→−7.4, NRG −9.1→−9.3, TL −6.2→−6.6 pts); max|bias| worsened
0.1478→0.1492 (TS). Mechanism dead — with 0.23% of maps censored there is
nothing for the likelihood to un-censor, and OT re-imputation is symmetric
noise on near-coin-flip maps.

## E2 round-level Bradley-Terry — TIE; the power-play claim is FALSIFIED

Joint fit: 4,349/5,140 maps (84.6%) as side-binomial round cells,
791 uncovered maps (all 26 corpus-addition events + 2026_ewc_qual gaps) as
race-function map-Bernoulli terms in the same fit; v6's exact per-day weights
(replication parity-gated at 0.0 rdiff gap); λ=2.0 ridge + 3λ region prior
(train-only, grid extended past its preregistered edge — logged); β=1.73
train-only; fitted attack advantage +0.035 logit (raw attack share 50.9%).
Holdout LL **0.64238** vs v6 0.64216 → **ΔLL −0.222m** (iid [−2.62, +2.11],
P(better) 0.44) — a dead tie, INSIDE the cross-family floor 5.89m; ROI
translation 0.000. Bias: elite five slightly WORSE (PRX −10.2→−10.9), max
|bias| 0.1515. Bucket-level: favorite [.8,.9) −12.8m (n=26, noise-exempt),
international −4.4m (n=122, floor ~5.9m) — nothing gateable.

**Measured effective-sample multiplier** (ref 2025-01-01, preregistered):
per-map Fisher info ratio 1.44 (mean); per-team SE ratio² median **1.25**
(Fisher, 33 teams, range 1.00–1.37); **cluster bootstrap by match: 0.80**
(B=200, CRN-derived stream). The brief's "~order of magnitude" claim needed
k ≥ 7: **FALSIFIED — the real multiplier is ≈1**. Rounds within a match are
correlated (economy/momentum), so the binomial's nominal information does not
survive match-level resampling. Consequently a round-level referee buys NO
MDE improvement (first-order: 1.98m within / 6.59m cross at k=0.80 — flat to
worse, covered subset only). No power play here.

## Standing conclusions

1. H1 is dead at the premise level: the elite under-rating (expanded-frame
   caterpillar: PRX −10.2, NRG −9.1, 100T −7.1, TL −6.2; T1 only −2.7 here —
   the brief's −7.6 was the old-frame number) is NOT a margin-censoring
   artifact. Two independent likelihood replacements (Tobit, round-BT) leave
   it intact; whatever causes it lives elsewhere (schedule/opponent-pool or
   prediction-layer — other Phase 5 agents' territory).
2. A structurally different likelihood (round-level BT with race-function
   fallback) lands within 0.2m of the tuned Massey champion — the map-level
   margin target already extracts essentially all rating information the
   rounds contain.
3. The round-level "order of magnitude more data" intuition is quantitatively
   refuted: k_eff ≈ 1.25 (Fisher) and ≈ 0.8 (clustered). Do not budget any
   future power gains on round-level evaluation.
