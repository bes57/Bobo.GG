# Phase 5 H2 — schedule connectivity: DEAD

agent:bias-h2, 2026-07-28. Preregistered in preregister.bias_h2.md (gate and
all specs written before any run). Frame: v8/data/frame_expanded/series.csv,
sha256 verified vs crn.json; holdout n=1217. Baseline: v6 config on the
expanded frame — β=0.1152 (train-fit), holdout LL 0.64216, per-team bias table
43 teams (min_n=25), max|bias| 14.8 prob-pts (TS), mean|bias| 4.7.

## Verdict

**H2 is dead.** Per-team bias does not track opponent-graph centrality, and
the preregistered gate stops the program here: no ridge ablation (E2), no
hierarchical pooling (E3), no rescue fits. Correlations over 43 teams, CIs
from 4000 team-resamples under the crn seed:

| target | feature | Spearman [95% CI] | Pearson [95% CI] | gates? |
|---|---|---|---|---|
| bias | eig_centrality | **−0.304 [−0.566, +0.005]** | −0.277 [−0.518, +0.002] | yes — FAIL |
| \|bias\| | eig_centrality | **−0.077 [−0.421, +0.281]** | −0.139 [−0.444, +0.197] | yes — FAIL |
| bias | opp_count | −0.314 [−0.575, −0.007] | −0.345 [−0.554, −0.094] | no |
| bias | opp_diversity | −0.076 [−0.379, +0.249] | −0.133 [−0.353, +0.163] | no |
| bias | xregion_share | −0.431 [−0.640, −0.157] | −0.420 [−0.595, −0.210] | no |
| \|bias\| | opp_count / diversity / xshare | all null (CIs span 0) | all null | no |

The signed bias–centrality correlation is a near-miss (upper CI +0.005), but
the mechanistically decisive row is |bias|: if poor connectivity degraded
rating identifiability, poorly connected teams would have LARGER errors —
they do not (ρ = −0.08, CI [−0.42, +0.28], as null as it gets at this n).

## Why the signed correlations light up anyway

The signed bias ~ opp_count and bias ~ xregion_share correlations (CIs exclude
0) are composition, not connectivity: teams that travel are disproportionately
elite (they qualify), and elite teams are under-rated; insular teams are
disproportionately floor, and floor teams are over-rated. Direct evidence:

- PRX has the HIGHEST centrality of the named teams (0.894) and is the MOST
  under-rated (−10.2 pts). T1 (0.634): −2.7.
- TE (0.708) and JDG (0.501) are well-connected inside the dense CN+EWC
  cluster yet over-rated (+7.2, +10.6). TS: peripheral AND over-rated (+14.8).
- C9 and FUR sit in the well-connected Americas pool and are over-rated
  (+11.7, +12.0) — insularity not required.
- Partial Spearman of bias ~ centrality controlling for mean v6 rating
  attenuates to −0.188 [−0.491, +0.161] — consistent with centrality proxying
  strength rather than carrying its own bias signal.
- Per-region mean signed bias is small (Americas +1.1, Pacific +0.2, CN +1.4,
  EMEA −2.7 pts): the region-prior ridge is not leaving a large regional
  signed residue. The bias axis is elite-vs-floor (H1's territory), not
  region-vs-region.

Small-N caveat applies throughout: 43 teams, CI half-widths ≈ ±0.3.

## Corpus expansion did move the graph — and still no relationship

At T=2026-07-01, removing the 25 corpus-addition events (EWC/off-season)
changes eigenvector centrality by mean |Δ| = 0.099 (max-normalized units)
across bias-table teams. The additions re-center the graph toward the CN/EWC
cluster: JDG 0.613 → 0.775 (rank 12→8), TE 0.559 → 0.740 (14→10), while
Western elite drop in relative terms (NRG 0.705 → 0.524, T1 0.433 → 0.356,
PRX 0.774 → 0.695; TL rank 30→30, TS 43→42). A ~0.1-unit graph perturbation
with visibly re-ranked hubs produced no bias-centrality coupling — additional
evidence the bias structure does not live on the schedule graph.

## What this kills and what it doesn't

- Killed: "per-team bias is explained by opponent-graph position" (H2 as
  posed), and with it the motivation for graph-targeted fixes (E2 ridge
  ablation, E3 hierarchical region pooling) **as bias remedies**.
- Not killed: the elite-compression bias itself (max|bias| 14.8 on the
  expanded frame — real and still the program's largest per-team distortion;
  H1's censoring/round-level mechanisms are the live explanations), and the
  narrow question of whether region_prior_ridge=1.5 is optimal for LL, which
  any future agent may test as a plain hyperparameter — just not under a
  connectivity story.

## Artifacts

- stats/h2_centrality.json — scatter (43 teams × 4 features + bias), full
  correlation table with CIs, gate record, expansion-shift table.
- stats/h2_bias_caterpillar.json — v6 expanded-frame caterpillar (43 teams);
  variant columns absent by design (gate).
- stats/h2_ridge_ablation.json, stats/h2_hierarchical.json — NOT-RUN markers
  citing the gate.
- preregister.bias_h2.md (+outcomes), logs/bias_h2.log, scratch/bias_h2/.

One holdout scoring was performed (the v6 baseline that defines the bias
table); no candidate was fit or scored. Nothing was tuned on holdout; market
data untouched.
