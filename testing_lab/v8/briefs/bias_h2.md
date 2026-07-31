# agent:bias-h2 — Phase 5 H2: schedule connectivity

Read briefs/wave2_common.md first; it is law. Scope: is per-team bias
explained by the opponent graph — regional pools connected only at
internationals, plus the region-prior ridge (1.5) pulling within-region
extremes toward the mean?

## Experiments (preregister each)
1. **Connectivity diagnostic.** Per team (train period, walk-forward at each
   holdout month): opponent count, opponent diversity (distinct opponents /
   games), eigenvector centrality in the opponent graph (weight = decayed
   game count), cross-region game share. Correlate per-team bias
   (referee.per_team_bias on v6 holdout predictions) against each; report
   coefficients with CIs (crn bootstrap over teams — note the small-N caveat,
   42 teams with min 25 series). The corpus expansion ADDED cross-cluster
   edges (EWC/off-season events) — quantify how much centrality moved for
   the named elite/floor teams vs the pre-expansion graph.
2. **Ridge ablation on the graph story.** region_prior_ridge ∈ {0, 0.75, 1.5}
   × plain ridge grid on TRAIN (walk-forward internal split), then the
   train-chosen config scored ONCE on holdout. Does weakening the pull
   decompress elite ratings (bias table) without wrecking thin-data teams?
3. **Hierarchical partial pooling.** Replace the fixed region-prior ridge
   with region random effects whose variance is FIT (empirical Bayes on
   train): team_rating ~ region_mean + deviation, deviation shrunk by
   fitted τ_region. Implement in scratch (your own solver or engine
   rd_custom/ridge machinery — preregister which). Judge: holdout LL, bias
   caterpillar, max|bias|, cold-start bucket.

If the diagnostic shows NO bias-centrality relationship, H2 is dead —
publish that with the correlation CI and stop; do not fit models to rescue it.

## Outputs (yours alone)
- stats/h2_centrality.json (scatter data: bias vs centrality, per team,
  chart-ready), h2_ridge_ablation.json, h2_hierarchical.json,
  h2_bias_caterpillar.json
- phase5_h2.md, preregister.bias_h2.md, logs/bias_h2.log, scratch/bias_h2/

Done when: the correlation is measured with honest CIs; the ablation and
pooling variants are scored on holdout with bias tables; verdict stated
plainly either direction.
