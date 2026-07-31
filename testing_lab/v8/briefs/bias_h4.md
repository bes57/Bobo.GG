# agent:bias-h4 — Phase 5 H4: series aggregation assumes iid maps

Read briefs/wave2_common.md first; it is law. Scope: the closed-form bo3/bo5
math treats maps as independent draws at one p; deeper map pools plausibly
lower cross-map variance, understating elite series probability. NOT the
rejected intra-series momentum (within-series carryover) — your writeup must
draw that line explicitly in its first paragraph.

## Experiments (preregister each)
1. **Dispersion diagnostic.** On train series: realized map-win counts per
   series vs the closed-form binomial-style implied distribution at the
   model's p (v6 predictions reconstructed on the frame). Estimate
   over/under-dispersion overall, then conditioned on: map-pool depth
   (team's distinct maps played in trailing 90d and veto-era pool from
   data/map_vetos.csv — filter the MapNum=="all" hazard when joining maps),
   favorite strength band, format. If maps are NOT under-dispersed for
   deep-pool favorites, H4's premise is weak — report and stop.
2. **Dispersion-parameterized series link.** Replace iid aggregation with a
   correlated-maps model: maps share a series-level random effect (e.g.
   p_map|series = sigmoid(logit(p) + u), u ~ N(0, σ_u) — closed-form via
   quadrature) or beta-binomial ρ. σ_u/ρ fit on TRAIN, optionally conditioned
   on map-pool depth (1 coefficient). Score holdout: LL overall, favorite
   bands (elite bias is the target: does P(bo5 win) for strong favorites
   rise?), per-team bias caterpillar, GF bucket.
3. **Interaction guard.** Whatever σ_u does must be reported alongside the
   b_pick/map surface note: no map-level double counting (the ledger's
   per-map+pick kill stays dead); your dispersion acts ONLY at the series
   aggregation layer over the single p.

## Outputs (yours alone)
- stats/h4_dispersion_diag.json (realized vs predicted variance by depth,
  chart-ready), h4_series_link.json, h4_bias_caterpillar.json
- phase5_h4.md (first paragraph: the momentum distinction),
  preregister.bias_h4.md, logs/bias_h4.log, scratch/bias_h4/

Done when: diagnostic measured with CIs; the correlated link scored on
holdout with bias/bucket tables; verdict plain, MDE-contextualized, both
units; the momentum distinction stated.
