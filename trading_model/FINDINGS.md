# FINDINGS.md — Complete research digest for the trading bot (BenPom v6)

> **Update 2026-07-30, operator decision: v6 deployed to the public site (site_model.json); site and bot now share the champion.**

Everything the bot (and the agent operating it) needs from the 2026-07-22 research program.
Sources: `docs/*.html` (full reports with charts), `stats/*.json` (raw numbers),
`model_snapshot.json` (the live model). Nothing here touches the public website's model.

---

## 1. The model (v6 — deployed in model_snapshot.json)

| Component | Value | Why |
|---|---|---|
| Rating solve | Massey, target sign(rd)·\|rd\|^0.75·2.5 | margins carry signal; ^0.75 beat sqrt & raw |
| Decay | games-counted, **consistency-conditioned**: HL 20 (result matches team level) / HL 12 (anomaly) | info arrives per game (break-proof); keeps elite wins AND floor losses as signal |
| Solve weights | playoffs/GF ×1.6 · Champions ×2 · ridge 0.5 · **region-prior ridge 1.5** | high-stakes games more informative; teams regress to region context |
| Roster | year-boundary continuity 0.3 | finer schemes all tested worse |
| Cold start | region 25th percentile | new orgs are below-average entrants |
| Cross-region | fitted offsets, CN pinned 0, refit each snapshot | replaces intl_exp/cn_dog entirely |
| Link | logistic; β in snapshot (0.1299; canonical full-stack refit 0.1256 — adopt at next rebuild) | 5 tail-modification families tested, all rejected |
| Series | closed form (bo1/bo3/bo5); GF +0.25 logit to upper | veto-MC adds no series-level info |
| Map-level | overall rating + pick bonus b_pick (snapshot) | picker advantage real & growing; per-map splits double-count it |

**Validation**: holdout 2025–26 LL 0.6409 vs production 0.6526 (+11.8m, p>0.998).
Kalshi VCT overlap: at/above market parity (model 0.6441 vs market 0.6457 pre-fix basis).
History: v5's win/loss asymmetric decay had a bug (erased floor teams' losses — EG rated
league-average on a 1-series-win season); operator-caught, fixed by consistency conditioning.

**Reference pricing = `predict.py`.** Any reimplementation must parity-test against it.

## 2. Where the model is strong/weak (stats/v6_profile.json)

- Improves 21/23 buckets vs production. Sharpest: huge gaps (LL 0.44), grand finals (0.60).
  Noisiest: EWC-class events (0.69), coin flips (0.69), playoffs (0.68).
- Self-calibration: all favorite bands inside CIs ([0.7,0.8): 0.744 pred / 0.750 won).
- **Team-bias watchlist (holdout)**: still UNDER-rates elite — T1 −7.6pts, PRX −7.3,
  100T −6.9, NRG −6.9, TL −6.5. OVER-rates: TS +16.3, JDG +12.6, TE +10.4, C9 +9.2, FUR +8.6.
  Quoting note: caution when the model fades T1/PRX-class teams; don't defend over-rated
  mid teams at full size. This is the #1 v7 target.

## 3. Deployment rules (docs/when_to_deploy.html — the 7 rules)

1. **Default on** in regular-season VCT windows + EWC-class events (every slice ROI-positive).
2. **Size the pocket**: model-backed sides at 25–45¢ with 5–10pt divergence (+35–41%, CI>0).
3. **Respect favorites** above ~55¢ (market near-fair; quote for spread, not edge).
4. **Quarter-size information-risk**: LANs, first week after 45+ day breaks, and any 15+pt
   divergence where the market is MORE confident (the May-fade shape — market knew).
5. **Quote early, size late**: quotes at listing (3¢ spreads, 3¢ drift left, ~5% of volume);
   scale size through 12h→2h (44% of volume); expire start−2h.
6. **Let liquidity in** (edge was larger in high-volume markets: +36% vs +20%).
7. **Re-underwrite monthly** as the prospective ledger grows.

Season ledger (taker-priced backtest): +10.25u on 35.8 staked, 90 trades, 46W/44L, +28.7%
ROI [CI +3%, +53%]. June (London LAN) was the only losing month (−12.4%).

## 4. Quoting margin — THE key config change (stats/quote_margin.json)

- **Flat 1¢ min-edge (old config) earned ≈ nothing (−0.6%) over the whole history.**
- **Winner: logit-space edge +0.5 to +0.6** — quote NO at the price implied by shifting the
  model's yes-logit up by δ. Only rule with CI above zero (+29.1% ROI, +8.12u).
- Cents table (per side, on the NO cap): ≈14¢ @ 50¢ · 12¢ @ 65¢ · 10¢ @ 75¢ · 8¢ @ 85¢ · 5¢ @ 92¢.
- Verified: both time halves positive, 1¢-trade-through fills +26.1%, with deployment
  exclusions +52.6%, extended window +36.9%.
- **Refinement: skip NO quotes on sides the model prices < ~45%** (the one negative fill
  pocket, −5.7%): selling underdogs is the wrong side of the market's favorite-bias.

## 5. Edge anatomy (stats/edge_anatomy.json, docs/favorites_lab.html)

- Golden cells: **35–65¢ price × 5–10pt divergence** (+51% ROI, win 62–88% vs implied).
- **2–5pt divergences are NEGATIVE at every price** — sub-threshold disagreement is model
  noise; never trade or thin-quote it.
- Deep dogs (<20¢) pay only at 10+pt divergence (+61%, 25% hit — lottery variance).
- Edge-by-price: model sides beat their price most at 35–45¢ (+12pts); underperform at
  coin flips; fair above 55¢.
- Market-referenced calibration (provisional, entangled with pre-fix model): 75–85¢ market
  favorites won 72.7%; 55–65¢ favorites were coin flips.

## 6. Microstructure (stats/deploy_micro.json)

| Hours to start | Spread | Volume share | Price move remaining |
|---|---|---|---|
| 48–24h | 3.0¢ | 4.5% | 3.0¢ |
| 24–12h | 2.0¢ | 8.5% | 2.0¢ |
| 12–6h | 1.0¢ | 15% | 1.5¢ |
| 6–2h | 1.0¢ | 29% | 1.0¢ |
| final 2h | 1.0¢ | 43% | 0.5¢ |

## 7. Telemetry the bot should start recording (full spec: docs/when_to_deploy.html §8)

Append-only SQLite, UTC, every row carries `ts_utc`, `mins_to_start`, `scheduled_start_utc`,
`model_version`. Tables: `book_snaps` (fixed horizons + deploy/undeploy + settlement, top-3
depth, **own orders flagged** — irreplaceable), `fv_log` (FV + full input provenance),
`quote_events`, `fill_markouts` (+5m/+30m/+2h mids after every fill), `deploy_events`
(book+FV+config at every arm/disarm), `market_meta` (listing/scheduled/actual/settlement
times), `roster_obs` (lineups + weekly off-week team-page scrapes), optional `trade_tape`
(aggressor side). Logging must never block the order path.

## 8. Do-not-retest ledger (docs/final_model.html §4 — 33 entries)

Highlights the bot-side agent must not re-litigate: shorter memory, calendar decay,
win/loss asymmetric decay (the v5 bug), lineup/roster reweighting, player-carryover priors,
post-break dampeners, favorite-margin discounts, rolling β/Platt refits, every tail-shape
link modification, margin-as-outcome fits (margins UNDERSTATE win prob — better teams win
the close maps), veto-rate shrinkage, per-map splits stacked with pick bonus, closed-form ×
MC ensembles, rematch terms. Re-open triggers: self-band drift, a new data source, or an
operator anomaly report (which gets case-level forensics FIRST — the v6 lesson).

## 9. Monitoring tripwires (monthly)

- Self-band calibration on accumulating results ([0.7,0.8) and [0.8,0.9) drift = investigate).
- Kalshi market-band re-measure with v6-era settles (favorite-bias claim is provisional).
- Prediction logger keeps running (`testing_lab/log_predictions.py`).
- β reconciliation: adopt 0.1256 at next snapshot rebuild.

## 10. Bundle layout

```
trading_model/
  model_snapshot.json      <- the live model (bot reads this)
  predict.py               <- reference pricing (parity-test any reimplementation)
  build_model_snapshot.py  <- rerun after every data refresh
  README.md                <- integration rules (8 rules; β discipline; GF handling)
  FINDINGS.md              <- this file
  docs/                    <- full report pages w/ charts (open in a browser; nav links
                              point at the password-gated site and won't resolve standalone)
  stats/                   <- raw JSONs behind every number above
```
