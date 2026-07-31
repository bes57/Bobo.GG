# Private Trading Model (BenPom v6) — Handoff for VCTMM

> **Update 2026-07-30, operator decision: v6 deployed to the public site (site_model.json); site and bot now share the champion.**

**This model is deliberately NOT deployed on bobo-gg.net.** The website continues to run
the public production model. This folder is the private, more accurate model for the
trading bot only. Do not surface its numbers on any public page.

## What this is

`benpom-v6-2026-07-22` — v5 plus the consistency-decay fix (see below), from the 9-round optimization program
(full record: `/testing/report/state_of_benpom` on the site, password-gated).

| Validation (walk-forward, holdout 2025–26) | Log-loss |
|---|---|
| Public production model | 0.6526 |
| **This model (v6)** | **0.6413** (+11.4 milli-LL vs production, p>0.998) |
| Kalshi pre-match market, VCT overlap n=86 | market 0.6457 vs **model 0.6441 — ahead** (v5 basis; v6 re-check pending new settles) |
| Simulated divergence trading (5+pt gaps, taker) | **+29% ROI** [CI +3%, +53%] |

What it does differently from production: decay by **games played, consistency-conditioned**
(results consistent with a team's level fade with a 20-game half-life, anomalies with 12 —
this replaced a flawed win/loss asymmetry that inflated chronically losing teams by letting
their losses decay away; operator-caught 2026-07-22) instead of calendar weeks; margin^0.75; playoff/GF games
weighted ×1.6 in the solve; teams regularized toward their **region's** trailing mean;
cold-start teams enter at their region's 25th percentile; **fitted cross-region offsets
replace intl_exp/cn_dog entirely**; EWC-class events (qualifiers, China Evolution Series,
EWC) included in the data.

## Start here

**`FINDINGS.md`** is the complete research digest — model spec, deployment rules, the
quoting-margin config change (logit +0.5–0.6, NOT flat cents), edge anatomy, team-bias
watchlist, telemetry spec, and the do-not-retest ledger. `docs/` holds the four full report
pages with charts; `stats/` holds the raw JSONs behind every number.

## Files

- `build_model_snapshot.py` — solves current ratings from PythonTest data and writes
  `model_snapshot.json`. **Run after every data refresh** (same cadence as the site scrape).
- `model_snapshot.json` — everything needed to price a match: ratings, region priors,
  cross-region offsets, β, pick bonus, GF logit. Embeds `generated_utc` and
  `ratings_as_of` — apply the bot's staleness rules against these.
- `predict.py` — the reference pricing implementation (unchanged by v6; snapshot carries the fix) (library + CLI). If the bot
  reimplements, it must match this file's math exactly (parity-test against it).

## Usage

```bash
python3 trading_model/build_model_snapshot.py          # refresh the snapshot
python3 trading_model/predict.py FNC TH bo3            # P(FNC beats TH)
python3 trading_model/predict.py LEV PRX bo5_gf --upper LEV
```

```python
from predict import load_model, series_probability, map_probability
m = load_model()
p = series_probability(m, "FNC", "TH", "bo3")          # series fair value
q = map_probability(m, "FNC", "TH", a_picked=True)     # map-level, pick-aware
```

## Integration notes for the trading terminal

1. **The series fair value is the closed form in `predict.py`** — NOT the 20k-sim veto MC.
   Tested head-to-head: the MC's per-map/veto layer adds no series-level accuracy on this
   model (every ensemble weight lost). If map-level or veto-conditional prices are ever
   needed, use `map_probability` with the pick bonus — and do NOT add per-map rating splits
   on top (they double-count the pick bonus; measured).
2. **β = 0.1299 is scale-bound to this exact config** (v6 refit; the snapshot carries it — always read beta from the snapshot, never hardcode). If any constant in
   `snapshot["config"]` is changed, β must be refit before quoting. Never mix this β with
   production ratings or vice versa.
3. **Cross-region matches**: the `xregion_offsets` adjustment in `predict.py` is the ONLY
   cross-region correction. Do not re-apply the old intl-experience/CN-dog offsets — they
   are superseded (intl_exp refits to zero on these ratings).
4. **Unknown/new orgs** (not in `ratings`): pass their region so they price at the region
   prior. If the region is unknown too, do not quote the market.
5. **Grand finals** (`bo5_gf`): pass `upper=<org>`; the +0.25 logit goes to the
   upper-bracket team.
6. **Data dependency**: the builder reads this repo's `data/` (games, vetoes, dates). The
   snapshot is only as fresh as the last scrape. Recommended chain per refresh:
   site scrape → `build_model_snapshot.py` → bot reloads `model_snapshot.json`.
7. **Deployment policy**: follow the Playbook (`/testing/playbook`) — regular-season
   windows by default; size concentrated on 25–45¢ model-backed sides at 5–10pt divergence;
   quarter-size at LANs, post-break weeks, and 15+pt gaps where the market is more
   confident; quote from listing, scale size 12h→2h, expire start−2h.
8. **Telemetry**: implement the collection spec (Playbook §8) so the next model iteration
   has markouts, book snapshots with own-order flags, deployment snapshots, and roster
   observations to learn from.

## Known limits

- Kalshi evidence spans ten weeks of one season (n=86 VCT matches; trade sim n=90).
  Positive and consistent, but wide CIs — re-underwrite monthly as the prediction logger
  and telemetry accumulate.
- The model reads only match results. Roster news, stand-ins, and LAN form reads are
  invisible to it — that's what the size-down rules exist for.
- `region_priors` / `xregion_offsets` / `b_pick` are refit inside every snapshot build
  (walk-forward-consistent); ratings move with every new match ingested.
