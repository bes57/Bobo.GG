# Pre-registration — agent:autopsy (Phase 7 live P&L autopsy)

Written 2026-07-28, BEFORE any P&L/calibration/markout computation. Inputs read so
far: brief, README, FINDINGS.md, quote_margin.json, trade_sim.json, incident memo,
local VCTMM config.toml + engine/quotes code (mechanics only), local db schema.
No fill rows have been aggregated yet.

## Question
Why is the live bot down money on real (dry_run=0) fills since July 2026:
model, config, execution (adverse selection), fees, or noise?

## Decomposition identity (fixed in advance)

Per event, real fills only (dry_run=0), NO-side prices in cents:

    realized P&L ≡ locked-pair margin + unhedged SETTLED P&L − fees
    total P&L    = realized P&L + open-inventory mark (reported SEPARATELY)

- Locked pairs: NO on both markets of an event pays 100¢/pair. Pairs = FIFO
  match of side-1 and side-2 fill qty per event (cross-check vs lots +
  deploy_index.locked_pairs/locked_profit_dollars; discrepancies REPORTED,
  never smoothed). Margin/pair = 100 − p1 − p2.
- Unhedged settled: unpaired contracts on settled markets; NO pays 100 iff
  the market's team lost. Settlement source: db market status/result, else
  Kalshi public /markets result, else VLR outcome via match_links.
- Open inventory: unsettled markets, marked at last trade (tape; else public
  trades API; else fill price, flagged). Mark P&L is NEVER mixed into realized.
- Fees: per-contract Kalshi fee (schedule verified from public API/docs this
  run; expected: taker ceil(0.07·P·(1−P)) per contract, maker 0 — VERIFY).
  Maker/taker per fill decided by trade_id join to tape.taker_side vs our
  resting side (bot posts limit orders ⇒ expected maker).
- Identity audit: components must sum to independently computed cash P&L
  (Σ settlements + Σ marks − Σ costs − fees). Residual > $1 ⇒ investigate,
  report as its own waterfall line.

## Adverse-selection test (fixed in advance)

1. Fill-conditional calibration: per real fill, model q = P(NO pays) =
   1 − p_model(team wins), at stated vintage. Contract-weighted reliability
   (Wilson CIs) of realized NO-pay rate vs mean q, on settled fills.
2. Unconditional benchmark: same model, ALL VCT Kalshi markets settled in the
   fill window (one obs per market side, and price-band matched view).
3. **Adverse-selection number** = (realized − predicted on fills) −
   (realized − predicted unconditional), in probability points; also stated
   as ¢/contract. Negative ⇒ fills are adversely selected.
4. Markout decomposition per fill from tape/public prints, NO terms
   (NO_mid = 100 − YES_mid): maker P&L = spread capture (mid@fill − price) +
   adverse move (mid@T − mid@fill), T ∈ {+5m, +30m, +2h, start−5m}. Slice by
   side_role, price band, minutes-to-start. Sparse tape ⇒ quantify the gap
   (coverage %), do not interpolate silently.

## Prediction vintage (stated per fill)
Primary: frozen v6 `trading_model/model_snapshot.json` via `predict.py`
(reference math). Fills while the VM served a drifted rebuild (2026-07-23
13:07 UTC → the post-sync re-enable; exact window from VM audit_log
model_rebuild entries) are bracketed: frozen v6 AND price-implied (tape mid
at fill). Any fill whose event can't be priced by v6 (missing org mapping)
is excluded and counted loudly.

## Blame rules (fixed in advance)
- **MODEL**: unconditional v6 calibration in the window is off (CI excludes 0)
  in the same direction as fill losses — the model is wrong everywhere, not
  just where it got filled.
- **CONFIG**: unconditional calibration fine, but losses concentrate in
  pockets research already flagged: fills that a logit +0.5/+0.6 cap would
  have refused, NO quotes on model-p<45% sides (rule confirmed ABSENT from
  live code), fills inside the expiry window, hedge margin not clearing the
  fee stack. Realized ROI inside flat-5¢ sim CI [−9.2%, +22.8%] but below
  logit+0.6 counterfactual ⇒ config gap, not model failure.
- **EXECUTION (adverse selection)**: adverse-selection number significantly
  negative AND markout adverse move exceeds spread capture; unconditional
  calibration fine.
- **FEES**: fee line ≥ 50% of gross loss, or 2¢ hedge margin − fees/pair ≤ 0.
- **NOISE**: bootstrap per-fill P&L under the flat-5¢ sim point edge
  (+6.7% ROI): p = P(cum P&L ≤ observed). p ≥ 0.10 ⇒ not distinguishable
  from noise; p < 0.05 ⇒ real underperformance; else weak evidence.
  Blame is a waterfall, not a single label; dollars per bucket.

## CRN
README rule 3: bootstrap randomness from v8/crn.json. Not yet present at
prereg time (power agent writes it). At variance-check time: re-check; if
still absent, use documented fallback seed 780728 and FLAG the deviation in
autopsy_variance.json + phase7_autopsy.md.

## Outputs
quote_density.json (early, for agent:referee: fill+quote density by NO price
band × side_role, real fills), autopsy_pnl/fees/fill_calib/markouts/
config_gap/variance JSONs, phase7_autopsy.md, snapshot db, log.
