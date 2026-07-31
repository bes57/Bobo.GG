# agent:autopsy — Phase 7: live P&L autopsy

## Scope (one question)
Why is the live bot down money on its fills since July 2026 — model, config,
execution (adverse selection), fees, or noise? One-line verdict, dollar
figures, independently of any modeling work.

## Context
- Rules: testing_lab/v8/README.md. VCTMM IS HANDS-OFF: read-only everywhere.
- LOCAL repo ~/VCTMM is the dev copy — its db (db/vctmm.sqlite3) has only 4
  fills. THE LIVE BOT RUNS ON A VM: vctmm@4.150.24.178 (Azure), repo at
  /home/vctmm/VCTMM, db at /home/vctmm/VCTMM/db/vctmm.sqlite3 (WAL). The
  brief says ~719 fills since July 2026 — VERIFY the real number, never
  assume.
- Consistent read-only snapshot procedure (the ONLY sanctioned VM touch):
  `ssh vctmm@4.150.24.178 "sqlite3 /home/vctmm/VCTMM/db/vctmm.sqlite3 \"VACUUM INTO '/tmp/v8_autopsy.db'\""`
  then scp that file to testing_lab/v8/data/vctmm_live.db, then
  `ssh ... "rm /tmp/v8_autopsy.db"`. Nothing else on the VM: no writes into
  the repo, no restarts, no order/cancel calls, no reading .env or keys/.
  If ssh fails, STOP the VM branch, say so, and deliver the local-db +
  public-API portions only.
- Local schema (VM's may differ — enumerate it fresh): events, markets,
  match_links (event_ticker↔vlr_match_id), fills (side_role, price_cents,
  qty, ts, dry_run — NO fee column), orders, tape (public trade prints:
  trade_id, market_ticker, yes_price_cents, count, taker_side, created_ts),
  deploy_index (locked_pairs, locked_profit_dollars), lots (qty_unhedged),
  volume_samples, audit_log. Filter dry_run correctly everywhere.
- Live config (Mac copy read this session — re-read the VM's config.toml via
  ssh cat for truth): hard_min_edge_cents=5 (flat), min_edge_cents=1,
  hedge_margin_cents=2, order_expiry_lead_hours=2, kelly_fraction=0.5,
  series KXVALORANTGAME, host api.elections.kalshi.com.
- Research recommendations to audit against (trading_model/FINDINGS.md §3-5):
  logit-space min edge +0.5–0.6 (≈14¢@50¢/8¢@85¢) vs live flat 5¢; expire
  start−2h (Rule 5); "skip NO quotes on model-p<~45% sides" refinement
  (−5.7% pocket) — confirm whether implemented by reading vctmm strategy
  code (local repo, read-only; grep for the rule); hedge margin 2¢ vs
  fees + measured adverse selection. Sim reference points: flat 5¢ = +6.7%
  [−9,+23]; logit +0.6 = +29.1% [+1,+61] (stats/quote_margin.json).
- Kalshi public API for anything the db lacks: markets/trades (prints),
  markets (settlement), series fee schedule. Public endpoints only, polite
  rate (≤2 req/s), never authenticated.
- Model predictions for fill-conditional calibration: the frozen snapshot
  trading_model/model_snapshot.json + trading_model/predict.py (reference
  math), testing_lab/out/prediction_log.csv, and (with match_links) VLR match
  outcomes from data/match_results.csv. State clearly which prediction
  vintage you use per fill; if the snapshot at fill time isn't recoverable
  (the 2026-07-23 incident replaced live snapshots for a window — read
  trading_model/INCIDENT_2026-07-23_vm_snapshot_drift.md), bracket results
  with both the frozen v6 and the price-implied model and say so.

## Pre-register first
testing_lab/v8/preregister.autopsy.md: the decomposition identity you'll use
(realized P&L ≡ locked-pair margin + unhedged settlement P&L − fees, with
open-inventory mark clearly separated), the adverse-selection test (fill-
conditional vs unconditional calibration; markout decomposition spread-capture
vs adverse-move), and what result would blame model vs config vs execution vs
noise.

## Work
1. **Snapshot + inventory.** VM db snapshot; enumerate schema; counts per
   table; date range of fills; verify "719 fills" and dollar volume. Also ssh
   cat the VM's live config.toml + `git -C /home/vctmm/VCTMM log --oneline
   -5` (read-only provenance of what code is live).
2. **P&L decomposition** with dollars: locked hedge pairs (arithmetic margin),
   unhedged settled inventory, fees (see 3), open inventory marked at last
   trade. Waterfall JSON.
3. **Fee verification, live.** Determine actual maker/taker fees paid: from
   the db if recorded; else reconcile implied cash vs settlements; AND fetch
   Kalshi's current fee schedule for the series from the public API/docs.
   State whether maker fees are zero and whether a 2¢ hedge margin clears
   the real cost stack.
4. **Fill-conditional calibration.** On filled markets: model p (stated
   vintage) vs settlement outcome, reliability with Wilson CIs, versus the
   same model on ALL VCT Kalshi markets in the window (unconditional). The
   delta IS the adverse-selection measurement. Slice by side_role
   (side1/hedge), price band, minutes-to-start.
5. **Markouts.** Per fill: mid (or last trade) at +5m, +30m, +2h, start−5m
   from tape + public prints; maker P&L = spread capture + adverse move;
   slice by minutes-to-start at fill. Identify the toxic window if any.
6. **Config-vs-research audit** with expected-ROI deltas from
   stats/quote_margin.json + out/trade_sim.json machinery: flat-5¢ vs logit
   +0.5/+0.6; expiry rule live vs recommended; skip-NO<45% implemented?;
   hedge margin. Table: live setting / recommended / expected ROI delta / CI.
7. **Variance check.** Bootstrap per-fill P&L under the sim's true-edge point
   estimate: P(cumulative loss ≥ observed | edge real). One sentence verdict:
   distinguishable from noise or not.

## Outputs (yours alone)
- testing_lab/v8/data/vctmm_live.db (snapshot)
- testing_lab/v8/stats/autopsy_pnl.json, autopsy_fees.json,
  autopsy_fill_calib.json, autopsy_markouts.json, autopsy_config_gap.json,
  autopsy_variance.json, quote_density.json (fill/quote density by price
  band — agent:referee consumes this)
- testing_lab/v8/phase7_autopsy.md (deliverable 3: dollars + one-line verdict)
- testing_lab/v8/preregister.autopsy.md, logs/autopsy.log

## Forbidden
Any VM write outside the single /tmp snapshot file (removed after). Any write
under ~/VCTMM. Authenticated Kalshi calls. Order actions. Secrets.

## Done criteria
Verified fill count + P&L with decomposition; fees verified from source;
fill-conditional calibration measured; markouts computed (or the data gap
quantified); config table with ROI deltas; noise verdict.

## Return format
≤500 words: the one-line verdict, the waterfall (dollars), adverse-selection
number, top config gap by expected ROI, artifact paths, gaps. No transcripts.
