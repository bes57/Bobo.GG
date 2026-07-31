# Phase 7 — Live P&L autopsy (agent:autopsy, 2026-07-28)

Pre-registered: `preregister.autopsy.md` (written before analysis). Data: VM db
snapshot `data/vctmm_live.db` (VACUUM INTO /tmp → scp → rm, the only VM touch;
plus read-only cat of config.toml), Kalshi public API (unauthenticated, ≤2 req/s),
frozen v6 `trading_model/model_snapshot.json` + `predict.py`.

## One-line verdict

**Not noise, not fees, not microstructure: the bot is down $309 realized because
its flat 5¢ edge floor let it accumulate every model–market disagreement, and in
this window the market — not BenPom v6 — was right; the already-researched
logit-space cap (+0.6) and skip-NO<45% rules would have refused the fills that
carried essentially all of the loss.**

## Verified facts

- **719 real fills** (dry_run=0; zero dry-run rows in the live db), 8,143
  contracts, $3,803.32 cost, 2026-07-21T19:11 → 2026-07-28T21:36 UTC, 69 markets
  / 37 events. The "~719" brief claim is exactly right.
- Live config (ssh cat, matches Mac copy): `hard_min_edge_cents=5` (flat),
  `min_edge_cents=1`, `hedge_margin_cents=2`, expiry start−2h, half-Kelly,
  `vm_model_rebuild=true`. VM repo has **no .git** — code provenance
  unverifiable by git-log (gap).
- Maker-only verified in code (`post_only=True` hardcoded; taker fill ⇒ halt)
  and in data (all fills join resting orders; tape shows counterparty
  taker_side=yes where recorded).

## Dollar waterfall (settled = Kalshi settlement, fees from verified schedule)

| Component | $ |
|---|---|
| Locked-pair margin (3,258.5 pairs @ avg 1.66¢) | **+53.98** |
| Unhedged settled inventory (on $529 cost) | **−362.67** |
| Fees (maker fee = $0 for this series; verified from series endpoint + bot settlement recon; taker counterfactual $154.29) | **0.00** |
| **Realized** | **−308.69** |
| Open-inventory mark (last trade; $69.80 cost) | −4.67 |
| **Total incl. mark** | **−313.35** |

Identity check: restricted to the bot's own 40 settled deployments my FIFO
reconstruction gives **−247.80 vs deploy_index −247.78** (2¢ rounding) — no
hidden cash leak. The remainder is 3 events that settled after the bot's last
reconcile (NAVI-GM −50.47, DRG-JDG −28.23, LOUD-SEN +10.75).
Worst events: EDG-FPX −54.0, NAVI-GM −50.5, BBL-FUT −47.3 (the 2026-07-23
incident's role-flip suspect), TH-EF −45.0, AG-EDG −42.7, NOVA-TYLOO −39.7.

## Adverse selection (pre-registered definition)

Fill-conditional gap (frozen v6, contract-weighted, settled fills n=664):
realized NO-pay 42.5% vs predicted 47.0% ⇒ −4.55pts. Unconditional benchmark
(v6 on all 38 settled linked events, NO-on-favorite): +3.94pts.
**Adverse-selection number = −8.5pts ≈ −8.5¢/contract.**
Sharpest slice: **side-1 −15.8pts** (realized 27.2% vs predicted 43.0%, Wilson
CI 23.2–31.6%); hedges mirror at +13.1pts. Price bands: NO≤60¢ (favorite-fades)
−11.5 to −12.6pts; toxic time window is **6–24h before start (−10.4pts,
2,697 contracts)** — the "size late" zone — not the final 2h (−4.8pts).
Micro markouts are CLEAN: spread capture +0.57¢; maker P&L +0.35¢@5m,
+0.42¢@30m, +0.86¢@2h (start−5m −0.57¢ at 12.7% tape coverage). So the
selection is at match/side level (the market knew the resolution), not
price-level toxic flow. Vintage slices exonerate the 2026-07-23 snapshot
drift: drifted-vm fills calibrate +7.8pts (n=28, $421 of contracts).
Unconditional v6 self-calibration in-window: favorites won 57.9% vs predicted
61.8% (−3.9pts, n=38, CI includes 0) — mild model overconfidence, ~¼ of the
fill-conditional gap; the rest is selection through the too-loose filter.

## Top config gap (expected ROI)

| Knob | Live | Recommended (FINDINGS §4) | Evidence |
|---|---|---|---|
| **Side-1 edge floor** | flat 5¢ | **logit +0.5/+0.6** | Sim: +6.7% [−9.2,+22.8] vs +29.1% [+0.6,+61.0] = +22.4pts. On our settled side-1 fills: logit+0.6 keeps $133 at **+$123 (+92%)**, refuses $1,500 at **−$549 (−36.6%)** |
| Skip NO on model-p<45% | ABSENT (code-verified) | skip | Our pocket: 83 fills, $709, **−$106 (−14.9%)** (research said −5.7%) |
| Hedge margin | 2¢ | adequate | fees $0; 1.66¢/pair realized; not the leak |
| Expiry | start−2h | start−2h | implemented; not the worst window anyway |

## Noise verdict

CRN bootstrap (crn.json seed 20260728, PCG64, B=4000; settled events n=28)
under H0 "flat-5¢ sim edge is real (+6.7% of side-1 stake ⇒ +$109 expected)":
**p(cum ≤ −315.75) = 0/4000 (<0.00025). Real underperformance — not variance.**
(First run used fallback seed 780728 before agent:power wrote crn.json mid-run;
superseded, both agree.)

## Blame waterfall (per pre-registered rules)

1. **Config ~60–70%**: the flat 5¢ floor admitted the −$549 beyond-logit-cap
   tail and the −$106 sub-45% pocket that FINDINGS §4 had already rejected.
2. **Model ~25%**: v6's divergences from market resolved against it broadly
   (in-window favorite calibration −3.9pts; known elite-team under-rating).
3. **Execution micro / fees / snapshot-drift / noise: exonerated** ($0 fees;
   positive markouts; drift fills profitable; p<0.00025).

## Artifacts
`stats/autopsy_pnl.json`, `autopsy_fees.json`, `autopsy_fill_calib.json`,
`autopsy_markouts.json`, `autopsy_config_gap.json`, `autopsy_variance.json`,
`quote_density.json` (shipped early for agent:referee), `data/vctmm_live.db`,
`data/kalshi_markets_meta.json`, `data/vm_config_and_gitlog.txt`,
scripts `autopsy_step[123]*.py`, journal `logs/autopsy.log`.

## Gaps
- VM code provenance unverifiable (no .git on VM).
- start−5m markouts only 12.7% tape coverage; book_snaps too sparse (498 rows).
- Unconditional benchmark is n=38 events (wide CI); bot-side fv missing for 88
  early fills (pre-telemetry) — frozen-v6 repricing used there (parity MAD
  2.3pts on the validated vintage).
- Open inventory small ($70 cost) — marks from last trade, not mid.
