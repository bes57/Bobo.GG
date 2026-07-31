# Kalshi VALORANT dataset summary

Generated: 2026-07-22 05:46 UTC

- Total markets in raw cache: **906**
- Total events (matches): **453** (duplicate event tickers: 0)
- Tier-1 events (both teams resolve to BenPom orgs): **170**
- Excluded events (GC / tier-2 / college): **283**
- Date range (close_time UTC): **2026-05-16 02:09 .. 2026-07-21 20:59**

## Volume (per event, contracts, all events)
- median: 71,148
- p90: 359,622
- max: 1,417,040

## Price coverage (tier-1 events)
- with close (t-5m) price: 170
- with t2h price: 170
- with BOTH close and t2h: 170
- with VWAP: 170

## Suspicious settlements (winner's close prob < 0.5)
- KXVALORANTGAME-26MAY171200THVIT (TH vs VIT): winner=TH close_prob=0.485 vol=540497.79

## Anomalies
- event KXVALORANTGAME-26JUL030700AGTH: scalar settlement (cancelled/forfeit) ({'KXVALORANTGAME-26JUL030700AGTH-AG': 'scalar', 'KXVALORANTGAME-26JUL030700AGTH-TH': 'scalar'})
- event KXVALORANTGAME-26JUL030700GEMIBR: scalar settlement (cancelled/forfeit) ({'KXVALORANTGAME-26JUL030700GEMIBR-GE': 'scalar', 'KXVALORANTGAME-26JUL030700GEMIBR-MIBR': 'scalar'})
- event KXVALORANTGAME-26JUL070400YIJUNK: scalar settlement (cancelled/forfeit) ({'KXVALORANTGAME-26JUL070400YIJUNK-UNK': 'scalar', 'KXVALORANTGAME-26JUL070400YIJUNK-YIJ': 'scalar'})
- event KXVALORANTGAME-26JUL070700NSPFCY: scalar settlement (cancelled/forfeit) ({'KXVALORANTGAME-26JUL070700NSPFCY-FCY': 'scalar', 'KXVALORANTGAME-26JUL070700NSPFCY-NSP': 'scalar'})

## Notes
- `status=finalized` is rejected by the API (400 invalid status filter); `status=settled`
  returns everything, with the market `status` field reading "finalized".
- Kalshi alias overlay applied (verified via rules_primary competition text):
  "DRX" -> KRX (Kiwoom DRX, VCT Pacific), "JD Gaming" -> JDG (VCT CN).
- All API prices are dollar-denominated strings (e.g. "0.9900"); no cents scaling needed.
- `price` object in candles is empty when a minute had no trades; bid/ask always present.
- Close prob = last minute-candle at/before close_time-5min; bid/ask midpoint preferred
  (spread<=0.30 and non-empty book), else last trade close.
- prob_a combines both markets: volume-weighted avg of P(yes_A) and 1-P(yes_B).
- Markets close only after a winner is declared, so close prices reflect in-match trading;
  t2h is the better pre/early-match snapshot.

## Validation (2026-07-22 run)
- 5 randomly sampled tier-1 events hand-checked: winner's market settled `yes`, winner's
  close (t-5m) probability > 0.5 in all 5.
- Full population: 167/168 tier-1 events with a winner have winner close-prob > 0.5.
  The single disagreement (KXVALORANTGAME-26MAY171200THVIT, TH 0.485 at t-5m) is a real
  photo-finish: TH traded 0.47-0.49 until t-2m, won the final round, snapped to 0.99.
- Independent cross-check vs BenPom Stage 2 series results (org pair + date +/-1 day):
  49 events matched, 49/49 winners agree.
- No duplicate event tickers (453 unique).
- Tier-1 volume: median 134,940 contracts/event, p90 540,498, min 265; only 1 tier-1
  event under 1,000 contracts.
