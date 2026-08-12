# Kalshi VALORANT dataset summary

Generated: 2026-08-12 20:40 UTC

- Total markets in raw cache: **1278**
- Total events (matches): **639** (duplicate event tickers: 0)
- Tier-1 events (both teams resolve to BenPom orgs): **257**
- Excluded events (GC / tier-2 / college): **382**
- Date range (close_time UTC): **2026-05-16 02:09 .. 2026-08-12 17:39**

## Volume (per event, contracts, all events)
- median: 78,713
- p90: 359,380
- max: 1,417,040

## Price coverage (tier-1 events)
- with close (t-5m) price: 257
- with t2h price: 257
- with BOTH close and t2h: 257
- with VWAP: 257

## Suspicious settlements (winner's close prob < 0.5)
- KXVALORANTGAME-26MAY171200THVIT (TH vs VIT): winner=TH close_prob=0.485 vol=540497.79

## Anomalies
- event KXVALORANTGAME-26AUG070400TECAG: scalar settlement (cancelled/forfeit) ({'KXVALORANTGAME-26AUG070400TECAG-AG': 'scalar', 'KXVALORANTGAME-26AUG070400TECAG-TEC': 'scalar'})
- event KXVALORANTGAME-26AUG070600TEEDG: scalar settlement (cancelled/forfeit) ({'KXVALORANTGAME-26AUG070600TEEDG-EDG': 'scalar', 'KXVALORANTGAME-26AUG070600TEEDG-TE': 'scalar'})
- event KXVALORANTGAME-26AUG101900AOLC: scalar settlement (cancelled/forfeit) ({'KXVALORANTGAME-26AUG101900AOLC-AO': 'scalar', 'KXVALORANTGAME-26AUG101900AOLC-LC': 'scalar'})
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
