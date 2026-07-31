# frame_expanded — canonical Wave 2 evaluation frame

Emitted by agent:power, 2026-07-28 21:51:31. The raw-corpus series
frame (2023-2026, train+holdout) that every Wave 2 agent evaluates against.
Production rating files (`data/rating_timeline*.json`) were NOT used and NOT rebuilt;
this frame carries no ratings.

## Row counts
- total: 2058
- train (date <= 2024-12-31): 841
- holdout (date > 2024-12-31): 1217
- raw series excluded by harness rules: 10 ({'org not in ORG_REGIONS': 7, 'org count != 2': 3}) — all
  pre-existing junk in pre-corpus events; 0 of the +335 corpus additions fail.

## Sources
- data/match_results.csv — sha256 4b1c7f62708e8c06e0e70443604a6f34da2dbc4dc1e80309e88dee1e3b0e41a5 · mtime 2026-07-28 19:05:21 · 330429 bytes
- data/match_dates.json — sha256 d9241c88a61512bb6a6616896e5b9d225cc4f06d5a74808ce5ec4b8e9c5fd467 · mtime 2026-07-28 19:05:38 · 53770 bytes
- data/maps/*.csv (56 files) — manifest sha256 27a1c716de5996c68d48be0a90f2288cfa07ec4a2447508a57c02942087ccbdb
  (sha256 over 'name:sha256' lines, name-sorted)
- testing_lab/harness.py (rule source, incl. the NEW exact-shape _is_intl_event) —
  sha256 db300d1c1e9d334decca4b5b8e974831cc333565fe3392ce716227f8a44bce8f · mtime 2026-07-28 21:48:33 · 10471 bytes

## Harness rules applied (mirrored exactly)
1. Per-map winners from match_results.csv EXCLUDING the MapNum=="all" aggregate row
   (one per match; it carries the series score and double-counts the winner if kept).
2. Org pair per match from data/maps/<event>.csv player rows; junk-org filter: both
   orgs must be in vctmm.benpom.teams.ORG_REGIONS.
3. Series score = per-org map-win counts; keep only w_maps > l_maps and
   w_maps in (1,2,3) (drops forfeits/incomplete/odd series).
4. fmt = bo1/bo3/bo5 from w_maps; bo5 + stage 'grand_final' => bo5_gf.
   stage = harness._stage(MatchName). n_maps_played = numeric map rows.
5. intl = harness._is_intl_event (EXACT-SHAPE version, 2026-07-28): YYYY_masters_*,
   YYYY_champions, YYYY_lock_in only. Verified intl=False for 2024_shanghai_masters,
   2025_shanghai_masters, 2025_super_champions_cup, 2023_china_champions_qual.
6. Sort (date, match_id); de-dup match_id keep-first. Columns: match_id, date, event_id, year, winner, loser, w_maps, l_maps, fmt, stage, match_name, reg_w, reg_l, intl, n_maps_played.
   `intl` serialized as True/False.

## Validation
- All 1695 frozen npz-era frame rows present; column-level agreement on
  date/event_id/winner/loser/w_maps/l_maps/fmt/stage: 0 mismatches.
  n_maps_played differs on 0 rows (timeline counts vs raw numeric map rows —
  remakes/replays; informational column only). intl differs on 0 rows
  (old frame predates the exact-shape fix).
- Holdout n = 1217 = 1007 frozen-npz + 28 organic post-npz + 182 corpus additions.

## Integrity
- series.csv sha256: ff772d417b714844aa1b11426d8c04917e1b2848c9c8df811cd9988694d55142
- Registered in testing_lab/v8/crn.json under "frame_expanded" — verify before use.
