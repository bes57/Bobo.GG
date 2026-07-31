# Data Patch Summary (TESTING ONLY — not production data)

Built 2026-07-22 by scraping VLR.gg (curl_cffi, chrome impersonation, ~1.5s between requests).
Raw HTML cached under `html/` so reruns are free. Pipeline: `scrape_patch.py` (stage 1 =
target-to-URL matching writes `_assign.csv`; `stage2` = match-page scrape writes the CSVs).

## Results

| Metric | Count |
|---|---|
| Targets in `data_patch_targets.csv` | 82 |
| Matched to a VLR match URL | **82 / 82** |
| Successfully scraped | **82 / 82** |
| Missed / not on VLR | 0 |
| Map rows (`patch_games.csv`) | 223 |
| Lineup rows (`patch_lineups.csv`) | 820 (exactly 5 per side, 10 per match) |
| Series rows (`patch_series.csv`) | 82 |

## Winner cross-check

Every scraped series winner was compared against `winner_org` in the target list at two
levels (event-list winner marker, and maps-won recomputation from per-map scores):
**82/82 agree, zero mismatches.** Scraped series scores also match the event-list series
scores for all 82, and per-map scorelines are all sane (13-x or overtime with a 2-round
margin; no forfeits/short maps).

## What these matches actually are (event identification)

The task brief guessed "Stage 1 playoffs", but the four regional Stage 1 playoffs ended
May 10-24 on VLR. The late-May block is actually:

- **Esports World Cup 2026 regional qualifiers** — Americas (event 2953), EMEA (2954),
  Pacific (2955), China (2956). Covers most of the May 16 - Jun 1 targets, including
  Challengers guests (FULL SENSE, VARREL, Team Secret, Nova, TEC, etc.).
- **China Evolution Series 2026 Act 2** (event 2988, May 21-31) — the 12 remaining China
  targets (AG/DRG, FPX/WOL, JDG/XLG, EDG/TYL, AG/FPX, TE/TEC, JDG/NOVA, BLG/TYL, AG/TEC,
  NOVA/TYL, AG/TYL, NOVA/TEC). VCT CN teams entered at the Ro16 alongside tier-2 teams.
- A few genuine Stage 1 playoff matches that were still missing locally
  (e.g. EG-SEN May 21 Americas, GEN-TS May 18 Pacific).
- **Esports World Cup 2026** proper (event 2952, Jul 2-12) for the `2026_ewc` block —
  all 22 July targets, all bo3/bo5 (no bo1s).

Event tags follow the brief: `2026_stage1_late` for the May block (60 series),
`2026_ewc` for July (22 series).

## Name-resolution notes

`vctmm.benpom.teams.resolve_team_name` handled everything except two hardcoded overrides
(per the brief): **'DRX' -> KRX** and **'JD Gaming' -> JDG**. VLR quirks that resolved fine
but are worth knowing: China events list XLG as **"Xi Lai Gaming"**, JDG as
**"JDG Esports"**, and TEC as **"Titan Esports Club"**. Unresolvable names encountered
(tier-2 China Evolution Series teams: Weibo Gaming, Any Questions Gaming, KeepBest
Gaming, Unsettled Resentment) never matched a target pair, so they were ignored.

## Scraping quirks hit along the way

- VLR event match-list pages **ignore the `page=` parameter** now — the default
  `series_id=all` view truncates at ~42-48 matches. Fix: fetch each `series_id=<N>`
  sub-list (Group Stage / Playoffs / Stage 1 / Stage 2) and union them.
- Match pages use the post-July-2026 `div.ovw-*` grid; lineups were taken from the
  `data-game-id="all"` overview (two `ovw-table` blocks, in match-header team order,
  verified against the per-player `ovw-player-tag`).
- `vm-stats-game` class must be matched exactly — a loose prefix regex also catches the
  `vm-stats-gamesnav` map-switcher buttons (which duplicate `data-game-id`s).
- Map names carried a nested `PICK` span; stripped. All 223 map names are canonical
  (pool seen: Breeze, Haven, Lotus, Ascent, Split, Pearl, Fracture, Sunset).

## Format distribution

bo3 x68, bo5 x14 (series scores: 2-0 x35, 2-1 x33, 3-0 x7, 3-2 x5, 3-1 x2). No bo1s.
Dates in all CSVs are the **target list's `date_utc` calendar date** (per the brief), which
can be one day after VLR's displayed (US/Eastern) date for late-night UTC matches.
