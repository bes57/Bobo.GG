# Riot Deep Stats (official telemetry)

Derives the round-by-round, positional stats that **do not exist on VLR** —
true saves, death locations, last-alive/clutch situations, and round-win impact —
from Riot's official VCT game telemetry (the public `vcthackathon-data` S3 bucket).

## Scope
- **Franchised international, 2023–2024 only.** 2022 (pre-franchising) is excluded
  by design; no free source covers 2025+ at this depth (that needs GRID, paid).

## Isolation
- Streams raw 15 MB game files from S3, parses, and **discards them** (only
  compact per-game output is kept).
- Reads/writes only under `data/riot/`. Not imported by the app except through
  `MatchDataExplorer` (the Match Data dashboard's Deep Stats tab).

## What it derives (per player)
- **Saving** — a true save = survived a round your team lost. `save rate = saves ÷ lost rounds`.
- **Last-alive / clutch** — how often a player was the sole survivor, the odds
  faced (1vN), and how often they converted (attempts *and* wins — VLR only has wins).
- **Death locations** — every death's (x, y), plotted on the minimap as a heatmap.
- **Round impact** — first-kill→round-win conversion, and round-win% when alive at round end.
- Plus first bloods, plants, defuses.

## Pipeline
```bash
python -m scrapers.riot.pipeline           # parse all 2023-2024 international games
python -m scrapers.riot.pipeline --status  # coverage
python -m scrapers.riot.pipeline --limit 50
```
- `esports-data` (players/teams/tournaments/mapping_data_v2) is cached to
  `data/riot/esports/` on first run.
- Each game → `data/riot/games/<platformGameId>.json` (map, tournament, per-player
  aggregates, death points). **Resumable** — re-running skips parsed games.
- Map minimap images + coordinate transforms are in `data/riot/map_coords.json`
  (from valorant-api.com), used by the death-map heatmap.

## Files
- `parse_game.py` — one game's event stream → derived per-player stats + death points.
- `pipeline.py` — S3 streaming driver.
- Dashboard aggregation + routes (`/match-data/api/deep`, `/api/deep/deaths`,
  `/api/deep/mapmeta`) live in `MatchDataExplorer.py`.
