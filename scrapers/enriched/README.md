# VLR Match Enrichment (Tier 1)

A **self-contained** pipeline that pulls the deeper per-map data VLR.gg shows on
individual match pages but does **not** expose in the event stat tables the main
site already scrapes. Built as a foundation for **new projects** — it is not
wired into the Flask app and does not modify any existing file.

## Isolation guarantees
- **Reads** only `data/match_results.csv` and `data/match_dates.json` (read-only).
- **Writes** only under `data/enriched/`.
- **Not imported** by `BobosHome` or any site module. Running it cannot affect
  the live website.

## What it collects (per map)
- **Round-by-round outcomes** — winning team, side (`attack`/`defense`), and win
  condition (`elimination` / `defuse` / `detonate` / `time`), plus running score.
- **Economy** — pistols won and eco / semi-eco / semi-buy / full-buy counts with
  wins, per team.
- **Performance** — per player: multikills (2K–5K), **clutches / last-alive**
  (1v1–1v5), econ rating, plants, defuses.
- **Kill matrix** — player-vs-player kills and deaths.

Everything keys on the existing **`MatchID`** (VLR match/series id) and
**`map_num`** (VLR game id == their `MapNum`), and players carry `player_id`
(VLR player id) — so all of it joins directly to the existing dataset.

## Running it
```bash
# enrich every match not done yet (dynamic: only processes new matches)
python -m scrapers.enriched.enrich

python -m scrapers.enriched.enrich --status        # counts, no fetching
python -m scrapers.enriched.enrich --limit 50      # cap this run
python -m scrapers.enriched.enrich --match 314622  # specific match(es)
python -m scrapers.enriched.enrich --refresh       # re-enrich existing
python -m scrapers.enriched.enrich --rebuild-csv   # rebuild flat CSVs only
```

It is **incremental and resumable**: matches already enriched are skipped, and
matches with no round data (forfeits / very old) are tombstoned so they aren't
retried. To keep it current, run it right after the site's own scrape step —
it will pick up only the newly-added matches. Anti-bot handling mirrors the main
site (curl_cffi Chrome impersonation → cloudscraper fallback, polite delays,
bounded retries).

## Outputs
### Source of truth — one JSON per match
`data/enriched/vlr/<MatchID>.json`
```jsonc
{
  "match_id": "314622",
  "event": "Champions Tour 2024: Americas Stage 1",
  "date": "2024-03-10",
  "team1": {"org": "C9", "name": "Cloud9"},
  "team2": {"org": "LEV", "name": "LEVIATÁN"},
  "n_maps": 3,
  "maps": [
    {
      "map_num": "163351",
      "map_name": "IceboxPICK",          // raw form, matches existing MapName
      "team1_org": "C9", "team2_org": "LEV",
      "score": {"LEV": 13, "C9": 7},
      "rounds": [
        {"round": 1, "winner_org": "LEV", "winner_side": "attack",
         "win_condition": "elimination", "raw_icon": "elim", "score_after": "0-1"}
      ],
      "economy": {
        "LEV": {"pistol_won": 1,
                "eco":      {"n": 3,  "won": 1},
                "semi_eco": {"n": 1,  "won": 0},
                "semi_buy": {"n": 1,  "won": 1},
                "full_buy": {"n": 15, "won": 11}}
      },
      "players": [
        {"player": "aspas", "org": "LEV", "player_id": "8480",
         "multikills": {"2k": 8, "3k": 1, "4k": 0, "5k": 1},
         "clutches": {"1v1": 0, "1v2": 0, "1v3": 0, "1v4": 0, "1v5": 0},
         "econ": 116, "plants": 0, "defuses": 0}
      ],
      "kill_matrix": [
        {"killer": "Xeppaa", "victim": "kiNgg", "kills": 6, "deaths": 2}
      ]
    }
  ]
}
```
`data/enriched/vlr/_index.json` — MatchID → summary (date, event, teams, n_maps).
`data/enriched/vlr/_tombstones.json` — MatchIDs skipped (no round data).

### Analyst-friendly flat CSVs (regenerated from the JSONs)
- `data/enriched/round_outcomes.csv` — one row per round
  (`match_id, map_num, map_name, round, winner_org, winner_side, win_condition, score_after, date, event`)
- `data/enriched/map_economy.csv` — one row per team per map
- `data/enriched/player_map_advanced.csv` — one row per player per map
- `data/enriched/kill_matrix.csv` — one row per killer→victim pair per map

## Validation
Derived per-map scores were cross-checked against the existing
`match_results.csv` (winner + score) with **zero mismatches**.

## Scope note
This is **Tier 1** (VLR match pages), which is live across all seasons incl. the
current one. The deeper **positional** layer ("where players die" / X-Y coords)
was intentionally *not* built here: rib.gg (the planned source) shut down, and
the only free alternative — Riot's official VCT hackathon S3 data
(`vcthackathon-data`) — is historical-only (2022–2024). It can be added later as
a separate module if a positional layer is wanted.
