# VLR Match Enrichment (Tier 1)

A **self-contained** pipeline that pulls the deeper per-map data VLR.gg shows on
individual match pages but does **not** expose in the event stat tables the main
site already scrapes. It backs the Match Data Explorer (`/match-data/`).

## Isolation guarantees
- **Reads** only `data/match_results.csv` and `data/match_dates.json` (read-only).
- **Writes** only under `data/enriched/`.
- **Not imported** by `BobosHome` or any site module — the only automated entry
  point is `scrapers/EnrichNewMatches.py` (below), spawned as its own detached
  process. A failure there cannot break a refresh or the live website.

## Staying current (automated)
`scrapers/EnrichNewMatches.py` runs an incremental pass of both collectors.
`RefreshLiveData` spawns it, detached, right after it finishes — so the
enrichment tracks the same match list as everything else on the site without
adding its 3-4 extra VLR fetches per match to the refresh the user is watching.
`/match-data/` caches key off the enriched file count, so new files are picked up
with no cache-busting.

It is time-budgeted (`--budget`, default 420 s, split between the two passes) and
`flock`'d, so a backlog drains over several refreshes instead of pinning one
process, and overlapping refreshes can't stack passes on VLR. Matches finished
within the last few days are **never** tombstoned/blanked and are re-checked if
only part of the series' stats have been published — VLR posts round, economy and
agent data some time after a match ends. Last-pass counts land in
`data/enriched/_last_run.json`; the log is `data/enriched/_enrich.log`.

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
retried. Keeping it current is automatic (see above); these commands are for
backfills and one-offs. Anti-bot handling mirrors the main site (curl_cffi Chrome
impersonation → cloudscraper fallback, polite delays, bounded retries).

**Parsing note:** VLR migrated match pages from `table.wf-table-inset.mod-overview`
(`td.mod-player`, `td.mod-agents`) to a `div.ovw-table` grid (`.ovw-cell.mod-player`,
with the agent icon now inside the player cell as `.ovw-agents img`) around
mid-2026. `parse.py` and `agents.py` match both shapes. If team orgs, map scores
or comps ever come back empty for new matches, that selector pair is the first
thing to check.

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
