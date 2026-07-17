"""Isolated VLR match-page enrichment subsystem (Tier 1).

Self-contained. Reads existing data/ files (match_results.csv, match_dates.json)
read-only, and writes ONLY under data/enriched/. Nothing here is imported by the
Flask app (BobosHome) — it is a standalone data pipeline for new projects.

Produces per-map data that VLR does NOT expose in the event stat tables the main
site already scrapes:
  - round-by-round outcomes: winning team, side (attack/defense), win condition
    (elimination / defuse / detonate / time)
  - economy: pistols won, eco / semi-eco / semi-buy / full-buy counts + wins
  - performance: multikills (2K-5K), clutches / last-alive (1v1..1v5), plants,
    defuses, econ rating, and a player-vs-player kill matrix
"""
