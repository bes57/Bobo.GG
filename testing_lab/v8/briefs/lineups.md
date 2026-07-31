# agent:lineups — Phase 3 data: per-match lineups

## Scope (one question)
Which five players did each org field in every match of the corpus, and the
walk-forward lineup features derived from that — from LOCAL data only.

## Context
- Repo: /Users/benny_es1/PythonTest. Rules: testing_lab/v8/README.md.
- The raw signal already exists locally: data/maps/<event_id>.csv has one row
  per player per map (Player, Org, ProfileURL, MatchID, MapNum, MapName...).
  testing_lab/engine.py load_match_lineups() shows the canonical extraction
  ((org, match_id) -> frozenset(ProfileURL)). data/enriched/vlr/<mid>.json
  adds per-map player lists with player_id for most matches. Dates:
  data/match_dates.json (MatchID -> YYYY-MM-DD). Event registry:
  MoreTestingMaybeFiles.ALL_EVENTS (import it; agent:corpus may be editing it
  concurrently — import ONCE at start, record which events you saw, and
  re-check at the end: if new events appeared, top up your table for them
  before finishing).
- ProfileURL is the player key (stable across name changes). player_id from
  enriched JSONs where joinable; missing is fine, record coverage.
- NO VLR ACCESS. agent:corpus owns the host. Any gap you can't fill locally
  goes in the coverage report as a named gap, never a fetch.

## Pre-register first
testing_lab/v8/preregister.lineups.md: feature definitions (exact), the
walk-forward rule (modal five computed from strictly-earlier matches only),
and the coverage bar you consider complete (e.g. ≥99% of engine games have a
lineup for both orgs; report the actual number).

## Work
1. **Lineup table** → testing_lab/v8/data/lineups.csv, one row per
   (match_id, org): date, event_id, org, players (sorted ProfileURLs,
   ';'-joined), player_ids where known, n_players, source. n_players > 5
   means a mid-series substitution — flag it, don't collapse it.
2. **Features** → testing_lab/v8/data/lineup_features.csv, one row per
   (match_id, org), ALL walk-forward:
   - modal5_30d: the org's modal five over matches in the preceding 30 days
     (exclude match day); overlap_modal = |lineup ∩ modal5|/5 (NaN + flag if
     no prior matches in window)
   - overlap_prev: overlap with the org's immediately previous match lineup
   - stand_in_flag: overlap_modal < 1 with modal defined; n_standins
   - games_since_change / matches_since_change: count since the fielded five
     last differed
   - first_after_break_45d: no org match in the preceding 45 days
   - offseason_absence_flag: stand_in_flag AND the event is ratings_only
     (EWC-class) — the "attendance anomaly" marker for Phase 2
3. **Coverage audit** → testing_lab/v8/stats/lineups_coverage.json: engine
   game list (via testing_lab/engine.py data loaders) vs your table — % of
   (match, org) sides covered, per event; named list of every gap; player_id
   join rate; count of >5-player matches. Also basic sanity distributions
   (share of matches with stand-ins by event class — EWC-class should be
   visibly elevated if the operator's seriousness story is real; report the
   number either way, it feeds Phase 2's §3 chart).
4. **Modal-five reference table** → v8/data/modal5_by_org_date.csv if cheap
   (org, date, modal five) — Phase 2 wants "lineup delta vs VCT modal five"
   for EWC participants; precompute per (org, event) at event start.

## Outputs (yours alone)
- testing_lab/v8/data/lineups.csv, lineup_features.csv, modal5_by_org_date.csv
- testing_lab/v8/stats/lineups_coverage.json
- testing_lab/v8/preregister.lineups.md, logs/lineups.log

## Forbidden
Network. Writing outside the paths above. Modifying data/ or any shared file.

## Done criteria
Coverage number stated against the engine game list; features defined
walk-forward with the definitions in the preregister file matching the code;
EWC stand-in rate reported.

## Return format
≤500 words: coverage %, gap list size, stand-in rate by event class (the
Phase 2 teaser number), artifact paths, anything pending corpus top-up.
