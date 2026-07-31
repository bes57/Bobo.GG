# Pre-registration — agent:lineups (written 2026-07-28, BEFORE building)

Scope: per-match fielded lineups + walk-forward lineup features, LOCAL data only.
Sources: data/maps/<event_id>.csv (canonical, mirrors engine.load_match_lineups),
data/match_dates.json, data/enriched/vlr/<mid>.json (validation + player_id),
MoreTestingMaybeFiles.ALL_EVENTS (imported once at start; re-checked at end),
BuildMapRatings.EVENT_DATES, engine game list via testing_lab/engine.load_games_real_dates().

## Keys, dates, ordering (fixed before any computation)

- **Player key**: `ProfileURL` verbatim from the maps CSV. `player_id` := the numeric
  segment of ProfileURL (`/player/(\d+)/`), which is VLR's player id (verified:
  koldamenta URL id 339 == enriched player_id "339"). The enriched JSONs are used as
  an independent *validation* join — (casefold(player name), org) within the match —
  and their join rate + id-mismatch count are reported in the coverage JSON. This is
  declared now because ProfileURL definitionally embeds the id; pretending the id is
  "unknown" without the enriched file would be a fake coverage number.
- **Fielded lineup** L(org, match) := set of ProfileURLs with ≥1 player-map row for
  that (Org, MatchID) — identical grouping to engine.load_match_lineups(). Union over
  maps; n_players > 5 ⇒ mid-series substitution, flagged, never collapsed.
- **Match date** D (day granularity), resolution order, with `date_source` recorded:
  1. `match_dates.json[str(mid)]` (`match_dates`)
  2. engine-interpolated date from load_games_real_dates() (`engine_interp`)
  3. `EVENT_DATES[event_id][0]` (`event_window`) — last resort, flagged.
- **Org sequence**: org's matches sorted ascending by (D, match_id).
- **Walk-forward rule (binding)**: every historical aggregate for a match at date D
  uses only the org's matches with **date strictly < D**. Same-day matches are never
  history for each other (day-granularity, matches v8 rule 1). The current match's
  own lineup is the row's *subject*, never part of its own history.
- **Event class**: `ewc` if the ALL_EVENTS entry has `ratings_only: True`, else `vct`.

## Artifact 1 — testing_lab/v8/data/lineups.csv

One row per (match_id, org): `match_id, org, date, date_source, event_id,
event_class, players` (';'-joined sorted ProfileURLs), `player_ids` (';'-joined,
aligned to players order), `n_players, n_maps` (distinct MapNum), `multi_lineup_flag`
(n_players > 5), `short_lineup_flag` (n_players < 5), `source` (= maps_csv).

## Artifact 2 — testing_lab/v8/data/lineup_features.csv

One row per (match_id, org). All history strictly earlier per the rule above.
W30 := org matches with 1 ≤ (D − date)days ≤ 30.

- **modal5_30d**: per player p over W30, c(p) = # matches whose L contains p;
  rank by (−c(p), −most-recent-appearance-date(p), ProfileURL asc); take top 5
  (fewer if <5 distinct players appeared; then `modal5_short=1`). Deterministic.
  If |W30| = 0: modal undefined → modal5_30d empty, `no_prior_30d=1`.
- **overlap_modal** = |L ∩ modal5_30d| / 5; NaN if modal undefined.
- **n_modal_matches** = |W30|.
- **overlap_prev** = min(1.0, |L ∩ L_prev| / 5); L_prev = lineup of the latest match
  with date < D (tie on date → larger match_id). NaN if no prior match.
- **stand_in_flag** = 1 if modal defined and overlap_modal < 1.0; 0 if modal defined
  and overlap_modal = 1.0; NaN if modal undefined. **n_standins** = |L \ modal5_30d|
  (NaN if modal undefined).
- **matches_since_change** = # consecutive matches at the tail of the strictly-earlier
  org sequence whose L equals this match's L (walk back while equal; 0 if the
  immediately previous lineup differs or no prior match).
  **games_since_change** = sum of n_maps over those counted matches.
- **first_after_break_45d** = 1 iff org has ≥1 prior match ever AND none with
  1 ≤ (D − date)days ≤ 45. `org_debut` = 1 iff no prior match ever (then
  first_after_break_45d = 0).
- **offseason_absence_flag** = 1 iff stand_in_flag = 1 AND event_class = ewc
  (brief-exact definition).
- **Robust supplement (declared now, because EWC follows ≥30 quiet days and the 30d
  modal is expected to be undefined there — the brief-exact flag alone would NaN out
  exactly where Phase 2 looks):**
  - **modal5_vct** := modal five (same count+tie-break rule) over the org's last ≤10
    matches in `vct`-class events with date < D; undefined if none; basis size
    `n_vct_basis`.
  - **overlap_vct_modal** = |L ∩ modal5_vct| / 5; **stand_in_vs_vct_flag** analogous
    to stand_in_flag; **offseason_absence_vs_vct** = stand_in_vs_vct_flag=1 AND ewc.
  Both the brief-exact and the supplement numbers are reported side by side; neither
  is tuned on anything.

## Artifact 3 — testing_lab/v8/data/modal5_by_org_date.csv

One row per (event_id, org) for every org appearing in that event in lineups.csv,
computed at S = EVENT_DATES[event_id].start using only matches with date < S:
`event_id, org, event_start, modal5_30d_pre` (30d window ending S−1), `n_30d,
modal5_last10_vct, n_vct_basis`. Same modal + tie-break rules.

## Artifact 4 — testing_lab/v8/stats/lineups_coverage.json

Audit against the engine's own game list (load_games_real_dates()): unique
(match_id, org) sides from engine games vs lineups.csv. Report: total sides, covered,
pct (overall + per event), engine game/match counts per event, named gap list
[{match_id, org, event_id, reason}], n_players>5 and <5 counts + lists, enriched
file-presence rate and player-level enriched join rate + id mismatches, stand-in
rates by event class (brief-exact and vs-VCT-modal, plus no_prior_30d share by
class), registry snapshot start/end + any top-up.

## Coverage bar (declared)

Complete = **≥99.0%** of engine (match_id, org) sides present in lineups.csv with
n_players ≥ 5. Actual number reported regardless; every miss named.

## Not done here

No network. No writes outside the six declared paths. No tuning of any constant
against anything — this is a data/feature deliverable; no holdout contact.
