# Pre-registration — agent:corpus (Phase 1: corpus expansion)

Written 2026-07-28, BEFORE any VLR fetch. Brief: testing_lab/v8/briefs/corpus.md.

## Enumeration methods (VLR only, never memory)
1. **Archive sweep**: https://www.vlr.gg/events/?page=N (completed section),
   walked sequentially from page 1 until an entire page's events end before
   2021-01-01 (plus one confirmation page). Every event row captured:
   vlr_event_id, name, dates, prize, region flag, status.
2. **Year hub cross-check**: the VCT circuit hub/series pages for
   2021, 2022, 2023, 2024, 2025, 2026 (vlr.gg/vct-<year> and the event-series
   listings they link). Any official-circuit event that the archive sweep
   missed gets added to the candidate list from here.
3. **Targeted event search**: vlr.gg event search for name families the brief
   flags: "Esports World Cup", "Evolution Series", "Ludwig", "Tarik",
   "OFF SEASON", "Home Ground". Search results only add candidates; they never
   remove any.
The candidate list is the union of all three. Enumeration output is written to
testing_lab/v8/stats/corpus_diff.json before any match scraping starts.

## Inclusion criteria for "tier-1" (decided per event, recorded in corpus_diff.json)
A candidate event is ADDED to the ratings corpus (2023-2026) iff:
- **C1 — VCT franchised circuit**: an official VCT franchise-era event
  (Kickoff / Stage league / Masters / Champions / China splits, incl. LOCK//IN
  class). These should already be registered; any hole found is backfilled.
- **C2 — EWC class**: Esports World Cup main events, their regional
  qualifiers, and China Evolution Series acts that form the EWC-CN qualifying
  chain. Treatment mirrors the existing 2026 entries: ratings_only + vct_only,
  NOT International-tagged; the regional qualifiers of one year are merged
  into ONE multi-region entry (2026_ewc_qual precedent).
- **C3 — Off-season / one-off with franchised participation**: competitive
  (non-showmatch) bracket events where >= 4 franchised VCT orgs participated,
  so that vct_only filtering yields a meaningful number of both-sides-
  franchised series. Ludwig x Tarik-class invitationals qualify under C3 if
  they meet the >= 4-org bar. Events failing the bar are EXCLUDED with reason.
- Anything else (tier-2 Challengers/Ascension, Game Changers, showmatches,
  collegiate, watch parties) is EXCLUDED with reason.
Pre-franchising 2021-2022: VCT Champions / Masters / Challengers main events
and tier-1 equivalents go ONLY to testing_lab/v8/data/prefranchise/ (separate
registry.json, never ALL_EVENTS, never data/). Priority if volume is huge:
Champions > Masters > regional main events; deferrals reported with counts.

## What counts as verification passing
- **Mechanical, every backfilled series**: winner org + map score parsed from
  the SAME HTML my scraper fetched (cached to disk at scrape time) must equal
  the winner org + score row that scrapers/BuildMatchResults.py (independent
  re-fetch + parse) wrote into data/match_results.csv — for the series row and
  every per-map row. AND the match date (data-utc-ts, ET calendar day) must
  land inside [event start - 1 day, event end + 1 day] of the registry entry.
- **Sample re-fetch**: per backfilled event, 10 series (or all, if fewer)
  drawn WITHOUT replacement using seed 20260728 (crn.json does not exist yet
  at pre-registration time — the power agent has not written it; this seed is
  fixed here instead, and this deviation is disclosed in the report), each
  re-fetched live from VLR and re-parsed; winner + score must match
  match_results.csv. Report N/N per event.
- Any mismatch = verification FAILURE for that event; reported loudly, never
  silently dropped or "fixed" by hand-editing CSVs.

## Falsifier
If VLR's own listings show no 2025 EWC regional qualifiers (or no 2025
Esports World Cup Valorant event, or no 2025 China Evolution Series act), then
the brief's premise about the 2025 EWC chain is wrong, and I report exactly
that — I do not substitute a different event to fill the slot. Same logic for
any expected event: absence on VLR is reported as absence.

## Fetch discipline
All fetching by my own scripts goes through scrapers/enriched/vlr_client.fetch
(sequential, >= 0.75 s between match pages, >= 1.0 s between listing pages).
The standard builders (BuildMatchResults) use their built-in RefreshLiveData
_fetch stack, run only after `pgrep -f RefreshLiveData.py` is empty.
Match dates/times for new MatchIDs are parsed from my cached match HTML with
RefreshLiveData step-5 semantics (data-utc-ts -> ET day in match_dates.json,
_et_walltime_to_utc -> UTC in match_times.json) — zero extra VLR load.
Forbidden and untouched: BuildRatingTimeline, BuildMapRatings, RefreshLiveData,
VCTMM, anything in testing_lab/ outside v8/, rewriting existing event CSVs.
