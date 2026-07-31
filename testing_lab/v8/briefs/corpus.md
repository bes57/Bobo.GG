# agent:corpus — Phase 1: corpus expansion

## Scope (one question)
What tier-1 Valorant results is the corpus missing, and backfill them —
EWC 2025 chain first, everything else recoverable second, pre-franchising
2021-22 into a SEPARATE corpus third.

## Context
- Repo: /Users/benny_es1/PythonTest. Rules: testing_lab/v8/README.md.
- Registry: MoreTestingMaybeFiles.py ALL_EVENTS. You are its SOLE writer this
  wave. Current EWC-class 2026 entries (your template — ratings_only +
  vct_only, NOT International-tagged): 2026_ewc_qual (ONE merged multi-region
  entry — the 2026 regional qualifiers were consolidated 2026-07-28; mirror
  that shape for 2025), 2026_china_evo_2, 2026_ewc.
- Scrape stack: scrapers/ScrapeMatchData.py scrape_event() — its match-page
  parsing delegates to RefreshLiveData._parse_match_html (current VLR div
  markup) with legacy-table fallback, and vct_only filtering already works
  (keeps matches where both orgs are in BuildMapRatings.TEAM_REGIONS, TYLOO
  allowed). Event stats pages (vlr.gg/event/stats/<id>/...) list the match
  URLs. Fetch layer: scrapers/enriched/vlr_client.fetch (curl_cffi →
  cloudscraper, retries). You OWN VLR this wave — no other agent scrapes.
  Keep sequential fetches with the existing polite delays.
- After maps/series CSVs land: run scrapers/BuildMatchResults.py then
  scrapers/ScrapeMatchDates.py semantics for NEW MatchIDs only (see how
  RefreshLiveData step 5 does dates incrementally — match_dates.json /
  match_times.json). BEFORE running builders, check no refresh is active
  (pgrep -f RefreshLiveData.py; wait if found).
- Do NOT run BuildRatingTimeline or BuildMapRatings. Rating rebuilds happen
  after the operator checkpoint. NOTE in your summary: the site's on-page-load
  refresh will rebuild ratings automatically once your registry entries +
  CSVs exist — the operator gets told this at checkpoint.

## Pre-register first
testing_lab/v8/preregister.corpus.md: enumeration method(s), inclusion
criteria for "tier-1", what counts as verification passing, and the falsifier
("if VLR shows no 2025 EWC regional qualifiers, the brief's premise about
them is wrong and I report that, not a substitute event").

## Work
1. **Enumerate from source.** VLR only, never memory: the /events archive
   (completed, by pages), year hub pages, and event search. Build the full
   candidate list of tier-1-relevant events 2021–2026: every VCT-franchised
   event (already registered ones included, as the diff baseline), EWC-class,
   off-season/one-off events with franchised-org participation (Ludwig x
   Tarik invitational-class events count as candidates; decide inclusion by
   your pre-registered criteria and record the decision per event).
2. **Diff vs ALL_EVENTS** → testing_lab/v8/stats/corpus_diff.json: per season,
   every candidate event: vlr_event_id, name, dates, in_registry (bool),
   decision add/exclude + reason, match count if added.
3. **Backfill the EWC 2025 chain** (expected but verify: Esports World Cup
   2025 Riyadh ~Jul 8-13 2025; 2025 regional qualifiers all regions; China
   Evolution Series Act 2 2025). Registry entries: ids 2025_ewc,
   2025_ewc_qual (merged multi-region), 2025_china_evo_2 — year 2025, real
   start/end from VLR, ratings_only: True, vct_only: True. Scrape maps/series
   CSVs, then the other recoverable holes your diff surfaced (same
   treatment; new ids follow existing naming).
4. **Verification.** For EVERY backfilled series: winner + score in the built
   match_results.csv must match the VLR match page you scraped (your parser
   already read it — assert consistency mechanically), and dates must land
   inside the event window. Additionally hand-verify a random 10-series
   sample per event against the live match pages (re-fetch) and report N/N.
5. **Pre-franchising 2021-2022** → SEPARATE corpus. Do NOT touch ALL_EVENTS
   or data/ for these. Write testing_lab/v8/data/prefranchise/registry.json
   (same entry shape) + maps_<id>.csv / series_<id>.csv there. VCT 2021-2022
   Challengers/Masters/Champions + tier-1 equivalents; record the map-pool /
   patch-regime caveat in the registry file. If volume is huge, prioritize:
   Champions > Masters > regional main events, and report exactly what was
   deferred (loudly, with counts).
6. **Value-of-data prep**: testing_lab/v8/stats/corpus_blocks.json listing
   each backfilled block (event ids, n_matches, n_maps, date span) so the
   later holdout-LL-with/without analysis can run per block.

## Outputs (yours alone)
- MoreTestingMaybeFiles.py (registry entries only — nothing else in the file)
- data/<id>.csv + data/maps/<id>.csv + data/series/<id>.csv for new events
- data/match_results.csv + data/match_dates.json + data/match_times.json
  (via the standard builders, incremental)
- testing_lab/v8/stats/corpus_diff.json, corpus_blocks.json
- testing_lab/v8/data/prefranchise/*
- testing_lab/v8/preregister.corpus.md, logs/corpus.log

## Forbidden
BuildRatingTimeline / BuildMapRatings / any rating rebuild. VCTMM. Editing
anything in testing_lab/ outside v8/. Running RefreshLiveData. Deleting or
rewriting existing event CSVs (append-new only).

## Done criteria
corpus_diff.json covers 2021-2026 with a decision per event; EWC 2025 chain
scraped + verified N/N; match_results/dates updated; prefranchise corpus
delivered or its deferral quantified.

## Return format
≤500 words: events added (ids + match counts), verification results, the
total new-match count 2023-2026 (this changes holdout n — say by how much,
approximately), prefranchise status, gaps. Artifact paths. No transcripts.
