"""
Run once to scrape all player headshot URLs across VCT 2023-2026 events
and save them to headshots.json. The VCT app loads from that file instead
of scraping headshots live.

Usage: python MoreTestingMaybeFiles.py
"""

import requests
from bs4 import BeautifulSoup
import json
import time
import os

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
}

OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "data", "headshots.json")

# All VCT franchised-era events (2023-current), most-recent first.
# League events list each region separately; international events use {"International": url}.
#
# Each entry can optionally include "start"/"end" YYYY-MM-DD strings.  The live-data
# refresh pipeline (scrapers/RefreshLiveData.py) treats an event as "live" when today
# is between (start - 7 days) and (end + 3 days), and scrapes completed + upcoming
# matches for every live event automatically.  To onboard a brand-new event, add one
# entry below with the correct VLR IDs and dates; no other code changes are needed.
ALL_EVENTS = [
    # ── EWC-class events (added 2026-07-22) ──────────────────────────
    # ratings_only: feed BenPom team ratings but hidden from all player-facing
    # UIs (leaderboards, dropdowns) — same treatment as CN-only events.
    # vct_only: ScrapeMatchData keeps only matches where BOTH teams are known
    # VCT orgs (these events include tier-2 guests we don't rate).
    # NOTE: deliberately NOT tagged "International" — that key would pull them
    # into INTL_EVENTS (CN-shrinkage intl weights), which is untested for
    # non-VCT events. All are past events; live-scrape never triggers on them.
    # The four regional EWC qualifiers are FOUR separate single-region events,
    # not one multi-region circuit. They were merged into a single entry on
    # 2026-07-28 and split back on 2026-08-12: every one of the 184 map
    # matchups is intra-regional (verified — zero cross-region pairings), so a
    # single entry spanning four regions misdescribed them. Each region also
    # ran its own window (CN finished 2026-05-20, the rest 05-31), which one
    # merged start/end could not express.
    #
    # Ratings are unaffected either way: the v6 solve reads region from
    # TEAM_REGIONS per team, never from the event's `regions` field, and these
    # events are not in INTL_EVENTS. Verified by rebuild — identical output.
    # Data lives in data/{,maps/,series/}2026_ewc_qual_<region>.csv.
    {"id": "2026_ewc_qual_americas", "label": "2026 EWC Qualifier — Americas",
     "year": 2026, "start": "2026-05-12", "end": "2026-05-31",
     "ratings_only": True, "vct_only": True,
     "regions": {
         "Americas": "https://www.vlr.gg/event/stats/2953/esports-world-cup-2026-americas-qualifier"}},
    {"id": "2026_ewc_qual_emea", "label": "2026 EWC Qualifier — EMEA",
     "year": 2026, "start": "2026-05-11", "end": "2026-05-31",
     "ratings_only": True, "vct_only": True,
     "regions": {
         "EMEA": "https://www.vlr.gg/event/stats/2954/esports-world-cup-2026-emea-qualifier"}},
    {"id": "2026_ewc_qual_pacific", "label": "2026 EWC Qualifier — Pacific",
     "year": 2026, "start": "2026-05-11", "end": "2026-05-31",
     "ratings_only": True, "vct_only": True,
     "regions": {
         "Pacific": "https://www.vlr.gg/event/stats/2955/esports-world-cup-2026-pacific-qualifier"}},
    {"id": "2026_ewc_qual_cn", "label": "2026 EWC Qualifier — China",
     "year": 2026, "start": "2026-05-12", "end": "2026-05-20",
     "ratings_only": True, "vct_only": True,
     "regions": {
         "CN": "https://www.vlr.gg/event/stats/2956/esports-world-cup-2026-china-qualifier"}},
    {"id": "2026_china_evo_2", "label": "2026 China Evolution Series Act 2",
     "year": 2026, "start": "2026-05-21", "end": "2026-05-31",
     "ratings_only": True, "vct_only": True,
     "regions": {"CN": "https://www.vlr.gg/event/stats/2988/china-evolution-series-2026-act-2"}},
    {"id": "2026_ewc", "label": "2026 Esports World Cup",
     "year": 2026, "start": "2026-07-02", "end": "2026-07-12",
     "ratings_only": True, "vct_only": True,
     "regions": {"Mixed": "https://www.vlr.gg/event/stats/2952/esports-world-cup-2026"}},
    # ── v8 corpus backfill (added 2026-07-28, agent:corpus) ──────────
    # Same treatment as the EWC-class block above: ratings_only + vct_only,
    # NOT International-tagged. All past events — live-scrape never triggers.
    # Decisions + provenance: testing_lab/v8/stats/corpus_diff.json.
    # NOTE on the 2025 EWC pair: VLR models the 2025 regional qualifiers as
    # series STAGES inside event 2449 (unlike 2026's separate events), so
    # 2025_ewc and 2025_ewc_qual share vlr id 2449. The maps/series CSVs were
    # split by VLR series_id (main: playoffs 4960 + group stage 4758; quals:
    # EMEA 4759 / Americas 4760 / Pacific-x-ACL 4757) and are DISJOINT — no
    # map is rated twice. Do NOT re-scrape either entry with ScrapeMatchData
    # (its match-list fetch is unfiltered and would conflate the two); the
    # ?series_id= params below only scope the event-stats pages.
    {"id": "2025_ewc", "label": "2025 Esports World Cup",
     "year": 2025, "start": "2025-07-08", "end": "2025-07-13",
     "ratings_only": True, "vct_only": True,
     "regions": {"Mixed": "https://www.vlr.gg/event/stats/2449/esports-world-cup-2025"}},
    {"id": "2025_ewc_qual", "label": "2025 EWC Qualifiers",
     "year": 2025, "start": "2025-05-16", "end": "2025-05-26",
     "ratings_only": True, "vct_only": True,
     "regions": {
         "Americas": "https://www.vlr.gg/event/stats/2449/esports-world-cup-2025?series_id=4760",
         "EMEA":     "https://www.vlr.gg/event/stats/2449/esports-world-cup-2025?series_id=4759",
         "Pacific":  "https://www.vlr.gg/event/stats/2449/esports-world-cup-2025?series_id=4757"}},
    {"id": "2025_china_evo_2", "label": "2025 China Evolution Series Act 2",
     # end = last match date on VLR (485380 played May 12; the archive's
     # "May 8-10" range undershoots by two days)
     "year": 2025, "start": "2025-05-08", "end": "2025-05-12",
     "ratings_only": True, "vct_only": True,
     "regions": {"CN": "https://www.vlr.gg/event/stats/2450/china-evolution-series-act-2-x-asian-champions-league"}},
    {"id": "2023_lcq", "label": "2023 Last Chance Qualifiers",
     "year": 2023, "start": "2023-07-15", "end": "2023-07-23",
     "ratings_only": True, "vct_only": True,
     "regions": {
         "Americas": "https://www.vlr.gg/event/stats/1658/champions-tour-2023-americas-last-chance-qualifier",
         "EMEA":     "https://www.vlr.gg/event/stats/1659/champions-tour-2023-emea-last-chance-qualifier",
         "Pacific":  "https://www.vlr.gg/event/stats/1660/champions-tour-2023-pacific-last-chance-qualifier"}},
    {"id": "2023_china_champions_qual", "label": "2023 Champions China Qualifier",
     "year": 2023, "start": "2023-06-01", "end": "2023-07-16",
     "ratings_only": True, "vct_only": True,
     "regions": {"CN": "https://www.vlr.gg/event/stats/1664/champions-tour-2023-champions-china-qualifier"}},
    {"id": "2023_china_evo_1", "label": "2023 China Evolution Series Act 1",
     "year": 2023, "start": "2023-09-07", "end": "2023-10-01",
     "ratings_only": True, "vct_only": True,
     "regions": {"CN": "https://www.vlr.gg/event/stats/1747/china-evolution-series-act-1-variation"}},
    {"id": "2023_china_evo_3", "label": "2023 China Evolution Series Act 3",
     "year": 2023, "start": "2023-11-02", "end": "2023-11-26",
     "ratings_only": True, "vct_only": True,
     "regions": {"CN": "https://www.vlr.gg/event/stats/1880/china-evolution-series-act-3-heritability"}},
    {"id": "2023_rbhg", "label": "2023 Red Bull Home Ground #4",
     "year": 2023, "start": "2023-09-11", "end": "2023-11-06",
     "ratings_only": True, "vct_only": True,
     "regions": {"Mixed": "https://www.vlr.gg/event/stats/1752/red-bull-home-ground-4"}},
    {"id": "2023_ten_global", "label": "2023 TEN Global Invitational",
     "year": 2023, "start": "2023-10-07", "end": "2023-10-08",
     "ratings_only": True, "vct_only": True,
     "regions": {"Mixed": "https://www.vlr.gg/event/stats/1807/ten-global-invitational-2023"}},
    {"id": "2023_convergence", "label": "2023 Convergence",
     "year": 2023, "start": "2023-12-14", "end": "2023-12-17",
     "ratings_only": True, "vct_only": True,
     "regions": {"Mixed": "https://www.vlr.gg/event/stats/1911/convergence-2023"}},
    {"id": "2024_fgc_inv", "label": "2024 FGC Invitational",
     "year": 2024, "start": "2024-10-30", "end": "2024-11-10",
     "ratings_only": True, "vct_only": True,
     "regions": {"CN": "https://www.vlr.gg/event/stats/2234/fgc-invitational-2024"}},
    {"id": "2024_ten_asia", "label": "2024 TEN Valorant Asia Invitational",
     "year": 2024, "start": "2024-10-26", "end": "2024-10-27",
     "ratings_only": True, "vct_only": True,
     "regions": {"Mixed": "https://www.vlr.gg/event/stats/2219/ten-valorant-asia-invitational"}},
    {"id": "2024_rbhg", "label": "2024 Red Bull Home Ground #5",
     # start = first match on the event's own VLR match list (its Sep-Oct
     # qualifier phase lives under the main event id, unlike the archive's
     # "Nov 20-23" main-stage-only date range)
     "year": 2024, "start": "2024-09-29", "end": "2024-11-23",
     "ratings_only": True, "vct_only": True,
     "regions": {"Mixed": "https://www.vlr.gg/event/stats/2171/red-bull-home-ground-5"}},
    {"id": "2024_radiant_asia", "label": "2024 Valorant Radiant Asia Invitational",
     "year": 2024, "start": "2024-11-21", "end": "2024-12-01",
     "ratings_only": True, "vct_only": True,
     "regions": {"Mixed": "https://www.vlr.gg/event/stats/2228/valorant-radiant-asia-invitational"}},
    {"id": "2024_shanghai_masters", "label": "2024 Shanghai Esports Masters",
     "year": 2024, "start": "2024-12-06", "end": "2024-12-07",
     "ratings_only": True, "vct_only": True,
     "regions": {"Mixed": "https://www.vlr.gg/event/stats/2265/shanghai-esports-masters-2024"}},
    {"id": "2025_china_evo_1", "label": "2025 China Evolution Series Act 1",
     "year": 2025, "start": "2025-02-08", "end": "2025-02-16",
     "ratings_only": True, "vct_only": True,
     "regions": {"CN": "https://www.vlr.gg/event/stats/2339/china-evolution-series-act-1"}},
    {"id": "2025_acl", "label": "2025 Asian Champions League",
     "year": 2025, "start": "2025-05-14", "end": "2025-05-25",
     "ratings_only": True, "vct_only": True,
     "regions": {"Mixed": "https://www.vlr.gg/event/stats/2402/hero-esports-asian-champions-league-2025"}},
    {"id": "2025_china_evo_3", "label": "2025 China Evolution Series Act 3",
     "year": 2025, "start": "2025-08-28", "end": "2025-09-07",
     "ratings_only": True, "vct_only": True,
     "regions": {"CN": "https://www.vlr.gg/event/stats/2590/china-evolution-series-act-3"}},
    {"id": "2025_ten_global", "label": "2025 TEN Global Invitational",
     "year": 2025, "start": "2025-11-07", "end": "2025-11-09",
     "ratings_only": True, "vct_only": True,
     "regions": {"Mixed": "https://www.vlr.gg/event/stats/2667/ten-global-invitational-2025"}},
    {"id": "2025_china_evo_epilogue", "label": "2025 China Evolution Series Epilogue",
     "year": 2025, "start": "2025-11-11", "end": "2025-11-16",
     "ratings_only": True, "vct_only": True,
     "regions": {"CN": "https://www.vlr.gg/event/stats/2720/china-evolution-series-epilogue"}},
    {"id": "2025_rbhg", "label": "2025 Red Bull Home Ground",
     "year": 2025, "start": "2025-11-13", "end": "2025-11-16",
     "ratings_only": True, "vct_only": True,
     "regions": {"Mixed": "https://www.vlr.gg/event/stats/2602/red-bull-home-ground-2025"}},
    {"id": "2025_super_champions_cup", "label": "2025 China Esports Festival Super Champions Cup",
     "year": 2025, "start": "2025-11-28", "end": "2025-11-30",
     "ratings_only": True, "vct_only": True,
     "regions": {"Mixed": "https://www.vlr.gg/event/stats/2735/china-esports-festival-super-champions-cup"}},
    {"id": "2025_shanghai_masters", "label": "2025 Shanghai Esports Masters",
     "year": 2025, "start": "2025-12-03", "end": "2025-12-05",
     "ratings_only": True, "vct_only": True,
     "regions": {"Mixed": "https://www.vlr.gg/event/stats/2734/shanghai-esports-masters-2025"}},
    {"id": "2025_radiant_intl", "label": "2025 Valorant Radiant International Invitational",
     "year": 2025, "start": "2025-12-17", "end": "2025-12-21",
     "ratings_only": True, "vct_only": True,
     "regions": {"Mixed": "https://www.vlr.gg/event/stats/2740/valorant-radiant-international-invitational"}},
    {"id": "2026_china_evo_1", "label": "2026 China Evolution Series Act 1",
     "year": 2026, "start": "2026-03-16", "end": "2026-03-22",
     "ratings_only": True, "vct_only": True,
     "regions": {"CN": "https://www.vlr.gg/event/stats/2894/china-evolution-series-2026-act-1"}},
    # ── 2026 ──────────────────────────────────────────────────────────
    {
        "id": "2026_champions",
        "label": "2026 Champions",
        "year": 2026,
        "start": "2026-09-24",
        "end":   "2026-10-18",
        "regions": {
            # Fill VLR URL when the event is posted on VLR. RefreshLiveData
            # will skip entries whose region URL is empty.
            "International": "",
        },
    },
    {
        "id": "2026_stage2",
        "label": "2026 Stage 2",
        "year": 2026,
        "start": "2026-07-15",
        "end":   "2026-09-06",
        "regions": {
            "Americas": "",
            "EMEA":     "",
            "Pacific":  "",
            "CN":       "",
        },
    },
    {
        "id": "2026_masters_london",
        "label": "2026 Masters London",
        "year": 2026,
        "start": "2026-06-05",
        "end":   "2026-06-21",
        "regions": {
            "International": "https://www.vlr.gg/event/stats/2765/valorant-masters-london-2026",
        },
    },
    {
        "id": "2026_stage1",
        "label": "2026 Stage 1",
        "year": 2026,
        "start": "2026-04-01",
        "end":   "2026-05-25",
        "regions": {
            "EMEA":     "https://www.vlr.gg/event/stats/2863/vct-2026-emea-stage-1",
            "Americas": "https://www.vlr.gg/event/stats/2860/vct-2026-americas-stage-1",
            "Pacific":  "https://www.vlr.gg/event/stats/2775/vct-2026-pacific-stage-1",
            "CN":       "https://www.vlr.gg/event/stats/2864/vct-2026-china-stage-1",
        },
    },
    {
        "id": "2026_masters_santiago",
        "label": "2026 Masters Santiago",
        "year": 2026,
        "start": "2026-02-28",
        "end":   "2026-03-15",
        "regions": {
            "International": "https://www.vlr.gg/event/stats/2760/valorant-masters-santiago-2026",
        },
    },
    {
        "id": "2026_china_kickoff",
        "label": "2026 China Kickoff",
        "year": 2026,
        "start": "2026-01-21",
        "end":   "2026-02-09",
        "regions": {
            "CN": "https://www.vlr.gg/event/stats/2685/vct-2026-china-kickoff",
        },
    },
    {
        "id": "2026_kickoff",
        "label": "2026 Kickoff",
        "year": 2026,
        "start": "2026-01-15",
        "end":   "2026-02-16",
        "regions": {
            "Americas": "https://www.vlr.gg/event/stats/2682/vct-2026-americas-kickoff",
            "EMEA":     "https://www.vlr.gg/event/stats/2684/vct-2026-emea-kickoff",
            "Pacific":  "https://www.vlr.gg/event/stats/2683/vct-2026-pacific-kickoff",
        },
    },
    # ── 2025 ──────────────────────────────────────────────────────────
    {
        "id": "2025_champions",
        "label": "2025 Champions",
        "year": 2025,
        "regions": {
            "International": "https://www.vlr.gg/event/stats/2283/valorant-champions-2025",
        },
    },
    {
        "id": "2025_stage2",
        "label": "2025 Stage 2",
        "year": 2025,
        "regions": {
            "Americas": "https://www.vlr.gg/event/stats/2501/vct-2025-americas-stage-2",
            "EMEA":     "https://www.vlr.gg/event/stats/2498/vct-2025-emea-stage-2",
            "Pacific":  "https://www.vlr.gg/event/stats/2500/vct-2025-pacific-stage-2",
        },
    },
    {
        "id": "2025_china_stage2",
        "label": "2025 China Stage 2",
        "year": 2025,
        "regions": {
            "CN": "https://www.vlr.gg/event/stats/2499/vct-2025-china-stage-2",
        },
    },
    {
        "id": "2025_masters_toronto",
        "label": "2025 Masters Toronto",
        "year": 2025,
        "regions": {
            "International": "https://www.vlr.gg/event/stats/2282/valorant-masters-toronto-2025",
        },
    },
    {
        "id": "2025_stage1",
        "label": "2025 Stage 1",
        "year": 2025,
        "regions": {
            "Americas": "https://www.vlr.gg/event/stats/2347/vct-2025-americas-stage-1",
            "EMEA":     "https://www.vlr.gg/event/stats/2380/vct-2025-emea-stage-1",
            "Pacific":  "https://www.vlr.gg/event/stats/2379/vct-2025-pacific-stage-1",
        },
    },
    {
        "id": "2025_china_stage1",
        "label": "2025 China Stage 1",
        "year": 2025,
        "regions": {
            "CN": "https://www.vlr.gg/event/stats/2359/vct-2025-china-stage-1",
        },
    },
    {
        "id": "2025_masters_bangkok",
        "label": "2025 Masters Bangkok",
        "year": 2025,
        "regions": {
            "International": "https://www.vlr.gg/event/stats/2281/valorant-masters-bangkok-2025",
        },
    },
    {
        "id": "2025_kickoff",
        "label": "2025 Kickoff",
        "year": 2025,
        "regions": {
            "Americas": "https://www.vlr.gg/event/stats/2274/vct-2025-americas-kickoff",
            "EMEA":     "https://www.vlr.gg/event/stats/2276/vct-2025-emea-kickoff",
            "Pacific":  "https://www.vlr.gg/event/stats/2277/champions-tour-2025-pacific-kickoff",
        },
    },
    {
        "id": "2025_china_kickoff",
        "label": "2025 China Kickoff",
        "year": 2025,
        "regions": {
            "CN": "https://www.vlr.gg/event/stats/2275/vct-2025-china-kickoff",
        },
    },
    # ── 2024 ──────────────────────────────────────────────────────────
    {
        "id": "2024_champions",
        "label": "2024 Champions",
        "year": 2024,
        "regions": {
            "International": "https://www.vlr.gg/event/stats/2097/valorant-champions-2024",
        },
    },
    {
        "id": "2024_stage2",
        "label": "2024 Stage 2",
        "year": 2024,
        "regions": {
            "Americas": "https://www.vlr.gg/event/stats/2095/champions-tour-2024-americas-stage-2",
            "EMEA":     "https://www.vlr.gg/event/stats/2094/champions-tour-2024-emea-stage-2",
            "Pacific":  "https://www.vlr.gg/event/stats/2005/champions-tour-2024-pacific-stage-2",
        },
    },
    {
        "id": "2024_china_stage2",
        "label": "2024 China Stage 2",
        "year": 2024,
        "regions": {
            "CN": "https://www.vlr.gg/event/stats/2096/champions-tour-2024-china-stage-2",
        },
    },
    {
        "id": "2024_masters_shanghai",
        "label": "2024 Masters Shanghai",
        "year": 2024,
        "regions": {
            "International": "https://www.vlr.gg/event/stats/1999/champions-tour-2024-masters-shanghai",
        },
    },
    {
        "id": "2024_stage1",
        "label": "2024 Stage 1",
        "year": 2024,
        "regions": {
            "Americas": "https://www.vlr.gg/event/stats/2004/champions-tour-2024-americas-stage-1",
            "EMEA":     "https://www.vlr.gg/event/stats/1998/champions-tour-2024-emea-stage-1",
            "Pacific":  "https://www.vlr.gg/event/stats/2002/champions-tour-2024-pacific-stage-1",
        },
    },
    {
        "id": "2024_china_stage1",
        "label": "2024 China Stage 1",
        "year": 2024,
        "regions": {
            "CN": "https://www.vlr.gg/event/stats/2006/champions-tour-2024-china-stage-1",
        },
    },
    {
        "id": "2024_masters_madrid",
        "label": "2024 Masters Madrid",
        "year": 2024,
        "regions": {
            "International": "https://www.vlr.gg/event/stats/1921/champions-tour-2024-masters-madrid",
        },
    },
    {
        "id": "2024_kickoff",
        "label": "2024 Kickoff",
        "year": 2024,
        "regions": {
            "Americas": "https://www.vlr.gg/event/stats/1923/champions-tour-2024-americas-kickoff",
            "EMEA":     "https://www.vlr.gg/event/stats/1925/champions-tour-2024-emea-kickoff",
            "Pacific":  "https://www.vlr.gg/event/stats/1924/champions-tour-2024-pacific-kickoff",
        },
    },
    {
        "id": "2024_china_kickoff",
        "label": "2024 China Kickoff",
        "year": 2024,
        "regions": {
            "CN": "https://www.vlr.gg/event/stats/1926/champions-tour-2024-china-kickoff",
        },
    },
    # ── 2023 ──────────────────────────────────────────────────────────
    {
        "id": "2023_champions",
        "label": "2023 Champions",
        "year": 2023,
        "regions": {
            "International": "https://www.vlr.gg/event/stats/1657/valorant-champions-2023",
        },
    },
    {
        "id": "2023_masters_tokyo",
        "label": "2023 Masters Tokyo",
        "year": 2023,
        "regions": {
            "International": "https://www.vlr.gg/event/stats/1494/champions-tour-2023-masters-tokyo",
        },
    },
    {
        "id": "2023_league",
        "label": "2023 League",
        "year": 2023,
        "regions": {
            "Americas": "https://www.vlr.gg/event/stats/1189/champions-tour-2023-americas-league",
            "EMEA":     "https://www.vlr.gg/event/stats/1190/champions-tour-2023-emea-league",
            "Pacific":  "https://www.vlr.gg/event/stats/1191/champions-tour-2023-pacific-league",
        },
    },
    {
        "id": "2023_lock_in",
        "label": "2023 LOCK//IN",
        "year": 2023,
        "regions": {
            "International": "https://www.vlr.gg/event/stats/1188/champions-tour-2023-lock-in-s-o-paulo",
        },
    },
]


def _parse_vlr_stats_url(url):
    """Extract (vlr_id, slug) from a VLR stats URL — returns (None, None) if blank."""
    import re as _re
    if not url:
        return None, None
    m = _re.search(r"/event/(?:stats|matches)/(\d+)/([^/?#]+)", url)
    if not m:
        return None, None
    return m.group(1), m.group(2)


def live_events_today(today=None, lead_days=7, trail_days=3):
    """
    Return events whose date window contains today (with `lead_days` of pre-roll
    and `trail_days` of post-roll), OR the single most-recent event by end date
    if nothing is currently within window (so the pipeline always has something
    to monitor between official splits).

    Used by scrapers/RefreshLiveData.py to drive dynamic scraping.
    """
    import datetime as _dt
    if today is None:
        today = _dt.date.today()
    elif isinstance(today, str):
        today = _dt.date.fromisoformat(today)

    dated = []
    for ev in ALL_EVENTS:
        s = ev.get("start"); e = ev.get("end")
        if not (s and e):
            continue
        try:
            sd = _dt.date.fromisoformat(s)
            ed = _dt.date.fromisoformat(e)
        except ValueError:
            continue
        dated.append((sd, ed, ev))

    live = [ev for sd, ed, ev in dated
            if (sd - _dt.timedelta(days=lead_days)) <= today
            <= (ed + _dt.timedelta(days=trail_days))]

    if live:
        return live

    # Fallback: most recent past event so upcoming-match scraping still has a
    # surface to listen on (handy in the gap between splits).
    past = [(sd, ed, ev) for sd, ed, ev in dated if ed <= today]
    if past:
        past.sort(key=lambda t: t[1], reverse=True)
        return [past[0][2]]
    return []


def scrape_profile_urls(url):
    try:
        res = requests.get(url, headers=HEADERS, timeout=15)
    except Exception as e:
        print(f"  Request failed for {url}: {e}")
        return []
    soup = BeautifulSoup(res.text, "html.parser")
    table = soup.find("table")
    if not table:
        print(f"  No table at {url}")
        return []
    urls = []
    for tr in table.find("tbody").find_all("tr"):
        tds = tr.find_all("td")
        if not tds:
            continue
        a = tds[0].find("a", href=True)
        if a:
            urls.append("https://www.vlr.gg" + a["href"])
    return urls


def fetch_headshot(profile_url):
    try:
        res = requests.get(profile_url, headers=HEADERS, timeout=15)
        soup = BeautifulSoup(res.text, "html.parser")
        og = soup.find("meta", property="og:image")
        if og:
            content = og.get("content", "")
            if "owcdn.net" in content:
                return content
    except Exception:
        pass
    return ""


def main():
    all_profile_urls = set()
    for event in ALL_EVENTS:
        for region, url in event["regions"].items():
            print(f"Scraping player list: {event['label']} / {region}...")
            urls = scrape_profile_urls(url)
            all_profile_urls.update(urls)
            time.sleep(1)

    print(f"\nFound {len(all_profile_urls)} unique player profiles across all events.")

    headshots = {}
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE) as f:
            headshots = json.load(f)
        print(f"Loaded {len(headshots)} cached entries — will skip already-fetched profiles.")

    to_fetch = [u for u in all_profile_urls if u not in headshots]
    print(f"Fetching headshots for {len(to_fetch)} new profiles...\n")

    for i, url in enumerate(to_fetch, 1):
        headshot = fetch_headshot(url)
        headshots[url] = headshot
        status = "found" if headshot else "none"
        print(f"  [{i}/{len(to_fetch)}] {url.split('/')[-1]} — {status}")
        if i % 10 == 0:
            with open(OUTPUT_FILE, "w") as f:
                json.dump(headshots, f, indent=2)
        time.sleep(0.5)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(headshots, f, indent=2)

    found = sum(1 for v in headshots.values() if v)
    print(f"\nDone. {found}/{len(headshots)} players have headshots.")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
