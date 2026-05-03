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

OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "headshots.json")

# All VCT franchised-era events (2023-current), most-recent first.
# League events list each region separately; international events use {"International": url}.
ALL_EVENTS = [
    # ── 2026 ──────────────────────────────────────────────────────────
    {
        "id": "2026_stage1",
        "label": "2026 Stage 1",
        "year": 2026,
        "regions": {
            "EMEA":     "https://www.vlr.gg/event/stats/2863/vct-2026-emea-stage-1",
            "Americas": "https://www.vlr.gg/event/stats/2860/vct-2026-americas-stage-1",
            "Pacific":  "https://www.vlr.gg/event/stats/2775/vct-2026-pacific-stage-1",
        },
    },
    {
        "id": "2026_masters_santiago",
        "label": "2026 Masters Santiago",
        "year": 2026,
        "regions": {
            "International": "https://www.vlr.gg/event/stats/2760/valorant-masters-santiago-2026",
        },
    },
    {
        "id": "2026_kickoff",
        "label": "2026 Kickoff",
        "year": 2026,
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
