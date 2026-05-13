"""
ScrapeLogos.py — Download team logos from VLR.gg event team pages.
Saves PNG files to static/logos/{ORG}.png and writes static/logos/logos.json.

KNOWN BREAK (2026-05-13): VLR retired the `/event/teams/{id}/...` URL pattern;
those return 404. The EVENT_URLS below are kept as historical references but
this script currently finds 0 orgs end-to-end. For new-org logo fetches use
the manual flow: scrape `/event/{id}/...` for `/team/{tid}/{slug}` links, then
fetch each team page's `og:image`. See git history of 2026-05-13 for an
example of that flow.
"""

import os
import json
import time
import urllib.request
import urllib.error
from html.parser import HTMLParser

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGOS_DIR = os.path.join(ROOT, "static", "logos")
os.makedirs(LOGOS_DIR, exist_ok=True)

# Target orgs to find
TARGET_ORGS = {
    "100T", "2G", "AG", "ALP", "APK", "ASE", "BBL", "BLD", "BLG", "BME",
    "BNY", "C9", "DFM", "DRG", "EDG", "EG", "FNC", "FPX", "FRA", "FRTTT",
    "FS", "FUR", "FUT", "G2", "GE", "GEN", "GIA", "GOA", "GX", "KC",
    "KRX", "KRÜ", "LEV", "LOUD", "M8", "MIBR", "MKOI", "NAVI", "NRG",
    "NS", "OME", "PCF", "PRX", "RRQ", "SEN", "SPB", "T1", "TE", "TH",
    "THAi", "TL", "TLN", "TS", "ULF", "VIT", "VL", "WOL", "WOR", "XLG", "ZETA",
    # CN orgs added 2026-05-13 — extra teams that appear in VCT CN events
    "JDG", "NOVA", "TEC", "TYL", "TYLOO",
}

# Known aliases: VLR abbreviation -> our abbreviation
VLR_ALIASES = {
    "KRU": "KRÜ",
    "KRÜ": "KRÜ",
    "THAI": "THAi",
    "THAi": "THAi",
}

EVENT_URLS = [
    "https://www.vlr.gg/event/teams/2682/vct-2026-americas-kickoff",
    "https://www.vlr.gg/event/teams/2684/vct-2026-emea-kickoff",
    "https://www.vlr.gg/event/teams/2683/vct-2026-pacific-kickoff",
    "https://www.vlr.gg/event/teams/2347/vct-2025-americas-stage-1",
    "https://www.vlr.gg/event/teams/2380/vct-2025-emea-stage-1",
    "https://www.vlr.gg/event/teams/2379/vct-2025-pacific-stage-1",
    # 2024 Stage 1
    "https://www.vlr.gg/event/teams/1921/vct-2024-americas-stage-1",
    "https://www.vlr.gg/event/teams/1923/vct-2024-emea-stage-1",
    "https://www.vlr.gg/event/teams/1922/vct-2024-pacific-stage-1",
    # 2024 Kickoff
    "https://www.vlr.gg/event/teams/1842/vct-2024-americas-kickoff",
    "https://www.vlr.gg/event/teams/1845/vct-2024-emea-kickoff",
    "https://www.vlr.gg/event/teams/1843/vct-2024-pacific-kickoff",
    # 2023 League
    "https://www.vlr.gg/event/teams/1198/vct-2023-americas-league",
    "https://www.vlr.gg/event/teams/1200/vct-2023-emea-league",
    "https://www.vlr.gg/event/teams/1199/vct-2023-pacific-league",
    # 2025 Stage 2
    "https://www.vlr.gg/event/teams/2384/vct-2025-americas-stage-2",
    "https://www.vlr.gg/event/teams/2382/vct-2025-emea-stage-2",
    "https://www.vlr.gg/event/teams/2383/vct-2025-pacific-stage-2",
    # 2026 Stage 1
    "https://www.vlr.gg/event/teams/2686/vct-2026-americas-stage-1",
    "https://www.vlr.gg/event/teams/2688/vct-2026-emea-stage-1",
    "https://www.vlr.gg/event/teams/2687/vct-2026-pacific-stage-1",
    # VCT CN — covers all CN-only orgs across 2024-2026
    "https://www.vlr.gg/event/teams/2864/vct-2026-china-stage-1",
    "https://www.vlr.gg/event/teams/2685/vct-2026-china-kickoff",
    "https://www.vlr.gg/event/teams/2499/vct-2025-china-stage-2",
    "https://www.vlr.gg/event/teams/2359/vct-2025-china-stage-1",
    "https://www.vlr.gg/event/teams/2275/vct-2025-china-kickoff",
    "https://www.vlr.gg/event/teams/2096/champions-tour-2024-china-stage-2",
    "https://www.vlr.gg/event/teams/2006/champions-tour-2024-china-stage-1",
    "https://www.vlr.gg/event/teams/1926/champions-tour-2024-china-kickoff",
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}


def fetch_url(url, retries=3):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=15) as resp:
                return resp.read()
        except Exception as e:
            print(f"  Attempt {attempt+1} failed for {url}: {e}")
            if attempt < retries - 1:
                time.sleep(2)
    return None


class TeamPageParser(HTMLParser):
    """
    Parse VLR.gg /event/teams/ page.
    Each team card looks like:
      <a class="event-teams-item wf-card" href="/team/...">
        <div class="event-teams-item-img">
          <img src="//owcdn.net/img/..." alt="Team Name">
        </div>
        <div class="event-teams-item-name">
          <div class="ge-text-light">ABBREV</div>
          ...
        </div>
      </a>
    """
    def __init__(self):
        super().__init__()
        self.teams = []  # list of {abbrev, img_url}
        self._in_item = False
        self._item_depth = 0
        self._depth = 0
        self._current_img = None
        self._in_abbrev_div = False
        self._abbrev_div_depth = None
        self._collecting_abbrev = False
        self._current_abbrev = ""

    def handle_starttag(self, tag, attrs):
        self._depth += 1
        attrs_dict = dict(attrs)

        if tag == "a":
            classes = attrs_dict.get("class", "")
            if "event-teams-item" in classes:
                self._in_item = True
                self._item_depth = self._depth
                self._current_img = None
                self._current_abbrev = ""

        if self._in_item:
            if tag == "img" and self._current_img is None:
                src = attrs_dict.get("src", "")
                if src:
                    if src.startswith("//"):
                        src = "https:" + src
                    self._current_img = src

            if tag == "div":
                classes = attrs_dict.get("class", "")
                if "ge-text-light" in classes and self._abbrev_div_depth is None:
                    self._in_abbrev_div = True
                    self._abbrev_div_depth = self._depth
                    self._collecting_abbrev = True

    def handle_endtag(self, tag):
        if self._in_item and self._in_abbrev_div and tag == "div":
            if self._depth == self._abbrev_div_depth:
                self._in_abbrev_div = False
                self._abbrev_div_depth = None
                self._collecting_abbrev = False

        if self._in_item and tag == "a" and self._depth == self._item_depth:
            # End of team card
            if self._current_img and self._current_abbrev:
                self.teams.append({
                    "abbrev": self._current_abbrev.strip(),
                    "img_url": self._current_img
                })
            self._in_item = False
            self._current_img = None
            self._current_abbrev = ""
            self._abbrev_div_depth = None
            self._in_abbrev_div = False

        self._depth -= 1

    def handle_data(self, data):
        if self._collecting_abbrev and self._in_abbrev_div:
            self._current_abbrev += data


def normalize_abbrev(abbrev):
    """Normalize VLR abbreviation to match our TARGET_ORGS."""
    abbrev = abbrev.strip()
    # Direct match
    if abbrev in TARGET_ORGS:
        return abbrev
    # Check aliases
    if abbrev in VLR_ALIASES:
        return VLR_ALIASES[abbrev]
    # Try upper
    upper = abbrev.upper()
    if upper in TARGET_ORGS:
        return upper
    if upper in VLR_ALIASES:
        return VLR_ALIASES[upper]
    # Case-insensitive scan of TARGET_ORGS
    abbrev_lower = abbrev.lower()
    for org in TARGET_ORGS:
        if org.lower() == abbrev_lower:
            return org
    return None


def download_image(url, dest_path, retries=3):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = resp.read()
            with open(dest_path, "wb") as f:
                f.write(data)
            return True
        except Exception as e:
            print(f"  Download attempt {attempt+1} failed for {url}: {e}")
            if attempt < retries - 1:
                time.sleep(1)
    return False


def main():
    # Map: org -> img_url (first found wins)
    found = {}

    remaining = set(TARGET_ORGS)

    for url in EVENT_URLS:
        if not remaining:
            print("All orgs found, stopping early.")
            break

        print(f"\nFetching: {url}")
        html = fetch_url(url)
        if not html:
            print("  FAILED to fetch page")
            continue

        try:
            html_text = html.decode("utf-8", errors="replace")
        except Exception:
            html_text = html.decode("latin-1", errors="replace")

        parser = TeamPageParser()
        parser.feed(html_text)

        if not parser.teams:
            print("  No teams parsed — page structure may differ")
            # Debug: show a snippet
            idx = html_text.find("event-teams-item")
            if idx >= 0:
                print(f"  HTML snippet: {html_text[idx:idx+500]}")
            continue

        print(f"  Parsed {len(parser.teams)} teams:")
        for team in parser.teams:
            abbrev = team["abbrev"]
            normalized = normalize_abbrev(abbrev)
            if normalized and normalized in remaining:
                found[normalized] = team["img_url"]
                remaining.discard(normalized)
                print(f"    + {abbrev} -> {normalized} : {team['img_url']}")
            else:
                marker = "(skip)" if normalized is None else f"(dupe: {normalized})"
                print(f"    - {abbrev} {marker}")

        time.sleep(1.5)

    print(f"\n--- Found {len(found)}/{len(TARGET_ORGS)} orgs ---")
    if remaining:
        print(f"Missing: {sorted(remaining)}")

    # Download all found logos
    logos_map = {}
    success = []
    failed = []

    print("\nDownloading logos...")
    for org, img_url in sorted(found.items()):
        dest = os.path.join(LOGOS_DIR, f"{org}.png")
        print(f"  {org}: {img_url}")
        ok = download_image(img_url, dest)
        if ok:
            logos_map[org] = f"{org}.png"
            success.append(org)
            print(f"    -> saved {org}.png")
        else:
            failed.append(org)
            print(f"    -> FAILED")
        time.sleep(0.5)

    # Merge with existing logos.json — never wipe entries we already have.
    # (VLR moved /event/teams/{id} → /event/{id} at some point; if every URL 404s,
    # we don't want to nuke the on-disk dict that other scripts depend on.)
    json_path = os.path.join(LOGOS_DIR, "logos.json")
    existing = {}
    if os.path.exists(json_path):
        try:
            with open(json_path) as f:
                existing = json.load(f) or {}
        except Exception:
            existing = {}
    merged = dict(existing)
    merged.update(logos_map)
    with open(json_path, "w") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False, sort_keys=True)
    print(f"\nWrote {json_path} ({len(merged)} entries; {len(logos_map)} new/updated)")

    print(f"\n=== Summary ===")
    print(f"Downloaded ({len(success)}): {sorted(success)}")
    if failed:
        print(f"Failed ({len(failed)}): {sorted(failed)}")
    not_found = TARGET_ORGS - set(found.keys())
    if not_found:
        print(f"Not found on any event page ({len(not_found)}): {sorted(not_found)}")


if __name__ == "__main__":
    main()
