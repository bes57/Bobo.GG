"""agent:corpus — backfill driver.

Scrapes one or more events end-to-end through scrapers/enriched/vlr_client.fetch
(sequential, polite), reusing the production parsers:
  - match list:   same selectors as ScrapeMatchData.get_match_urls
  - match stats:  RefreshLiveData._parse_match_html (ovw grid) with the legacy
                  <table> fallback copied from ScrapeMatchData.scrape_match
  - winner/score: BuildMatchResults logic (ovw ._first_org) with legacy fallback
  - date ts:      data-utc-ts capture (RefreshLiveData._scrape_date semantics)

Every fetched match page is cached to the session scratchpad so verification
can re-parse the exact HTML this scrape saw. Per-event ground truth (winner,
score, ts per MatchID) goes to testing_lab/v8/stats/scrape_verify_<id>.json.

Usage: python3 scrape_backfill.py <plan.json>
plan.json = [{"id","label","vct_only":bool,"dest":"data"|"prefranchise",
              "regions":{name:stats_url}, "drop_showmatch":bool}]
"""
import csv
import json
import os
import re
import sys
import time

ROOT = "/Users/benny_es1/PythonTest"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scrapers"))
from bs4 import BeautifulSoup  # noqa: E402
from scrapers.enriched.vlr_client import fetch  # noqa: E402
from scrapers.RefreshLiveData import _parse_match_html  # noqa: E402
from scrapers.BuildMapRatings import TEAM_REGIONS  # noqa: E402

CACHE = ("/private/tmp/claude-501/-Users-benny-es1-PythonTest/"
         "1a0d507c-1768-436d-a3aa-05cca76a816a/scratchpad/html_cache")
V8STATS = os.path.join(ROOT, "testing_lab", "v8", "stats")
PREFR = os.path.join(ROOT, "testing_lab", "v8", "data", "prefranchise")
LOG = os.path.join(ROOT, "testing_lab", "v8", "logs", "corpus.log")

COLS = ["Player", "Org", "ProfileURL", "Region", "MatchID", "MapNum", "MapName",
        "SeriesFormat", "R2.0", "ACS", "K", "D", "A", "KAST", "ADR", "HS%",
        "FK", "FD", "K:D"]

MATCH_DELAY = 0.9
LIST_DELAY = 1.1


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def get_soup(url, cache_name):
    path = os.path.join(CACHE, cache_name)
    if os.path.exists(path):
        html = open(path).read()
    else:
        html = fetch(url)
        with open(path, "w") as f:
            f.write(html)
        time.sleep(MATCH_DELAY)
    return BeautifulSoup(html, "html.parser")


def get_match_urls(numeric_id, slug):
    url = f"https://www.vlr.gg/event/matches/{numeric_id}/{slug}/"
    soup = get_soup(url, f"evmatches_{numeric_id}.html")
    urls = []
    for a in soup.select("a.wf-module-item.match-item"):
        href = a.get("href", "")
        status_el = a.select_one(".ml-status")
        if status_el and status_el.get_text(strip=True).lower() != "completed":
            continue
        if re.match(r"^/\d+/", href):
            full = "https://www.vlr.gg" + href
            if full not in urls:
                urls.append(full)
    return urls


# ── legacy <table> fallbacks (pre-ovw markup), copied from ScrapeMatchData ──

def _stat(td):
    span = td.find("span", class_=lambda c: c and "mod-both" in c.split())
    return span.get_text(strip=True) if span else td.get_text(strip=True)


def parse_legacy_tables(soup, match_url, region_tag):
    fmt_el = soup.select_one(".match-header-vs-note")
    fmt_raw = fmt_el.get_text(strip=True).lower() if fmt_el else ""
    series_fmt = "bo5" if "5" in fmt_raw else ("bo3" if "3" in fmt_raw else "bo1")
    m = re.search(r"vlr\.gg/(\d+)/", match_url)
    match_id = m.group(1) if m else ""
    map_rows, series_rows = [], []
    for game_div in soup.select("div.vm-stats-game"):
        game_id = game_div.get("data-game-id", "")
        is_all = (game_id == "all")
        map_name = ""
        if not is_all:
            hdr = game_div.select_one(".vm-stats-game-header .map")
            if hdr:
                fd = hdr.find("div")
                if fd:
                    map_name = fd.get_text(strip=True)
        for table in game_div.select("table.wf-table-inset.mod-overview"):
            tbody = table.find("tbody")
            if not tbody:
                continue
            for tr in tbody.find_all("tr"):
                tds = tr.find_all("td")
                if len(tds) < 10:
                    continue
                ptd = tds[0]
                pname = ptd.select_one(".text-of")
                porg = ptd.select_one(".ge-text-light")
                pa = ptd.find("a", href=True)
                player = pname.get_text(strip=True) if pname else ""
                org = porg.get_text(strip=True) if porg else ""
                if not player:
                    continue
                k = _stat(tds[4]); d = _stat(tds[5])
                try:
                    kd = round(int(k) / int(d), 2) if int(d) > 0 else float(int(k))
                except Exception:
                    kd = ""
                row = {"Player": player, "Org": org,
                       "ProfileURL": ("https://www.vlr.gg" + pa["href"]) if pa else "",
                       "Region": region_tag, "MatchID": match_id, "MapNum": game_id,
                       "MapName": map_name, "SeriesFormat": series_fmt,
                       "R2.0": _stat(tds[2]), "ACS": _stat(tds[3]),
                       "K": k, "D": d, "A": _stat(tds[6]),
                       "KAST": _stat(tds[8]) if len(tds) > 8 else "",
                       "ADR": _stat(tds[9]) if len(tds) > 9 else "",
                       "HS%": _stat(tds[10]) if len(tds) > 10 else "",
                       "FK": _stat(tds[11]) if len(tds) > 11 else "",
                       "FD": _stat(tds[12]) if len(tds) > 12 else "",
                       "K:D": kd}
                (series_rows if is_all else map_rows).append(row)
    return map_rows, series_rows


def _first_org_ovw(table):
    for row in table.select(".ovw-row"):
        pcell = row.select_one(".ovw-cell.mod-player") or row.select_one(".mod-player")
        if pcell is None:
            continue
        porg = pcell.select_one(".ge-text-light")
        if porg:
            return porg.get_text(strip=True)
    return None


def _first_org_legacy(table):
    tbody = table.find("tbody")
    if not tbody:
        return None
    for tr in tbody.find_all("tr"):
        td = tr.find("td")
        if td is None:
            continue
        porg = td.select_one(".ge-text-light")
        if porg:
            return porg.get_text(strip=True)
    return None


def parse_results(soup):
    """BuildMatchResults semantics on an already-parsed soup.
    Returns (per_map {mapnum: [winner, 'w-l']}, series [winner, 'wm-lm'], match_name)."""
    match_name = ""
    name_el = soup.select_one(".match-header-event-series")
    if name_el:
        match_name = re.sub(r"\s+", " ", name_el.get_text()).strip()
    per_map = {}
    for game_div in soup.select("div.vm-stats-game"):
        game_id = game_div.get("data-game-id", "")
        if game_id == "all":
            continue
        tables = game_div.select(".ovw-table")
        first_org = _first_org_ovw
        if len(tables) < 2:
            tables = game_div.select("table.wf-table-inset.mod-overview")
            first_org = _first_org_legacy
        if len(tables) < 2:
            continue
        org1, org2 = first_org(tables[0]), first_org(tables[1])
        if not org1 or not org2:
            continue
        score_els = game_div.select(".vm-stats-game-header .team .score")
        if len(score_els) < 2:
            continue
        try:
            s1 = int(score_els[0].get_text(strip=True) or 0)
            s2 = int(score_els[1].get_text(strip=True) or 0)
        except Exception:
            continue
        if s1 > s2:
            per_map[game_id] = [org1, f"{s1}-{s2}"]
        elif s2 > s1:
            per_map[game_id] = [org2, f"{s2}-{s1}"]
    series = None
    if per_map:
        from collections import Counter
        c = Counter(w for w, _ in per_map.values())
        sw = c.most_common(1)[0][0]
        series = [sw, f"{c[sw]}-{sum(c.values()) - c[sw]}"]
    return per_map, series, match_name


def get_utc_ts(soup):
    el = soup.find("div", class_="moment-tz-convert", attrs={"data-utc-ts": True})
    return el["data-utc-ts"] if el else None


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in COLS})


def scrape_event(ev):
    eid = ev["id"]
    dest = ev.get("dest", "data")
    if dest == "prefranchise":
        maps_path = os.path.join(PREFR, f"maps_{eid}.csv")
        series_path = os.path.join(PREFR, f"series_{eid}.csv")
    else:
        maps_path = os.path.join(ROOT, "data", "maps", f"{eid}.csv")
        series_path = os.path.join(ROOT, "data", "series", f"{eid}.csv")
    verify_path = os.path.join(V8STATS, f"scrape_verify_{eid}.json")
    if os.path.exists(maps_path) and os.path.exists(series_path) and os.path.exists(verify_path):
        log(f"SKIP {eid} (maps/series/verify already exist)")
        return

    log(f"EVENT {eid} start — {ev['label']}")
    all_map_rows, all_series_rows = [], []
    verify = {}
    kept = skipped_vct = skipped_show = empty = 0

    explicit = ev.get("match_urls") or {}
    region_iter = (list(explicit.items()) if explicit
                   else list(ev["regions"].items()))
    for region_name, region_src in region_iter:
        region_tag = region_name if region_name not in ("International",) else ""
        if explicit:
            murls = region_src  # pre-resolved list (e.g. series_id-scoped stages)
            log(f"  {eid}/{region_name}: {len(murls)} matches (explicit list)")
        else:
            m = re.search(r"/event/(?:stats/)?(\d+)/([^/?#]+)", region_src)
            if not m:
                log(f"  BAD region url for {eid}/{region_name}: {region_src}")
                continue
            numeric_id, slug = m.group(1), m.group(2)
            murls = get_match_urls(numeric_id, slug)
            log(f"  {eid}/{region_name}: {len(murls)} completed matches listed")
            time.sleep(LIST_DELAY - MATCH_DELAY if LIST_DELAY > MATCH_DELAY else 0)

        for i, murl in enumerate(murls, 1):
            mid = re.search(r"vlr\.gg/(\d+)/", murl).group(1)
            soup = get_soup(murl, f"match_{mid}.html")
            mrows, srows, _disp = _parse_match_html(soup, murl, region_tag)
            parser_used = "ovw"
            if not mrows and not srows:
                mrows, srows = parse_legacy_tables(soup, murl, region_tag)
                parser_used = "legacy"
            if not mrows:
                empty += 1
                log(f"    [{i}/{len(murls)}] {mid} EMPTY (no stat rows; forfeit/showpage?)")
                continue
            per_map, series_res, match_name = parse_results(soup)
            if ev.get("drop_showmatch", True) and "showmatch" in match_name.lower():
                skipped_show += 1
                log(f"    [{i}/{len(murls)}] {mid} skipped (showmatch: {match_name})")
                continue
            if ev.get("vct_only"):
                orgs = {r["Org"] for r in mrows if r.get("Org")}
                unknown = {o for o in orgs if o not in TEAM_REGIONS and o != "TYLOO"}
                if unknown or len(orgs) < 2:
                    skipped_vct += 1
                    log(f"    [{i}/{len(murls)}] {mid} skipped (non-VCT: {sorted(unknown)})")
                    continue
            if not per_map or not series_res:
                empty += 1
                log(f"    [{i}/{len(murls)}] {mid} NO RESULTS PARSED — excluded, flagged")
                continue
            all_map_rows.extend(mrows)
            all_series_rows.extend(srows)
            kept += 1
            verify[mid] = {"event_id": eid, "region": region_name, "url": murl,
                           "maps": per_map, "series": series_res,
                           "match_name": match_name, "utc_ts": get_utc_ts(soup),
                           "parser": parser_used,
                           "orgs": sorted({r["Org"] for r in mrows if r.get("Org")})}
            log(f"    [{i}/{len(murls)}] {mid} ok ({parser_used}) "
                f"{series_res[0]} {series_res[1]} | {len(per_map)} maps")

    if all_map_rows:
        write_csv(maps_path, all_map_rows)
        write_csv(series_path, all_series_rows)
        with open(verify_path, "w") as f:
            json.dump(verify, f, indent=1)
        log(f"EVENT {eid} done: kept={kept} series, map_rows={len(all_map_rows)}, "
            f"series_rows={len(all_series_rows)}, skipped_vct={skipped_vct}, "
            f"showmatch={skipped_show}, empty={empty}")
        log(f"  -> {maps_path}")
        log(f"  -> {series_path}")
        log(f"  -> {verify_path}")
    else:
        log(f"EVENT {eid} produced NO ROWS — nothing written (kept={kept}, "
            f"skipped_vct={skipped_vct}, showmatch={skipped_show}, empty={empty})")


if __name__ == "__main__":
    plan = json.load(open(sys.argv[1]))
    for ev in plan:
        scrape_event(ev)
    log("scrape_backfill plan complete")
