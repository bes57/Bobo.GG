"""
Build agent/role data for all domestic VCT events for the role scatter charts.
Saves:
  data/article_all_roles_raw.json   — raw agent cache (resumable, keyed by event|match|game|profile)
  data/article_all_roles_data.json  — final scatter data organized by role

Run from project root: python scrapers/BuildAllRolesData.py
"""

import os, re, sys, time, json
import pandas as pd
import requests
from bs4 import BeautifulSoup
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DATA_DIR   = os.path.join(ROOT, "data")
RAW_CACHE  = os.path.join(DATA_DIR, "article_all_roles_raw.json")
FINAL_DATA = os.path.join(DATA_DIR, "article_all_roles_data.json")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
}

DOMESTIC_EVENTS = [
    "2023_league", "2024_kickoff", "2024_stage1", "2024_stage2",
    "2025_kickoff", "2025_stage1", "2025_stage2", "2026_kickoff",
]

EVENT_LABELS = {
    "2023_league":  "2023 League",
    "2024_kickoff": "2024 Kickoff",
    "2024_stage1":  "2024 Stage 1",
    "2024_stage2":  "2024 Stage 2",
    "2025_kickoff": "2025 Kickoff",
    "2025_stage1":  "2025 Stage 1",
    "2025_stage2":  "2025 Stage 2",
    "2026_kickoff": "2026 Kickoff",
}

AGENT_ROLES = {
    "jett": "Duelist", "reyna": "Duelist", "raze": "Duelist",
    "neon": "Duelist", "yoru": "Duelist", "iso": "Duelist", "waylay": "Duelist",
    "sova": "Initiator", "breach": "Initiator", "fade": "Initiator",
    "kayo": "Initiator", "gekko": "Initiator", "tejo": "Initiator",
    "omen": "Controller", "brimstone": "Controller", "viper": "Controller",
    "astra": "Controller", "harbor": "Controller", "clove": "Controller",
    "killjoy": "Sentinel", "cypher": "Sentinel", "sage": "Sentinel",
    "chamber": "Sentinel", "deadlock": "Sentinel", "vyse": "Sentinel",
}


def load_raw():
    if os.path.exists(RAW_CACHE):
        with open(RAW_CACHE) as f:
            return json.load(f)
    return {}


def save_raw(cache):
    with open(RAW_CACHE, "w") as f:
        json.dump(cache, f)


def scrape_agents(match_id, event_id, raw):
    """Scrape one match page for agent data. Updates raw in place. Returns True if network needed."""
    prefix = f"{event_id}|{match_id}|"
    if any(k.startswith(prefix) for k in raw):
        return False
    url = f"https://www.vlr.gg/{match_id}/"
    try:
        soup = BeautifulSoup(requests.get(url, headers=HEADERS, timeout=20).text, "html.parser")
        for gdiv in soup.select("div.vm-stats-game"):
            gid = gdiv.get("data-game-id", "")
            if gid == "all":
                continue
            for tbl in gdiv.select("table.wf-table-inset.mod-overview"):
                for tr in tbl.select("tbody tr"):
                    tds = tr.find_all("td")
                    if len(tds) < 3:
                        continue
                    a = tds[0].find("a", href=True)
                    if not a:
                        continue
                    profile = "https://www.vlr.gg" + a["href"]
                    for td in tds:
                        for img in td.find_all("img"):
                            m = re.search(r'/agents/([^.]+)\.png', img.get("src", ""))
                            if m:
                                raw[f"{event_id}|{match_id}|{gid}|{profile}"] = m.group(1).lower()
                                break
    except Exception as e:
        print(f"    error {match_id}: {e}")
    return True


def compute_org_ratios(maps_df, results_df, match_ids):
    """Return {org: round_win_ratio} for the given match IDs."""
    ev_res = results_df[
        results_df["MatchID"].isin(match_ids) & (results_df["MapNum"] != "all")
    ].copy()
    map_orgs = maps_df.groupby(["MatchID", "MapNum"])["Org"].apply(
        lambda x: list(x.unique())
    ).reset_index()
    merged = ev_res.merge(map_orgs, on=["MatchID", "MapNum"], how="inner")
    rounds = {}
    for _, row in merged.iterrows():
        winner = row["WinnerOrg"]
        try:
            w, l = [int(x) for x in str(row["Score"]).split("-")]
        except Exception:
            continue
        losers = [o for o in row["Org"] if o != winner]
        if not losers:
            continue
        loser = losers[0]
        for org, won, lost in [(winner, w, l), (loser, l, w)]:
            rounds.setdefault(org, [0, 0])
            rounds[org][0] += won
            rounds[org][1] += lost
    return {
        org: round(s[0] / (s[0] + s[1]), 4)
        for org, s in rounds.items() if (s[0] + s[1]) > 0
    }


def build_event(event_id, raw, results_df):
    label      = EVENT_LABELS[event_id]
    maps_path  = os.path.join(DATA_DIR, "maps",  f"{event_id}.csv")
    event_path = os.path.join(DATA_DIR, f"{event_id}.csv")
    if not os.path.exists(maps_path) or not os.path.exists(event_path):
        print(f"  [{label}] missing CSVs, skipping")
        return []

    maps_df = pd.read_csv(maps_path)
    maps_df = maps_df[maps_df["Region"].isin(["Americas", "EMEA", "Pacific"])]
    maps_df = maps_df[maps_df["MapNum"].astype(str) != "all"]
    maps_df["MatchID"] = maps_df["MatchID"].astype(str)
    maps_df["MapNum"]  = maps_df["MapNum"].astype(str)

    event_df = pd.read_csv(event_path)
    event_df = event_df[event_df["Region"].isin(["Americas", "EMEA", "Pacific"])]
    event_df["R2.0"] = pd.to_numeric(event_df["R2.0"], errors="coerce")
    event_df["Rnd"]  = pd.to_numeric(event_df["Rnd"],  errors="coerce")
    event_df = event_df.dropna(subset=["R2.0"])
    event_df = event_df[event_df["Rnd"] > 75]

    match_ids = maps_df["MatchID"].unique()
    print(f"  [{label}] {len(match_ids)} matches to check")

    scraped = 0
    for i, mid in enumerate(match_ids, 1):
        if scrape_agents(mid, event_id, raw):
            scraped += 1
            if scraped % 20 == 0:
                save_raw(raw)
                print(f"    checkpoint: scraped {scraped} new ({i}/{len(match_ids)} total), cache={len(raw)}")
            time.sleep(0.6)
    if scraped > 0:
        save_raw(raw)
        print(f"    scraped {scraped} new matches for {label}")

    # Count agent usage per player across the event
    player_agents = {}
    for _, row in maps_df.iterrows():
        key   = f"{event_id}|{row['MatchID']}|{row['MapNum']}|{row['ProfileURL']}"
        agent = raw.get(key)
        if not agent:
            continue
        player_agents.setdefault(row["ProfileURL"], Counter())[agent] += 1

    org_ratio = compute_org_ratios(maps_df, results_df, set(match_ids))

    points = []
    for _, row in event_df.iterrows():
        profile = row["ProfileURL"]
        agents  = player_agents.get(profile)
        if not agents:
            continue

        # Build role distribution from agent usage
        role_counts = Counter()
        for ag, count in agents.items():
            role = AGENT_ROLES.get(ag)
            if role:
                role_counts[role] += count
        if not role_counts:
            continue

        total_role_maps = sum(role_counts.values())
        top_role, top_count = role_counts.most_common(1)[0]
        top_pct = top_count / total_role_maps

        ratio = org_ratio.get(row["Org"])
        if ratio is None:
            continue

        base = {
            "player":          row["Player"],
            "org":             row["Org"],
            "region":          row["Region"],
            "profile":         profile,
            "event":           label,
            "rating":          round(float(row["R2.0"]), 3),
            "round_win_ratio": ratio,
        }

        if top_pct >= 0.5:
            top_agent = next(
                (ag for ag, _ in agents.most_common() if AGENT_ROLES.get(ag) == top_role), None
            )
            points.append({**base, "role": top_role, "agent": top_agent or ""})
        else:
            role_pct = {
                role: round(count / total_role_maps * 100, 1)
                for role, count in role_counts.items()
            }
            points.append({**base, "role": "Flex", "roles": role_pct})

    print(f"    {len(points)} player-event points with role data")
    return points


if __name__ == "__main__":
    raw = load_raw()
    print(f"Raw cache: {len(raw)} existing entries\n")

    results_df = pd.read_csv(os.path.join(DATA_DIR, "match_results.csv"))
    results_df["MatchID"] = results_df["MatchID"].astype(str)
    results_df["MapNum"]  = results_df["MapNum"].astype(str)

    all_points = []
    for eid in DOMESTIC_EVENTS:
        pts = build_event(eid, raw, results_df)
        all_points.extend(pts)

    by_role = {"Initiator": [], "Controller": [], "Duelist": [], "Sentinel": [], "Flex": []}
    for pt in all_points:
        role = pt.get("role")
        if role not in by_role:
            continue
        entry = {
            "x":       pt["round_win_ratio"],
            "y":       pt["rating"],
            "player":  pt["player"],
            "org":     pt["org"],
            "region":  pt["region"],
            "profile": pt["profile"],
            "event":   pt["event"],
        }
        if role == "Flex":
            entry["roles"] = pt.get("roles", {})
        else:
            entry["agent"] = pt.get("agent", "")
        by_role[role].append(entry)

    with open(FINAL_DATA, "w") as f:
        json.dump(by_role, f)

    print(f"\nDone! Saved {len(all_points)} total points to {FINAL_DATA}")
    for role, pts in by_role.items():
        print(f"  {role}: {len(pts)} points")
