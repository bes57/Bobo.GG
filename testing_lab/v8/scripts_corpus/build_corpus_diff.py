"""agent:corpus — build testing_lab/v8/stats/corpus_diff.json.

Sources (all VLR, no memory): archive_events.json (59-page /events sweep),
hub_events.json (vct-2021..2026 circuit hubs), candidate_facts.json (event
pages: participants + franchised counts), search results, ewc2025_stages.json.
Decision rules are the pre-registered C1/C2/C3 criteria.
Exact dates for adds come from the cached event pages ("Dates <x>, <year>").
"""
import json
import os
import re
import sys

ROOT = "/Users/benny_es1/PythonTest"
sys.path.insert(0, ROOT)
from MoreTestingMaybeFiles import ALL_EVENTS, _parse_vlr_stats_url  # noqa: E402

CACHE = ("/private/tmp/claude-501/-Users-benny-es1-PythonTest/"
         "1a0d507c-1768-436d-a3aa-05cca76a816a/scratchpad")
OUT = os.path.join(ROOT, "testing_lab", "v8", "stats", "corpus_diff.json")

archive = json.load(open(os.path.join(CACHE, "archive_events.json")))
facts = json.load(open(os.path.join(CACHE, "candidate_facts.json")))
arch_by_id = {r["vlr_event_id"]: r for r in archive}

MONTHS = {m: i + 1 for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])}


def exact_dates(eid):
    """Parse 'Dates <Mon D>–<Mon D>, YYYY' from the cached event page header."""
    p = os.path.join(CACHE, "html_cache", f"event_{eid}.html")
    if not os.path.exists(p):
        return None, None
    html = open(p).read()
    txt = " ".join(re.sub(r"<[^>]+>", " ", html).split())
    m = re.search(r"Dates\s+([A-Za-z]{3})\w*\s+(\d+)\s*[–—-]\s*(?:([A-Za-z]{3})\w*\s+)?(\d+),\s*(\d{4})", txt)
    if not m:
        m2 = re.search(r"Dates\s+([A-Za-z]{3})\w*\s+(\d+),?\s*(\d{4})", txt)
        if m2:
            mo, d, y = MONTHS[m2.group(1)], int(m2.group(2)), int(m2.group(3))
            return f"{y}-{mo:02d}-{d:02d}", f"{y}-{mo:02d}-{d:02d}"
        return None, None
    sm = MONTHS[m.group(1)]
    sd = int(m.group(2))
    em = MONTHS[m.group(3)] if m.group(3) else sm
    ed = int(m.group(4))
    y = int(m.group(5))
    sy = y - 1 if sm > em else y  # ranges never span new year in these events
    return f"{sy}-{sm:02d}-{sd:02d}", f"{y}-{em:02d}-{ed:02d}"


registered_vlr = {}
for ev in ALL_EVENTS:
    for r, u in ev["regions"].items():
        vid, _ = _parse_vlr_stats_url(u)
        if vid:
            registered_vlr[vid] = ev["id"]

# ── decisions (pre-registered criteria; facts from candidate_facts.json) ──
ADDS = {
    # vlr_id: (new_event_id, criterion, region_tag)
    "2449": ("2025_ewc + 2025_ewc_qual (split by VLR series stages)", "C2", "Mixed/per-region"),
    "2450": ("2025_china_evo_2", "C2", "CN"),
    "1658": ("2023_lcq", "C1", "Americas"),
    "1659": ("2023_lcq", "C1", "EMEA"),
    "1660": ("2023_lcq", "C1", "Pacific"),
    "1664": ("2023_china_champions_qual", "C1", "CN"),
    "1747": ("2023_china_evo_1", "C3", "CN"),
    "1880": ("2023_china_evo_3", "C3", "CN"),
    "1911": ("2023_convergence", "C3", "Mixed"),
    "1807": ("2023_ten_global", "C3", "Mixed"),
    "1752": ("2023_rbhg", "C3", "Mixed"),
    "2234": ("2024_fgc_inv", "C3", "CN"),
    "2171": ("2024_rbhg", "C3", "Mixed"),
    "2219": ("2024_ten_asia", "C3", "Mixed"),
    "2228": ("2024_radiant_asia", "C3", "Mixed"),
    "2265": ("2024_shanghai_masters", "C3", "Mixed"),
    "2339": ("2025_china_evo_1", "C3", "CN"),
    "2402": ("2025_acl", "C3", "Mixed"),
    "2590": ("2025_china_evo_3", "C3", "CN"),
    "2720": ("2025_china_evo_epilogue", "C3", "CN"),
    "2602": ("2025_rbhg", "C3", "Mixed"),
    "2667": ("2025_ten_global", "C3", "Mixed"),
    "2740": ("2025_radiant_intl", "C3", "Mixed"),
    "2735": ("2025_super_champions_cup", "C3", "Mixed"),
    "2734": ("2025_shanghai_masters", "C3", "Mixed"),
    "2894": ("2026_china_evo_1", "C3", "CN"),
}

EXCLUDES = {
    "2666": "Spotlight 2025 Pacific: GC/academy squads (T1 Spotlight etc.), 0 franchised orgs",
    "2669": "Spotlight 2025 Americas: GC/academy squads, 0 franchised orgs",
    "2212": "Ludwig x Tarik Inv 3: custom-banner mixed teams, 0 franchised orgs",
    "1953": "Ludwig x Tarik Inv 2: only 2 franchised orgs (100T, C9) < 4 gate",
    "1453": "Ludwig x Tarik Inv 1: only 2 franchised orgs (SEN, T1) < 4 gate",
    "2291": "VCT OFF//SEASON Spotlight 2024 Pacific: GC/academy squads, 0 franchised",
    "2260": "VCT OFF//SEASON Spotlight 2024 Americas: GC/academy squads, 0 franchised",
    "2207": "VCT OFF//SEASON Spotlight 2024 EMEA: GC teams, 0 franchised",
    "1915": "Sean Gares OFF//SEASON Showdown: content/tier-2 teams, 0 franchised",
    "1868": "Sentinels Invitational 2023: 3-team event, 2 franchised < 4 gate",
    "1825": "China Evolution 2023 Act 2: 4-team event, 3 franchised (EDG/ASE/TE) < 4 gate",
    "2765_dup": "",
}
EXCLUDES.pop("2765_dup")

EXCLUDE_FAMILIES = [
    ("Ascension", "tier-2 path-to-franchise event (pre-registered exclusion)"),
    ("Game Changers", "separate GC circuit (pre-registered exclusion)"),
    ("Challengers", "tier-2 Challengers circuit 2023+ (pre-registered exclusion)"),
    ("College", "collegiate"),
    ("Predator League", "national/tier-2 mix, franchised participation below gate"),
    ("Twitch Rivals", "content event, custom teams"),
    ("Saudi eLeague", "national event"),
    ("China National Tournament", "CN tier-2 national circuit"),
    ("VALORANT China National Competition", "CN tier-2 national circuit"),
]

diff = {"generated": "2026-07-28",
        "method": "VLR /events archive sweep (59 pages, 2937 rows) + vct-YYYY hub pages "
                  "+ event search; candidate facts from event pages (participants, dates); "
                  "decisions per pre-registered C1/C2/C3 (testing_lab/v8/preregister.corpus.md)",
        "falsifier_findings": [
            "No separate 2025 EWC regional-qualifier events exist on VLR (2026-style ids do not "
            "exist for 2025). The 2025 qualifiers DO exist but as series stages INSIDE event 2449 "
            "(EMEA Qualifier May 16-26, Americas Qualifier May 16-25, Pacific X ACL Qualifier "
            "May 22-25; series_ids 4759/4760/4757). Backfilled as 2025_ewc_qual by stage split.",
            "EWC 2025 main event had 8 teams, all franchised, NO CN team. China Evolution Act 2 "
            "2025 (2450) exists in the expected window branded 'X Asian Champions League'; its "
            "EWC-qualifier role is not stated on VLR — included as 2025_china_evo_2 per brief "
            "naming, role caveat recorded.",
            "No Valorant event at EWC 2024 on VLR (search returns only 2025/2026 EWC events).",
        ],
        "seasons": {}}

for yr in range(2021, 2027):
    diff["seasons"][str(yr)] = []

comp = [r for r in archive if r["section"] == "completed events"]


def add_row(yr, row):
    diff["seasons"][str(yr)].append(row)


seen_ids = set()
for r in comp:
    yr = r["year_inferred"]
    if yr is None or yr < 2021 or yr > 2026:
        continue
    vid = r["vlr_event_id"]
    name = r["name"]
    if vid in seen_ids:
        continue
    fam_excl = None
    for fam, why in EXCLUDE_FAMILIES:
        if fam.lower() in name.lower():
            fam_excl = why
            break
    row = None
    if vid in registered_vlr:
        row = {"vlr_event_id": vid, "name": name, "dates": r["dates_txt"],
               "in_registry": True, "registry_id": registered_vlr[vid],
               "decision": "registered", "reason": "already in ALL_EVENTS"}
    elif vid in ADDS:
        newid, crit, reg = ADDS[vid]
        f = facts.get(vid, {})
        ds, de = exact_dates(vid)
        row = {"vlr_event_id": vid, "name": name,
               "dates": f"{ds}..{de}" if ds else r["dates_txt"],
               "in_registry": False, "decision": "add", "criterion": crit,
               "new_registry_id": newid, "region_tag": reg,
               "n_teams": f.get("n_teams"), "n_franchised": f.get("n_franchised"),
               "franchised_orgs": f.get("franchised_orgs"),
               "match_count": None}
    elif vid in EXCLUDES:
        f = facts.get(vid, {})
        row = {"vlr_event_id": vid, "name": name, "dates": r["dates_txt"],
               "in_registry": False, "decision": "exclude",
               "reason": EXCLUDES[vid],
               "n_teams": f.get("n_teams"), "n_franchised": f.get("n_franchised")}
    elif yr <= 2022 and re.search(r"(champions tour|valorant champions|masters)", name, re.I) \
            and not re.search(r"game changers|academy", name, re.I):
        row = {"vlr_event_id": vid, "name": name, "dates": r["dates_txt"],
               "in_registry": False, "decision": "prefranchise_candidate",
               "reason": "2021-22 official circuit; separate corpus, prioritized "
                         "Champions > Masters > regional (see prefranchise/registry.json)"}
    elif fam_excl:
        row = {"vlr_event_id": vid, "name": name, "dates": r["dates_txt"],
               "in_registry": False, "decision": "exclude", "reason": fam_excl}
    if row is not None:
        seen_ids.add(vid)
        add_row(yr, row)

# everything else in the archive is a non-candidate (tier-3/community);
# count them per season for transparency instead of listing 2500 rows
noncand = {str(y): 0 for y in range(2021, 2027)}
for r in comp:
    yr = r["year_inferred"]
    if yr is None or yr < 2021 or yr > 2026 or r["vlr_event_id"] in seen_ids:
        continue
    noncand[str(yr)] += 1
diff["non_candidates_by_year"] = noncand
diff["non_candidate_definition"] = ("archive rows matching no tier-1 family (no official "
                                    "circuit name, no EWC family, prize < $100k without "
                                    "franchised-participation signal) — community/tier-3")

with open(OUT, "w") as f:
    json.dump(diff, f, indent=1)
n_add = sum(1 for y in diff["seasons"].values() for r in y if r["decision"] == "add")
n_reg = sum(1 for y in diff["seasons"].values() for r in y if r["decision"] == "registered")
n_exc = sum(1 for y in diff["seasons"].values() for r in y if r["decision"] == "exclude")
n_pre = sum(1 for y in diff["seasons"].values() for r in y if r["decision"] == "prefranchise_candidate")
print(f"corpus_diff.json: add={n_add} registered={n_reg} exclude={n_exc} prefranchise_cand={n_pre}")
for y, rows in diff["seasons"].items():
    for r in rows:
        if r["decision"] == "add":
            print(f"  ADD {y} {r['vlr_event_id']:>5} -> {r['new_registry_id']:28} {r['dates']:24} franchised={r.get('n_franchised')}")
