"""agent:corpus — post-scrape patch of corpus_diff.json:
fill match_count (kept series) per added event from scrape_verify_*.json,
correct the EWC-2025 falsifier wording (16-team main event, group stage incl
4 CN orgs — the event-page 8-team module was playoffs-only), and add
prefranchise scraped/deferred status.
"""
import json
import os
import re

ROOT = "/Users/benny_es1/PythonTest"
V8STATS = os.path.join(ROOT, "testing_lab", "v8", "stats")
DIFF = os.path.join(V8STATS, "corpus_diff.json")

verify = {}
for fn in sorted(os.listdir(V8STATS)):
    m = re.match(r"scrape_verify_(.+)\.json$", fn)
    if m:
        verify[m.group(1)] = json.load(open(os.path.join(V8STATS, fn)))

diff = json.load(open(DIFF))

diff["falsifier_findings"] = [
    "No separate 2025 EWC regional-qualifier events exist on VLR (the 2026-style "
    "standalone qualifier events have no 2025 counterparts). The 2025 qualifiers DO exist "
    "as series stages INSIDE event 2449: EMEA Qualifier May 16-26 (series 4759, 18 "
    "matches), Americas Qualifier May 16-25 (4760, 16), Pacific X ACL Qualifier May 22-25 "
    "(4757, 15). Backfilled as 2025_ewc_qual via series_id split; disjoint from 2025_ewc.",
    "EWC 2025 main event (Jul 8-13) is a 16-team group stage + 8-team playoff; the scraped "
    "field includes 4 CN orgs (BLG/EDG/XLG/TEC). The event page's 8-team module lists only "
    "the playoff field. (An earlier note claiming 'no CN team at EWC 2025' was wrong and is "
    "corrected here.) No CN qualifier stage exists inside 2449; China Evolution Series Act 2 "
    "2025 (2450, 'X Asian Champions League') is the in-window CN Evo act and is backfilled "
    "as 2025_china_evo_2 — VLR does not state an explicit EWC-qualifier role for it.",
    "No Valorant event at EWC 2024 on VLR (search returns only 2025/2026 EWC events).",
]

VLR_TO_EID = {
    "2449": ["2025_ewc", "2025_ewc_qual"],
    "2450": ["2025_china_evo_2"],
    "1658": ["2023_lcq"], "1659": ["2023_lcq"], "1660": ["2023_lcq"],
    "1664": ["2023_china_champions_qual"],
    "1747": ["2023_china_evo_1"], "1880": ["2023_china_evo_3"],
    "1911": ["2023_convergence"], "1807": ["2023_ten_global"], "1752": ["2023_rbhg"],
    "2234": ["2024_fgc_inv"], "2171": ["2024_rbhg"], "2219": ["2024_ten_asia"],
    "2228": ["2024_radiant_asia"], "2265": ["2024_shanghai_masters"],
    "2339": ["2025_china_evo_1"], "2402": ["2025_acl"], "2590": ["2025_china_evo_3"],
    "2720": ["2025_china_evo_epilogue"], "2602": ["2025_rbhg"],
    "2667": ["2025_ten_global"], "2740": ["2025_radiant_intl"],
    "2735": ["2025_super_champions_cup"], "2734": ["2025_shanghai_masters"],
    "2894": ["2026_china_evo_1"],
}

for season, rows in diff["seasons"].items():
    for r in rows:
        if r.get("decision") != "add":
            continue
        eids = VLR_TO_EID.get(r["vlr_event_id"], [])
        counts = {}
        for eid in eids:
            if eid in verify:
                if r["vlr_event_id"] in ("1658", "1659", "1660"):
                    # 2023_lcq is one merged registry event across 3 VLR events;
                    # split count by the region recorded per match
                    reg = {"1658": "Americas", "1659": "EMEA", "1660": "Pacific"}[r["vlr_event_id"]]
                    counts[eid] = sum(1 for v in verify[eid].values() if v["region"] == reg)
                else:
                    counts[eid] = len(verify[eid])
        r["match_count"] = sum(counts.values()) if counts else None
        r["match_count_by_registry_id"] = counts if counts else None

# prefranchise status
pre_reg_p = os.path.join(ROOT, "testing_lab", "v8", "data", "prefranchise", "registry.json")
if os.path.exists(pre_reg_p):
    pre = json.load(open(pre_reg_p))
    scraped = {e["id"]: len(verify.get(e["id"], {})) for e in pre["events"]}
    diff["prefranchise_status"] = {
        "scraped_events": scraped,
        "deferred": pre["deferred"]["deferred_counts"],
        "corpus_path": "testing_lab/v8/data/prefranchise/",
    }

with open(DIFF, "w") as f:
    json.dump(diff, f, indent=1)
adds = [(r["new_registry_id"], r["match_count"])
        for rows in diff["seasons"].values() for r in rows if r.get("decision") == "add"]
print("patched match counts:")
for nid, mc in adds:
    print(f"  {nid}: {mc}")
