"""agent:corpus — testing_lab/v8/stats/corpus_blocks.json.
One block per backfilled unit so the later holdout-LL with/without analysis
can toggle blocks. Blocks group related events (EWC chains, CN Evo family,
off-season one-offs, official-circuit holes, prefranchise LANs).
n_matches = kept series; n_maps = per-map games; span from scrape-time ts.
"""
import json
import os
import re
import sys

ROOT = "/Users/benny_es1/PythonTest"
V8STATS = os.path.join(ROOT, "testing_lab", "v8", "stats")
OUT = os.path.join(V8STATS, "corpus_blocks.json")

BLOCKS = {
    "ewc_2025_chain": ["2025_ewc", "2025_ewc_qual", "2025_china_evo_2"],
    "vct_2023_qualifiers": ["2023_lcq", "2023_china_champions_qual"],
    "cn_evolution_family": ["2023_china_evo_1", "2023_china_evo_3",
                            "2025_china_evo_1", "2025_china_evo_3",
                            "2025_china_evo_epilogue", "2026_china_evo_1",
                            "2024_fgc_inv"],
    "offseason_2023": ["2023_rbhg", "2023_ten_global", "2023_convergence"],
    "offseason_2024": ["2024_rbhg", "2024_ten_asia", "2024_radiant_asia",
                       "2024_shanghai_masters"],
    "offseason_2025": ["2025_acl", "2025_rbhg", "2025_ten_global",
                       "2025_china_evo_epilogue", "2025_super_champions_cup",
                       "2025_shanghai_masters", "2025_radiant_intl"],
    "prefranchise_2021": ["2021_masters2_reykjavik", "2021_masters3_berlin",
                          "2021_champions"],
    "prefranchise_2022": ["2022_masters1_reykjavik", "2022_masters2_copenhagen",
                          "2022_champions"],
}
# 2025_china_evo_epilogue sits in both cn_evolution_family and offseason_2025;
# keep it only in the CN family to keep blocks disjoint.
BLOCKS["offseason_2025"] = [e for e in BLOCKS["offseason_2025"]
                            if e != "2025_china_evo_epilogue"]


def main():
    verify = {}
    for fn in sorted(os.listdir(V8STATS)):
        m = re.match(r"scrape_verify_(.+)\.json$", fn)
        if m:
            verify[m.group(1)] = json.load(open(os.path.join(V8STATS, fn)))

    out = {"generated": "2026-07-28",
           "note": "blocks are disjoint; every backfilled event appears exactly once. "
                   "prefranchise blocks live in testing_lab/v8/data/prefranchise/ and are "
                   "NOT part of data/ or ALL_EVENTS.",
           "blocks": {}}
    listed = set()
    for bname, eids in BLOCKS.items():
        evs = {}
        n_matches = n_maps = 0
        dates = []
        for eid in eids:
            v = verify.get(eid)
            if v is None:
                continue
            listed.add(eid)
            em = len(v)
            emaps = sum(len(r["maps"]) for r in v.values())
            n_matches += em
            n_maps += emaps
            ds = [r["utc_ts"][:10] for r in v.values() if r.get("utc_ts")]
            dates.extend(ds)
            evs[eid] = {"n_matches": em, "n_maps": emaps,
                        "span": [min(ds), max(ds)] if ds else None}
        if not evs:
            continue
        out["blocks"][bname] = {"events": evs, "n_matches": n_matches,
                                "n_maps": n_maps,
                                "span": [min(dates), max(dates)] if dates else None}
    orphans = sorted(set(verify) - listed)
    if orphans:
        out["orphan_events_not_in_any_block"] = orphans
    tot_m = sum(b["n_matches"] for b in out["blocks"].values())
    tot_g = sum(b["n_maps"] for b in out["blocks"].values())
    pre_m = sum(out["blocks"].get(b, {}).get("n_matches", 0)
                for b in ("prefranchise_2021", "prefranchise_2022"))
    out["totals"] = {"all_matches": tot_m, "all_maps": tot_g,
                     "matches_2023_2026": tot_m - pre_m,
                     "matches_prefranchise": pre_m}
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out["totals"], indent=1))
    if orphans:
        print("ORPHANS (loud):", orphans)


if __name__ == "__main__":
    main()
