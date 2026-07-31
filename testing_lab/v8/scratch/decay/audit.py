"""agent:decay — pre-registration data audit (NO experiments, no holdout LL).
Facts needed to write preregister.decay.md precisely."""
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

TL = "/Users/benny_es1/PythonTest/testing_lab"
V8 = os.path.join(TL, "v8")
sys.path.insert(0, TL)

# 1. frame + sha verify
fp = os.path.join(V8, "data", "frame_expanded", "series.csv")
sha = hashlib.sha256(open(fp, "rb").read()).hexdigest()
crn = json.load(open(os.path.join(V8, "crn.json")))
want = crn["frame_expanded"]["series_csv_sha256"]
assert sha == want, f"FRAME SHA MISMATCH {sha} != {want}"
frame = pd.read_csv(fp, dtype={"date": str})
hold = (frame.date > "2024-12-31").values
print(f"frame ok sha={sha[:12]} n={len(frame)} train={int((~hold).sum())} holdout={int(hold.sum())}")

# 2. engine game corpus
from engine import Engine  # noqa: E402
eng = Engine()
print(f"engine games={len(eng.games)} teams={len(eng.teams)}")
ev_games = Counter(g["event_id"] for g in eng.games)
for e in ("2023_lcq", "2023_china_evo_1", "2025_ewc", "2026_ewc", "2024_shanghai_masters",
          "2025_super_champions_cup", "2026_china_evo_2", "2026_stage2"):
    print(f"  games[{e}] = {ev_games.get(e, 0)}")
frame_ev = set(frame.event_id)
game_ev = set(ev_games)
missing = frame_ev - game_ev
print(f"frame events with NO games: {sorted(missing) if missing else 'none'}")
pred_days = sorted(frame.date.unique())
print(f"pred days = {len(pred_days)} [{pred_days[0]} .. {pred_days[-1]}]")

# maps per game available?
mn = Counter(g["map_name"] for g in eng.games)
print(f"map names: {len(mn)} distinct; top: {mn.most_common(14)}")

# 3. lineup tables coverage of frame match_ids
lf = pd.read_csv(os.path.join(V8, "data", "lineup_features.csv"))
lu = pd.read_csv(os.path.join(V8, "data", "lineups.csv"))
lf_m = set(lf.match_id)
lu_m = set(lu.match_id)
fm = set(frame.match_id)
fh = set(frame.match_id[hold])
print(f"lineup_features: frame coverage {len(fm & lf_m)}/{len(fm)}, holdout {len(fh & lf_m)}/{len(fh)}")
print(f"lineups.csv    : frame coverage {len(fm & lu_m)}/{len(fm)}, holdout {len(fh & lu_m)}/{len(fh)}")
miss_ev = Counter(frame[~frame.match_id.isin(lf_m)].event_id)
print(f"  lineup_features missing by event: {dict(miss_ev)}")
# matches_since_change availability
msc = lf.groupby("match_id")["matches_since_change"].min()
print(f"  matches_since_change non-null on {int(msc.notna().sum())}/{len(msc)} covered matches")

# 4. round_outcomes coverage (5c side-conditional)
ro = pd.read_csv(os.path.join("/Users/benny_es1/PythonTest/data/enriched/round_outcomes.csv"))
ro_m = set(ro.match_id)
cov = frame.match_id.isin(ro_m)
print(f"round_outcomes: frame coverage {int(cov.sum())}/{len(frame)}")
by_ev = frame.groupby("event_id").apply(lambda d: d.match_id.isin(ro_m).mean())
nocov = by_ev[by_ev < 0.5]
print(f"  events <50% covered: n={len(nocov)}")
print("  " + ", ".join(f"{k}={v:.2f}" for k, v in sorted(nocov.items())[:40]))
# side columns
print(f"  ro cols: {list(ro.columns)}")

# 5. quote density real?
qd = os.path.join(V8, "stats", "quote_density.json")
d = json.load(open(qd))
print(f"quote_density: keys {list(d)[:6]}")
if "bands" in d:
    print(f"  bands: {list(d['bands'])[:12]}")

# 6. maps csv player columns (player trajectories)
mp = pd.read_csv("/Users/benny_es1/PythonTest/data/maps/2026_stage2.csv", nrows=3)
print(f"maps csv cols: {list(mp.columns)}")

# 7. map rotation windows: first/last appearance + gaps > 60d
gdates = defaultdict(list)
for g in eng.games:
    gdates[g["map_name"]].append(g["date_s"])
rows = []
for m, ds in gdates.items():
    ds = sorted(ds)
    d0 = np.array(ds, dtype="datetime64[D]")
    gaps = (d0[1:] - d0[:-1]).astype(int)
    big = [(str(d0[i]), str(d0[i + 1]), int(gaps[i])) for i in np.where(gaps >= 60)[0]]
    rows.append((m, ds[0], ds[-1], len(ds), big))
for r in sorted(rows, key=lambda r: r[1]):
    print(f"  map {r[0]:<10} {r[1]} .. {r[2]} n={r[3]} gaps60={r[4]}")

# 8. v7 continuity artifacts present
print("v7_probs.npz exists:", os.path.exists(os.path.join(TL, "out", "v7_probs.npz")))
print("quote_margin.json exists:", os.path.exists(os.path.join(TL, "out", "quote_margin.json")))
