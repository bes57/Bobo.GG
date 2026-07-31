"""bias_h1 smoke + coverage audit (pre-preregistration data availability check).

1. Engine() loads expanded game set? (EWC-class events present, game counts)
2. frame_expanded loads, holdout n=1217
3. round_outcomes.csv coverage vs the frame: per-event, per-region, train/holdout
No experiment outcomes computed here.
"""
import json, os, sys
import numpy as np
import pandas as pd

TL = "/Users/benny_es1/PythonTest/testing_lab"
V8 = os.path.join(TL, "v8")
sys.path.insert(0, TL)
sys.path.insert(0, V8)

frame = pd.read_csv(os.path.join(V8, "data/frame_expanded/series.csv"))
print("frame rows", len(frame), "train", (frame.date <= "2024-12-31").sum(),
      "holdout", (frame.date > "2024-12-31").sum())
print("frame cols", frame.columns.tolist())

from engine import Engine
eng = Engine()
ev = pd.Series([g["event_id"] for g in eng.games])
print("engine games:", len(eng.games), "events:", ev.nunique())
for e in ["2025_ewc", "2025_ewc_qual", "2026_ewc", "2025_china_evo_1",
          "2023_lcq", "2024_shanghai_masters", "2026_china_evo_2"]:
    print(f"  {e}: {int((ev == e).sum())} maps")

# frame matches vs engine games
gmids = {g["match_id"] for g in eng.games}
missing = set(frame.match_id) - gmids
print("frame matches missing from engine games:", len(missing))

# ---- round_outcomes coverage vs frame
ro = pd.read_csv("/Users/benny_es1/PythonTest/data/enriched/round_outcomes.csv")
ro_mids = set(ro.match_id.unique())
frame["has_rounds"] = frame.match_id.isin(ro_mids)
frame["split"] = np.where(frame.date > "2024-12-31", "holdout", "train")
cov = frame.groupby(["split"]).has_rounds.agg(["mean", "sum", "count"])
print("\ncoverage by split:\n", cov)

# per-event coverage (train side is what the BT fit uses for history)
evcov = (frame.groupby("event_id").has_rounds.agg(["mean", "sum", "count"])
         .sort_values("mean"))
print("\nevents with <100% coverage:")
print(evcov[evcov["mean"] < 1.0].to_string())

# region view: CN involvement
cn_any = (frame.reg_w == "CN") | (frame.reg_l == "CN")
print("\nCN-involved series coverage:",
      frame[cn_any].has_rounds.mean(), "n", int(cn_any.sum()))
print("non-CN series coverage:", frame[~cn_any].has_rounds.mean())

# round rows sanity
print("\nround rows:", len(ro), "matches", ro.match_id.nunique(),
      "sides:", ro.winner_side.value_counts().to_dict())
# rounds per map sanity
per_map = ro.groupby(["match_id", "map_num"]).size()
print("rounds per map: mean %.1f min %d max %d" % (per_map.mean(), per_map.min(), per_map.max()))

# does map_num join to anything? check a match's maps vs match_results
mr = pd.read_csv("/Users/benny_es1/PythonTest/data/match_results.csv")
print("\nmatch_results cols:", mr.columns.tolist())
print(mr[mr.MatchID == ro.match_id.iloc[0]].to_string())
sub = ro[ro.match_id == ro.match_id.iloc[0]]
print(sub.groupby(["map_num", "map_name"]).size())
