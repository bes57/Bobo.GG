"""Validate the fast engine against production timeline ratings: same config
must reproduce winner_before/loser_before closely (non-CN teams; the engine
omits CN intl-shrinkage by design — measured here)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import Engine

eng = Engine()
cfg_prod = {"decay": {"kind": "exp", "hl": 6.0},
            "rd": {"power": 0.5, "scale": 2.5},
            "roster_mode": "year", "roster_persistence": 0.3,
            "ridge": 0.5, "champ_mult": 2.0, "beta": 0.170}
out = eng.run(cfg_prod)

s = eng.series
sys.path.insert(0, "/Users/benny_es1/VCTMM")
from vctmm.benpom.teams import ORG_REGIONS  # noqa: E402

cn_w = s.winner.map(lambda o: ORG_REGIONS.get(o) == "CN")
cn_l = s.loser.map(lambda o: ORG_REGIONS.get(o) == "CN")
non_cn = (~cn_w & ~cn_l).values
valid = ~np.isnan(out["rat_w"])

for label, mask in [("non-CN", non_cn & valid), ("CN-involved", (~non_cn) & valid)]:
    dw = out["rat_w"][mask] - s.r_w.values[mask]
    dl = out["rat_l"][mask] - s.r_l.values[mask]
    d = np.concatenate([dw, dl])
    print(f"{label}: n={mask.sum()}  mean|diff|={np.abs(d).mean():.4f}  "
          f"p90|diff|={np.quantile(np.abs(d), .9):.4f}  max={np.abs(d).max():.4f}")

# correlation of rating diffs
rd_eng = out["rat_w"] - out["rat_l"]
rd_tl = (s.r_w - s.r_l).values
m = valid & non_cn
print(f"corr(rating-diff engine vs timeline, non-CN): "
      f"{np.corrcoef(rd_eng[m], rd_tl[m])[0,1]:.5f}")

print(f"\nengine prod-config: beta={out['beta']} ll_test={out['ll_test']} "
      f"brier_test={out['brier_test']} (n={out['n_test']})")

# production surface scored on the same test split for reference
from harness import BETA_LIVE, intl_attendance_asof, predict, summarize  # noqa: E402
att = intl_attendance_asof(s)
p_prod = predict(s, beta=BETA_LIVE, gating="backend", attendance=att)
test = out["test_mask"]
print("timeline-prod same split:", summarize(p_prod[test], "prod timeline"))
