"""Kalshi vs BenPom head-to-head on the 2026 overlap window.

Joins testing_lab/kalshi/kalshi_matches.csv (tier-1 events, T-2h + close +
vwap market probs) to the walk-forward series dataset (production BenPom
probs). Scores log-loss both ways, fits the optimal logit blend, and ranks
the biggest divergences for manual taxonomy. Writes out/kalshi_compare.json."""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from harness import (BETA_LIVE, brier, intl_attendance_asof, load_series,
                     logloss, paired_bootstrap, predict)

OUT = os.path.join(HERE, "out")

k = pd.read_csv(os.path.join(HERE, "kalshi", "kalshi_matches.csv"))
k = k[(k.excluded != True) & k.winner_org.notna()].copy()  # noqa: E712
print(f"kalshi tier-1 events with winner: {len(k)}")

s = load_series()
att = intl_attendance_asof(s)
s["p_prod"] = predict(s, beta=BETA_LIVE, gating="backend", attendance=att)
s26 = s[s.year == 2026].copy()
s26["pair"] = [frozenset((a, b)) for a, b in zip(s26.winner, s26.loser)]
s26["d"] = pd.to_datetime(s26.date)

k["pair"] = [frozenset((a, b)) for a, b in zip(k.org_a, k.org_b)]
k["d"] = pd.to_datetime(k.date_utc.str[:10])

rows = []
for _, kr in k.iterrows():
    cand = s26[(s26.pair == kr["pair"]) &
               (abs((s26.d - kr.d).dt.days) <= 1)]
    if len(cand) == 0:
        continue
    sr = cand.iloc[0]
    # market prob of the eventual WINNER (kalshi row is org_a-referenced)
    def wprob(p_a):
        if pd.isna(p_a):
            return np.nan
        return p_a if kr.winner_org == kr.org_a else 1.0 - p_a
    rows.append({
        "event_ticker": kr.event_ticker, "date": sr.date,
        "match_id": sr.match_id, "event_id": sr.event_id,
        "winner": sr.winner, "loser": sr.loser, "fmt": sr.fmt,
        "stage": sr.stage, "intl": bool(sr.intl),
        "p_benpom": float(sr.p_prod),
        "pk_t2h": wprob(kr.prob_a_t2h),
        "pk_close": wprob(kr.prob_a_close),
        "pk_vwap": wprob(kr.prob_a_vwap),
        "volume": float(kr.volume_total),
    })
m = pd.DataFrame(rows)
print(f"joined: {len(m)} events")
m = m.dropna(subset=["pk_t2h"])
res = {"n_joined": len(m)}

# clip market probs away from 0/1 (1c tick floor)
for c in ("pk_t2h", "pk_close", "pk_vwap"):
    m[c] = m[c].clip(0.01, 0.99)

res["benpom"] = {"logloss": round(logloss(m.p_benpom.values), 5),
                 "brier": round(brier(m.p_benpom.values), 5)}
res["kalshi_t2h"] = {"logloss": round(logloss(m.pk_t2h.values), 5),
                     "brier": round(brier(m.pk_t2h.values), 5)}
res["kalshi_vwap"] = {"logloss": round(logloss(m.pk_vwap.values), 5),
                      "brier": round(brier(m.pk_vwap.values), 5)}
res["boot_kalshi_vs_benpom"] = paired_bootstrap(m.pk_t2h.values, m.p_benpom.values)
print("\nBenPom:", res["benpom"])
print("Kalshi t2h:", res["kalshi_t2h"])
print("Kalshi vwap:", res["kalshi_vwap"])
print("boot (kalshi_t2h better?):", res["boot_kalshi_vs_benpom"])

# volume split — is the market only better when liquid?
med = m.volume.median()
for lab, mask in [("high_vol", m.volume >= med), ("low_vol", m.volume < med)]:
    res[f"{lab}"] = {
        "n": int(mask.sum()),
        "benpom_ll": round(logloss(m.p_benpom[mask].values), 5),
        "kalshi_ll": round(logloss(m.pk_t2h[mask].values), 5)}
    print(lab, res[lab])

# favorite-band comparison: does the market price favorites higher?
pb_fav = np.maximum(m.p_benpom, 1 - m.p_benpom)
pk_fav_same_side = np.where(m.p_benpom >= 0.5, m.pk_t2h, 1 - m.pk_t2h)
res["fav_shift"] = []
for lo, hi in [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.01)]:
    msk = (pb_fav >= lo) & (pb_fav < hi)
    if msk.sum():
        res["fav_shift"].append({
            "band": f"[{lo},{hi})", "n": int(msk.sum()),
            "benpom_fav_mean": round(float(pb_fav[msk].mean()), 4),
            "kalshi_same_side_mean": round(float(pk_fav_same_side[msk].mean()), 4),
            "emp_fav_won": round(float((m.p_benpom[msk] >= 0.5).mean()), 4)})
print("\nfav bands (benpom fav side vs kalshi same side vs empirical):")
for r in res["fav_shift"]:
    print(" ", r)

# optimal logit blend weight (in-sample on this window — diagnostic only)
from scipy.optimize import minimize_scalar
lb_ = np.log(np.clip(m.p_benpom, 1e-6, 1 - 1e-6) / (1 - np.clip(m.p_benpom, 1e-6, 1 - 1e-6)))
lk_ = np.log(m.pk_t2h / (1 - m.pk_t2h))


def bl_ll(w):
    p = 1 / (1 + np.exp(-(w * lb_ + (1 - w) * lk_)))
    return -np.mean(np.log(np.clip(p, 1e-9, 1)))


r = minimize_scalar(bl_ll, bounds=(0.0, 1.0), method="bounded")
res["blend_w_benpom"] = round(float(r.x), 4)
res["blend_ll"] = round(float(bl_ll(r.x)), 5)
print(f"\noptimal blend weight on BenPom: {res['blend_w_benpom']} "
      f"(ll {res['blend_ll']} vs benpom {res['benpom']['logloss']} "
      f"kalshi {res['kalshi_t2h']['logloss']})")

# biggest divergences for taxonomy
m["div"] = (m.p_benpom - m.pk_t2h).abs()
top = m.nlargest(20, "div")[["date", "winner", "loser", "fmt", "stage",
                             "event_id", "p_benpom", "pk_t2h", "div", "volume"]]
res["top_divergences"] = top.to_dict("records")
print("\nTOP DIVERGENCES (benpom prob vs kalshi t2h, winner-referenced):")
for r_ in res["top_divergences"][:15]:
    print(f"  {r_['date']} {r_['winner']} bt {r_['loser']}: benpom "
          f"{r_['p_benpom']:.2f} kalshi {r_['pk_t2h']:.2f} ({r_['event_id']})")

m.to_csv(os.path.join(OUT, "kalshi_joined.csv"), index=False)
with open(os.path.join(OUT, "kalshi_compare.json"), "w") as f:
    json.dump(res, f, indent=1, default=str)
print("\nsaved out/kalshi_compare.json + kalshi_joined.csv")
