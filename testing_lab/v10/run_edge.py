"""Edge Lab — is there a real BenPom edge over Kalshi, and how sure can we be?

The prior read showed +34.7% ROI at a >=5c edge, but 91% of those bets were on
the underdog. That raises one question above all others:

    Did the MODEL pick good underdogs, or were underdogs simply cheap?

Everything here is built to separate those. The controls are:

  blind_dog   back EVERY underdog at the market price, ignoring the model.
              If this returns the same as the model-selected bets, the model
              contributed nothing and the "edge" is a market-wide bias.
  blind_fav   the mirror, for completeness.
  sharpened   the model recalibrated to the market's confidence level. If the
              edge is only under-confidence, sharpening destroys it.

Reporting only. Market data is never a fitting target or selection signal.
Writes testing_lab/v10/stats/v10_edge.json.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from engine import Engine   # noqa: E402

OUT = os.path.join(HERE, "stats")
V6 = {"decay": {"kind": "games", "consistency": (20.0, 12.0)},
      "rd": {"power": 0.75, "scale": 2.5}, "roster_mode": "year",
      "roster_persistence": 0.3, "ridge": 0.5, "champ_mult": 2.0,
      "region_prior_ridge": 1.5}
RNG = np.random.default_rng(20260812)

eng = Engine()
s = eng.series.reset_index(drop=True)
rdiff = eng.run(dict(V6))["rdiff"]
fmts, FIT1 = s.fmt.values, (s.date <= "2024-12-31").values
ok = np.isfinite(rdiff)


def p_vec(b, m):
    pm = 1 / (1 + np.exp(-b * rdiff[m])); fm = fmts[m]
    return np.where(np.isin(fm, ("bo5", "bo5_gf")),
                    pm ** 3 * (1 + 3 * (1 - pm) + 6 * (1 - pm) ** 2),
                    np.where(fm == "bo1", pm, pm ** 2 * (3 - 2 * pm)))


beta = float(minimize_scalar(
    lambda b: -np.mean(np.log(np.clip(p_vec(b, ok & FIT1), 1e-9, 1))),
    bounds=(0.001, 1.0), method="bounded").x)
pw = np.full(len(s), np.nan); pw[ok] = p_vec(beta, ok)
s = s.assign(pb=pw)

k = pd.read_csv(os.path.join(os.path.dirname(HERE), "kalshi", "kalshi_matches.csv"))
k = k[(k.excluded != True) & k.winner_org.notna()].copy()   # noqa: E712
k["pair"] = [frozenset((a, b)) for a, b in zip(k.org_a, k.org_b)]
k["d"] = pd.to_datetime(k.date_utc.str[:10])
s["pair"] = [frozenset((a, b)) for a, b in zip(s.winner, s.loser)]
s["d"] = pd.to_datetime(s.date)

rows = []
for _, kr in k.iterrows():
    c = s[(s.pair == kr["pair"]) & (abs((s.d - kr.d).dt.days) <= 1) & s.pb.notna()]
    if not len(c):
        continue
    sr = c.iloc[0]

    def wp(pa):
        return np.nan if pd.isna(pa) else (pa if kr.winner_org == kr.org_a else 1.0 - pa)
    rows.append({"date": sr.date, "fmt": sr.fmt, "intl": bool(sr.intl),
                 "event_id": sr.event_id, "vol": float(kr.volume_total),
                 "pb": float(sr.pb), "pk": wp(kr.prob_a_t2h)})
m = pd.DataFrame(rows).dropna(subset=["pk"])
m["pk"] = m.pk.clip(0.01, 0.99)
print(f"n = {len(m)}   {m.date.min()} .. {m.date.max()}")

pb, pk = m.pb.values, m.pk.values


def roi(cost, won, label, n_boot=4000):
    """cost/won are per-bet arrays, winner-referenced already resolved."""
    if len(cost) == 0:
        return {"label": label, "n": 0}
    prof = np.where(won, 100 - cost, -cost)
    idx = RNG.integers(0, len(cost), size=(n_boot, len(cost)))
    bs = 100 * prof[idx].sum(1) / cost[idx].sum(1)
    return {"label": label, "n": int(len(cost)),
            "hit_pct": round(float(100 * won.mean()), 1),
            "mean_cost": round(float(cost.mean()), 1),
            "roi_pct": round(float(100 * prof.sum() / cost.sum()), 2),
            "ci95": [round(float(np.percentile(bs, 2.5)), 2),
                     round(float(np.percentile(bs, 97.5)), 2)],
            "p_profit": round(float((bs > 0).mean()), 3)}


res = {"n": int(len(m)), "beta": round(beta, 6),
       "window": [str(m.date.min()), str(m.date.max())]}

# ── 1. THE CONTROL: back every underdog, model ignored ───────────────────────
dog = pk < 0.5                     # the eventual winner was the market underdog
cost_blind = np.where(dog, pk * 100, (1 - pk) * 100)
won_blind = dog                    # backing the dog wins iff the dog won
res["blind_dog"] = roi(cost_blind, won_blind, "back every underdog (no model)")
res["blind_fav"] = roi(np.where(~dog, pk * 100, (1 - pk) * 100), ~dog,
                       "back every favourite (no model)")

# ── 2. the model's own >=5c bets, split by which side ────────────────────────
edge = (pb - pk) * 100
sel = np.abs(edge) >= 5
cost_m = np.where(edge >= 5, pk * 100, (1 - pk) * 100)[sel]
won_m = (edge >= 5)[sel]
res["model_5c"] = roi(cost_m, won_m, "model, >=5c edge")

took_dog = ((edge >= 5) & dog) | ((edge <= -5) & ~dog)
sel_dog = sel & took_dog
res["model_dog_only"] = roi(np.where(edge >= 5, pk * 100, (1 - pk) * 100)[sel_dog],
                            (edge >= 5)[sel_dog], "model, >=5c, underdog side only")

# ── 3. does the model SELECT better underdogs than blind? ────────────────────
# same universe (underdog bets), model-selected subset vs everything else
dog_all = np.where(dog, pk * 100, (1 - pk) * 100)
dog_won = dog
picked = sel_dog
res["dog_picked_by_model"] = roi(dog_all[picked], dog_won[picked],
                                 "underdogs the model liked")
res["dog_not_picked"] = roi(dog_all[~picked], dog_won[~picked],
                            "underdogs the model did NOT like")

# ── 4. sharpening: recalibrate the model to the market's confidence ──────────
# logit-scale pb by a single slope fitted so its mean confidence matches the
# market's. If the "edge" is only under-confidence, this kills it.
lg = np.log(np.clip(pb, 1e-6, 1 - 1e-6) / (1 - np.clip(pb, 1e-6, 1 - 1e-6)))


def sharp(a):
    return 1 / (1 + np.exp(-a * lg))


a_star = float(minimize_scalar(
    lambda a: (np.mean(sharp(a)) - pk.mean()) ** 2,
    bounds=(0.2, 6.0), method="bounded").x)
pb_s = sharp(a_star)
e_s = (pb_s - pk) * 100
sel_s = np.abs(e_s) >= 5
res["sharpened"] = roi(np.where(e_s >= 5, pk * 100, (1 - pk) * 100)[sel_s],
                       (e_s >= 5)[sel_s], f"model sharpened x{a_star:.2f}, >=5c")
res["sharpen_slope"] = round(a_star, 3)
res["mean_conf"] = {"benpom": round(float(pb.mean()), 4),
                    "benpom_sharpened": round(float(pb_s.mean()), 4),
                    "kalshi": round(float(pk.mean()), 4)}

# ── 5. discrimination: who actually ranks matchups better (AUC) ──────────────
def auc(p):
    # winner-referenced: AUC vs a coin, = P(p > 1-p) with ties at .5
    return float(np.mean((p > 0.5) + 0.5 * (p == 0.5)))


res["discrimination"] = {"benpom_acc": round(100 * auc(pb), 1),
                         "kalshi_acc": round(100 * auc(pk), 1),
                         "benpom_logloss": round(float(-np.mean(np.log(pb))), 5),
                         "kalshi_logloss": round(float(-np.mean(np.log(pk))), 5)}

# ── 6. segments (reported WITH the multiple-comparison caveat) ───────────────
segs = {}
segs["Bo3"] = m.fmt == "bo3"
segs["Bo5"] = m.fmt.isin(["bo5", "bo5_gf"])
segs["international"] = m.intl
segs["domestic"] = ~m.intl
segs["high volume"] = m.vol >= m.vol.median()
segs["low volume"] = m.vol < m.vol.median()
segs["price 1-20c"] = (np.minimum(pk, 1 - pk) * 100 <= 20)
segs["price 20-40c"] = ((np.minimum(pk, 1 - pk) * 100 > 20) &
                        (np.minimum(pk, 1 - pk) * 100 <= 40))
segs["price 40-50c"] = (np.minimum(pk, 1 - pk) * 100 > 40)
res["segments"] = []
for name, msk in segs.items():
    mm = msk.values if hasattr(msk, "values") else msk
    ss = sel & mm
    if ss.sum() >= 12:
        r = roi(np.where(edge >= 5, pk * 100, (1 - pk) * 100)[ss],
                (edge >= 5)[ss], name)
        res["segments"].append(r)

# ── 7. how sure can we be? power at this sample size ─────────────────────────
# null: bets are fair coins at their price. simulate ROI under the null.
null = []
cost_sel = np.where(edge >= 5, pk * 100, (1 - pk) * 100)[sel]
p_fair = cost_sel / 100.0
for _ in range(4000):
    w = RNG.random(len(cost_sel)) < p_fair
    null.append(100 * np.where(w, 100 - cost_sel, -cost_sel).sum() / cost_sel.sum())
null = np.array(null)
res["null_test"] = {
    "n_bets": int(sel.sum()),
    "null_roi_sd": round(float(null.std()), 2),
    "null_roi_p95": round(float(np.percentile(null, 95)), 2),
    "observed_roi": res["model_5c"]["roi_pct"],
    "p_value_one_sided": round(float((null >= res["model_5c"]["roi_pct"]).mean()), 4),
    "mde_roi_at_n": round(float(1.645 * null.std()), 2)}

with open(os.path.join(OUT, "v10_edge.json"), "w") as f:
    json.dump(res, f, indent=1)

print("\n== THE CONTROL ==")
for kk in ("blind_dog", "blind_fav", "model_5c", "model_dog_only"):
    r = res[kk]
    print(f"  {r['label']:42} n={r['n']:3}  hit {r['hit_pct']:5.1f}%  "
          f"ROI {r['roi_pct']:+7.2f}%  CI [{r['ci95'][0]:+.0f},{r['ci95'][1]:+.0f}]")
print("\n== DOES THE MODEL SELECT BETTER UNDERDOGS? ==")
for kk in ("dog_picked_by_model", "dog_not_picked"):
    r = res[kk]
    print(f"  {r['label']:42} n={r['n']:3}  hit {r['hit_pct']:5.1f}%  ROI {r['roi_pct']:+7.2f}%")
print("\n== SHARPENED (edge from under-confidence only?) ==")
r = res["sharpened"]
print(f"  {r['label']:42} n={r['n']:3}  ROI {r['roi_pct']:+7.2f}%  CI {r['ci95']}")
print(f"  mean confidence: {res['mean_conf']}")
print("\n== DISCRIMINATION ==", res["discrimination"])
print("\n== NULL TEST ==", res["null_test"])
print(f"\nwrote {OUT}/v10_edge.json")
