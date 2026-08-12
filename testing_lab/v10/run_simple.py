"""The operator's spec, exactly as stated:

    Take the YES price an hour or two before the match starts. Bet the side
    where BenPom differs from that price by 5 percentage points or more.
    Assume no fees.

The 5pp threshold is GIVEN, not fitted, so there is no selection problem and
the whole sample is usable. Prices come from prematch_book.csv, which is
rebuilt from raw candles against the real scheduled start -- so unlike the
shipped columns, nothing here is sampled after the match began.

Writes testing_lab/v10/stats/v10_simple.json
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
THRESH = 5.0            # percentage points, GIVEN

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
s = s.assign(pw=pw)
s["pair"] = [frozenset((a, b)) for a, b in zip(s.winner, s.loser)]
s["d"] = pd.to_datetime(s.date)

bk = pd.read_csv(os.path.join(OUT, "prematch_book.csv"))
bk["d"] = pd.to_datetime(bk.start_utc, unit="s").dt.normalize()
bk["pair"] = [frozenset((a, b)) for a, b in zip(bk.org_a, bk.org_b)]

legs = []
for _, r in bk.iterrows():
    c = s[(s.pair == r["pair"]) & (abs((s.d - r.d).dt.days) <= 1) & s.pw.notna()]
    if not len(c):
        continue
    sr = c.iloc[0]
    won = (r.side_org == r.winner_org)
    legs.append({"date": sr.date, "event": r.event_ticker, "side": r.side_org,
                 "won": bool(won),
                 "p_model": float(sr.pw) if won else 1.0 - float(sr.pw),
                 "yes_2h": float(r.mid),
                 "yes_1h": float(r.mid_1h) if pd.notna(r.mid_1h) else np.nan,
                 "ask_2h": float(r.yes_ask), "vol": float(r.volume_total)})
L = pd.DataFrame(legs)


def sim(df, col, thresh=THRESH):
    d = df.dropna(subset=[col])
    edge = (d.p_model * 100) - (d[col] * 100)
    sel = (edge >= thresh).values
    if sel.sum() == 0:
        return None
    cost = d[col].values[sel] * 100          # no fees, per spec
    won = d.won.values[sel]
    prof = np.where(won, 100 - cost, -cost)
    idx = RNG.integers(0, len(cost), size=(4000, len(cost)))
    bs = 100 * prof[idx].sum(1) / cost[idx].sum(1)
    return {"n_bets": int(sel.sum()), "n_available": int(len(d)),
            "hit_pct": round(float(100 * won.mean()), 1),
            "mean_price": round(float(cost.mean()), 1),
            "staked": round(float(cost.sum()), 1),
            "profit": round(float(prof.sum()), 1),
            "roi_pct": round(float(100 * prof.sum() / cost.sum()), 2),
            "ci95": [round(float(np.percentile(bs, 2.5)), 2),
                     round(float(np.percentile(bs, 97.5)), 2)],
            "p_profit": round(float((bs > 0).mean()), 3)}


res = {"spec": "YES price 1-2h pre-match, bet on >=5pp model-vs-market gap, no fees",
       "threshold_pp": THRESH, "given_not_fitted": True,
       "beta": round(beta, 6), "n_matches": int(L.event.nunique()),
       "n_legs": int(len(L)), "window": [str(L.date.min()), str(L.date.max())],
       "T2h": sim(L, "yes_2h"), "T1h": sim(L, "yes_1h"),
       "T2h_paying_ask": sim(L, "ask_2h")}

# stability: split the window in half, same fixed threshold
dates = sorted(L.date.unique())
cut = dates[len(dates) // 2]
res["halves"] = {"first": sim(L[L.date <= cut], "yes_2h"),
                 "second": sim(L[L.date > cut], "yes_2h"), "split": str(cut)}
# by month, same fixed rule
res["by_month"] = []
for mo in sorted({d[:7] for d in L.date}):
    r = sim(L[L.date.str[:7] == mo], "yes_2h")
    if r:
        res["by_month"].append({"month": mo, **r})
# is the model or the price doing the work? same rule, underdog side only
d2 = L.dropna(subset=["yes_2h"])
dog = (d2.yes_2h < 0.5).values
edge = (d2.p_model * 100 - d2.yes_2h * 100).values
selall = edge >= THRESH
for lab, m in (("on_underdogs", selall & dog), ("on_favourites", selall & ~dog)):
    if m.sum():
        cost = d2.yes_2h.values[m] * 100
        won = d2.won.values[m]
        prof = np.where(won, 100 - cost, -cost)
        res[lab] = {"n": int(m.sum()), "hit_pct": round(float(100 * won.mean()), 1),
                    "roi_pct": round(float(100 * prof.sum() / cost.sum()), 2)}
# control: back every underdog at the same prices, model ignored.
# One BET per underdog LEG: you buy that side's YES at its own price and it
# pays iff that side actually won. (An earlier draft set won = dog, which
# asserts every underdog wins -- nonsense, and it produced a fake +43%.)
cost = d2.yes_2h.values[dog] * 100
won_c = d2.won.values[dog]
prof = np.where(won_c, 100 - cost, -cost)
res["control_blind_underdog"] = {
    "n": int(dog.sum()),
    "hit_pct": round(float(100 * won_c.mean()), 1),
    "mean_price": round(float(cost.mean()), 1),
    "roi_pct": round(float(100 * prof.sum() / cost.sum()), 2)}
# and the mirror, backing every favourite
fav = ~dog
cf = d2.yes_2h.values[fav] * 100
wf = d2.won.values[fav]
pf = np.where(wf, 100 - cf, -cf)
res["control_blind_favourite"] = {
    "n": int(fav.sum()),
    "hit_pct": round(float(100 * wf.mean()), 1),
    "roi_pct": round(float(100 * pf.sum() / cf.sum()), 2)}

with open(os.path.join(OUT, "v10_simple.json"), "w") as f:
    json.dump(res, f, indent=1)

print(f"matches {res['n_matches']}  legs {res['n_legs']}  "
      f"{res['window'][0]} .. {res['window'][1]}\n")
for k in ("T2h", "T1h", "T2h_paying_ask"):
    r = res[k]
    print(f"  {k:15} n={r['n_bets']:3}  hit {r['hit_pct']:5.1f}%  "
          f"avg price {r['mean_price']:4.1f}c  ROI {r['roi_pct']:+7.2f}%  "
          f"CI [{r['ci95'][0]:+.0f},{r['ci95'][1]:+.0f}]  p {r['p_profit']}")
print("\n  halves (same fixed 5pp rule):")
for k in ("first", "second"):
    r = res["halves"][k]
    print(f"    {k:7} n={r['n_bets']:3}  ROI {r['roi_pct']:+7.2f}%  CI [{r['ci95'][0]:+.0f},{r['ci95'][1]:+.0f}]")
print("\n  by month:")
for r in res["by_month"]:
    print(f"    {r['month']}  n={r['n_bets']:3}  ROI {r['roi_pct']:+7.2f}%")
print(f"\n  bets on underdogs : {res.get('on_underdogs')}")
print(f"  bets on favourites: {res.get('on_favourites')}")
c=res['control_blind_underdog']; fv=res['control_blind_favourite']
print(f"  CONTROL back every underdog : n={c['n']:3} hit {c['hit_pct']:4.1f}% "
      f"avg {c['mean_price']:.1f}c  ROI {c['roi_pct']:+.2f}%")
print(f"  CONTROL back every favourite: n={fv['n']:3} hit {fv['hit_pct']:4.1f}% "
      f"ROI {fv['roi_pct']:+.2f}%")
print(f"\nwrote {OUT}/v10_simple.json")
