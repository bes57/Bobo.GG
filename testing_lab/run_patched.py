"""Rebuild v3 with the scraped data patch (late Stage-1 playoffs + EWC) and
re-measure everything: holdout (patch rows excluded from metrics), the Kalshi
overlap (now joinable for patched matches too), EWC weight variants.
Writes out/patched.json."""
import bisect
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "/Users/benny_es1/VCTMM")
from engine import Engine
from harness import paired_bootstrap
from scipy.optimize import minimize, minimize_scalar
from vctmm.benpom.teams import ORG_REGIONS

OUT = os.path.join(HERE, "out")
PATCH = os.path.join(HERE, "data_patch")

eng = Engine(patch_dir=PATCH)

# extend the series table with patch series (for Kalshi joining + ratings use)
ps = pd.read_csv(os.path.join(PATCH, "patch_series.csv"))
s = eng.series.reset_index(drop=True)
add = []
for r in ps.itertuples(index=False):
    wm, lm = map(int, r.series_score.split("-"))
    add.append({"match_id": int(r.match_id), "date": r.date,
                "event_id": r.event_tag, "year": int(r.date[:4]),
                "winner": r.winner_org,
                "loser": r.org_a if r.winner_org == r.org_b else r.org_b,
                "w_maps": wm, "l_maps": lm, "r_w": np.nan, "r_l": np.nan,
                "fmt": r.fmt, "stage": "playoffs" if "stage1" in r.event_tag
                else "other", "match_name": r.event_tag, "n_maps_played": wm + lm,
                "intl": False, "reg_w": ORG_REGIONS.get(r.winner_org, "?"),
                "reg_l": ORG_REGIONS.get(
                    r.org_a if r.winner_org == r.org_b else r.org_b, "?")})
add_df = pd.DataFrame(add)
add_df["patch"] = True
s["patch"] = False
s2 = pd.concat([s, add_df], ignore_index=True).sort_values(
    ["date", "match_id"]).reset_index(drop=True)
eng.series = s2
eng.pred_days = sorted(s2.date.unique())
# rebuild per-day series index used by run()
print(f"series: {len(s)} + {len(add_df)} patch = {len(s2)}")

fmts = s2.fmt.values
patch_row = s2.patch.values
train_v = (s2.date <= "2024-12-31").values & ~patch_row
test_v = (s2.date > "2024-12-31").values & ~patch_row

stage_by_mid = dict(zip(s2.match_id, s2.stage))
g_stage = np.array([stage_by_mid.get(g["match_id"], "groups") for g in eng.games])
g_event = np.array([g["event_id"] for g in eng.games])


def series_pv(b, rdv, mask):
    pm = 1 / (1 + np.exp(-b * rdv[mask]))
    fm = fmts[mask]
    return np.where(np.isin(fm, ("bo5", "bo5_gf")),
                    pm ** 3 * (1 + 3 * (1 - pm) + 6 * (1 - pm) ** 2),
                    np.where(fm == "bo1", pm, pm ** 2 * (3 - 2 * pm)))


def nll(b, rdv, mask):
    return -np.mean(np.log(np.clip(series_pv(b, rdv, mask), 1e-9, 1)))


def v3_run(ewc_w):
    wc = np.where(np.isin(g_stage, ("playoffs", "grand_final")), 1.6, 1.0)
    wc = wc * np.where(g_event == "2026_ewc", ewc_w, 1.0)
    out = eng.run({"decay": {"kind": "games", "hl_games": 20.0,
                             "hl_games_loss": 12.0},
                   "rd": {"power": 0.75, "scale": 2.5}, "roster_mode": "year",
                   "roster_persistence": 0.3, "ridge": 0.5, "champ_mult": 2.0,
                   "w_custom": wc, "daily_out": True})
    return out


res = {}
runs = {}
for ewc_w in (0.0, 0.6, 1.0):
    out = v3_run(ewc_w)
    rd = out["rdiff"]
    # cold-start + x-region layers (same as v3)
    daily = out["daily_r"]
    days_sorted = sorted(daily.keys())
    first_game = {org: min(eng.g_date[i] for i in rows_)
                  for org, rows_ in eng.team_game_rows.items()}
    for i, row in enumerate(s2.itertuples(index=False)):
        if np.isnan(rd[i]):
            continue
        for org, sign in ((row.winner, 1.0), (row.loser, -1.0)):
            if first_game.get(org, "9999") < row.date:
                continue
            j = bisect.bisect_left(days_sorted, row.date) - 1
            if j < 0:
                continue
            rv = daily[days_sorted[j]]
            reg = ORG_REGIONS.get(org)
            regs = [rv[eng.tidx[t]] for t in eng.teams
                    if ORG_REGIONS.get(t) == reg
                    and first_game.get(t, "9999") < row.date]
            if len(regs) >= 6:
                rd[i] += sign * float(np.percentile(regs, 25))
    REGS = ["Americas", "EMEA", "Pacific", "CN"]
    cross = (s2.reg_w != s2.reg_l).values
    iw = s2.reg_w.map({r_: i for i, r_ in enumerate(REGS)}).fillna(0).astype(int).values
    il = s2.reg_l.map({r_: i for i, r_ in enumerate(REGS)}).fillna(0).astype(int).values
    vX = ~np.isnan(rd)
    b0 = float(minimize_scalar(lambda x: nll(x, rd, vX & train_v),
                               bounds=(0.02, 0.6), method="bounded").x)
    dmap = {}
    for mo in sorted({d[:7] for d in s2.date}):
        hist = vX & cross & (s2.date < mo + "-01").values & ~patch_row
        if hist.sum() < 60:
            dmap[mo] = np.zeros(4)
            continue

        def nd(d3):
            d4 = np.append(d3, 0.0)
            adj = rd.copy()
            adj[hist] = rd[hist] + d4[iw[hist]] - d4[il[hist]]
            return nll(b0, adj, hist)
        dmap[mo] = np.append(minimize(nd, np.zeros(3), method="Nelder-Mead").x, 0.0)
    rd3 = rd.copy()
    for i in np.where(vX & cross)[0]:
        d4 = dmap[s2.date.iloc[i][:7]]
        rd3[i] = rd[i] + d4[iw[i]] - d4[il[i]]
    b = float(minimize_scalar(lambda x: nll(x, rd3, vX & train_v),
                              bounds=(0.02, 0.6), method="bounded").x)
    ll_t = float(nll(b, rd3, vX & test_v))
    res[f"ewc{ewc_w}"] = {"beta": round(b, 4), "ll_test": round(ll_t, 5)}
    runs[ewc_w] = (rd3, b)
    print(f"patched v3 (ewc_w={ewc_w}): ll_test={ll_t:.5f} beta={b:.3f}", flush=True)

# ── Kalshi overlap with expanded joins ───────────────────────────────────────
mt = json.load(open(os.path.join(HERE, "..", "data", "match_times.json")))
try:
    mt.update(json.load(open(os.path.join(PATCH, "patch_times.json"))))
except FileNotFoundError:
    pass
k = pd.read_csv(os.path.join(HERE, "kalshi", "kalshi_matches.csv"))
k = k[(k.excluded != True) & k.winner_org.notna()].copy()  # noqa: E712
raw = {}
with open(os.path.join(HERE, "kalshi", "markets_raw.jsonl")) as f:
    for line in f:
        m_ = json.loads(line)
        raw.setdefault(m_["event_ticker"], []).append(m_)
k["pair"] = [frozenset((a, b)) for a, b in zip(k.org_a, k.org_b)]
k["close_dt"] = pd.to_datetime(k.date_utc, utc=True)

s26 = s2[s2.year == 2026].copy()
s26["pair"] = [frozenset((a, b)) for a, b in zip(s26.winner, s26.loser)]
s26["d"] = pd.to_datetime(s26.date)
used = set()
joins = []
for _, kr in k.sort_values("date_utc").iterrows():
    cand = s26[(s26.pair == kr["pair"]) & (~s26.match_id.isin(used))].copy()
    if len(cand) == 0:
        continue
    cand["dd"] = (cand.d - kr.close_dt.tz_localize(None).normalize()).abs()
    cand = cand[cand.dd <= pd.Timedelta(days=1)].sort_values("dd")
    if len(cand) == 0:
        continue
    sr = cand.iloc[0]
    used.add(sr.match_id)
    joins.append((kr, sr))
print(f"joined events now: {len(joins)}")


def yes_mid_at(candles, ts_target, max_lookback_min=240):
    best = None
    for c in candles:
        if c["end_period_ts"] <= ts_target:
            if best is None or c["end_period_ts"] > best["end_period_ts"]:
                best = c
    if best is None or ts_target - best["end_period_ts"] > max_lookback_min * 60:
        return None
    try:
        bid = float(best["yes_bid"]["close_dollars"])
        ask = float(best["yes_ask"]["close_dollars"])
    except Exception:
        bid, ask = 0.0, 1.0
    if (ask - bid) <= 0.30 and not (bid <= 0.0 and ask >= 1.0):
        return (bid + ask) / 2.0
    pr = best.get("price") or {}
    for key in ("close_dollars", "previous_dollars"):
        if pr.get(key):
            return float(pr[key])
    return None


rows = []
sidx = {m: i for i, m in enumerate(s2.match_id.values)}
rd_best, b_best = runs[1.0]
for kr, sr in joins:
    t_real = mt.get(str(sr.match_id))
    if t_real:
        start = datetime.strptime(t_real, "%Y-%m-%d %H:%M:%S").replace(
            tzinfo=timezone.utc)
    else:
        hrs = 5 if sr.fmt in ("bo5", "bo5_gf") else 4
        start = kr.close_dt.to_pydatetime() - timedelta(hours=hrs)
    if (kr.close_dt.to_pydatetime() - start) < timedelta(minutes=30):
        continue
    t0 = int(start.timestamp())
    ps_ = {}
    for mkt in raw.get(kr.event_ticker, []):
        team = mkt.get("no_sub_title", "")
        path = os.path.join(HERE, "kalshi", "candles", f"{mkt['ticker']}.json")
        if not os.path.exists(path):
            continue
        candles = json.load(open(path)).get("candlesticks", [])
        p5 = yes_mid_at(candles, t0 - 300)
        org = kr.org_a if team == kr.team_a_raw else (
            kr.org_b if team == kr.team_b_raw else None)
        if org:
            ps_[org] = p5
    vals = []
    if ps_.get(kr.org_a) is not None:
        vals.append(ps_[kr.org_a])
    if ps_.get(kr.org_b) is not None:
        vals.append(1 - ps_[kr.org_b])
    if not vals:
        continue
    pa = float(np.mean(vals))
    i = sidx[sr.match_id]
    if np.isnan(rd_best[i]):
        continue
    pw_all = series_pv(b_best, rd_best, np.arange(len(s2)) == i)
    p_model = float(pw_all[0])
    pk = pa if kr.winner_org == kr.org_a else 1 - pa
    rows.append({"event_ticker": kr.event_ticker, "date": sr.date,
                 "match_id": sr.match_id, "patch": bool(sr.patch),
                 "winner": sr.winner, "loser": sr.loser,
                 "p_model": p_model, "pk_pre": min(max(pk, 0.01), 0.99)})
m = pd.DataFrame(rows)
print(f"scoreable overlap: {len(m)} (was 86)")


def ll(x):
    return float(-np.mean(np.log(np.clip(x, 1e-9, 1))))


res["overlap"] = {"n": len(m), "model_ll": round(ll(m.p_model.values), 5),
                  "kalshi_ll": round(ll(m.pk_pre.values), 5)}
res["overlap_boot"] = paired_bootstrap(m.p_model.values, m.pk_pre.values)
print("OVERLAP:", res["overlap"])
print("boot (model better than market?):", res["overlap_boot"])

# windows
m["window"] = np.where(m.date < "2026-06-05", "Stage1/lateplayoffs",
                np.where(m.date < "2026-07-01", "London", "Stage2+EWC"))
res["overlap_windows"] = {}
for w, grp in m.groupby("window"):
    res["overlap_windows"][w] = {
        "n": len(grp), "model": round(ll(grp.p_model.values), 5),
        "kalshi": round(ll(grp.pk_pre.values), 5)}
    print(f"  {w:<20} n={len(grp):<3} model {res['overlap_windows'][w]['model']:.4f} "
          f"kalshi {res['overlap_windows'][w]['kalshi']:.4f}")

m.to_csv(os.path.join(OUT, "kalshi_joined4.csv"), index=False)
np.save(os.path.join(OUT, "rd_patched_v3.npy"), rd_best)
s2[["match_id", "date", "patch"]].to_csv(os.path.join(OUT, "series2_index.csv"),
                                         index=False)
with open(os.path.join(OUT, "patched.json"), "w") as f:
    json.dump(res, f, indent=1, default=str)
print("saved out/patched.json")
