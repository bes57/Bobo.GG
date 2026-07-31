"""Re-validate everything on the rebuilt production data (EWC-class events now
native): baseline vs key decay ladder vs the full v3 stack; fine-tune the
stack's knobs on the complete data; Kalshi overlap v5 with native joins.
Writes out/revalidate.json."""
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

eng = Engine()  # no patch dir — data is native now
s = eng.series.reset_index(drop=True)
print(f"series (native): {len(s)}  2026: {(s.year==2026).sum()}")
fmts = s.fmt.values
train_v = (s.date <= "2024-12-31").values
test_v = (s.date > "2024-12-31").values

stage_by_mid = dict(zip(s.match_id, s.stage))
g_stage = np.array([stage_by_mid.get(g["match_id"], "groups") for g in eng.games])
g_event = np.array([g["event_id"] for g in eng.games])
res = {"n_series": len(s), "n_2026": int((s.year == 2026).sum())}


def series_pv(b, rdv, mask):
    pm = 1 / (1 + np.exp(-b * rdv[mask]))
    fm = fmts[mask]
    return np.where(np.isin(fm, ("bo5", "bo5_gf")),
                    pm ** 3 * (1 + 3 * (1 - pm) + 6 * (1 - pm) ** 2),
                    np.where(fm == "bo1", pm, pm ** 2 * (3 - 2 * pm)))


def nll(b, rdv, mask):
    return -np.mean(np.log(np.clip(series_pv(b, rdv, mask), 1e-9, 1)))


def fit_score(name, rdv):
    v = ~np.isnan(rdv)
    b = float(minimize_scalar(lambda x: nll(x, rdv, v & train_v),
                              bounds=(0.02, 0.6), method="bounded").x)
    r = {"beta": round(b, 4), "ll_test": round(float(nll(b, rdv, v & test_v)), 5)}
    res[name] = r
    print(f"{name:<30} b={b:.3f} ll={r['ll_test']:.5f}", flush=True)
    return rdv, b, v


PO = np.where(np.isin(g_stage, ("playoffs", "grand_final")), 1.6, 1.0)
BASE = {"rd": {"power": 0.75, "scale": 2.5}, "roster_mode": "year",
        "roster_persistence": 0.3, "ridge": 0.5, "champ_mult": 2.0}

# ladder re-check on native data
rd_p, b_p, v_p = fit_score("prod_hl6_pow05", eng.run(
    {**BASE, "decay": {"kind": "exp", "hl": 6.0},
     "rd": {"power": 0.5, "scale": 2.5}})["rdiff"])
fit_score("hl13_pow075", eng.run(
    {**BASE, "decay": {"kind": "exp", "hl": 13.0}})["rdiff"])
fit_score("games16", eng.run(
    {**BASE, "decay": {"kind": "games", "hl_games": 16.0}})["rdiff"])
out_asym = eng.run({**BASE, "decay": {"kind": "games", "hl_games": 20.0,
                                      "hl_games_loss": 12.0},
                    "w_custom": PO, "daily_out": True})
rd_a, b_a, v_a = fit_score("asym_w20l12_po16", out_asym["rdiff"])

# fine-tune asym knobs on complete data (train-fit, test-scored)
for hw, hlz in ((18.0, 11.0), (22.0, 13.0), (20.0, 10.0), (24.0, 14.0)):
    fit_score(f"asym_w{int(hw)}l{int(hlz)}", eng.run(
        {**BASE, "decay": {"kind": "games", "hl_games": hw, "hl_games_loss": hlz},
         "w_custom": PO})["rdiff"])
for pw in (1.4, 1.8):
    POx = np.where(np.isin(g_stage, ("playoffs", "grand_final")), pw, 1.0)
    fit_score(f"asym_po{pw}", eng.run(
        {**BASE, "decay": {"kind": "games", "hl_games": 20.0, "hl_games_loss": 12.0},
         "w_custom": POx})["rdiff"])
# EWC ablation: zero-weight the new events (does their inclusion help natively?)
EWC0 = PO * np.where(np.isin(g_event, ("2026_ewc", "2026_ewc_qual",
                                       "2026_china_evo_2")), 0.0, 1.0)
fit_score("asym_no_ewc_events", eng.run(
    {**BASE, "decay": {"kind": "games", "hl_games": 20.0, "hl_games_loss": 12.0},
     "w_custom": EWC0})["rdiff"])

# full stack (candidate v4-native): asym + po1.6 + coldstart + xregion
rd = out_asym["rdiff"].copy()
daily = out_asym["daily_r"]
days_sorted = sorted(daily.keys())
first_game = {org: min(eng.g_date[i] for i in rows_)
              for org, rows_ in eng.team_game_rows.items()}
for i, row in enumerate(s.itertuples(index=False)):
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
                if ORG_REGIONS.get(t) == reg and first_game.get(t, "9999") < row.date]
        if len(regs) >= 6:
            rd[i] += sign * float(np.percentile(regs, 25))
REGS = ["Americas", "EMEA", "Pacific", "CN"]
cross = (s.reg_w != s.reg_l).values
iw = s.reg_w.map({r_: i for i, r_ in enumerate(REGS)}).fillna(0).astype(int).values
il = s.reg_l.map({r_: i for i, r_ in enumerate(REGS)}).fillna(0).astype(int).values
vX = ~np.isnan(rd)
b0 = float(minimize_scalar(lambda x: nll(x, rd, vX & train_v),
                           bounds=(0.02, 0.6), method="bounded").x)
dmap = {}
for mo in sorted({d[:7] for d in s.date}):
    hist = vX & cross & (s.date < mo + "-01").values
    if hist.sum() < 60:
        dmap[mo] = np.zeros(4)
        continue

    def nd(d3):
        d4 = np.append(d3, 0.0)
        adj = rd.copy()
        adj[hist] = rd[hist] + d4[iw[hist]] - d4[il[hist]]
        return nll(b0, adj, hist)
    dmap[mo] = np.append(minimize(nd, np.zeros(3), method="Nelder-Mead").x, 0.0)
rd4 = rd.copy()
for i in np.where(vX & cross)[0]:
    d4 = dmap[s.date.iloc[i][:7]]
    rd4[i] = rd[i] + d4[iw[i]] - d4[il[i]]
rd4v, b4, v4m = fit_score("V4_NATIVE_full_stack", rd4)
res["boot_v4_vs_prod"] = paired_bootstrap(
    series_pv(b4, rd4, v4m & v_p & test_v), series_pv(b_p, rd_p, v4m & v_p & test_v))
print("boot v4-native vs prod:", res["boot_v4_vs_prod"])
np.save(os.path.join(OUT, "rd_v4_native.npy"), rd4)
json.dump({"beta": b4}, open(os.path.join(OUT, "v4_native_beta.json"), "w"))

# ── Kalshi overlap v5 (native joins) ─────────────────────────────────────────
mt = json.load(open(os.path.join(HERE, "..", "data", "match_times.json")))
try:
    mt.update(json.load(open(os.path.join(HERE, "data_patch", "patch_times.json"))))
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
s26 = s[s.year == 2026].copy()
s26["pair"] = [frozenset((a, b)) for a, b in zip(s26.winner, s26.loser)]
s26["d"] = pd.to_datetime(s26.date)
used, joins = set(), []
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
print(f"native joins: {len(joins)}")


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


sidx = {m_: i for i, m_ in enumerate(s.match_id.values)}
rows = []
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
    if np.isnan(rd4[i]):
        continue
    p_model = float(series_pv(b4, rd4, np.arange(len(s)) == i)[0])
    pk = pa if kr.winner_org == kr.org_a else 1 - pa
    rows.append({"event_ticker": kr.event_ticker, "date": sr.date,
                 "match_id": sr.match_id, "event_id": sr.event_id,
                 "winner": sr.winner, "loser": sr.loser,
                 "vct": not sr.event_id.startswith(("2026_ewc", "2026_china_evo")),
                 "p_model": p_model, "pk_pre": min(max(pk, 0.01), 0.99)})
m = pd.DataFrame(rows)


def ll(x):
    return float(-np.mean(np.log(np.clip(x, 1e-9, 1))))


res["overlap_all"] = {"n": len(m), "model": round(ll(m.p_model.values), 5),
                      "kalshi": round(ll(m.pk_pre.values), 5),
                      **paired_bootstrap(m.p_model.values, m.pk_pre.values)}
vct = m[m.vct]
res["overlap_vct"] = {"n": len(vct), "model": round(ll(vct.p_model.values), 5),
                      "kalshi": round(ll(vct.pk_pre.values), 5),
                      **paired_bootstrap(vct.p_model.values, vct.pk_pre.values)}
s2v = vct[vct.date >= "2026-07-01"]
res["overlap_vct_stage2"] = {"n": len(s2v), "model": round(ll(s2v.p_model.values), 5),
                             "kalshi": round(ll(s2v.pk_pre.values), 5)}
print("OVERLAP all:", res["overlap_all"])
print("OVERLAP vct:", res["overlap_vct"])
print("OVERLAP vct stage2:", res["overlap_vct_stage2"])
m.to_csv(os.path.join(OUT, "kalshi_joined5.csv"), index=False)

with open(os.path.join(OUT, "revalidate.json"), "w") as f:
    json.dump(res, f, indent=1, default=str)
print("saved out/revalidate.json")
