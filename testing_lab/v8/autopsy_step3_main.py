#!/usr/bin/env python3
"""agent:autopsy master analysis — P&L waterfall, fees, calibration, markouts,
config gap, variance. Reads the VM snapshot read-only. Preregistered:
testing_lab/v8/preregister.autopsy.md."""
import sqlite3, json, math, csv, os, sys, glob
from collections import defaultdict
from datetime import datetime, timedelta, timezone

V8 = "/Users/benny_es1/PythonTest/testing_lab/v8"
PT = "/Users/benny_es1/PythonTest"
sys.path.insert(0, f"{PT}/trading_model")
import predict as P6
M6 = P6.load_model()

db = sqlite3.connect(f"file:{V8}/data/vctmm_live.db?mode=ro", uri=True)
db.row_factory = sqlite3.Row
KM = json.load(open(f"{V8}/data/kalshi_markets_meta.json"))

def iso(t): return datetime.fromisoformat(t.replace("Z", "+00:00"))
def wilson(k, n, z=1.96):
    if n == 0: return (None, None)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 4), round(c + h, 4))
def logit(p): p = min(max(p, 1e-9), 1 - 1e-9); return math.log(p / (1 - p))
def sig(x): return 1 / (1 + math.exp(-x))

# ── reference joins ─────────────────────────────────────────────────────────
mkt_team = {r["market_ticker"]: (r["event_ticker"], r["team_code"], r["team_name"])
            for r in db.execute("SELECT market_ticker,event_ticker,team_code,team_name FROM markets")}
tm = {r["kalshi_name"]: r["benpom_org"] for r in db.execute("SELECT kalshi_name,benpom_org FROM team_mappings")}
links = {r["event_ticker"]: dict(r) for r in db.execute("SELECT * FROM match_links")}
ev_mkts = defaultdict(list)
for mt, (ev, tc, tn) in mkt_team.items(): ev_mkts[ev].append(mt)

# vlr format map from series csvs
fmt_by_vlr = {}
for f in glob.glob(f"{PT}/data/series/*.csv"):
    with open(f, encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            mid, sf = row.get("MatchID"), (row.get("SeriesFormat") or "").lower()
            if mid and sf.startswith("bo"): fmt_by_vlr[str(mid)] = sf.split("_")[0]

def org_of(mt):
    ev, tc, tn = mkt_team[mt]
    if tn in tm: return tm[tn]
    lk = links.get(ev)
    if lk:  # match team_code prefix against org codes
        for o in (lk["org_a"], lk["org_b"]):
            if o.upper() == tc.upper(): return o
    return tm.get(tc, tc)

def v6_prob(mt):
    """Frozen-v6 P(this market's team wins the series)."""
    ev = mkt_team[mt][0]; lk = links.get(ev)
    if not lk: return None, None
    me = org_of(mt)
    opp = lk["org_b"] if me == lk["org_a"] else lk["org_a"]
    fmt = fmt_by_vlr.get(str(lk["vlr_match_id"]), "bo3")
    try: return P6.series_probability(M6, me, opp, fmt), fmt
    except Exception: return None, fmt

# ── fills + vintage + fv_at_fill ────────────────────────────────────────────
fills = [dict(r) for r in db.execute(
    "SELECT f.*, m.event_ticker ev FROM fills f JOIN markets m USING(market_ticker) "
    "WHERE f.dry_run=0 ORDER BY f.ts")]
fm = {r["trade_id"]: dict(r) for r in db.execute("SELECT * FROM fill_markouts") if r["trade_id"]}
DRIFT_LO, DRIFT_HI = "2026-07-23 00:00:00", "2026-07-23 14:10:16"
V6_PUSH = "2026-07-22T23:03:53"
def vintage(f):
    mo = fm.get(f["trade_id"])
    if mo and mo["model_version"]:
        v = mo["model_version"]
        gen = v.split("@")[1] if "@" in v else v.replace("T", " ")[:19]
        if v.startswith("benpom-v5") or gen[:19] < "2026-07-22 23:03:53": return "pre_v6"
        if DRIFT_LO <= gen <= DRIFT_HI: return "drifted_vm"
        return "v6_valid" if gen[:19] == "2026-07-22 23:03:53" else "v6_vm_rebuild"
    ts = f["ts"][:19].replace("T", " ")
    if ts < "2026-07-22 23:03:53": return "pre_v6"
    if "2026-07-23 09:41:35" <= ts <= "2026-07-23 14:20:00": return "drifted_vm"
    return "v6_vm_rebuild"
for f in fills:
    f["vin"] = vintage(f)
    mo = fm.get(f["trade_id"])
    f["fv_bot"] = mo["fv_at_fill"] if mo else None
    f["mts"] = mo["mins_to_start"] if mo else None
    f["mid_at_fill"] = mo["mid_at_fill"] if mo else None
    f["mo"] = mo
    p, fmt = v6_prob(f["market_ticker"]); f["fv_v6"] = p; f["fmt"] = fmt

res = {t: KM[t]["result"] for t in KM}
last_px = {t: KM[t]["last_price"] for t in KM}
tape_last = {r["market_ticker"]: r["yes_price_cents"] for r in db.execute(
    "SELECT market_ticker, yes_price_cents FROM tape t WHERE created_ts=(SELECT MAX(created_ts) FROM tape WHERE market_ticker=t.market_ticker)")}

# ── A. P&L decomposition (FIFO pairing per event) ───────────────────────────
ev_pnl = {}
for ev in sorted({f["ev"] for f in fills}):
    efs = [f for f in fills if f["ev"] == ev]
    mts = sorted({f["market_ticker"] for f in efs})
    q = {m: [] for m in ev_mkts[ev]}   # unpaired queues: [qty, price, side_role]
    pairs = 0.0; pair_margin = 0.0; s1c = s2c = 0.0
    for f in efs:
        (s1c, s2c) = (s1c + f["qty"] * f["price_cents"] / 100, s2c) if f["side_role"] == 1 else (s1c, s2c + f["qty"] * f["price_cents"] / 100)
        m = f["market_ticker"]; other = [x for x in ev_mkts[ev] if x != m]
        rem = f["qty"]
        oq = q[other[0]] if other else []
        while rem > 1e-9 and oq:
            h = oq[0]; take = min(rem, h[0])
            pairs += take; pair_margin += take * (100 - f["price_cents"] - h[1]) / 100
            h[0] -= take; rem -= take
            if h[0] <= 1e-9: oq.pop(0)
        if rem > 1e-9: q[m].append([rem, f["price_cents"], f["side_role"]])
    unh_set = unh_set_cost = unh_open = unh_open_cost = mark = 0.0
    unh_detail = []
    for m, lots_ in q.items():
        for (qty, px, sr) in lots_:
            r = res.get(m)
            if r in ("yes", "no"):
                pay = qty * (1.0 if r == "no" else 0.0)
                unh_set += pay - qty * px / 100; unh_set_cost += qty * px / 100
                unh_detail.append((m, sr, qty, px, r))
            else:
                yp = tape_last.get(m, last_px.get(m))
                nv = (100 - yp) if yp is not None else px
                unh_open += qty * (nv - px) / 100; unh_open_cost += qty * px / 100
    settled = all(res.get(m) in ("yes", "no") for m in ev_mkts[ev])
    ev_pnl[ev] = dict(pairs=pairs, pair_margin=pair_margin, unh_settled_pnl=unh_set,
                      unh_settled_cost=unh_set_cost, unh_open_mark=unh_open,
                      unh_open_cost=unh_open_cost, s1_cost=s1c, s2_cost=s2c,
                      settled=settled, n_fills=len(efs), unh_detail=unh_detail)

tot = lambda k: sum(e[k] for e in ev_pnl.values())
settled_evs = {e: v for e, v in ev_pnl.items() if v["settled"]}
realized = tot("pair_margin") + tot("unh_settled_pnl")   # fees = 0, verified
di = db.execute("SELECT ROUND(SUM(COALESCE(settle_pnl_dollars,0)),2) s, COUNT(*) n FROM deploy_index WHERE dry_run=0 AND settled_ts IS NOT NULL").fetchone()
pnl = {
 "window_utc": [fills[0]["ts"], fills[-1]["ts"]],
 "fills": len(fills), "contracts": round(sum(f["qty"] for f in fills), 1),
 "cost_dollars": round(sum(f["qty"] * f["price_cents"] / 100 for f in fills), 2),
 "events": len(ev_pnl), "events_settled": len(settled_evs),
 "waterfall": {
   "locked_pair_margin": round(tot("pair_margin"), 2),
   "locked_pairs": round(tot("pairs"), 1),
   "margin_cents_per_pair": round(100 * tot("pair_margin") / tot("pairs"), 2),
   "unhedged_settled_pnl": round(tot("unh_settled_pnl"), 2),
   "unhedged_settled_cost": round(tot("unh_settled_cost"), 2),
   "fees": 0.0,
   "realized_total": round(realized, 2),
   "open_inventory_mark_pnl": round(tot("unh_open_mark"), 2),
   "open_inventory_cost": round(tot("unh_open_cost"), 2),
   "total_incl_open_mark": round(realized + tot("unh_open_mark"), 2)},
 "identity_check": {
   "deploy_index_settle_pnl_sum": di["s"], "deploy_index_settled_n": di["n"],
   "note": "independent FIFO reconstruction vs bot's own deploy_index settlement records"},
 "per_event_settled": {e: {k: round(v[k], 2) for k in ("pair_margin", "unh_settled_pnl", "s1_cost", "s2_cost")} | {"pairs": round(v["pairs"], 1)} for e, v in sorted(settled_evs.items())},
}
json.dump(pnl, open(f"{V8}/stats/autopsy_pnl.json", "w"), indent=1)
print("PNL:", json.dumps(pnl["waterfall"]))
print("identity:", di["s"], "n=", di["n"])

# ── B. fees ─────────────────────────────────────────────────────────────────
cf_taker = sum(math.ceil(0.07 * (f["price_cents"]/100) * (1 - f["price_cents"]/100) * 100) / 100 * f["qty"] for f in fills)
taker_join = db.execute("SELECT COALESCE(t.taker_side,'missing') s, COUNT(*) n, SUM(f.qty) q FROM fills f LEFT JOIN tape t ON t.trade_id=f.trade_id WHERE f.dry_run=0 GROUP BY 1").fetchall()
fees = {
 "fee_schedule_source": "GET /trade-api/v2/series/KXVALORANTGAME (public, 2026-07-28)",
 "fee_type": "quadratic", "fee_multiplier": 1,
 "formula": "taker: ceil_cents(0.07*C*P*(1-P)); maker: none for this series (no maker-fee field; Kalshi maker fees apply only to designated series)",
 "maker_only_evidence": ["vctmm/kalshi/orders.py post_only=True hardcoded; taker fill raises OrderRejected and halts",
                          "all 719 fills join to resting orders (0 orphans)",
                          "tape taker_side on our fills: " + json.dumps({r["s"]: [r["n"], round(r["q"],1)] for r in taker_join}) + " (taker_side=yes means counterparty took YES against our resting NO bid)"],
 "fees_paid_dollars": 0.0,
 "counterfactual_taker_fees_dollars": round(cf_taker, 2),
 "hedge_margin_vs_cost_stack": "2c hedge margin clears fees trivially (fees=0); real cost stack is adverse selection only — see autopsy_markouts.json",
 "settlement_fees": "none on Kalshi for this series (settlement_value null; no fee fields on settled markets)"}
json.dump(fees, open(f"{V8}/stats/autopsy_fees.json", "w"), indent=1)
print("FEES: paid=0, counterfactual taker=", round(cf_taker, 2))

# ── C. fill-conditional calibration vs unconditional ────────────────────────
def calib(rows, key):  # rows: (q_pred, paid 0/1, weight)
    W = sum(r[2] for r in rows); n = len(rows)
    if not n or W == 0: return {"n": 0}
    qbar = sum(r[0] * r[2] for r in rows) / W
    paid = sum(r[1] * r[2] for r in rows) / W
    k_eff = paid * n
    lo, hi = wilson(k_eff, n)
    return {"n": n, "contracts": round(W, 1), "predicted_no_pay": round(qbar, 4),
            "realized_no_pay": round(paid, 4), "gap": round(paid - qbar, 4),
            "wilson_ci_realized": [lo, hi], "key": key}

sf = [f for f in fills if res.get(f["market_ticker"]) in ("yes", "no")]
def rows_for(fs, fvk):
    out = []
    for f in fs:
        fv = f[fvk]
        if fv is None: continue
        out.append((1 - fv, 1.0 if res[f["market_ticker"]] == "no" else 0.0, f["qty"]))
    return out

fill_calib = {"settled_fills": len(sf), "unsettled_fills": len(fills) - len(sf),
              "vintage_counts": {v: sum(1 for f in fills if f["vin"] == v) for v in ("pre_v6", "v6_valid", "drifted_vm", "v6_vm_rebuild")},
              "by_model": {}}
for fvk, name in (("fv_bot", "bot_believed_fv"), ("fv_v6", "frozen_v6")):
    fc = {"all": calib(rows_for(sf, fvk), "all")}
    for sr in (1, 2):
        fc[f"side{sr}"] = calib(rows_for([f for f in sf if f["side_role"] == sr], fvk), f"side{sr}")
    for lo in range(0, 100, 20):
        sub = [f for f in sf if lo <= f["price_cents"] < lo + 20]
        if sub: fc[f"px_{lo}_{lo+20}"] = calib(rows_for(sub, fvk), "px")
    bands = [(0, 120, "lt2h"), (120, 360, "2-6h"), (360, 1440, "6-24h"), (1440, 1e9, "gt24h")]
    for a, b, nm in bands:
        sub = [f for f in sf if f["mts"] is not None and a <= f["mts"] < b]
        if sub: fc[f"mts_{nm}"] = calib(rows_for(sub, fvk), "mts")
    for vin in ("v6_valid", "drifted_vm", "v6_vm_rebuild", "pre_v6"):
        sub = [f for f in sf if f["vin"] == vin]
        if sub: fc[f"vin_{vin}"] = calib(rows_for(sub, fvk), "vin")
    fill_calib["by_model"][name] = fc

# unconditional: frozen v6 on every settled linked event in window (favorite side)
unc_rows = []; unc_miss = 0
for ev, lk in links.items():
    mts_ = ev_mkts.get(ev, [])
    if len(mts_) != 2 or not all(res.get(m) in ("yes", "no") for m in mts_): continue
    p0, _ = v6_prob(mts_[0])
    if p0 is None: unc_miss += 1; continue
    fav = mts_[0] if p0 >= 0.5 else mts_[1]
    pf = p0 if p0 >= 0.5 else 1 - p0
    unc_rows.append((1 - pf, 1.0 if res[fav] == "no" else 0.0, 1.0))  # NO-on-favorite orientation
unc = calib(unc_rows, "uncond_fav_no")
# same orientation stats in win terms
fav_win = 1 - unc["realized_no_pay"] if unc.get("n") else None
fav_pred = 1 - unc["predicted_no_pay"] if unc.get("n") else None
F = fill_calib["by_model"]["frozen_v6"]["all"]["gap"]
Fb = fill_calib["by_model"]["bot_believed_fv"]["all"]["gap"]
U = unc["gap"]
fill_calib["unconditional_frozen_v6"] = unc | {"favorite_pred_win": round(fav_pred, 4), "favorite_real_win": round(fav_win, 4), "events": unc["n"], "unpriceable_events": unc_miss}
fill_calib["adverse_selection"] = {
 "definition": "(realized-predicted NO-pay on filled contracts, frozen v6) - (same on unconditional NO-on-v6-favorite benchmark)",
 "fill_gap_frozen_v6": F, "fill_gap_bot_fv": Fb, "uncond_gap": U,
 "adverse_selection_pts": round(F - U, 4),
 "adverse_selection_cents_per_contract": round(100 * (F - U), 2),
 "orientation_note": "fills are NO buys; negative = filled contracts pay less than model says vs baseline"}
json.dump(fill_calib, open(f"{V8}/stats/autopsy_fill_calib.json", "w"), indent=1)
print("CALIB fill(v6)=", F, "fill(bot)=", Fb, "uncond=", U, "AS=", round(F - U, 4))

# ── D. markouts ─────────────────────────────────────────────────────────────
tape_all = defaultdict(list)
for r in db.execute("SELECT market_ticker,yes_price_cents,created_ts FROM tape ORDER BY created_ts"):
    tape_all[r["market_ticker"]].append((r["created_ts"], r["yes_price_cents"]))
def tape_at(m, when):
    lastp = None
    for ts, p in tape_all.get(m, []):
        if iso(ts) <= when: lastp = p
        else: break
    return lastp
mk_rows = []
for f in fills:
    mo = f["mo"]
    if not mo or mo["mid_at_fill"] is None: continue
    row = {"side": f["side_role"], "px": f["price_cents"], "qty": f["qty"], "mts": mo["mins_to_start"],
           "spread_capture": (100 - mo["mid_at_fill"]) - f["price_cents"]}
    for h in ("5m", "30m", "2h"):
        v = mo[f"mid_{h}"]
        row[f"move_{h}"] = (mo["mid_at_fill"] - v) if v is not None else None
    sm5 = None
    if mo["scheduled_start_utc"]:
        t5 = iso(mo["scheduled_start_utc"]) - timedelta(minutes=5)
        if iso(f["ts"]) < t5:
            p5 = tape_at(f["market_ticker"], t5)
            if p5 is not None: sm5 = mo["mid_at_fill"] - p5
    row["move_startm5"] = sm5
    mk_rows.append(row)
def agg(rows):
    o = {"n_fills": len(rows), "contracts": round(sum(r["qty"] for r in rows), 1)}
    W = sum(r["qty"] for r in rows) or 1
    o["spread_capture_c"] = round(sum(r["spread_capture"] * r["qty"] for r in rows) / W, 2)
    for h in ("5m", "30m", "2h", "startm5"):
        sub = [r for r in rows if r[f"move_{h}"] is not None]
        w = sum(r["qty"] for r in sub)
        o[f"move_{h}_c"] = round(sum(r[f"move_{h}"] * r["qty"] for r in sub) / w, 2) if w else None
        o[f"cov_{h}"] = round(len(sub) / len(rows), 3) if rows else None
        if w: o[f"maker_pnl_{h}_c"] = round(o["spread_capture_c"] + o[f"move_{h}_c"], 2)
    return o
mko = {"coverage": {"fills": len(fills), "with_markout_row": len(mk_rows),
       "note": "fill_markouts written by bot (YES-mid); moves converted to NO-holder terms: move=mid_fill-mid_T, +=good; start-5m from public tape"},
       "all": agg(mk_rows),
       "side1": agg([r for r in mk_rows if r["side"] == 1]),
       "side2": agg([r for r in mk_rows if r["side"] == 2])}
for a, b, nm in [(0, 120, "lt2h"), (120, 360, "2_6h"), (360, 1440, "6_24h"), (1440, 1e9, "gt24h")]:
    sub = [r for r in mk_rows if r["mts"] is not None and a <= r["mts"] < b]
    if sub: mko[f"mts_{nm}"] = agg(sub)
for lo in (0, 20, 40, 60, 80):
    sub = [r for r in mk_rows if lo <= r["px"] < lo + 20]
    if sub: mko[f"px_{lo}_{lo+20}"] = agg(sub)
json.dump(mko, open(f"{V8}/stats/autopsy_markouts.json", "w"), indent=1)
print("MARKOUTS all:", json.dumps(mko["all"]))

# ── E. config gap ───────────────────────────────────────────────────────────
def logit_cap_no(fv, delta):  # max NO price we would pay under logit rule
    return 100 * (1 - sig(logit(fv) + delta))
def cf_filter(delta, fvk):
    keep = drop = 0.0; keep_pnl = drop_pnl = 0.0; kept_fills = 0
    for f in sf:
        if f["side_role"] != 1: continue
        fv = f[fvk]
        if fv is None: continue
        pnl_f = f["qty"] * ((100 if res[f["market_ticker"]] == "no" else 0) - f["price_cents"]) / 100
        if f["price_cents"] <= logit_cap_no(fv, delta):
            keep += f["qty"] * f["price_cents"] / 100; keep_pnl += pnl_f; kept_fills += 1
        else:
            drop += f["qty"] * f["price_cents"] / 100; drop_pnl += pnl_f
    return {"kept_fills": kept_fills, "kept_cost": round(keep, 2), "kept_pnl": round(keep_pnl, 2),
            "kept_roi": round(keep_pnl / keep, 4) if keep else None,
            "dropped_cost": round(drop, 2), "dropped_pnl": round(drop_pnl, 2),
            "dropped_roi": round(drop_pnl / drop, 4) if drop else None}
s1_settled = [f for f in sf if f["side_role"] == 1]
s1_cost = sum(f["qty"] * f["price_cents"] / 100 for f in s1_settled)
s1_pnl = sum(f["qty"] * ((100 if res[f["market_ticker"]] == "no" else 0) - f["price_cents"]) / 100 for f in s1_settled)
# skip-NO<45% pocket
pocket = [f for f in s1_settled if (f["fv_bot"] or f["fv_v6"] or 1) < 0.45]
pk_cost = sum(f["qty"] * f["price_cents"] / 100 for f in pocket)
pk_pnl = sum(f["qty"] * ((100 if res[f["market_ticker"]] == "no" else 0) - f["price_cents"]) / 100 for f in pocket)
inside2h = [f for f in fills if f["mts"] is not None and f["mts"] < 120]
cfg = {
 "live_config_verified": {"source": "ssh cat VM config.toml 2026-07-28 (identical to Mac copy)",
   "hard_min_edge_cents": 5, "min_edge_cents": 1, "hedge_margin_cents": 2,
   "order_expiry_lead_hours": 2, "kelly_fraction": 0.5, "vm_model_rebuild": True,
   "vm_git_provenance": "UNAVAILABLE — /home/vctmm/VCTMM is not a git repo"},
 "side1_settled_actual": {"cost": round(s1_cost, 2), "pnl": round(s1_pnl, 2),
                          "roi": round(s1_pnl / s1_cost, 4)},
 "rows": [
  {"knob": "side-1 min edge", "live": "flat 5c (hard_min_edge_cents=5)",
   "recommended": "logit +0.5..+0.6 (FINDINGS 4; ~14c@50c, 8c@85c)",
   "sim_roi": {"flat_5c": [0.067, [-0.092, 0.228]], "logit_0.6": [0.2911, [0.006, 0.61]], "logit_0.4": [0.1498, [-0.05, 0.364]]},
   "expected_roi_delta_pts": 22.4,
   "counterfactual_on_our_fills": {"logit_0.5_v6": cf_filter(0.5, "fv_v6"), "logit_0.6_v6": cf_filter(0.6, "fv_v6"), "logit_0.6_botfv": cf_filter(0.6, "fv_bot")},
   "implemented": False},
  {"knob": "skip NO on model-p<45% sides", "live": "ABSENT (code-verified: no threshold in quotes.py/engine.py)",
   "recommended": "skip (FINDINGS 4 refinement; pocket -5.7% in sim)",
   "our_pocket": {"fills": len(pocket), "cost": round(pk_cost, 2), "pnl": round(pk_pnl, 2),
                  "roi": round(pk_pnl / pk_cost, 4) if pk_cost else None},
   "implemented": False},
  {"knob": "order expiry", "live": "start-2h", "recommended": "start-2h (Rule 5)",
   "fills_inside_2h": {"n": len(inside2h), "contracts": round(sum(f["qty"] for f in inside2h), 1)},
   "implemented": True},
  {"knob": "hedge margin", "live": "2c", "recommended": "2c must clear fees+adverse; fees=0 verified",
   "realized_margin_c_per_pair": pnl["waterfall"]["margin_cents_per_pair"], "implemented": True},
 ]}
json.dump(cfg, open(f"{V8}/stats/autopsy_config_gap.json", "w"), indent=1)
print("CFG s1 settled roi:", round(s1_pnl / s1_cost, 4), "| logit0.6 cf:", json.dumps(cfg["rows"][0]["counterfactual_on_our_fills"]["logit_0.6_v6"]))
print("pocket<45%:", json.dumps(cfg["rows"][1]["our_pocket"]))

# ── F. variance ─────────────────────────────────────────────────────────────
crn_path = f"{V8}/crn.json"
if os.path.exists(crn_path):
    seed = json.load(open(crn_path)).get("master_seed", 780728); crn_src = "crn.json"
else:
    seed = 780728; crn_src = "FALLBACK 780728 — crn.json absent at run time (power agent had not written it); flagged per prereg"
import random
rng = random.Random(seed)
evs = list(settled_evs.items())
obs = sum(v["pair_margin"] + v["unh_settled_pnl"] for _, v in evs)
costs_s1 = [v["s1_cost"] for _, v in evs]
pnls = [v["pair_margin"] + v["unh_settled_pnl"] for _, v in evs]
mu = sum(pnls) / len(pnls)
h0 = [0.067 * c for c in costs_s1]           # flat-5c sim edge on side-1 stake
B = 10000; worse = 0; sims = []
cent = [p - mu for p in pnls]
for _ in range(B):
    s = 0.0
    for i in range(len(evs)):
        j = rng.randrange(len(evs)); s += cent[j] + h0[i]
    sims.append(s)
    if s <= obs: worse += 1
sims.sort()
var_out = {"crn_source": crn_src, "seed": seed, "B": B, "unit": "settled event (n=%d)" % len(evs),
 "observed_realized": round(obs, 2),
 "H0": "per-event E[pnl] = 6.7% of side-1 cost (flat-5c sim point edge, quote_margin.json)",
 "H0_expected_total": round(sum(h0), 2),
 "p_cum_le_observed": round(worse / B, 4),
 "sim_quantiles": {q: round(sims[int(q * B)], 2) for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
 "verdict": None}
p = var_out["p_cum_le_observed"]
var_out["verdict"] = ("real underperformance (p<0.05)" if p < 0.05 else
                      "not distinguishable from noise (p>=0.10)" if p >= 0.10 else
                      "weak evidence (0.05<=p<0.10)")
json.dump(var_out, open(f"{V8}/stats/autopsy_variance.json", "w"), indent=1)
print("VARIANCE:", json.dumps(var_out, default=str)[:400])
