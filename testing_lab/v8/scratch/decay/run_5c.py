"""agent:decay — 5c: performance-based form (rd-margin, side-conditional,
player R2.0), refit at the probability layer on the v6 rdiff."""
import json
import math
import os
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.optimize import minimize

SCR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCR)
from decay_lib import Runner, jlog, sp, V8, TL, PROBS  # noqa: E402

sys.path.insert(0, V8)
import referee  # noqa: E402

STATS = os.path.join(V8, "stats")
DATA = "/Users/benny_es1/PythonTest/data"
LN2 = math.log(2)
FAMILY_MDE_WITHIN = 1.773

t0 = time.time()
rn = Runner()
v6 = rn.run_cfg("v6_consist_20_12", rn.lam_arrays("consist"))
p6, rd6 = v6["p"], v6["rdiff"]
s = rn.frame
fmts = rn.fmts
hold = rn.test_v
ev_hold = s.event_id.values
sdates = sorted(set(s.date))
games_sorted = sorted(rn.games, key=lambda g: (g["date_s"], g["match_id"]))
jlog(f"5c runner ready ({time.time()-t0:.0f}s)")


def series_arrays(at, default=0.5):
    w = np.array([at.get((r.winner, r.date), default) for r in s.itertuples(index=False)])
    l = np.array([at.get((r.loser, r.date), default) for r in s.itertuples(index=False)])
    return w, l


def decayed_stat(hl, value_fn, den_gt=3.0):
    """Generic exp-decayed per-team mean over MAPS, as-of each series date
    (stage-2 wr_series machinery, value_fn(g, org) -> contribution)."""
    lam = LN2 / hl
    state = defaultdict(lambda: [0.0, 0.0])
    at = {}
    si = 0
    for g in games_sorted:
        while si < len(sdates) and sdates[si] <= g["date_s"]:
            for t_, (n_, d_) in state.items():
                if d_ > den_gt:
                    at[(t_, sdates[si])] = n_ / d_
            si += 1
        for org in (g["winner"], g["loser"]):
            st = state[org]
            st[0] = st[0] * math.exp(-lam) + value_fn(g, org)
            st[1] = st[1] * math.exp(-lam) + 1.0
    while si < len(sdates):
        for t_, (n_, d_) in state.items():
            if d_ > den_gt:
                at[(t_, sdates[si])] = n_ / d_
        si += 1
    return at


def fit_form(name, dform, mask_covered=None):
    """(beta, b_form) joint Nelder-Mead on train; holdout judge vs v6-alone."""
    v = ~np.isnan(rd6) & ~np.isnan(dform)
    m_tr = v & rn.train_v

    def nll(params, mask):
        b, bf = params
        z = b * rd6[mask] + bf * dform[mask]
        pm = 1 / (1 + np.exp(-z))
        return -np.mean(np.log(np.clip(sp(pm, fmts[mask]), 1e-9, 1)))

    fit = minimize(nll, x0=[0.13, 0.0], args=(m_tr,), method="Nelder-Mead",
                   options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 4000})
    b, bf = fit.x
    from scipy.optimize import minimize_scalar
    nll0 = float(minimize_scalar(lambda x: nll((x, 0.0), m_tr),
                                 bounds=(0.03, 0.6), method="bounded").fun)
    train_gain = (nll0 - float(nll((b, bf), m_tr))) * 1000
    with np.errstate(invalid="ignore"):
        p = sp(1 / (1 + np.exp(-(b * rd6 + bf * dform))), fmts)
    m_te = v & hold
    d = referee.delta_vector(p[m_te], p6[m_te])
    n = int(m_te.sum())
    l6 = referee.per_series_ll(p6[m_te])
    Xc = (l6 - l6.mean()).reshape(-1, 1)
    d_cv = d - Xc @ np.linalg.lstsq(Xc, d, rcond=None)[0]
    b_iid = referee.paired_bootstrap_crn(d, mode="iid")
    b_blk = referee.paired_bootstrap_crn(d, mode="block_event", event_ids=ev_hold[m_te])
    mde_raw = 2.8016 * float(np.std(d, ddof=1)) / np.sqrt(n) * 1000
    mde_cv = 2.8016 * float(np.std(d_cv, ddof=1)) / np.sqrt(n) * 1000
    dm = float(np.mean(d)) * 1000
    if (abs(dm) >= mde_raw and
            ((b_iid["p_better"] >= 0.95 and b_blk["p_better"] >= 0.95) or
             (b_iid["p_better"] <= 0.05 and b_blk["p_better"] <= 0.05))):
        verdict = "WIN" if dm > 0 else "KILL"
    else:
        verdict = "INSIDE NOISE FLOOR"
    roi = referee.expected_roi_of_dll(float(np.mean(d)), p6[m_te])
    np.savez(os.path.join(PROBS, f"{name}.npz"), rdiff=rd6, beta=b, p=p)
    row = {"beta": round(float(b), 4), "b_form": round(float(bf), 4),
           "train_gain_from_form_milli": round(train_gain, 4),
           "form_identified_on_train": bool(train_gain >= 0.01),
           "ll_train": round(float(nll((b, bf), m_tr)), 5),
           "ll_test": round(rn.ll(p, m_te), 5), "n": n,
           "delta_milli_vs_v6": round(dm, 3),
           "family_mde_milli": FAMILY_MDE_WITHIN,
           "pair_mde_raw_milli": round(mde_raw, 3),
           "pair_mde_cv_milli": round(mde_cv, 3),
           "boot_iid": {k: b_iid[k] for k in ("mean_delta", "ci_lo", "ci_hi", "p_better")},
           "boot_block": {k: b_blk[k] for k in ("mean_delta", "ci_lo", "ci_hi", "p_better")},
           "verdict": verdict, "expected_roi_delta": roi["expected_roi_delta"]}
    if mask_covered is not None:
        row["holdout_rows_with_form_defined_both_teams"] = int((mask_covered & m_te).sum())
        row["holdout_coverage_pct"] = round(100 * float((mask_covered & m_te).sum()) / n, 1)
    jlog(f"5c {name}: b_form={row['b_form']:+.4f} ll={row['ll_test']} "
         f"d={dm:+.2f}m ({verdict})")
    return row, p


results = {}

# ── 1. wr-form continuity replication ───────────────────────────────────────
wr16 = decayed_stat(16.0, lambda g, o: 1.0 if g["winner"] == o else 0.0)
w16w, w16l = series_arrays(wr16)
for hs in (3, 5, 8):
    wrS = decayed_stat(float(hs), lambda g, o: 1.0 if g["winner"] == o else 0.0)
    wsw, wsl = series_arrays(wrS)
    dform = (wsw - w16w) - (wsl - w16l)
    results[f"form_wr{hs}"], _ = fit_form(f"form_wr{hs}", dform)
results["continuity_note"] = ("old-frame reference b_form(wr,HL3) = -0.0872 "
                              "(out/v7_stage2.json, v6+form3, delta -0.25m n.s.)")

# ── 2. rd-margin form (PRIMARY) ─────────────────────────────────────────────
def margin_val(g, org):
    rd = g["wr"] - g["lr"]
    m_t = math.copysign(abs(rd) ** 0.75 * 2.5, rd)
    return m_t if g["winner"] == org else -m_t

md16 = decayed_stat(16.0, margin_val)
m16w, m16l = series_arrays(md16, default=0.0)
rd_rows = {}
for hs in (3, 5, 8):
    mdS = decayed_stat(float(hs), margin_val)
    msw, msl = series_arrays(mdS, default=0.0)
    dform = (msw - m16w) - (msl - m16l)
    rd_rows[hs] = fit_form(f"form_rd{hs}", dform)
    results[f"form_rd{hs}"] = rd_rows[hs][0]
best_rd = min((3, 5, 8), key=lambda h: results[f"form_rd{h}"]["ll_train"])
results["form_rd_selected"] = f"form_rd{best_rd} (argmin ll_train)"

# ── 3. side-conditional form (round_outcomes) ───────────────────────────────
ro = pd.read_csv(os.path.join(DATA, "enriched", "round_outcomes.csv"))
side_stats = {}  # (match_id, org) -> per-map dict later; build per (mid, map_num)
per_map = defaultdict(lambda: defaultdict(lambda: [0, 0, 0, 0]))  # (mid,mapnum)->org->[aw,an,dw,dn]
mid_orgs = {}
for g in rn.games:
    mid_orgs.setdefault(g["match_id"], set()).update((g["winner"], g["loser"]))
n_bad_side = 0
for r in ro.itertuples(index=False):
    orgs = mid_orgs.get(r.match_id)
    if not orgs or r.winner_org not in orgs or len(orgs) != 2:
        n_bad_side += 1
        continue
    other = next(o for o in orgs if o != r.winner_org)
    key = (r.match_id, r.map_num)
    ws = str(r.winner_side).lower()
    if ws not in ("attack", "defense"):
        n_bad_side += 1
        continue
    st_w = per_map[key][r.winner_org]
    st_l = per_map[key][other]
    if ws == "attack":
        st_w[0] += 1; st_w[1] += 1          # won on attack
        st_l[2] += 0; st_l[3] += 1          # lost on defense
    else:
        st_w[2] += 1; st_w[3] += 1          # won on defense
        st_l[0] += 0; st_l[1] += 1          # lost on attack
jlog(f"5c side: round rows used={len(ro)-n_bad_side}, dropped={n_bad_side}; "
     f"maps covered={len(per_map)}")

# order per-map side stats into each team's game sequence; decay per played map
def side_form(hl):
    lam = LN2 / hl
    state = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])  # aw,an,dw,dn decayed
    at = {}
    si = 0
    # map_num join: games list has no map_num; walk per (mid) maps in order of
    # round file map_num for the same match — decay steps on EVERY played map
    # from the games list; data added when that (mid) map exists in per_map.
    per_mid = defaultdict(list)
    for (mid, mn), d in per_map.items():
        per_mid[mid].append((mn, d))
    for mid in per_mid:
        per_mid[mid].sort()
    used = defaultdict(int)  # how many maps of this mid consumed
    for g in games_sorted:
        while si < len(sdates) and sdates[si] <= g["date_s"]:
            for t_, st in state.items():
                if st[1] > 12 and st[3] > 12:
                    at[(t_, sdates[si])] = (st[0] / st[1], st[2] / st[3])
            si += 1
        maps_ = per_mid.get(g["match_id"], [])
        k = used[g["match_id"]]
        data = maps_[k][1] if k < len(maps_) else None
        used[g["match_id"]] += 1
        for org in (g["winner"], g["loser"]):
            st = state[org]
            for j in range(4):
                st[j] *= math.exp(-lam)
            if data and org in data:
                aw, an, dw, dn = data[org]
                st[0] += aw; st[1] += an; st[2] += dw; st[3] += dn
    while si < len(sdates):
        for t_, st in state.items():
            if st[1] > 12 and st[3] > 12:
                at[(t_, sdates[si])] = (st[0] / st[1], st[2] / st[3])
        si += 1
    return at

sf16 = side_form(16.0)
sf5 = side_form(5.0)
def side_delta(team, date):
    a16 = sf16.get((team, date))
    a5 = sf5.get((team, date))
    if a16 is None or a5 is None:
        return None
    return (a5[0] - a16[0]) + (a5[1] - a16[1])
sd_w = np.array([side_delta(r.winner, r.date) for r in s.itertuples(index=False)],
                dtype=object)
sd_l = np.array([side_delta(r.loser, r.date) for r in s.itertuples(index=False)],
                dtype=object)
cov_both = np.array([a is not None and b is not None for a, b in zip(sd_w, sd_l)])
dform_side = np.array([(a if a is not None else 0.0) - (b if b is not None else 0.0)
                       for a, b in zip(sd_w, sd_l)])
jlog(f"5c side coverage: both-teams-defined frame={int(cov_both.sum())}/{len(s)}, "
     f"holdout={int((cov_both & hold).sum())}/{int(hold.sum())}")
results["form_side5"], _ = fit_form("form_side5", dform_side, mask_covered=cov_both)
results["form_side5"]["coverage_note"] = (
    "round_outcomes covers 1707/2058 frame matches; all 25 corpus-addition "
    "events 0% (incl CN evo/EWC) — uncovered teams contribute neutral 0")

# ── 4. player-form (R2.0 trajectories) ──────────────────────────────────────
from engine import Engine  # noqa: E402  (registry pinned by decay_lib import)
maps_dir = os.path.join(DATA, "maps")
mid_date = {}
for g in rn.games:
    mid_date.setdefault(g["match_id"], g["date_s"])
rows = []
import MoreTestingMaybeFiles  # noqa: E402 — not needed; use registry via engine
for fn in sorted(os.listdir(maps_dir)):
    if not fn.endswith(".csv"):
        continue
    df = pd.read_csv(os.path.join(maps_dir, fn),
                     usecols=["ProfileURL", "MatchID", "MapNum", "R2.0"])
    df = df[df["MapNum"].astype(str) != "all"]
    rows.append(df)
pm = pd.concat(rows, ignore_index=True)
pm["date"] = pm["MatchID"].map(mid_date)
pm = pm.dropna(subset=["date", "R2.0", "ProfileURL"])
pm = pm.sort_values(["date", "MatchID", "MapNum"], kind="mergesort")
jlog(f"5c player: {len(pm)} player-map rows with R2.0 + date")

def player_form(hl):
    lam = LN2 / hl
    state = defaultdict(lambda: [0.0, 0.0])
    at = {}
    si = 0
    for r in pm.itertuples(index=False):
        while si < len(sdates) and sdates[si] <= r.date:
            for pl, (n_, d_) in state.items():
                if d_ > 3:
                    at[(pl, sdates[si])] = n_ / d_
            si += 1
        st = state[r.ProfileURL]
        st[0] = st[0] * math.exp(-lam) + float(r._3)   # R2.0 column
        st[1] = st[1] * math.exp(-lam) + 1.0
    while si < len(sdates):
        for pl, (n_, d_) in state.items():
            if d_ > 3:
                at[(pl, sdates[si])] = n_ / d_
        si += 1
    return at

pf16 = player_form(16.0)
pf5 = player_form(5.0)
lups = rn.eng.lineups
def team_pform(org, mid, date):
    L = lups.get((org, mid))
    if not L:
        return None
    vals = []
    for pl in L:
        a5, a16 = pf5.get((pl, date)), pf16.get((pl, date))
        if a5 is not None and a16 is not None:
            vals.append(a5 - a16)
    return float(np.mean(vals)) if len(vals) >= 3 else None
pf_w = [team_pform(r.winner, r.match_id, r.date) for r in s.itertuples(index=False)]
pf_l = [team_pform(r.loser, r.match_id, r.date) for r in s.itertuples(index=False)]
cov_p = np.array([a is not None and b is not None for a, b in zip(pf_w, pf_l)])
dform_p = np.array([(a if a is not None else 0.0) - (b if b is not None else 0.0)
                    for a, b in zip(pf_w, pf_l)])
jlog(f"5c player coverage: both-defined frame={int(cov_p.sum())}/{len(s)}, "
     f"holdout={int((cov_p & hold).sum())}/{int(hold.sum())}")
results["form_player5"], _ = fit_form("form_player5", dform_p, mask_covered=cov_p)

# ── 5. combined (beta, b_rd_best, b_side) ───────────────────────────────────
mdS = decayed_stat(float(best_rd), margin_val)
msw, msl = series_arrays(mdS, default=0.0)
dform_rd_best = (msw - m16w) - (msl - m16l)
v = ~np.isnan(rd6)
m_tr = v & rn.train_v

def nll3(params, mask):
    b, brd, bsd = params
    z = b * rd6[mask] + brd * dform_rd_best[mask] + bsd * dform_side[mask]
    pmv = 1 / (1 + np.exp(-z))
    return -np.mean(np.log(np.clip(sp(pmv, fmts[mask]), 1e-9, 1)))

fit3 = minimize(nll3, x0=[0.13, 0.0, 0.0], args=(m_tr,), method="Nelder-Mead",
                options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 6000})
b3, brd3, bsd3 = fit3.x
with np.errstate(invalid="ignore"):
    p3 = sp(1 / (1 + np.exp(-(b3 * rd6 + brd3 * dform_rd_best + bsd3 * dform_side))), fmts)
m_te = v & hold
d3 = referee.delta_vector(p3[m_te], p6[m_te])
bi3 = referee.paired_bootstrap_crn(d3, mode="iid")
bb3 = referee.paired_bootstrap_crn(d3, mode="block_event", event_ids=ev_hold[m_te])
mde3 = 2.8016 * float(np.std(d3, ddof=1)) / np.sqrt(int(m_te.sum())) * 1000
dm3 = float(np.mean(d3)) * 1000
verdict3 = ("WIN" if dm3 > 0 else "KILL") if (
    abs(dm3) >= mde3 and ((bi3["p_better"] >= 0.95 and bb3["p_better"] >= 0.95)
                          or (bi3["p_better"] <= 0.05 and bb3["p_better"] <= 0.05))
) else "INSIDE NOISE FLOOR"
np.savez(os.path.join(PROBS, "form_combined.npz"), rdiff=rd6, beta=b3, p=p3)
results["form_combined"] = {
    "beta": round(float(b3), 4), "b_rd": round(float(brd3), 4),
    "b_side": round(float(bsd3), 4), "short_hl_rd": best_rd,
    "ll_train": round(float(nll3(fit3.x, m_tr)), 5),
    "ll_test": round(rn.ll(p3, m_te), 5),
    "delta_milli_vs_v6": round(dm3, 3), "pair_mde_raw_milli": round(mde3, 3),
    "family_mde_milli": FAMILY_MDE_WITHIN,
    "boot_iid": {k: bi3[k] for k in ("mean_delta", "ci_lo", "ci_hi", "p_better")},
    "boot_block": {k: bb3[k] for k in ("mean_delta", "ci_lo", "ci_hi", "p_better")},
    "verdict": verdict3,
    "expected_roi_delta": referee.expected_roi_of_dll(
        float(np.mean(d3)), p6[m_te])["expected_roi_delta"]}
jlog(f"5c combined: b_rd={brd3:+.4f} b_side={bsd3:+.4f} d={dm3:+.2f}m ({verdict3})")

out = {"preregistered": "testing_lab/v8/preregister.decay.md",
       "base_rdiff": "v6_consist_20_12 (within-family prob-layer terms)",
       "v6_ll_test": round(rn.ll(p6, v & hold), 5),
       "reference_old_frame": {"b_form_wr3": -0.0872, "delta_milli": -0.25,
                               "source": "out/v7_stage2.json"},
       "results": results}
json.dump(out, open(os.path.join(STATS, "decay_form.json"), "w"), indent=1)
jlog("wrote stats/decay_form.json")
