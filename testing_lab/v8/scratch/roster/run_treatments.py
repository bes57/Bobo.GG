"""agent:roster step 5 — the <=4 EXPLORATORY holdout reads (preregister + operator addendum).

Read 1 (b) graded change-point continuity  — engine, gamma train-fit
Read 2 (c) change-triggered partial cold start (mean-blend) — post-solve, train-fit
Read 3 (d) PHASE-RESET FILTER (operator-specified) — variance injection, g train-fit
Read 4 reserve — only on a <0.1m train tie (preregistered rule)

LOOK HYGIENE: non-selected grid points are evaluated on TRAIN ONLY (any
holdout number an engine call returns for them is scrubbed unrecorded).
Exactly one holdout read per treatment, at the train-selected params.
All CRN judging via referee.py. Baseline = stored v6 (bias_h3/v6_baseline.npz).
Writes stats/roster_treatments.json + stats/roster_looks.json and appends
phase-reset overlays to the case JSONs. Everything EXPLORATORY — holdout spent.
"""
import bisect
import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

HERE = os.path.dirname(os.path.abspath(__file__))
V8 = os.path.dirname(os.path.dirname(HERE))
TL = os.path.dirname(V8)
STATS = os.path.join(V8, "stats")
sys.path.insert(0, TL)
sys.path.insert(0, V8)

import referee  # noqa: E402

FRAME = os.path.join(V8, "data", "frame_expanded", "series.csv")
crn = json.load(open(os.path.join(V8, "crn.json")))
sha = hashlib.sha256(open(FRAME, "rb").read()).hexdigest()
assert sha == crn["frame_expanded"]["series_csv_sha256"], f"FRAME SHA MISMATCH {sha}"
frame = pd.read_csv(FRAME).reset_index(drop=True)
N = len(frame)
hold = (frame.date > "2024-12-31").values
train_m = ~hold
fmts = frame.fmt.values
event_ids = frame.event_id.values

v6 = np.load(os.path.join(V8, "scratch", "bias_h3", "v6_baseline.npz"))
p_v6 = v6["p_all"]
rat_w6, rat_l6 = v6["rat_w"], v6["rat_l"]
valid_v6 = v6["valid"]
zd = np.load(os.path.join(HERE, "v6_daily.npz"), allow_pickle=True)
Rday, days_list, teams = zd["R"], list(zd["days"]), list(zd["teams"])
region_idx = zd["region_idx"]
day_pos = {d: i for i, d in enumerate(days_list)}
tpos = {t: i for i, t in enumerate(teams)}

ep = pd.read_csv(os.path.join(HERE, "episodes.csv"))
st = pd.read_csv(os.path.join(HERE, "team_match_state.csv"))
st = st.sort_values(["org", "date", "match_id"]).reset_index(drop=True)
st_ix = {(r.org, r.match_id): (int(r.msc), int(r.episode_idx),
                               float(r.ov) if r.ov == r.ov else np.nan,
                               int(r.sustained))
         for r in st.itertuples(index=False)}

# per-org sequences + walk-forward episode structures (preregister §1)
org_dates = {o: sorted(g.date.tolist()) for o, g in st.groupby("org")}
org_seq = {o: list(zip(g.sort_values(["date", "match_id"]).date,
                       g.sort_values(["date", "match_id"]).match_id))
           for o, g in st.groupby("org")}
eps_by_org = defaultdict(list)
ep_rows = list(ep.itertuples(index=False))
for k, e in enumerate(ep_rows):
    seq = org_seq[e.org]
    pos = next(i for i, (d, m) in enumerate(seq) if m == e.change_match_id)
    d_conf = seq[pos + 2][0] if e.run_len >= 3 and pos + 2 < len(seq) else None
    d_dead = seq[pos + e.run_len][0] if pos + e.run_len < len(seq) else None
    eps_by_org[e.org].append({
        "d": e.change_date, "mid": int(e.change_match_id), "ov": float(e.ov),
        "d_conf": d_conf, "d_dead": d_dead, "row": k})
for o in eps_by_org:
    eps_by_org[o].sort(key=lambda x: x["d"])


def sustained_at(e, D):
    """Walk-forward: confirmed (3rd run match played before D) or still alive."""
    if e["d_conf"] is not None and e["d_conf"] < D:
        return True
    return e["d_dead"] is None or e["d_dead"] >= D


# retrospective improvement flag per episode (contrast panel ONLY — diagnostic)
p_team_of = {}
for i, r in enumerate(frame.itertuples(index=False)):
    p_team_of[(r.winner, r.match_id)] = (1, float(p_v6[i]))
    p_team_of[(r.loser, r.match_id)] = (0, 1.0 - float(p_v6[i]))
imp_flag = {}
for o, eps in eps_by_org.items():
    seq = org_seq[o]
    for e in eps:
        pos = next(i for i, (d, m) in enumerate(seq) if m == e["mid"])
        vals = []
        for d, m in seq[pos:pos + 3]:
            if (o, m) in p_team_of:
                won, pt = p_team_of[(o, m)]
                vals.append(won - pt)
        if vals:
            imp_flag[e["row"]] = bool(np.mean(vals) > 0)

# ── per-row side states + bucket masks ───────────────────────────────────────
msc_w = np.full(N, np.nan)
msc_l = np.full(N, np.nan)
side_info = []          # per row: list of (org, msc, ep_row_idx, ov, sustained)
for i, r in enumerate(frame.itertuples(index=False)):
    inf = []
    for org, arr in ((r.winner, msc_w), (r.loser, msc_l)):
        s = st_ix.get((org, r.match_id))
        if s is not None:
            arr[i] = s[0]
            inf.append((org, s[0], s[1], s[2], s[3]))
        else:
            inf.append((org, None, -1, np.nan, 0))
    side_info.append(inf)
mn = np.fmin(msc_w, msc_l)
post_le3 = ~np.isnan(mn) & (mn <= 3)
post_4_10 = ~np.isnan(mn) & (mn >= 4) & (mn <= 10)
stable = ~np.isnan(mn) & (mn > 10)


def min_side(i):
    a, b = side_info[i]
    if a[1] is None:
        return b
    if b[1] is None:
        return a
    return a if a[1] <= b[1] else b


def magc(ov):
    return "keep4" if ov >= 0.8 else ("keep3" if ov >= 0.6 else "overhaul")


mag_of_row = np.array([
    (magc(min_side(i)[3]) if post_le3[i] and min_side(i)[2] >= 0
     and min_side(i)[4] == 1 else "") for i in range(N)])
# gated rows: either side within first 3 matches (msc<=2) of sustained ep ov<=0.6
gated = np.zeros(N, dtype=bool)
chg_sides = [[] for _ in range(N)]   # sides in first-3 of a sustained episode
for i in range(N):
    for (org, msc, epr, ov, sus) in side_info[i]:
        if msc is not None and msc <= 2 and epr >= 0 and sus == 1:
            chg_sides[i].append((org, epr, ov))
            if ov <= 0.6:
                gated[i] = True
improve_rows = np.zeros(N, dtype=bool)
degrade_rows = np.zeros(N, dtype=bool)
for i in range(N):
    if len(chg_sides[i]) == 1:
        epr = chg_sides[i][0][1]
        if epr in imp_flag:
            (improve_rows if imp_flag[epr] else degrade_rows)[i] = True

BUCKETS = [
    ("post-change <=3 (either team, power def)", post_le3),
    ("post-change <=3 · keep4", post_le3 & (mag_of_row == "keep4")),
    ("post-change <=3 · keep3", post_le3 & (mag_of_row == "keep3")),
    ("post-change <=3 · overhaul", post_le3 & (mag_of_row == "overhaul")),
    ("post-change 4-10", post_4_10),
    ("stable (>10)", stable),
    ("gated (first-3 of sustained ov<=0.6 change)", gated),
    ("improvement cases (retrospective slice)", improve_rows),
    ("degradation cases (retrospective slice)", degrade_rows),
]

MDE = {"within_milli": 1.773, "cross_milli": 5.889,
       "post_le3_within_milli": 2.52, "post_le3_cross_milli": 7.81,
       "sources": ["stats/power_mde_expanded.json checkpoint_quote (n=1217)",
                   "stats/power_mde.json bucket 'roster change <=3 matches ago' (n=598, frozen-npz holdout)"]}

EXPL = ("EXPLORATORY — the holdout is spent (398 recorded looks before this "
        "file; adversary_report.md). This read is counted in "
        "stats/roster_looks.json, is NOT confirmatory and NOT promotable; "
        "adjudication requires the preregistered prospective test on fresh "
        "series (roster_integration.json).")


def p_series_closed(beta, rdiff, fm=None):
    fm = fmts if fm is None else fm
    pm = 1.0 / (1.0 + np.exp(-beta * rdiff))
    return np.where(np.isin(fm, ("bo5", "bo5_gf")),
                    pm ** 3 * (1 + 3 * (1 - pm) + 6 * (1 - pm) ** 2),
                    np.where(fm == "bo1", pm, pm ** 2 * (3 - 2 * pm)))


def fit_beta_train(rdiff, valid):
    m = valid & train_m

    def nll(b):
        p = p_series_closed(b, rdiff[m], fmts[m])
        return -np.mean(np.log(np.clip(p, 1e-9, 1)))
    b = float(minimize_scalar(nll, bounds=(0.03, 0.6), method="bounded").x)
    return b, float(nll(b))


def judge(p_cand, valid_cand, label):
    m = hold & valid_v6 & valid_cand & ~np.isnan(p_cand)
    d = referee.delta_vector(p_cand[m], p_v6[m])
    iid = referee.paired_bootstrap_crn(d, mode="iid")
    blk = referee.paired_bootstrap_crn(d, mode="block_event", event_ids=event_ids[m])
    roi = referee.expected_roi_of_dll(float(d.mean()), p_v6[m])
    ll_c = float(referee.per_series_ll(p_cand[m]).mean())
    buckets = []
    for name, mask in BUCKETS:
        mm = m & mask
        nn = int(mm.sum())
        if nn < 10:
            buckets.append({"name": name, "n": nn, "note": "n<10, suppressed"})
            continue
        db = referee.delta_vector(p_cand[mm], p_v6[mm])
        bb = referee.paired_bootstrap_crn(db, mode="iid")
        buckets.append({"name": name, "n": nn,
                        "delta_milli": round(float(db.mean()) * 1000, 2),
                        "ci_lo_milli": round(bb["ci_lo"] * 1000, 2),
                        "ci_hi_milli": round(bb["ci_hi"] * 1000, 2),
                        "p_better": bb["p_better"]})
    return {"label": label, "epistemic_status": EXPL, "n_scored": int(m.sum()),
            "ll_holdout": round(ll_c, 5),
            "ll_v6_same_rows": round(float(referee.per_series_ll(p_v6[m]).mean()), 5),
            "delta_milli": round(float(d.mean()) * 1000, 3),
            "iid": iid, "block_event": blk,
            "expected_roi": roi, "buckets": buckets, "mde_context": MDE}, m, d


results = {}
looks = []
t_all = time.time()

# ═════ Read 1 — (b) graded change-point continuity ══════════════════════════
from engine import Engine  # noqa: E402


class EngineChange(Engine):
    def enable_change_mode(self, eps_by_team, gamma, floor=0.2):
        self.change_gamma = gamma
        self._ce = {}
        for t, eps in eps_by_team.items():
            if t in self.tidx:
                self._ce[t] = [dict(e, logf=gamma * math.log(max(e["ov"], floor)))
                               for e in eps]
        self._rows_of = {}
        for t in self._ce:
            ti = self.tidx[t]
            self._rows_of[t] = np.where((self.wi == ti) | (self.li == ti))[0]

    def _continuity_vec(self, ref_date_s, mode, persistence):
        if getattr(self, "change_gamma", None) is None:
            return super()._continuity_vec(ref_date_s, mode, persistence)
        n = len(self.games)
        cw = np.ones(n)
        cl = np.ones(n)
        D = ref_date_s
        for t, eps in self._ce.items():
            act = [(e["d"], e["logf"]) for e in eps
                   if e["d"] < D and sustained_at(e, D)]
            if not act:
                continue
            rows = self._rows_of[t]
            if len(rows) == 0:
                continue
            dts = np.array([a for a, _ in act])
            lgf = np.array([b for _, b in act])
            suffix = np.concatenate([np.cumsum(lgf[::-1])[::-1], [0.0]])
            gd = self.g_date[rows]
            idx = np.searchsorted(dts, gd, side="right")
            fac = np.exp(suffix[idx])
            ti = self.tidx[t]
            ws = self.wi[rows] == ti
            cw[rows[ws]] = fac[ws]
            cl[rows[~ws]] = fac[~ws]
        return cw, cl


print("=== Read 1 (b): graded change-point continuity ===", flush=True)
stage_by_mid = dict(zip(frame.match_id, frame.stage))
b_grid = []
b_runs = {}
for gamma in (0.5, 1.0, 2.0):
    eng = EngineChange()
    eng.series = frame.copy().reset_index(drop=True)
    eng.pred_days = sorted(frame.date.unique())
    g_stage = np.array([stage_by_mid.get(g["match_id"], "groups") for g in eng.games])
    PO = np.where(np.isin(g_stage, ("playoffs", "grand_final")), 1.6, 1.0)
    eng.enable_change_mode(eps_by_org, gamma)
    cfg = {"rd": {"power": 0.75, "scale": 2.5}, "roster_mode": "year",
           "roster_persistence": 0.3, "ridge": 0.5, "champ_mult": 2.0,
           "region_prior_ridge": 1.5, "w_custom": PO,
           "decay": {"kind": "games", "consistency": (20.0, 12.0)}}
    out = eng.run(cfg)
    # LOOK HYGIENE: scrub holdout numbers for grid points immediately
    ll_tr = out["ll_train"]
    b_runs[gamma] = {"rdiff": out["rdiff"], "beta": out["beta"]}
    for k in ("ll_test", "brier_test", "p_test"):
        out.pop(k, None)
    b_grid.append({"gamma": gamma, "beta": b_runs[gamma]["beta"],
                   "ll_train": ll_tr})
    print(f"  gamma={gamma}: ll_train={ll_tr} beta={b_runs[gamma]['beta']} "
          f"(holdout scrubbed)", flush=True)
b_grid.sort(key=lambda x: x["ll_train"])
sel_b = b_grid[0]
tie_b = (b_grid[1]["ll_train"] - b_grid[0]["ll_train"]) * 1000 < 0.1
gamma_b = sel_b["gamma"]
rd_b = b_runs[gamma_b]["rdiff"]
valid_b = ~np.isnan(rd_b)
p_b = np.full(N, np.nan)
p_b[valid_b] = p_series_closed(b_runs[gamma_b]["beta"], rd_b[valid_b], fmts[valid_b])
res_b, m_b, d_b = judge(p_b, valid_b, f"(b) change-point continuity gamma={gamma_b}")
res_b["spec"] = {"replaces": "v6 year-boundary continuity (roster_mode year, 0.3)",
                 "factor": "prod over sustained-at-D episodes e with g.date < d_e <= D of max(ov_e, 0.2)^gamma",
                 "gamma_grid_train_only": b_grid, "selected_gamma": gamma_b,
                 "train_tie_lt_0p1m": bool(tie_b), "beta_refit_train": sel_b["beta"]}
results["read1_b_continuity"] = res_b
looks.append({"read": 1, "config": f"change_continuity_gamma{gamma_b}",
              "ll_holdout": res_b["ll_holdout"], "delta_milli_vs_v6": res_b["delta_milli"],
              "status": "EXPLORATORY"})
print(f"READ 1: gamma={gamma_b} delta={res_b['delta_milli']}m "
      f"iid CI [{res_b['iid']['ci_lo']*1000:.2f},{res_b['iid']['ci_hi']*1000:.2f}]m", flush=True)

# ═════ Read 2 — (c) change-triggered partial cold start (mean-blend) ════════
print("=== Read 2 (c): partial cold start mean-blend ===", flush=True)
REGS = ["Americas", "EMEA", "Pacific", "CN"]
reg_mean_cache = {}


def region_mean(day, reg):
    k = (day, reg)
    if k not in reg_mean_cache:
        di = day_pos.get(day)
        if di is None:
            reg_mean_cache[k] = None
        else:
            m = region_idx == reg
            reg_mean_cache[k] = float(Rday[di][m].mean()) if m.sum() >= 4 else None
    return reg_mean_cache[k]


def blend_adjust(a0, M):
    """Returns adjusted rdiff (walk-forward, day granularity)."""
    rw = rat_w6.copy()
    rl = rat_l6.copy()
    n_adj = 0
    for i, r in enumerate(frame.itertuples(index=False)):
        D = r.date
        for org, arr in ((r.winner, rw), (r.loser, rl)):
            if org not in tpos or np.isnan(arr[i]):
                continue
            eps = eps_by_org.get(org, [])
            for e in reversed(eps):
                if e["d"] >= D:
                    continue
                if not sustained_at(e, D) or e["ov"] > 0.6:
                    continue
                dd = org_dates[org]
                n_since = bisect.bisect_left(dd, D) - bisect.bisect_left(dd, e["d"])
                if n_since >= M:
                    break
                reg = region_idx[tpos[org]]
                rm = region_mean(D, reg) if reg >= 0 else None
                if rm is None:
                    break
                a = a0 * (1.0 - e["ov"]) * (1.0 - n_since / M)
                arr[i] = (1 - a) * arr[i] + a * rm
                n_adj += 1
                break
    return rw - rl, n_adj


c_grid = []
c_store = {}
for a0 in (0.3, 0.6, 1.0):
    for M in (3, 6):
        rd_c, n_adj = blend_adjust(a0, M)
        vv = ~np.isnan(rd_c)
        beta_c, trnll = fit_beta_train(rd_c, vv)
        c_grid.append({"a0": a0, "M": M, "beta": beta_c,
                       "ll_train": round(trnll, 6), "n_team_adjustments": n_adj})
        c_store[(a0, M)] = (rd_c, beta_c)
        print(f"  a0={a0} M={M}: ll_train={trnll:.6f} n_adj={n_adj}", flush=True)
c_grid.sort(key=lambda x: x["ll_train"])
sel_c = c_grid[0]
tie_c = (c_grid[1]["ll_train"] - c_grid[0]["ll_train"]) * 1000 < 0.1
rd_c, beta_c = c_store[(sel_c["a0"], sel_c["M"])]
valid_c = ~np.isnan(rd_c)
p_c = np.full(N, np.nan)
p_c[valid_c] = p_series_closed(beta_c, rd_c[valid_c], fmts[valid_c])
res_c, m_c, d_c = judge(p_c, valid_c,
                        f"(c) partial cold start a0={sel_c['a0']} M={sel_c['M']}")
res_c["spec"] = {"trigger": "sustained-at-D episode, ov<=0.6, n_since<M, day-granularity walk-forward",
                 "blend": "r' = (1-a) r + a region_mean(day); a = a0 (1-ov)(1-n_since/M)",
                 "grid_train_only": c_grid, "selected": {"a0": sel_c["a0"], "M": sel_c["M"]},
                 "train_tie_lt_0p1m": bool(tie_c), "beta_refit_train": beta_c}
results["read2_c_coldstart"] = res_c
looks.append({"read": 2, "config": f"coldstart_a{sel_c['a0']}_M{sel_c['M']}",
              "ll_holdout": res_c["ll_holdout"], "delta_milli_vs_v6": res_c["delta_milli"],
              "status": "EXPLORATORY"})
print(f"READ 2: a0={sel_c['a0']} M={sel_c['M']} delta={res_c['delta_milli']}m", flush=True)

# ═════ Read 3' — (d) PHASE-RESET FILTER (operator-specified) ════════════════
print("=== Read 3' (d): phase-reset filter ===", flush=True)
sys.path.insert(0, os.path.join(V8, "scratch", "bias_h3"))
import lib_h3  # noqa: E402

gd = lib_h3.GameData()
core = json.load(open(os.path.join(V8, "scratch", "bias_h3", "sweep_core.json")))
b1 = core["best"]["primary_cfg"]
q0 = b1["q_over_R"] * gd.R
V0 = b1["V0_over_R"] * gd.R
qcal = b1["q_cal_week"]        # stored ABSOLUTE (run_core_sweep passes qc*gd.R)
f0 = gd.run_filter(q0, V0, q_cal_week=qcal)
beta0, tr0 = gd.fit_beta(f0["mu"], f0["s2"])
sc0 = gd.score(beta0, f0["mu"], f0["s2"])
assert abs(sc0["ll_train"] - b1["ll_train"]) < 2e-5, \
    f"1b base train mismatch {sc0['ll_train']} vs {b1['ll_train']}"
assert abs(sc0["ll_holdout"] - b1["ll_holdout"]) < 2e-5, \
    f"1b base holdout mismatch {sc0['ll_holdout']} vs {b1['ll_holdout']} (stored-number reverify)"
print(f"  base 1b verified: ll_train={sc0['ll_train']:.6f} "
      f"ll_holdout={sc0['ll_holdout']:.6f} (stored reuse, not a new look)", flush=True)

# injection map: (game_row, side) per rotation-guarded episode (alive at injection)
gmid = gd.g_mid
inj = []
for o, eps in eps_by_org.items():
    ti = gd.tidx.get(o)
    if ti is None:
        continue
    for e in eps:
        rows = np.where(gmid == e["mid"])[0]
        rows = [j for j in rows if gd.wi[j] == ti or gd.li[j] == ti]
        if not rows:
            continue
        j = min(rows)
        side = 0 if gd.wi[j] == ti else 1
        inj.append((j, side, 1.0 - e["ov"], e["row"], o))

d_grid = []
d_store = {}
for g in (0.5, 1.0, 2.0, 4.0):
    qv = np.full((gd.n_games, 2), q0)
    for j, side, sev, _, _ in inj:
        qv[j, side] += g * sev * gd.R
    f = gd.run_filter(q0, V0, q_vec=qv, q_cal_week=qcal)
    beta_g, trnll = gd.fit_beta(f["mu"], f["s2"])
    d_grid.append({"g": g, "beta": beta_g, "ll_train": round(trnll, 6)})
    d_store[g] = (f, beta_g, qv)
    print(f"  g={g}: ll_train={trnll:.6f} (train-only eval; no holdout computed)",
          flush=True)
d_grid.sort(key=lambda x: x["ll_train"])
sel_d = d_grid[0]
tie_d = (d_grid[1]["ll_train"] - d_grid[0]["ll_train"]) * 1000 < 0.1
g_d = sel_d["g"]
f_d, beta_d, qv_d = d_store[g_d]
sc_d = gd.score(beta_d, f_d["mu"], f_d["s2"])
p_d = sc_d["p"]
valid_d = ~np.isnan(p_d)
res_d, m_d, d_dvec = judge(p_d, valid_d, f"(d) phase-reset filter g={g_d}")
res_d["spec"] = {
    "operator_amendment": "phase-reset filter: mean kept (reference point), "
                          "variance injected dq = g*(1-ov)*R at the team's "
                          "first map of the change match; elevated Kalman "
                          "gain over-reacts to first post-change results in "
                          "whichever direction they point, decaying as "
                          "evidence accumulates",
    "base": {"family": "h3 core 1b (train-selected primary, reused)",
             "q_over_R": b1["q_over_R"], "V0_over_R": b1["V0_over_R"],
             "q_cal_week_abs": qcal,
             "stored_ll_train": b1["ll_train"], "stored_ll_holdout": b1["ll_holdout"],
             "stored_note": "1b holdout number was already recorded by bias_h3 "
                            "(one of the 398) — reused, not a new look"},
    "g_grid_train_only": d_grid, "selected_g": g_d,
    "train_tie_lt_0p1m": bool(tie_d), "beta_refit_train": beta_d,
    "n_injections": len(inj),
    "injection_rule": "every rotation-guarded change event (alive at injection "
                      "by construction; later-transient episodes cannot be "
                      "un-injected — walk-forward honesty)",
}
# injection's own contribution vs stored 1b base (same rows)
m_1b = hold & valid_d & ~np.isnan(sc0["p"])
d_vs_base = referee.delta_vector(p_d[m_1b], sc0["p"][m_1b])
bb = referee.paired_bootstrap_crn(d_vs_base, mode="iid")
res_d["vs_own_base_1b"] = {
    "note": "injection's marginal effect vs the stored no-injection filter "
            "(mechanism view; base holdout was already recorded by bias_h3)",
    "delta_milli": round(float(d_vs_base.mean()) * 1000, 3),
    "ci_lo_milli": round(bb["ci_lo"] * 1000, 2),
    "ci_hi_milli": round(bb["ci_hi"] * 1000, 2), "p_better": bb["p_better"]}
results["read3_d_phase_reset"] = res_d
looks.append({"read": 3, "config": f"phase_reset_g{g_d}",
              "ll_holdout": res_d["ll_holdout"], "delta_milli_vs_v6": res_d["delta_milli"],
              "status": "EXPLORATORY", "operator_specified": True})
print(f"READ 3': g={g_d} delta vs v6={res_d['delta_milli']}m; "
      f"vs 1b base={res_d['vs_own_base_1b']['delta_milli']}m", flush=True)

# ═════ contrast (d) vs (c) — preregistered addendum prediction ══════════════
mm = hold & valid_c & valid_d
d_dc = referee.per_series_ll(p_c[mm]) - referee.per_series_ll(p_d[mm])
contrast = {"prediction": "variance-spike (d) beats mean-blend (c) on "
                          "improvement cases; (c) pre-judges toward the prior",
            "sign_convention": "positive = (d) better than (c)"}
for nm, mask in (("improvement cases", improve_rows),
                 ("degradation cases", degrade_rows),
                 ("post-change <=3 (all)", post_le3)):
    k = mm & mask
    if k.sum() >= 10:
        dv = (referee.per_series_ll(p_c[k]) - referee.per_series_ll(p_d[k]))
        bb = referee.paired_bootstrap_crn(dv, mode="iid")
        contrast[nm] = {"n": int(k.sum()),
                        "d_minus_c_milli": round(float(dv.mean()) * 1000, 2),
                        "ci_lo_milli": round(bb["ci_lo"] * 1000, 2),
                        "ci_hi_milli": round(bb["ci_hi"] * 1000, 2)}
contrast["note"] = ("improvement/degradation classified retrospectively from "
                    "each episode's own first-3 outcomes — diagnostic slice of "
                    "already-counted vectors, not a new look")
results["contrast_d_vs_c"] = contrast
print("contrast:", json.dumps({k: v for k, v in contrast.items()
                               if isinstance(v, dict)}, indent=1), flush=True)

# ═════ learning-rate telemetry + per-team filter trajectories ═══════════════
print("=== telemetry loop (selected g) ===", flush=True)


def filter_telemetry(q, V0, q_vec, q_cal_week):
    """lib_h3.run_filter update math + gain/trajectory collection. Verified
    against lib output below."""
    nT = gd.n_teams
    r = np.zeros(nT)
    v = np.full(nT, float(V0))
    last_day = np.full(nT, np.nan)
    R = gd.R
    mu = np.full(N, np.nan)
    s2 = np.full(N, np.nan)
    gains = []                       # (team_idx, match_id, gain)
    traj = defaultdict(list)         # team -> [(day, r_post)]
    d_ord = pd.to_datetime(gd.days).values.astype("datetime64[D]").astype(int)
    for di, day in enumerate(gd.days):
        dnum = d_ord[di]
        rows = gd.rows_by_day.get(day)
        if rows is not None:
            for i in rows:
                a, b = gd.f_wi[i], gd.f_li[i]
                mu[i] = r[a] - r[b]
                s2[i] = v[a] + v[b]
        gs = gd.games_by_day.get(day)
        if gs is None:
            continue
        touched = set()
        for j in gs:
            a, b = gd.wi[j], gd.li[j]
            v[a] += q_vec[j, 0]
            v[b] += q_vec[j, 1]
            if q_cal_week > 0.0:
                for t in (a, b):
                    if not np.isnan(last_day[t]):
                        v[t] += q_cal_week * (dnum - last_day[t]) / 7.0
            e = gd.y[j] - (r[a] - r[b])
            S = v[a] + v[b] + R / gd.w[j]
            ka = v[a] / S
            kb = v[b] / S
            gains.append((a, gd.g_mid[j], ka))
            gains.append((b, gd.g_mid[j], kb))
            r[a] += ka * e
            r[b] -= kb * e
            v[a] -= v[a] * v[a] / S
            v[b] -= v[b] * v[b] / S
            last_day[a] = dnum
            last_day[b] = dnum
            touched.update((a, b))
        for t in touched:
            traj[t].append((day, float(r[t])))
    return mu, s2, gains, traj


mu_t, s2_t, gains, traj = filter_telemetry(q0, V0, qv_d, qcal)
ok = ~np.isnan(f_d["mu"])
assert np.allclose(mu_t[ok], f_d["mu"][ok], atol=1e-10), "telemetry mu mismatch"
assert np.allclose(s2_t[ok], f_d["s2"][ok], atol=1e-10), "telemetry s2 mismatch"
print("  telemetry verified against lib_h3.run_filter (mu, s2 identical)", flush=True)

gain_cells = defaultdict(list)
gain_stable = []
for (t, mid, k) in gains:
    org = gd.teams[t]
    s = st_ix.get((org, int(mid)))
    if s is None:
        continue
    msc, epr, ov, sus = s
    if msc >= 10:
        gain_stable.append(k)
    elif epr >= 0 and sus == 1 and msc <= 9:
        gain_cells[(magc(ov), msc)].append(k)
lr_curve = []
for mag in ("keep4", "keep3", "overhaul"):
    pts = []
    for m in range(10):
        v = gain_cells.get((mag, m), [])
        if len(v) >= 5:
            pts.append({"m": m, "n": len(v), "gain": round(float(np.mean(v)), 4)})
        else:
            pts.append({"m": m, "n": len(v)})
    lr_curve.append({"magnitude": mag, "points": pts})
results["learning_rate_curve"] = {
    "label": "DESCRIPTIVE filter telemetry at the train-selected g "
             "(operator-required chart: 'overreact a bit, scaled by continuity')",
    "gain_definition": "Kalman gain k = v/S per team-map update; x = matches "
                       "since change of that team at that match",
    "by_magnitude": lr_curve,
    "stable_reference_gain": {"n": len(gain_stable),
                              "gain": round(float(np.mean(gain_stable)), 4)},
    "selected_g": g_d,
}

# ═════ overlays for the named cases + ENVY chain ════════════════════════════
def team_overlay(org, d0, d1):
    ti = gd.tidx.get(org)
    tj = tpos.get(org)
    fpath = [{"d": d, "r": round(rr, 3)} for d, rr in traj.get(ti, [])
             if d0 <= d <= d1] if ti is not None else []
    vpath = [{"d": d, "r": round(float(Rday[k, tj]), 3)}
             for k, d in enumerate(days_list) if d0 <= d <= d1] if tj is not None else []
    return {"filter_path": fpath, "v6_path": vpath,
            "note": "both in transformed round-margin units; filter = "
                    "phase-reset posterior mean (post-day), v6 = daily Massey solve"}


def match_p_overlay(org, d0, d1):
    out = []
    for i, r in enumerate(frame.itertuples(index=False)):
        if not (d0 <= r.date <= d1) or org not in (r.winner, r.loser):
            continue
        won = int(r.winner == org)
        pv = float(p_v6[i]) if won else 1 - float(p_v6[i])
        pf = (float(p_d[i]) if won else 1 - float(p_d[i])) if not np.isnan(p_d[i]) else None
        out.append({"date": r.date, "match_id": int(r.match_id), "won": won,
                    "opponent": r.loser if won else r.winner,
                    "p_v6": round(pv, 3),
                    "p_phase_reset": round(pf, 3) if pf is not None else None,
                    "holdout_descriptive": bool(r.date > "2024-12-31")})
    return out


OVERLAY_LABEL = ("DESCRIPTIVE overlay of the EXPLORATORY phase-reset filter "
                 "(train-selected g) vs v6 through the change window — case "
                 "panel, not a scored read")
for fn, orgs in (("roster_case_envy.json",
                  [("ENVY", "2026-01-01", "2026-07-28")]),
                 ("roster_case_gallery.json", None)):
    path = os.path.join(STATS, fn)
    doc = json.load(open(path))
    if orgs is not None:
        for o, d0, d1 in orgs:
            doc["phase_reset_overlay"] = {"label": OVERLAY_LABEL, "org": o,
                                          "selected_g": g_d,
                                          **team_overlay(o, d0, d1),
                                          "matches": match_p_overlay(o, d0, d1)}
    else:
        for c in doc.get("named_cases", []) + doc.get("cases", []):
            d0 = (np.datetime64(c["change_date"]) - np.timedelta64(60, "D")).astype(str)
            d1 = (np.datetime64(c["change_date"]) + np.timedelta64(120, "D")).astype(str)
            c["phase_reset_overlay"] = {"label": OVERLAY_LABEL, "selected_g": g_d,
                                        **team_overlay(c["org"], d0, d1),
                                        "matches": match_p_overlay(c["org"], d0, d1)}
    with open(path, "w") as f:
        json.dump(doc, f, indent=1)
    print(f"  overlays appended to {fn}", flush=True)

# ═════ reserve read decision (preregistered rule) ═══════════════════════════
reserve = {"used": False,
           "rule": "only on a <0.1m train tie in Read 1 or Read 2 grids "
                   "(addendum: also g tie)",
           "ties": {"read1_b": bool(tie_b), "read2_c": bool(tie_c),
                    "read3_d": bool(tie_d)}}
results["read4_reserve"] = reserve

# ═════ write treatments + looks ═════════════════════════════════════════════
results["_meta"] = {
    "written_by": "agent:roster", "written": time.strftime("%Y-%m-%d %H:%M:%S"),
    "epistemic_status": EXPL,
    "preregistered": "preregister.roster.md (+ operator addendum, ordering in logs/roster.log)",
    "baseline": "stored scratch/bias_h3/v6_baseline.npz (beta 0.1152, holdout LL 0.64216, n=1217)",
    "frame_sha256": sha, "n_holdout": int(hold.sum()),
    "crn": "referee.paired_bootstrap_crn (iid + block_event), crn.json governs",
    "runtime_s": round(time.time() - t_all, 1),
}
with open(os.path.join(STATS, "roster_treatments.json"), "w") as f:
    json.dump(results, f, indent=1)

prior = json.load(open(os.path.join(STATS, "compose_looks.json")))
looks_doc = {
    "written_by": "agent:roster",
    "written": time.strftime("%Y-%m-%d %H:%M:%S"),
    "successor_of": "stats/compose_looks.json (grand total 398 at Wave 3 close)",
    "prior_grand_total_recorded_holdout_numbers": prior["totals"]["grand_total_recorded_holdout_numbers"],
    "unit_definition": prior["unit_definition"],
    "epistemic_status": ("ALL reads below are EXPLORATORY on a SPENT holdout — "
                         "never confirmatory, never promotable "
                         "(adversary_report.md; preregister.roster.md)"),
    "new_reads": looks,
    "new_primary_looks": len(looks),
    "grid_points_train_only": {
        "read1_gammas": [g["gamma"] for g in b_grid],
        "read2_grid": [[g["a0"], g["M"]] for g in c_grid],
        "read3_gs": [g["g"] for g in d_grid],
        "note": "non-selected grid points were evaluated on TRAIN only; the "
                "engine/filter holdout numbers for them were scrubbed before "
                "any recording (logs/roster.log)"},
    "stored_reuse_not_new_looks": [
        "v6 baseline (bias_h3/v6_baseline.npz)",
        "h3 core 1b holdout (sweep_core.json, already among the 398)"],
    "grand_total_after": prior["totals"]["grand_total_recorded_holdout_numbers"] + len(looks),
}
with open(os.path.join(STATS, "roster_looks.json"), "w") as f:
    json.dump(looks_doc, f, indent=1)

print(f"\nDONE in {results['_meta']['runtime_s']}s — {len(looks)} exploratory "
      f"reads recorded; grand total {looks_doc['grand_total_after']}", flush=True)
for k in ("read1_b_continuity", "read2_c_coldstart", "read3_d_phase_reset"):
    r = results[k]
    print(f"{k}: delta {r['delta_milli']}m  iid p_better {r['iid']['p_better']:.3f}  "
          f"post<=3 bucket: "
          f"{next((b for b in r['buckets'] if b['name'].startswith('post-change <=3 (')), None)}",
      flush=True)
