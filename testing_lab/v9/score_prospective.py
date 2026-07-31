"""v9 STANDING PROSPECTIVE EVALUATOR — the only confirmatory instrument left.

Implements testing_lab/v9/stats/v9_prospective_protocol.json VERBATIM.
Owner: agent:v9-finish. One writer: this script owns
stats/v9_prospective_scoreboard.json and logs/prospective.log; it appends
checkpoint reads to stats/v9_looks.json prospective_reads (the protocol
routes them there). Any deviation from the protocol is a violation and must
be logged in v9/logs/.

RUN CADENCE: manual, or cron weekly. Safe to run at ANY time —
  * idempotent: same data in, same scoreboard out (only last_run + a log
    line move); frozen betas and past checkpoint reads are loaded from the
    existing scoreboard and NEVER recomputed;
  * read-only over data/ (and over every other agent's artifact);
  * refits NOTHING: v6's beta comes from stats/v9_ladder.json verbatim
    (0.128512); reference-arm betas were frozen once, on the protocol
    freeze window (2023-01-01..2026-07-28, protocol beta method), at first
    initialization, and are reused verbatim forever after;
  * before n=100 it reports "accumulating (n=X)" and evaluates nothing —
    no aggregate over post-2026-07-28 rows is computed or displayed outside
    the three protocol checkpoints (per-series predictions are generated
    and stored without aggregation, which the protocol permits).

ARMS
  candidate ladder (stats/v9_ladder.json, FROZEN): v6 alone — zero of five
  v9 candidates advanced, so there is nothing to promote; v6 is arm 0 and
  the paired baseline.
  reference arms (v8/stats/roster_integration.json arms_frozen, all
  preregistered with fixed configs BEFORE 2026-07-28): B_continuity,
  C_coldstart, D_phase_reset, E_atlas_replication, F_v6_overreact,
  H_specrun. Scored under the same machinery, clearly separated from the
  (empty) candidate set; they are NOT v9 candidates and are NOT promotable
  under the v9 protocol — their checkpoint numbers are recorded for the
  record (the roster_integration prospective plan).
  D_phase_reset is registered NOT_SCOREABLE_LIVE: its base is the h3-1b
  round-level state-space filter whose round enrichment does not exist for
  CN events or the live post-freeze corpus (disclosed, not substituted).

MECHANISM PORTS (code moves with attribution, zero new modeling decisions):
  EngineV9._continuity_vec spec hook   <- v8/scratch/roster/spec_run/runner.EngineSpec
  EngineV9._continuity_vec overreact   <- v8/scratch/roster/run_treatment_e.EngineOverreact
  EngineV9._continuity_vec change hook <- v8/scratch/roster/run_treatments.EngineChange
  build_episodes / msc                 <- v8/scratch/roster/build_changes.py
  coldstart blend                      <- v8/scratch/roster/run_treatments.blend_adjust
  frame recipe                         <- v8/data/frame_expanded/README.md (verbatim)
"""
import hashlib
import json
import os
import sys
import time
from bisect import bisect_left, bisect_right
from collections import defaultdict
from datetime import date as _date

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

HERE = os.path.dirname(os.path.abspath(__file__))          # testing_lab/v9
TL = os.path.dirname(HERE)                                 # testing_lab
V8 = os.path.join(TL, "v8")
ROOT = os.path.dirname(TL)                                 # PythonTest
DATA = os.path.join(ROOT, "data")
STATS = os.path.join(HERE, "stats")
LOGS = os.path.join(HERE, "logs")
SPEC_RUN = os.path.join(V8, "scratch", "roster", "spec_run")
for _p in (TL, V8, SPEC_RUN):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import referee                                   # noqa: E402  (v8 CRN judge)
from engine import Engine                        # noqa: E402
from harness import _stage, _is_intl_event       # noqa: E402
sys.path.insert(0, "/Users/benny_es1/VCTMM")
from vctmm.benpom.teams import ORG_REGIONS       # noqa: E402
import speclib                                   # noqa: E402  (frozen classifier)

PROTOCOL = json.load(open(os.path.join(STATS, "v9_prospective_protocol.json")))
LADDER = json.load(open(os.path.join(STATS, "v9_ladder.json")))
ROSTER_INT = json.load(open(os.path.join(V8, "stats", "roster_integration.json")))
SCOREBOARD_PATH = os.path.join(STATS, "v9_prospective_scoreboard.json")
LOOKS_PATH = os.path.join(STATS, "v9_looks.json")
LOG_PATH = os.path.join(LOGS, "prospective.log")
FROZEN_FRAME = os.path.join(V8, "data", "frame_expanded", "series.csv")

CUTOFF = "2026-07-28"           # scoring rows are date > CUTOFF (protocol)
FREEZE_LO = "2023-01-01"        # beta freeze window (protocol beta_freeze)
CHECKPOINTS = [int(x) for x in PROTOCOL["checkpoints"]["at_scored_n"]]
G1_THR = {int(k): float(v)
          for k, v in PROTOCOL["promotion_rule"]["G1_sequential_evidence"]["thresholds"].items()}
MDE_WITHIN = {int(k): float(v)
              for k, v in PROTOCOL["checkpoints"]["mde_at_n_milli"]["within_family"].items()}
MDE_CROSS = {int(k): float(v)
             for k, v in PROTOCOL["checkpoints"]["mde_at_n_milli"]["cross_family"].items()}
G4_MIN_N = {100: 10, 200: 10, 400: 15}          # protocol G4_team_bias
G4_TOL = 0.02
BETA_V6 = float([a for a in LADDER["arms"] if a["id"] == "v6"][0]["beta_frozen"])
assert LADDER["verdict_sentence"] == "the ladder is v6 alone", \
    "ladder file changed under the evaluator — refuse to run"


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


# ── scoring population: frame recipe VERBATIM from live data ────────────────

def build_live_frame():
    """frame_expanded README recipe, mirrored exactly, over the LIVE files:
    per-map winners from match_results.csv excluding MapNum=='all'; org pair
    from data/maps/<event>.csv player rows; both orgs in ORG_REGIONS; series
    score w_maps > l_maps, w_maps in {1,2,3}; fmt from w_maps (+ bo5_gf on
    grand finals); stage via harness._stage; intl via harness._is_intl_event;
    sort (date, match_id), dedup match_id keep-first. Read-only."""
    mr = pd.read_csv(os.path.join(DATA, "match_results.csv"), dtype={"MapNum": str})
    dates = json.load(open(os.path.join(DATA, "match_dates.json")))
    orgs_of, event_of = {}, {}
    maps_dir = os.path.join(DATA, "maps")
    for fn in sorted(os.listdir(maps_dir)):
        if not fn.endswith(".csv"):
            continue
        eid = fn[:-4]
        try:
            df = pd.read_csv(os.path.join(maps_dir, fn), usecols=["Org", "MatchID"])
        except Exception:
            continue
        for mid, grp in df.groupby("MatchID"):
            mid = int(mid)
            if mid not in orgs_of:                      # keep-first, name-sorted
                orgs_of[mid] = sorted(grp["Org"].dropna().unique().tolist())
                event_of[mid] = eid
    name_of = dict(mr.drop_duplicates("MatchID")[["MatchID", "MatchName"]].values)
    maps_only = mr[mr["MapNum"] != "all"]
    rows, excl = [], defaultdict(int)
    for mid, grp in maps_only.groupby("MatchID"):
        mid = int(mid)
        pair = orgs_of.get(mid)
        if pair is None or len(pair) != 2:
            excl["org count != 2"] += 1
            continue
        if not all(o in ORG_REGIONS for o in pair):
            excl["org not in ORG_REGIONS"] += 1
            continue
        d = dates.get(str(mid))
        if not d:
            excl["no date"] += 1
            continue
        wins = {o: int((grp["WinnerOrg"] == o).sum()) for o in pair}
        w, l = (pair[0], pair[1]) if wins[pair[0]] >= wins[pair[1]] else (pair[1], pair[0])
        wm, lm = wins[w], wins[l]
        if wm <= lm or wm not in (1, 2, 3):
            excl["forfeit/incomplete/odd"] += 1
            continue
        mname = str(name_of.get(mid, ""))
        stage = _stage(mname)
        fmt = {1: "bo1", 2: "bo3", 3: "bo5"}[wm]
        if fmt == "bo5" and stage == "grand_final":
            fmt = "bo5_gf"
        eid = event_of[mid]
        rows.append({"match_id": mid, "date": d, "event_id": eid,
                     "year": int(d[:4]), "winner": w, "loser": l,
                     "w_maps": wm, "l_maps": lm, "fmt": fmt, "stage": stage,
                     "match_name": mname, "reg_w": ORG_REGIONS.get(w, "?"),
                     "reg_l": ORG_REGIONS.get(l, "?"),
                     "intl": _is_intl_event(eid), "n_maps_played": int(len(grp))})
    df = (pd.DataFrame(rows).sort_values(["date", "match_id"])
          .drop_duplicates("match_id", keep="first").reset_index(drop=True))
    return df, dict(excl)


def frame_fixture(frame):
    """Column-agreement check vs the FROZEN frame on common match_ids (the
    recipe's reproduction guard). Disagreement is disclosed loudly, never
    silently absorbed."""
    fz = pd.read_csv(FROZEN_FRAME)
    cols = ["date", "event_id", "winner", "loser", "w_maps", "l_maps", "fmt", "stage"]
    mine = frame.set_index("match_id")
    common = fz[fz.match_id.isin(mine.index)]
    bad = 0
    for r in common.itertuples(index=False):
        me = mine.loc[r.match_id]
        if any(str(getattr(r, c)) != str(me[c]) for c in cols):
            bad += 1
    return {"frozen_rows": int(len(fz)), "common_rows": int(len(common)),
            "column_mismatches": int(bad),
            "live_rows_le_cutoff": int((frame.date <= CUTOFF).sum()),
            "live_rows_post_cutoff": int((frame.date > CUTOFF).sum())}


# ── episode structures (build_changes.py port, live + walk-forward) ─────────

def build_episodes(eng):
    lineups = eng.lineups
    g_date = {g["match_id"]: g["date_s"] for g in eng.games}
    seq = defaultdict(list)
    for (org, mid) in lineups:
        if mid in g_date:
            seq[org].append((g_date[mid], mid))
    for org in seq:
        seq[org].sort()

    def overlap(a, b):
        return len(a & b) / max(len(a), len(b), 5)

    eps_by_org, st_ix, org_dates = defaultdict(list), {}, {}
    for org, rows in seq.items():
        Ls = [lineups.get((org, mid)) for _, mid in rows]
        n = len(rows)
        org_dates[org] = [d for d, _ in rows]
        msc = []
        for i in range(n):
            d, mid = rows[i]
            m, j = 0, i - 1
            while j >= 0:
                dj, mj = rows[j]
                if dj >= d:
                    j -= 1
                    continue
                if Ls[j] == Ls[i] and Ls[i] is not None:
                    m += 1
                    j -= 1
                else:
                    break
            msc.append(m)
        ep_of_run = [None] * n
        org_eps = []
        for i in range(1, n):
            if Ls[i] is None or Ls[i - 1] is None:
                continue
            if Ls[i] == Ls[i - 1]:
                ep_of_run[i] = ep_of_run[i - 1]
                continue
            recent = {Ls[k] for k in range(max(0, i - 3), i - 1)}
            if Ls[i] in recent:                      # rotation guard
                ep_of_run[i] = None
                continue
            R = 1
            while i + R < n and Ls[i + R] == Ls[i]:
                R += 1
            censored = (i + R == n)
            d_conf = rows[i + 2][0] if R >= 3 and i + 2 < n else None
            d_dead = rows[i + R][0] if i + R < n else None
            org_eps.append({"d": rows[i][0], "mid": rows[i][1],
                            "ov": round(overlap(Ls[i], Ls[i - 1]), 4),
                            "run_len": R, "censored": censored,
                            "sustained": int(R >= 3 or censored),
                            "d_conf": d_conf, "d_dead": d_dead})
            ep_of_run[i] = len(org_eps) - 1
        org_eps.sort(key=lambda x: x["d"])
        eps_by_org[org] = org_eps
        for i in range(n):
            e = ep_of_run[i]
            ee = org_eps[e] if e is not None else None
            st_ix[(org, rows[i][1])] = (
                msc[i], -1 if ee is None else e,
                ee["ov"] if ee else float("nan"), ee["sustained"] if ee else 0)
    return eps_by_org, org_dates, st_ix


def sustained_at(e, D):
    """Walk-forward: confirmed (3rd run match played before D) or still alive
    (run_treatments.py port)."""
    if e["d_conf"] is not None and e["d_conf"] < D:
        return True
    return e["d_dead"] is None or e["d_dead"] >= D


# ── engine with the three frozen per-side hooks ─────────────────────────────

class EngineV9(Engine):
    """v6 base + exactly one of the three frozen mechanisms per run (ports
    with attribution, see module docstring). All hooks disabled => pure v6
    parent path (nesting asserted bitwise in v9_fixtures.json V1)."""

    def prepare_rows(self):
        self._org_rows = {}
        for org, ti in self.tidx.items():
            seq = self.team_match_seq.get(org, [])
            pos_of = {mid: i for i, (d, mid) in enumerate(seq)}
            rw = np.where(self.wi == ti)[0]
            rl = np.where(self.li == ti)[0]
            pw = np.array([pos_of[self.games[r]["match_id"]] for r in rw], dtype=np.int64)
            pl = np.array([pos_of[self.games[r]["match_id"]] for r in rl], dtype=np.int64)
            self._org_rows[org] = (rw, pw, rl, pl)
        self._rows_any = {t: np.where((self.wi == ti) | (self.li == ti))[0]
                          for t, ti in self.tidx.items()}
        self._spec = self._over = self._change = None

    def enable_spec(self, plan, a, tau, s, n_min=3, cap=None):
        self._spec = {"plan": plan, "a": a, "tau": tau, "s": s,
                      "n_min": n_min, "cap": cap}

    def enable_overreact(self, a, tau, eps_by_org, org_dates):
        self._over = {"a": a, "tau": tau, "eps": eps_by_org, "dates": org_dates}

    def enable_change(self, eps_by_org, gamma, floor=0.2):
        import math
        ce = {}
        for t, eps in eps_by_org.items():
            if t in self.tidx:
                ce[t] = [dict(e, logf=gamma * math.log(max(e["ov"], floor)))
                         for e in eps]
        self._change = {"gamma": gamma, "ce": ce}

    def disable_all(self):
        self._spec = self._over = self._change = None

    def _continuity_vec(self, ref_date_s, mode, persistence):
        import math
        cw, cl = super()._continuity_vec(ref_date_s, mode, persistence)
        D = ref_date_s
        sp = self._spec
        if sp is not None and not (sp["a"] == 0 and sp["s"] == 0):
            # runner.EngineSpec port (H_specrun)
            plan = sp["plan"]
            for org in self._org_rows.keys():
                ver = plan.version_asof(org, D)
                if ver is None:
                    continue
                mult = plan.multipliers(ver, sp["a"], sp["tau"], sp["s"],
                                        n_min=sp["n_min"], cap=sp["cap"])
                if not (mult != 1.0).any():
                    continue
                m2 = mult * mult
                rw, pw, rl, pl = self._org_rows[org]
                selw = pw < ver["nvis"]
                if selw.any():
                    mw = m2[pw[selw]]
                    if (mw != 1.0).any():
                        cw[rw[selw]] = cw[rw[selw]] * mw
                sell = pl < ver["nvis"]
                if sell.any():
                    ml = m2[pl[sell]]
                    if (ml != 1.0).any():
                        cl[rl[sell]] = cl[rl[sell]] * ml
        ov_ = self._over
        if ov_ is not None:
            # run_treatment_e.EngineOverreact port (F_v6_overreact)
            a, tau = ov_["a"], ov_["tau"]
            for org, eps in ov_["eps"].items():
                if org not in self.tidx:
                    continue
                act = None
                for e in reversed(eps):
                    if e["d"] < D and sustained_at(e, D):
                        act = e
                        break
                if act is None or act["ov"] >= 1.0:
                    continue
                dd = ov_["dates"][org]
                n_since = bisect_left(dd, D) - bisect_left(dd, act["d"])
                if n_since < 1 or n_since / tau > 12:
                    continue
                m = 1.0 + a * (1.0 - act["ov"]) * math.exp(-n_since / tau)
                if m <= 1.0 + 1e-12:
                    continue
                rows = self._rows_any[org]
                post = rows[self.g_date[rows] >= act["d"]]
                if len(post) == 0:
                    continue
                ti = self.tidx[org]
                ws = self.wi[post] == ti
                cw[post[ws]] *= m * m
                cl[post[~ws]] *= m * m
        ch = self._change
        if ch is not None:
            # run_treatments.EngineChange port (B_continuity): REPLACES the
            # year-boundary factor for teams with visible sustained episodes
            n = len(self.games)
            cw = np.ones(n)
            cl = np.ones(n)
            for t, eps in ch["ce"].items():
                act = [(e["d"], e["logf"]) for e in eps
                       if e["d"] < D and sustained_at(e, D)]
                if not act:
                    continue
                rows = self._rows_any[t]
                if len(rows) == 0:
                    continue
                dts = np.array([x for x, _ in act])
                lgf = np.array([y for _, y in act])
                suffix = np.concatenate([np.cumsum(lgf[::-1])[::-1], [0.0]])
                gd = self.g_date[rows]
                idx = np.searchsorted(dts, gd, side="right")
                fac = np.exp(suffix[idx])
                ti = self.tidx[t]
                ws = self.wi[rows] == ti
                cw[rows[ws]] = fac[ws]
                cl[rows[~ws]] = fac[~ws]
        return cw, cl


def p_series_closed(beta, rdiff, fm):
    pm = 1.0 / (1.0 + np.exp(-beta * rdiff))
    return np.where(np.isin(fm, ("bo5", "bo5_gf")),
                    pm ** 3 * (1 + 3 * (1 - pm) + 6 * (1 - pm) ** 2),
                    np.where(fm == "bo1", pm, pm ** 2 * (3 - 2 * pm)))


def fit_beta_freeze(rdiff, frame, valid):
    """Protocol beta method on the freeze window ONLY (2023-01-01..2026-07-28).
    Called exactly once per reference arm, at first initialization."""
    fmts = frame.fmt.values
    m = valid & (frame.date >= FREEZE_LO).values & (frame.date <= CUTOFF).values

    def nll(b):
        p = p_series_closed(b, rdiff[m], fmts[m])
        return -np.mean(np.log(np.clip(p, 1e-9, 1.0)))

    r = minimize_scalar(nll, bounds=(0.001, 1.0), method="bounded",
                        options={"xatol": 1e-6})
    return round(float(r.x), 6), int(m.sum())


V6_CFG = {"rd": {"power": 0.75, "scale": 2.5}, "roster_mode": "year",
          "roster_persistence": 0.3, "ridge": 0.5, "champ_mult": 2.0,
          "region_prior_ridge": 1.5,
          "decay": {"kind": "games", "consistency": (20.0, 12.0)}}


def run_arm(eng, frame, daily=False):
    """One walk-forward solve. beta is PINNED (no internal refit); every
    engine-computed evaluation metric is popped UNSEEN (runner.run_config
    look-hygiene pattern)."""
    eng._prev_rvec = None
    stage_by_mid = dict(zip(frame.match_id, frame.stage))
    g_stage = np.array([stage_by_mid.get(g["match_id"], "groups") for g in eng.games])
    cfg = dict(V6_CFG, w_custom=np.where(
        np.isin(g_stage, ("playoffs", "grand_final")), 1.6, 1.0), beta=BETA_V6)
    if daily:
        cfg["daily_out"] = True
    out = eng.run(cfg)
    for kk in ("ll_test", "brier_test", "p_test", "test_mask", "ll_train"):
        out.pop(kk, None)                       # LOOK HYGIENE: never seen
    return out


def coldstart_rdiff(frame, out_v6, eng, eps_by_org, org_dates, a0, M):
    """run_treatments.blend_adjust port (C_coldstart): post-solve blend of a
    fresh-roster team's rating toward its region's daily mean."""
    daily = out_v6["daily_r"]
    days_list = sorted(daily.keys())
    day_pos = {d: i for i, d in enumerate(days_list)}
    Rday = np.stack([daily[d] for d in days_list])
    region_idx = eng.team_region_idx
    tpos = eng.tidx
    cache = {}

    def region_mean(day, reg):
        k = (day, reg)
        if k not in cache:
            di = day_pos.get(day)
            if di is None:
                cache[k] = None
            else:
                mm = region_idx == reg
                cache[k] = float(Rday[di][mm].mean()) if mm.sum() >= 4 else None
        return cache[k]

    rw = out_v6["rat_w"].copy()
    rl = out_v6["rat_l"].copy()
    for i, r in enumerate(frame.itertuples(index=False)):
        D = r.date
        for org, arr in ((r.winner, rw), (r.loser, rl)):
            if org not in tpos or np.isnan(arr[i]):
                continue
            for e in reversed(eps_by_org.get(org, [])):
                if e["d"] >= D:
                    continue
                if not sustained_at(e, D) or e["ov"] > 0.6:
                    continue
                dd = org_dates[org]
                n_since = bisect_left(dd, D) - bisect_left(dd, e["d"])
                if n_since >= M:
                    break
                reg = region_idx[tpos[org]]
                rm = region_mean(D, reg) if reg >= 0 else None
                if rm is None:
                    break
                a = a0 * (1.0 - e["ov"]) * (1.0 - n_since / M)
                arr[i] = (1 - a) * arr[i] + a * rm
                break
    return rw - rl


# ── checkpoint machinery (protocol promotion/kill VERBATIM) ─────────────────

def checkpoint_read(cp, arm_id, role, d, ev, p_arm, p_v6, sub, rd_v6):
    """One arm's numbers at checkpoint cp, on EXACTLY the first cp scored
    rows. G1-G4 clause-by-clause; kill rule; MDE next to every number."""
    iid = referee.paired_bootstrap_crn(d, mode="iid")
    blk = referee.paired_bootstrap_crn(d, mode="block_event", event_ids=ev)
    delta_m = round(float(d.mean()) * 1000, 3)
    roi = referee.expected_roi_of_dll(float(d.mean()), p_v6)
    g1 = bool(iid["p_better"] >= G1_THR[cp] and blk["p_better"] >= G1_THR[cp])
    g2 = bool(delta_m >= MDE_WITHIN[cp])
    bres = referee.bucketed(sub, p_arm, p_ref=p_v6, rdiff=rd_v6,
                            holdout=np.ones(len(sub), dtype=bool), min_n=30)
    majors = [{"name": b["name"], "n": b["n"], "delta_milli": b["delta_milli"]}
              for b in bres["buckets"] if "delta_milli" in b and b["n"] >= 30
              and b["delta_milli"] <= (-4.0 if b["n"] >= 100 else -8.0)]
    g3 = len(majors) == 0
    w, l = sub.winner.values, sub.loser.values
    ones = np.ones(len(sub), dtype=bool)
    bias_c = referee.per_team_bias(p_arm, w, l, holdout=ones, min_n=G4_MIN_N[cp])
    bias_v = referee.per_team_bias(p_v6, w, l, holdout=ones, min_n=G4_MIN_N[cp])
    g4 = bool(bias_c["max_abs_bias"] <= bias_v["max_abs_bias"] + G4_TOL)
    killed = bool(blk["ci_hi"] < 0)
    return {
        "arm": arm_id, "role": role, "checkpoint_n": cp,
        "delta_milli": delta_m,
        "mde_within_milli": MDE_WITHIN[cp], "mde_cross_milli": MDE_CROSS[cp],
        "expected_roi_delta": roi["expected_roi_delta"],
        "iid": {"ci_milli": [round(iid["ci_lo"] * 1000, 2), round(iid["ci_hi"] * 1000, 2)],
                "p_better": iid["p_better"]},
        "block_event": {"ci_milli": [round(blk["ci_lo"] * 1000, 2), round(blk["ci_hi"] * 1000, 2)],
                        "p_better": blk["p_better"]},
        "inside_noise_floor": bool(abs(delta_m) < MDE_WITHIN[cp]),
        "G1_pass": g1, "G1_threshold": G1_THR[cp],
        "G2_pass": g2, "G3_pass": g3, "G3_major_regressions": majors,
        "G4_pass": g4, "G4_max_abs_bias": {"arm": bias_c["max_abs_bias"],
                                           "v6": bias_v["max_abs_bias"],
                                           "tolerance": G4_TOL},
        "KILL_ci_hi_lt_0": killed,
        "promotable": bool(role == "candidate" and g1 and g2 and g3 and g4),
    }


# ── main ────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    prior = json.load(open(SCOREBOARD_PATH)) if os.path.exists(SCOREBOARD_PATH) else {}
    frame, excl = build_live_frame()
    fx = frame_fixture(frame)
    if fx["column_mismatches"] > 0:
        log(f"WARNING frame fixture: {fx['column_mismatches']} column mismatches "
            f"vs frozen frame on common match_ids (data revision?) — disclosed, not absorbed")
    post = (frame.date > CUTOFF).values
    n_post = int(post.sum())
    log(f"run start: live frame {len(frame)} rows (excl {excl}); "
        f"settled post-{CUTOFF}: {n_post}")

    arms = [
        {"id": "v6", "role": "baseline", "scoreable": True,
         "label": LADDER["arms"][0]["label"],
         "config": {"mechanism": "v6", "note": "pure v6 (a=s=0)"},
         "beta_frozen": BETA_V6,
         "beta_provenance": "stats/v9_ladder.json verbatim (frozen by agent:v9-search)"},
        {"id": "B_continuity", "role": "reference", "scoreable": True,
         "label": ROSTER_INT["prospective_validation_plan"]["arms_frozen"]["B_continuity"],
         "config": {"mechanism": "change_continuity", "gamma": 2.0, "floor": 0.2}},
        {"id": "C_coldstart", "role": "reference", "scoreable": True,
         "label": ROSTER_INT["prospective_validation_plan"]["arms_frozen"]["C_coldstart"],
         "config": {"mechanism": "coldstart_blend", "a0": 1.0, "M": 6}},
        {"id": "D_phase_reset", "role": "reference", "scoreable": False,
         "label": ROSTER_INT["prospective_validation_plan"]["arms_frozen"]["D_phase_reset"],
         "config": {"mechanism": "phase_reset_filter", "g": 0.5, "base": "h3-1b"},
         "status_note": ("NOT_SCOREABLE_LIVE — base is the h3-1b round-level "
                         "state-space filter; round enrichment absent for CN "
                         "events and the live post-freeze corpus (disclosed in "
                         "preregister.finish.md; never silently substituted)")},
        {"id": "E_atlas_replication", "role": "reference", "scoreable": True,
         "label": ROSTER_INT["prospective_validation_plan"]["arms_frozen"]["E_atlas_replication"],
         "config": {"mechanism": "non_model_check",
                    "stat": "mean(won - p_v6), team-obs with msc<=2 in sustained episodes",
                    "replicates_iff": "pooled bias > 0 with 95% CI excluding 0"}},
        {"id": "F_v6_overreact", "role": "reference", "scoreable": True,
         "label": ROSTER_INT["prospective_validation_plan"]["arms_frozen"]["F_v6_overreact"],
         "config": {"mechanism": "v6_overreact", "a": 2.0, "tau": 5.0}},
        {"id": "H_specrun", "role": "reference", "scoreable": True,
         "label": ROSTER_INT["prospective_validation_plan"]["arms_frozen"]["H_specrun"],
         "config": {"mechanism": "solve_side_spec", "policy": "p1", "W": 8, "c": 5,
                    "a": 28.0, "tau": 13.0, "s": 0.7, "n_min": 3, "cap": 1.5}},
    ]
    prior_arms = {a["id"]: a for a in prior.get("arms", [])}

    # one engine, one episode build; every solve-side arm reruns the solve
    eng = EngineV9()
    eng.series = frame.copy().reset_index(drop=True)
    eng.pred_days = sorted(frame.date.unique())
    eng.prepare_rows()
    eps_by_org, org_dates, st_ix = build_episodes(eng)
    corpus = {"team_match_seq": {o: list(s) for o, s in eng.team_match_seq.items()},
              "lineups": eng.lineups}

    rdiffs, valids = {}, {}
    eng.disable_all()
    out_v6 = run_arm(eng, frame, daily=True)
    rdiffs["v6"] = out_v6["rdiff"]
    eng.disable_all()
    eng.enable_change(eps_by_org, gamma=2.0)
    rdiffs["B_continuity"] = run_arm(eng, frame)["rdiff"]
    eng.disable_all()
    rdiffs["C_coldstart"] = coldstart_rdiff(frame, out_v6, eng, eps_by_org,
                                            org_dates, a0=1.0, M=6)
    eng.enable_overreact(2.0, 5.0, eps_by_org, org_dates)
    rdiffs["F_v6_overreact"] = run_arm(eng, frame)["rdiff"]
    eng.disable_all()
    plan = speclib.SpecPlan("p1", W=8, c=5, corpus=corpus)
    eng.enable_spec(plan, a=28.0, tau=13.0, s=0.7, n_min=3, cap=1.5)
    rdiffs["H_specrun"] = run_arm(eng, frame)["rdiff"]
    eng.disable_all()
    for k, v in rdiffs.items():
        valids[k] = ~np.isnan(v)

    # beta freeze: ladder verbatim for v6 (+ reproduction fixture); reference
    # arms frozen ONCE (reused from the prior scoreboard on every later run)
    beta_fix = prior.get("beta_freeze_fixture")
    if beta_fix is None:
        b_chk, n_chk = fit_beta_freeze(rdiffs["v6"], frame, valids["v6"])
        beta_fix = {"ladder_beta_v6": BETA_V6, "protocol_refit_on_live_frame": b_chk,
                    "n_freeze_rows": n_chk, "gap": round(abs(b_chk - BETA_V6), 6),
                    "rule": "ladder value used VERBATIM regardless; gap > 0.001 would be logged loudly"}
        if beta_fix["gap"] > 1e-3:
            log(f"WARNING v6 beta reproduction gap {beta_fix['gap']} > 1e-3 "
                f"(live-frame revision?) — ladder beta still used verbatim")
    for a in arms:
        if a["id"] == "v6" or not a["scoreable"] or a["id"] == "E_atlas_replication":
            continue
        pa = prior_arms.get(a["id"], {})
        if pa.get("beta_frozen") is not None:
            a["beta_frozen"] = pa["beta_frozen"]
            a["beta_provenance"] = pa.get("beta_provenance", "frozen at first init, reused")
        else:
            b, nfit = fit_beta_freeze(rdiffs[a["id"]], frame, valids[a["id"]])
            a["beta_frozen"] = b
            a["beta_provenance"] = (f"frozen ONCE at first init ({time.strftime('%Y-%m-%d')}) "
                                    f"on {FREEZE_LO}..{CUTOFF} ({nfit} rows), protocol beta "
                                    f"method; reused verbatim on every later run")
    arm_by_id = {a["id"]: a for a in arms}
    arm_by_id["E_atlas_replication"]["beta_frozen"] = None

    # paired scoring rows: NaN for ANY scoreable model arm drops the row for ALL
    model_arms = [a["id"] for a in arms if a["scoreable"] and a["id"] != "E_atlas_replication"]
    paired_ok = np.ones(len(frame), dtype=bool)
    for k in model_arms:
        paired_ok &= valids[k]
    scored_m = post & paired_ok
    n_scored = int(scored_m.sum())
    n_dropped = int(post.sum() - n_scored)
    fmts = frame.fmt.values

    per_series = []
    idx_scored = np.where(scored_m)[0]
    for i in idx_scored:
        r = frame.iloc[i]
        row = {"match_id": int(r.match_id), "date": r.date, "event_id": r.event_id,
               "fmt": r.fmt,
               "p": {k: round(float(p_series_closed(arm_by_id[k]["beta_frozen"],
                                                    np.array([rdiffs[k][i]]),
                                                    np.array([fmts[i]]))[0]), 6)
                     for k in model_arms}}
        # arm E raw material (non-aggregated): per-side post-change obs
        eobs = []
        for org, won in ((r.winner, 1), (r.loser, 0)):
            s = st_ix.get((org, int(r.match_id)))
            if s is None or s[1] < 0 or s[0] > 2:
                continue
            e = eps_by_org[org][s[1]]
            if sustained_at(e, r.date):
                eobs.append({"org": org, "won": won, "msc": s[0], "ov": s[2]})
        if eobs:
            row["atlas_obs"] = eobs
        per_series.append(row)

    # checkpoint logic — the ONLY place any post-cutoff aggregate is computed
    reads = list(prior.get("checkpoint_reads", []))
    done_cps = {r["checkpoint_n"] for r in reads}
    statuses = {a["id"]: dict(prior_arms.get(a["id"], {}).get("live_status", {}))
                for a in arms}
    for cp in CHECKPOINTS:
        if n_scored < cp or cp in done_cps:
            continue
        rows_cp = idx_scored[:cp]
        sub = frame.iloc[rows_cp].reset_index(drop=True)
        # protocol integrity.journal: ledger BEFORE any delta
        log(f"CHECKPOINT n={cp} ledger: dropped(pairing)={n_dropped}; "
            f"match_ids={sub.match_id.tolist()}")
        p_v6 = p_series_closed(BETA_V6, rdiffs["v6"][rows_cp], fmts[rows_cp])
        rd_v6 = rdiffs["v6"][rows_cp]
        ev = sub.event_id.values
        cp_out = {"checkpoint_n": cp, "read_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                  "n_dropped_pairing": n_dropped, "arms": []}
        for a in arms:
            if not a["scoreable"] or a["id"] in ("v6", "E_atlas_replication"):
                continue
            p_a = p_series_closed(a["beta_frozen"], rdiffs[a["id"]][rows_cp], fmts[rows_cp])
            d = referee.delta_vector(p_a, p_v6)
            r_ = checkpoint_read(cp, a["id"], a["role"], d, ev, p_a, p_v6, sub, rd_v6)
            cp_out["arms"].append(r_)
            st = statuses.setdefault(a["id"], {})
            if r_["KILL_ci_hi_lt_0"]:
                st["status"] = "KILLED"
                st["killed_at_n"] = cp
            elif r_["promotable"] and st.get("status") != "KILLED":
                st["status"] = "PROMOTED"       # G5 vacuous: zero candidates
                st["promoted_at_n"] = cp
            elif st.get("status") not in ("KILLED", "PROMOTED"):
                st["status"] = "ALIVE"
        # arm E at checkpoint: pooled atlas bias with iid CRN CI
        obs = [(o["won"], row["p"]["v6"] if o["won"] else 1 - row["p"]["v6"])
               for row in per_series[:] for o in row.get("atlas_obs", [])
               if row["match_id"] in set(sub.match_id)]
        if obs:
            dv = np.array([w - p for w, p in obs])
            bb = referee.paired_bootstrap_crn(dv, mode="iid")
            cp_out["E_atlas_replication"] = {
                "n_obs": len(obs), "bias_pp": round(float(dv.mean()) * 100, 2),
                "ci_pp": [round(bb["ci_lo"] * 100, 2), round(bb["ci_hi"] * 100, 2)],
                "replicates": bool(bb["ci_lo"] > 0)}
        else:
            cp_out["E_atlas_replication"] = {"n_obs": 0, "note": "no qualifying team-obs"}
        reads.append(cp_out)
        done_cps.add(cp)
        # protocol: every checkpoint read appends to v9_looks prospective_reads
        looks = json.load(open(LOOKS_PATH))
        looks["prospective_reads"]["entries"].append(
            {"checkpoint_n": cp, "date": time.strftime("%Y-%m-%d"),
             "by": "score_prospective.py",
             "arms": {r_["arm"]: {"delta_milli": r_["delta_milli"],
                                  "blk_ci_milli": r_["block_event"]["ci_milli"],
                                  "status": statuses[r_["arm"]].get("status")}
                      for r_ in cp_out["arms"]}})
        with open(LOOKS_PATH, "w") as f:
            json.dump(looks, f, indent=1)
        log(f"CHECKPOINT n={cp} read complete; appended to v9_looks prospective_reads")

    next_cp = next((c for c in CHECKPOINTS if c not in done_cps), None)
    for a in arms:
        st = statuses.setdefault(a["id"], {})
        if a["id"] == "v6":
            st["status"] = "BASELINE"
        elif not a["scoreable"]:
            st["status"] = "NOT_SCOREABLE_LIVE"
        elif not st.get("status"):
            st["status"] = "ACCUMULATING"
        a["live_status"] = st

    verdict = (f"accumulating (n={n_scored})" if n_scored < CHECKPOINTS[0]
               else f"scored through n={n_scored}")
    board = {
        "written_by": "agent:v9-finish (testing_lab/v9/score_prospective.py)",
        "last_run": time.strftime("%Y-%m-%d %H:%M:%S"),
        "protocol": "stats/v9_prospective_protocol.json — implemented verbatim; "
                    "this file is the standing scoreboard it feeds",
        "run_cadence": "manual or cron weekly; idempotent, read-only over data/, refits nothing",
        "ladder": {"file": "stats/v9_ladder.json",
                   "verdict_sentence": LADDER["verdict_sentence"],
                   "n_candidate_arms_beyond_v6": 0,
                   "note": "zero of 5 v9 candidates advanced; there is nothing to promote"},
        "candidates": [],
        "arms": arms,
        "population": {"rule": f"settled series date > {CUTOFF}, frame recipe verbatim",
                       "n_settled_post_cutoff": n_post,
                       "n_scored_paired": n_scored,
                       "n_dropped_pairing": n_dropped,
                       "frame_fixture": fx, "exclusions": excl},
        "beta_freeze_fixture": beta_fix,
        "checkpoints": {"at_scored_n": CHECKPOINTS,
                        "G1_thresholds": G1_THR,
                        "alpha_spend": PROTOCOL["promotion_rule"]["G1_sequential_evidence"]["alpha_spending"],
                        "mde_within_milli": MDE_WITHIN, "mde_cross_milli": MDE_CROSS,
                        "kill_rule": PROTOCOL["kill_rule"]["rule"],
                        "next_checkpoint": next_cp,
                        "reads_taken": sorted(done_cps),
                        "reads_remaining": PROTOCOL["checkpoints"]["reads"]},
        "checkpoint_reads": reads,
        "per_series": per_series,
        "no_peeking": "no aggregate over post-cutoff rows exists outside checkpoint_reads; "
                      "per-series predictions above are stored without aggregation "
                      "(protocol integrity clause)",
        "termination": PROTOCOL["termination"],
        "verdict": verdict,
    }
    with open(SCOREBOARD_PATH, "w") as f:
        json.dump(board, f, indent=1)
    log(f"run done in {time.time() - t0:.1f}s: n_settled={n_post} n_scored={n_scored} "
        f"dropped={n_dropped} next_checkpoint={next_cp} verdict='{verdict}' "
        f"statuses={{{', '.join(a['id'] + ':' + a['live_status']['status'] for a in arms)}}}")


if __name__ == "__main__":
    main()
