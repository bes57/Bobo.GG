"""agent:decay runner — engine.Engine.run solve loop, replicated with per-game
half-life and per-(game,day) weight hooks, vectorized per org.

VALIDATION GATE (preregister.decay.md): reproduce eng.run(v6) rdiff to atol
1e-9 + identical train beta before any variant is trusted.
"""
import hashlib
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

TL = "/Users/benny_es1/PythonTest/testing_lab"
V8 = os.path.join(TL, "v8")
SCR = os.path.join(V8, "scratch", "decay")
PROBS = os.path.join(SCR, "probs")
LOG = os.path.join(V8, "logs", "decay.log")
FRAME = os.path.join(V8, "data", "frame_expanded", "series.csv")
os.makedirs(PROBS, exist_ok=True)
sys.path.insert(0, TL)

LN2 = math.log(2)


def jlog(msg):
    from datetime import datetime
    line = f"[{datetime.now().strftime('%F %T')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def load_frame():
    sha = hashlib.sha256(open(FRAME, "rb").read()).hexdigest()
    crn = json.load(open(os.path.join(V8, "crn.json")))
    want = crn["frame_expanded"]["series_csv_sha256"]
    if sha != want:
        raise RuntimeError(f"FRAME SHA MISMATCH {sha} != {want} — aborting")
    f = pd.read_csv(FRAME, dtype={"date": str})
    assert len(f) == 2058 and int((f.date > "2024-12-31").sum()) == 1217
    return f


def sp(pm, fm):
    return np.where(np.isin(fm, ("bo5", "bo5_gf")),
                    pm ** 3 * (1 + 3 * (1 - pm) + 6 * (1 - pm) ** 2),
                    np.where(fm == "bo1", pm, pm ** 2 * (3 - 2 * pm)))


class Runner:
    """Read-only shared precomputation over the engine's game corpus +
    expanded frame. One instance shared across configs (state per run_cfg call
    is local; region-prior prev-vec is reset inside run_cfg)."""

    def __init__(self):
        from engine import Engine
        eng = Engine()
        frame = load_frame()
        eng.series = frame.reset_index(drop=True)
        eng.pred_days = sorted(frame.date.unique())
        self.eng = eng
        self.frame = eng.series
        self.games = eng.games
        self.n_g = len(self.games)
        self.g_dnum = eng.g_dnum
        assert np.all(np.diff(self.g_dnum) >= 0), "games not date-prefix-sorted"
        self.wi, self.li = eng.wi, eng.li
        self.teams, self.tidx = eng.teams, eng.tidx
        self.n_t = len(self.teams)
        self.champ = eng.champ
        self.rd_raw = eng.rd_raw
        self.team_region_idx = eng.team_region_idx
        # rd transform (BASE: power .75 scale 2.5)
        self.rd_t = np.copysign(np.abs(self.rd_raw) ** 0.75 * 2.5, self.rd_raw)
        # playoff multiplier from the frame's stage
        stage_by_mid = dict(zip(self.frame.match_id, self.frame.stage))
        g_stage = np.array([stage_by_mid.get(g["match_id"], "groups")
                            for g in self.games])
        po = np.where(np.isin(g_stage, ("playoffs", "grand_final")), 1.6, 1.0)
        self.mult = np.where(self.champ, 2.0, 1.0) * po
        # per-org structures (rows ascending == date order)
        self.team_game_rows = eng.team_game_rows
        self.orgs = sorted(self.team_game_rows)
        self.org_rows = {o: np.array(r) for o, r in self.team_game_rows.items()}
        self.org_dnum = {o: self.g_dnum[r] for o, r in self.org_rows.items()}
        self.org_is_win = {o: np.array([self.games[ri]["winner"] == o
                                        for ri in r])
                           for o, r in self.org_rows.items()}
        self.org_rank = {o: np.arange(len(r), dtype=float)
                         for o, r in self.org_rows.items()}
        # v6 consistency flags (engine._game_consist replica, walk-forward)
        self.consist = {}
        lam_wr = LN2 / 16.0
        for org, rows_all in self.team_game_rows.items():
            wr_num = wr_den = 0.0
            for ri in rows_all:
                won = self.games[ri]["winner"] == org
                wr = wr_num / wr_den if wr_den > 0.5 else 0.5
                self.consist[(org, ri)] = (
                    (won and wr >= 0.5) or ((not won) and wr < 0.5))
                wr_num = wr_num * math.exp(-lam_wr) + (1.0 if won else 0.0)
                wr_den = wr_den * math.exp(-lam_wr) + 1.0
        # year continuity per ref_year (engine _continuity_vec 'year' replica)
        self.year_cont = eng.year_cont
        gyears = eng.g_date.astype("U4").astype(int)
        self.gyears = gyears
        self.cont_by_refyear = {}
        for ry in (2023, 2024, 2025, 2026):
            cache = {}

            def f(org, gy, ry=ry, cache=cache):
                key = (org, gy)
                if key not in cache:
                    v = 1.0
                    for by in range(gy + 1, ry + 1):
                        c = self.year_cont.get((org, by))
                        if c is not None:
                            v *= c
                    cache[key] = v
                return cache[key]
            cw = np.array([f(g["winner"], gyears[i])
                           for i, g in enumerate(self.games)])
            cl = np.array([f(g["loser"], gyears[i])
                           for i, g in enumerate(self.games)])
            self.cont_by_refyear[ry] = np.sqrt(cw * cl)
        # series bookkeeping
        s = self.frame
        self.fmts = s.fmt.values
        self.train_v = (s.date <= "2024-12-31").values
        self.test_v = (s.date > "2024-12-31").values
        self.s_by_day = defaultdict(list)
        for i, r in enumerate(s.itertuples(index=False)):
            self.s_by_day[r.date].append(i)
        self.pred_days = eng.pred_days
        self.pred_dnum = {d: int(np.datetime64(d, "D").astype(int))
                          for d in self.pred_days}

    # ── per-side lam arrays for a config ────────────────────────────────────
    def lam_arrays(self, mode, hl=20.0, hl_c=20.0, hl_a=12.0,
                   hl_anom_override=None, class_mult=None):
        """Returns lam_org[org] arrays aligned to org_rows[org].
        mode 'sym': constant hl. mode 'consist': hl_c/hl_a per consist flag;
        hl_anom_override[(org, ri)] replaces hl_a where present.
        class_mult: per-game HL multiplier array (aligned to games)."""
        lam = {}
        for org, rows in self.org_rows.items():
            if mode == "sym":
                h = np.full(len(rows), float(hl))
            elif mode == "consist":
                h = np.empty(len(rows))
                for k, ri in enumerate(rows):
                    if self.consist[(org, ri)]:
                        h[k] = hl_c
                    else:
                        h[k] = (hl_anom_override.get((org, ri), hl_a)
                                if hl_anom_override else hl_a)
            else:
                raise ValueError(mode)
            if class_mult is not None:
                h = h * class_mult[rows]
            lam[org] = LN2 / h
        return lam

    # ── one walk-forward run ────────────────────────────────────────────────
    def run_cfg(self, name, lam_org, year_cont=True, lineup_ov=None,
                lineup_gamma=None, rot_dates=None, rot_gamma=None,
                daily_out=False, cache=True):
        """lam_org: {org: lam array}. lineup_ov: {org: (state_dnum, OV matrix
        states x org_games)} raw overlap ratios; factor = max(ov,.04)^gamma.
        rot_dates: sorted int day-nums; weight *= rot_gamma^(#rot in (g,D])."""
        cpath = os.path.join(PROBS, f"{name}.npz")
        if cache and os.path.exists(cpath):
            z = np.load(cpath)
            return {"name": name, "rdiff": z["rdiff"], "beta": float(z["beta"]),
                    "p": z["p"], "cached": True}
        n_g, n_t = self.n_g, self.n_t
        s = self.frame
        rat_w = np.full(len(s), np.nan)
        rat_l = np.full(len(s), np.nan)
        w_w = np.ones(n_g)
        w_l = np.ones(n_g)
        R_g = None
        if rot_dates is not None:
            R_g = np.searchsorted(rot_dates, self.g_dnum, side="right")
        daily = {} if daily_out else None
        prev_rvec = None
        for day in self.pred_days:
            dn = self.pred_dnum[day]
            n_hist = int(np.searchsorted(self.g_dnum, dn, side="left"))
            if n_hist < 30:
                continue
            for org in self.orgs:
                od = self.org_dnum[org]
                k = int(np.searchsorted(od, dn, side="left"))
                if k == 0:
                    continue
                ago = (k - 1) - self.org_rank[org][:k]
                wv = np.exp(-lam_org[org][:k] * ago)
                if lineup_ov is not None:
                    sdn, OV = lineup_ov[org]
                    si = int(np.searchsorted(sdn, dn, side="left")) - 1
                    if si >= 0:
                        wv = wv * np.maximum(OV[si, :k], 0.04) ** lineup_gamma
                rows = self.org_rows[org][:k]
                iw = self.org_is_win[org][:k]
                w_w[rows[iw]] = wv[iw]
                w_l[rows[~iw]] = wv[~iw]
            base = np.sqrt(w_w[:n_hist] * w_l[:n_hist])
            if year_cont:
                base = base * self.cont_by_refyear[int(day[:4])][:n_hist]
            if rot_dates is not None:
                R_D = int(np.searchsorted(rot_dates, dn, side="right"))
                base = base * rot_gamma ** (R_D - R_g[:n_hist])
            w = base * self.mult[:n_hist]
            wi, li = self.wi[:n_hist], self.li[:n_hist]
            rdv = self.rd_t[:n_hist]
            M = np.zeros((n_t, n_t))
            p = np.zeros(n_t)
            np.add.at(M, (wi, wi), w)
            np.add.at(M, (li, li), w)
            np.add.at(M, (wi, li), -w)
            np.add.at(M, (li, wi), -w)
            np.add.at(p, wi, w * rdv)
            np.add.at(p, li, -w * rdv)
            M[np.diag_indices(n_t)] += 0.5          # ridge
            rpr = 1.5                                # region_prior_ridge
            prior = np.zeros(n_t)
            if prev_rvec is not None:
                for ri_ in range(4):
                    m_reg = self.team_region_idx == ri_
                    if m_reg.sum() >= 4:
                        prior[m_reg] = prev_rvec[m_reg].mean()
            M[np.diag_indices(n_t)] += rpr
            p += rpr * prior
            M[-1, :] = 1.0
            p[-1] = 0.0
            try:
                r_vec = np.linalg.solve(M, p)
            except np.linalg.LinAlgError:
                r_vec, *_ = np.linalg.lstsq(M, p, rcond=None)
            prev_rvec = r_vec.copy()
            if daily is not None:
                daily[day] = r_vec.copy()
            for i in self.s_by_day.get(day, ()):
                row = s.iloc[i]
                rat_w[i] = r_vec[self.tidx[row.winner]] if row.winner in self.tidx else 0.0
                rat_l[i] = r_vec[self.tidx[row.loser]] if row.loser in self.tidx else 0.0
        rdiff = rat_w - rat_l
        valid = ~np.isnan(rdiff)
        train = valid & self.train_v
        fmts = self.fmts

        def nll(beta, mask):
            pm = 1 / (1 + np.exp(-beta * rdiff[mask]))
            pv = sp(pm, fmts[mask])
            return -np.mean(np.log(np.clip(pv, 1e-9, 1)))

        beta = float(minimize_scalar(lambda b: nll(b, train),
                                     bounds=(0.03, 0.6), method="bounded").x)
        with np.errstate(invalid="ignore"):
            p_all = sp(1 / (1 + np.exp(-beta * rdiff)), fmts)
        res = {"name": name, "rdiff": rdiff, "beta": beta, "p": p_all,
               "cached": False}
        if daily is not None:
            res["daily"] = daily
        if cache:
            np.savez(cpath, rdiff=rdiff, beta=beta, p=p_all)
        return res

    def ll(self, p, mask):
        return float(-np.mean(np.log(np.clip(p[mask], 1e-9, 1))))


# ── lineup machinery ─────────────────────────────────────────────────────────

def build_lineup_tables(rn):
    """Unified per-(org, match) lineup sets from the engine's maps-CSV load
    (identical grouping to lineups.csv), + per-org state sequences and
    overlap matrices for the continuity axis. Also writes the corpus-addition
    top-up audit to scratch."""
    eng = rn.eng
    lups = eng.lineups  # (org, mid) -> frozenset
    org_matches = {}    # org -> [(dnum, mid, lineup)]
    seen = set()
    for g in rn.games:
        for org in (g["winner"], g["loser"]):
            k = (org, g["match_id"])
            if k in seen:
                continue
            seen.add(k)
            org_matches.setdefault(org, []).append(
                (int(np.datetime64(g["date_s"], "D").astype(int)),
                 g["match_id"], lups.get(k)))
    for org in org_matches:
        org_matches[org].sort(key=lambda t: (t[0], t[1]))
    # overlap matrices: states = org's own matches (post-match lineup known)
    lineup_ov = {}
    for org, seq in org_matches.items():
        sdn = np.array([t[0] for t in seq])
        state_l = [t[2] for t in seq]
        rows = rn.org_rows[org]
        game_l = [lups.get((org, rn.games[ri]["match_id"])) for ri in rows]
        OV = np.ones((len(seq), len(rows)))
        for si, sl in enumerate(state_l):
            if not sl:
                continue
            for gi, gl in enumerate(game_l):
                if gl:
                    OV[si, gi] = len(sl & gl) / max(len(sl), len(gl), 5)
        lineup_ov[org] = (sdn, OV)
    return org_matches, lineup_ov


def matches_since_change(org_matches):
    """lineups-agent rule verbatim on the expanded table: walk back through
    the strictly-earlier org sequence while lineup equal. Returns
    {(org, mid): msc} (NaN-> None where lineup missing)."""
    out = {}
    for org, seq in org_matches.items():
        for i, (dn, mid, L) in enumerate(seq):
            if L is None:
                out[(org, mid)] = None
                continue
            msc = 0
            j = i - 1
            while j >= 0:
                dj, mj, Lj = seq[j]
                if dj >= dn:      # same-day: never history
                    j -= 1
                    continue
                if Lj is not None and Lj == L:
                    msc += 1
                    j -= 1
                else:
                    break
            out[(org, mid)] = msc
    return out


# ── rotation derivation (preregistered mechanical rule) ─────────────────────

def rotation_dates(rn, gap_days=60, cluster_days=14, min_date="2023-03-15"):
    by_map = defaultdict(list)
    for g in rn.games:
        if g["map_name"] == "TBD":
            continue
        by_map[g["map_name"]].append(g["date_s"])
    bounds = []
    for m, ds in by_map.items():
        ds = sorted(set(ds))
        d0 = np.array(ds, dtype="datetime64[D]")
        bounds.append(str(d0[0]))                       # entry
        gaps = (d0[1:] - d0[:-1]).astype(int)
        for i in np.where(gaps >= gap_days)[0]:
            bounds.append(str(d0[i] + 1))               # exit (day after last)
            bounds.append(str(d0[i + 1]))               # re-entry
    bounds = sorted(b for b in set(bounds) if b > min_date)
    clusters = []
    for b in bounds:
        if clusters and (np.datetime64(b) - np.datetime64(clusters[-1][0])
                         ).astype(int) <= cluster_days:
            clusters[-1].append(b)
        else:
            clusters.append([b])
    rots = [c[0] for c in clusters]
    return rots, {r: c for r, c in zip(rots, clusters)}
