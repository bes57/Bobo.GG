"""agent:bias-h3 shared machinery — state-space rating filter on the expanded frame.

Everything here follows preregister.bias_h3.md exactly:
  observation y = sign(rd)|rd|^0.75 * 2.5 per map, R fixed = Var(y) on TRAIN games,
  R_i = R / w_i with v6's hand-set weights (champ x2 exact-shape, playoffs x1.6),
  q per map-game (both participants tick at each of their own maps),
  strict-day leak rule (predictions for day D see states before any day-D game),
  p_series = GH-20 integral of series_wp(sigmoid(beta*delta)) over N(dr, vA+vB),
  beta refit train-only per config.
"""
import hashlib
import json
import math
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

HERE = os.path.dirname(os.path.abspath(__file__))
TL = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
V8 = os.path.join(TL, "v8")
DATA = os.path.join(os.path.dirname(TL), "data")
sys.path.insert(0, TL)
if V8 not in sys.path:
    sys.path.insert(0, V8)

FRAME_PATH = os.path.join(V8, "data", "frame_expanded", "series.csv")
TRAIN_END = "2024-12-31"

_GH_X, _GH_W = np.polynomial.hermite.hermgauss(20)
_GH_WN = _GH_W / math.sqrt(math.pi)          # normalized weights, sum = 1


def load_frame(verify=True):
    if verify:
        crn = json.load(open(os.path.join(V8, "crn.json")))
        sha = hashlib.sha256(open(FRAME_PATH, "rb").read()).hexdigest()
        if sha != crn["frame_expanded"]["series_csv_sha256"]:
            raise RuntimeError(f"FRAME SHA MISMATCH: {sha}")
    return pd.read_csv(FRAME_PATH)


class GameData:
    """Immutable arrays for the filter. Built once from Engine's game list."""

    def __init__(self, eng=None, frame=None):
        if frame is None:
            frame = load_frame()
        if eng is None:
            from engine import Engine
            eng = Engine()
        self.frame = frame.reset_index(drop=True)
        self.teams = eng.teams
        self.tidx = eng.tidx
        self.n_teams = len(eng.teams)
        g = eng.games
        self.n_games = len(g)
        self.g_mid = np.array([x["match_id"] for x in g])
        self.g_date = np.array([x["date_s"] for x in g])
        self.wi = eng.wi.copy()
        self.li = eng.li.copy()
        rd = np.array([x["wr"] - x["lr"] for x in g], dtype=float)
        self.y = np.sign(rd) * np.abs(rd) ** 0.75 * 2.5    # winner-referenced
        # v6 hand-set weights
        champ = np.array([re.fullmatch(r"\d{4}_champions", x["event_id"]) is not None
                          for x in g])
        stage_by_mid = dict(zip(self.frame.match_id, self.frame.stage))
        g_stage = np.array([stage_by_mid.get(x["match_id"], "groups") for x in g])
        po = np.where(np.isin(g_stage, ("playoffs", "grand_final")), 1.6, 1.0)
        self.w = np.where(champ, 2.0, 1.0) * po
        # train-only observation-noise scale (identification constant)
        g_train = self.g_date <= TRAIN_END
        self.R = float(np.var(self.y[g_train]))
        self.n_games_train = int(g_train.sum())
        # frame row indexing
        self.f_wi = np.array([self.tidx[t] for t in self.frame.winner])
        self.f_li = np.array([self.tidx[t] for t in self.frame.loser])
        self.f_date = self.frame.date.values
        self.fmts = self.frame.fmt.values
        self.train_mask = (self.frame.date <= TRAIN_END).values
        self.holdout_mask = (self.frame.date > TRAIN_END).values
        # day-grouped iteration order (engine games already sorted (date_s, mid))
        days = sorted(set(self.g_date) | set(self.f_date))
        self.days = days
        idx_by_day = {}
        for j in range(self.n_games):
            idx_by_day.setdefault(self.g_date[j], []).append(j)
        self.games_by_day = {d: np.array(v) for d, v in idx_by_day.items()}
        rows_by_day = {}
        for j, d in enumerate(self.f_date):
            rows_by_day.setdefault(d, []).append(j)
        self.rows_by_day = {d: np.array(v) for d, v in rows_by_day.items()}
        # region trailing info for the debut-prior variant
        self.team_region_idx = eng.team_region_idx.copy()

    # ── the filter ──────────────────────────────────────────────────────────
    def run_filter(self, q, V0, q_vec=None, q_cal_week=0.0, debut_region_prior=False,
                   collect_z=False):
        """One walk-forward pass. Returns per-frame-row (mu, s2) pre-match,
        per-row pre-match team variances (v_w, v_l), and optionally the
        per-game standardized innovations (for the volatility axis).

        q_vec: optional per-game per-side process noise, shape (n_games, 2)
        [winner-side q, loser-side q]; overrides scalar q when given.
        q_cal_week: extra variance per calendar week since the team's last game.
        debut_region_prior: initialize a debuting team's mean at the trailing
        mean rating of already-observed teams of its region (walk-forward).
        """
        nT = self.n_teams
        r = np.zeros(nT)
        v = np.full(nT, float(V0))
        seen = np.zeros(nT, dtype=bool)
        last_day = np.full(nT, np.nan)     # ordinal day of team's last game
        n_upd = np.zeros(nT, dtype=int)
        R = self.R
        mu = np.full(len(self.frame), np.nan)
        s2 = np.full(len(self.frame), np.nan)
        vw_out = np.full(len(self.frame), np.nan)
        vl_out = np.full(len(self.frame), np.nan)
        z_rec = ([], [], []) if collect_z else None   # (game_idx, side, z)
        if not hasattr(self, "_d_ord"):
            self._d_ord = pd.to_datetime(self.days).values.astype(
                "datetime64[D]").astype(int)
        d_ord = self._d_ord

        for di, day in enumerate(self.days):
            dnum = d_ord[di]
            # 1) predictions for this day from the pre-day snapshot
            rows = self.rows_by_day.get(day)
            if rows is not None:
                for i in rows:
                    a, b = self.f_wi[i], self.f_li[i]
                    ra = self._debut_mean(a, r, seen) if (debut_region_prior and not seen[a]) else r[a]
                    rb = self._debut_mean(b, r, seen) if (debut_region_prior and not seen[b]) else r[b]
                    mu[i] = ra - rb
                    s2[i] = v[a] + v[b]
                    vw_out[i] = v[a]
                    vl_out[i] = v[b]
            # 2) absorb this day's games
            gs = self.games_by_day.get(day)
            if gs is None:
                continue
            for j in gs:
                a, b = self.wi[j], self.li[j]
                # debut initialization (mean only; variance stays V0)
                for t in (a, b):
                    if debut_region_prior and not seen[t]:
                        r[t] = self._debut_mean(t, r, seen)
                    seen[t] = True
                # process noise ticks (per own game)
                qa = q_vec[j, 0] if q_vec is not None else q
                qb = q_vec[j, 1] if q_vec is not None else q
                v[a] += qa
                v[b] += qb
                if q_cal_week > 0.0:
                    for t, in ((a,), (b,)):
                        if not np.isnan(last_day[t]):
                            v[t] += q_cal_week * (dnum - last_day[t]) / 7.0
                e = self.y[j] - (r[a] - r[b])
                S = v[a] + v[b] + R / self.w[j]
                if collect_z:
                    z = e / math.sqrt(S)
                    z_rec[0].append(j)
                    z_rec[1].append(a)
                    z_rec[2].append(z)
                    z_rec[0].append(j)
                    z_rec[1].append(b)
                    z_rec[2].append(-z)
                ka = v[a] / S
                kb = v[b] / S
                r[a] += ka * e
                r[b] -= kb * e
                v[a] -= v[a] * v[a] / S
                v[b] -= v[b] * v[b] / S
                last_day[a] = dnum
                last_day[b] = dnum
                n_upd[a] += 1
                n_upd[b] += 1
        out = {"mu": mu, "s2": s2, "v_w": vw_out, "v_l": vl_out,
               "r_final": r, "v_final": v, "n_upd": n_upd}
        if collect_z:
            out["z"] = (np.array(z_rec[0]), np.array(z_rec[1]),
                        np.array(z_rec[2], dtype=float))
        return out

    def _debut_mean(self, t, r, seen):
        reg = self.team_region_idx[t]
        if reg < 0:
            return 0.0
        m = seen & (self.team_region_idx == reg)
        return float(r[m].mean()) if m.sum() >= 4 else 0.0

    # ── probability surface ────────────────────────────────────────────────
    def p_series_gh(self, beta, mu, s2, mask):
        """Integrated series prob for masked rows (vectorized over GH nodes)."""
        m_mu = mu[mask][:, None]
        m_sd = np.sqrt(s2[mask])[:, None]
        delta = m_mu + math.sqrt(2.0) * m_sd * _GH_X[None, :]
        pm = 1.0 / (1.0 + np.exp(-beta * delta))
        fm = self.fmts[mask]
        is5 = np.isin(fm, ("bo5", "bo5_gf"))[:, None]
        is1 = (fm == "bo1")[:, None]
        ps = np.where(is5, pm ** 3 * (1 + 3 * (1 - pm) + 6 * (1 - pm) ** 2),
                      np.where(is1, pm, pm ** 2 * (3 - 2 * pm)))
        return ps @ _GH_WN

    def fit_beta(self, mu, s2, mask=None):
        """Train-only beta fit on the integrated surface (engine bounds)."""
        m = self.train_mask if mask is None else mask
        m = m & ~np.isnan(mu)

        def nll(b):
            p = self.p_series_gh(b, mu, s2, m)
            return -np.mean(np.log(np.clip(p, 1e-9, 1)))

        res = minimize_scalar(nll, bounds=(0.03, 0.6), method="bounded")
        return float(res.x), float(res.fun)

    def score(self, beta, mu, s2):
        """Full-frame integrated probs + train/holdout NLL."""
        ok = ~np.isnan(mu)
        p = np.full(len(self.frame), np.nan)
        p[ok] = self.p_series_gh(beta, mu, s2, ok)
        tr = ok & self.train_mask
        ho = ok & self.holdout_mask
        lltr = float(-np.mean(np.log(np.clip(p[tr], 1e-9, 1))))
        llho = float(-np.mean(np.log(np.clip(p[ho], 1e-9, 1))))
        return {"p": p, "ll_train": lltr, "ll_holdout": llho,
                "n_train": int(tr.sum()), "n_holdout": int(ho.sum())}

    def eval_config(self, q, V0, **kw):
        """filter -> beta fit (train) -> scores. The unit of every sweep."""
        f = self.run_filter(q, V0, **kw)
        beta, tr_nll = self.fit_beta(f["mu"], f["s2"])
        sc = self.score(beta, f["mu"], f["s2"])
        return {"q": q, "V0": V0, "beta": beta, **{k: sc[k] for k in
                ("ll_train", "ll_holdout", "n_train", "n_holdout")},
                "p": sc["p"], "filter": f}


def implied_half_life(q, R):
    """Steady-state Riccati at R_bar=R, w=1: HL_games = ln2 / -ln(1-K*)."""
    P = R
    for _ in range(500):
        Pn = (P + q) * R / (P + q + R)
        if abs(Pn - P) < 1e-14:
            P = Pn
            break
        P = Pn
    K = (P + q) / (P + q + R)
    if K >= 1.0:
        return 0.0
    return math.log(2) / -math.log(1.0 - K)


def steady_state_neff(q, R):
    """R / P* — effective sample size of the steady-state posterior."""
    P = R
    for _ in range(500):
        P = (P + q) * R / (P + q + R)
    return R / P
