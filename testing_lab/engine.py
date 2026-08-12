"""Fast vectorized walk-forward Massey engine for rating-level experiments.

Mirrors scrapers/BuildMapRatings.massey_ratings semantics (weights, RD
transform, roster continuity, ridge, prestige multipliers) but solves every
match day of 2023-2026 in one pass per config, so decay shapes / roster modes
/ margin transforms can be compared walk-forward in seconds.

Leak rule: ratings used for matches on day D are solved from games with
date < D (same as the timeline's `prev_ratings`).

Evaluation: beta fit on predictions through BETA_TRAIN_END, scored on the
holdout (2025-2026). Series-level closed form, identical to production's
past-match surface.
"""
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
sys.path.insert(0, os.path.join(ROOT, "scrapers"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from harness import load_series, series_wp  # noqa: E402

BETA_TRAIN_END = "2024-12-31"


# ── data assembly ────────────────────────────────────────────────────────────

def _pin_pythontest_modules():
    """Force MoreTestingMaybeFiles / BuildMapRatings / BuildRatingTimeline to
    come from THIS repo. harness puts /Users/benny_es1/VCTMM on sys.path (for
    teams.py), and importing vctmm.benpom adds its vendored dir — whose STALE
    copies of these modules (old event registry, old data snapshot) otherwise
    shadow the live ones."""
    for name in ("MoreTestingMaybeFiles", "BuildMapRatings",
                 "BuildRatingTimeline", "RefreshLiveData"):
        mod = sys.modules.get(name)
        if mod is not None and not str(getattr(mod, "__file__", "")).startswith(ROOT):
            del sys.modules[name]
    for p in (ROOT, os.path.join(ROOT, "scrapers")):
        while p in sys.path:
            sys.path.remove(p)
    sys.path.insert(0, ROOT)
    sys.path.insert(0, os.path.join(ROOT, "scrapers"))


def load_games_real_dates():
    _pin_pythontest_modules()
    """Map-level games with real dates (match_dates.json), as the timeline
    builder loads them. Import the site's own loader for bit-parity."""
    import BuildRatingTimeline as BRT
    games = BRT.load_all_games()
    for g in games:
        g["date_s"] = g["date"].strftime("%Y-%m-%d")
    games.sort(key=lambda g: (g["date_s"], g["match_id"]))
    return games


def load_event_rosters():
    """(org, event_id) -> frozenset(ProfileURL) from data/<eid>.csv files,
    plus EVENT_DATES for boundary ordering."""
    _pin_pythontest_modules()
    from BuildMapRatings import ALL_EVENTS, EVENT_DATES
    rosters = {}
    for ev in ALL_EVENTS:
        eid = ev["id"]
        path = os.path.join(DATA, f"{eid}.csv")
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path, usecols=["Org", "ProfileURL"])
        except Exception:
            continue
        for org, grp in df.groupby("Org"):
            urls = frozenset(grp["ProfileURL"].dropna().unique())
            if urls:
                rosters[(org, eid)] = urls
    return rosters, EVENT_DATES


def load_match_lineups():
    """(org, match_id) -> frozenset(ProfileURL) actually fielded, from
    maps/<eid>.csv player rows. The finest-grained roster signal we have."""
    _pin_pythontest_modules()
    from BuildMapRatings import ALL_EVENTS
    lineups = {}
    for ev in ALL_EVENTS:
        path = os.path.join(DATA, "maps", f"{ev['id']}.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, usecols=["Org", "ProfileURL", "MatchID"])
        for (org, mid), grp in df.groupby(["Org", "MatchID"]):
            lineups[(org, int(mid))] = frozenset(grp["ProfileURL"].dropna())
    return lineups


# ── decay families ───────────────────────────────────────────────────────────

def decay_weight(weeks, kind, **kw):
    """Vector weight for game age in weeks (np array)."""
    if kind == "exp":
        lam = math.log(2) / kw.get("hl", 6.0)
        return np.exp(-lam * weeks)
    if kind == "power":  # heavy tail: (1 + w/tau)^-alpha
        tau, alpha = kw.get("tau", 4.0), kw.get("alpha", 2.0)
        return (1.0 + weeks / tau) ** (-alpha)
    if kind == "linear":
        W = kw.get("W", 40.0)
        return np.clip(1.0 - weeks / W, 0.0, None)
    if kind == "boxexp":  # flat for c weeks, then exp
        c, hl = kw.get("c", 4.0), kw.get("hl", 6.0)
        lam = math.log(2) / hl
        return np.where(weeks <= c, 1.0, np.exp(-lam * (weeks - c)))
    raise ValueError(kind)


# ── engine ───────────────────────────────────────────────────────────────────

class Engine:
    def __init__(self, patch_dir=None):
        self.games = load_games_real_dates()
        _pin_pythontest_modules()
        from BuildMapRatings import INTL_EVENTS
        self.INTL = set(INTL_EVENTS)
        self.patch_lineups = {}
        if patch_dir and os.path.exists(os.path.join(patch_dir, "patch_games.csv")):
            pg = pd.read_csv(os.path.join(patch_dir, "patch_games.csv"))
            from datetime import datetime as _dt
            for r in pg.itertuples(index=False):
                self.games.append({
                    "match_id": int(r.match_id), "event_id": str(r.event_tag),
                    "map_name": r.map_name, "winner": r.winner, "loser": r.loser,
                    "wr": int(r.wr), "lr": int(r.lr),
                    "date": _dt.strptime(r.date, "%Y-%m-%d"), "date_s": r.date})
            self.games.sort(key=lambda g: (g["date_s"], g["match_id"]))
            lu_path = os.path.join(patch_dir, "patch_lineups.csv")
            if os.path.exists(lu_path):
                pl = pd.read_csv(lu_path)
                for (org, mid), grp in pl.groupby(["org", "match_id"]):
                    self.patch_lineups[(org, int(mid))] = frozenset(
                        grp["ProfileURL"].dropna())
            print(f"  [patch] +{len(pg)} maps, {pg.match_id.nunique()} matches "
                  f"from {patch_dir}")
        self.series = load_series()  # harness df with outcomes + formats
        # global team index
        teams = sorted({g["winner"] for g in self.games} |
                       {g["loser"] for g in self.games})
        self.teams = teams
        self.tidx = {t: i for i, t in enumerate(teams)}
        n_g = len(self.games)
        self.wi = np.array([self.tidx[g["winner"]] for g in self.games])
        self.li = np.array([self.tidx[g["loser"]] for g in self.games])
        self.rd_raw = np.array([g["wr"] - g["lr"] for g in self.games], dtype=float)
        self.g_date = np.array([g["date_s"] for g in self.games])
        self.g_dnum = pd.to_datetime(self.g_date).values.astype("datetime64[D]").astype(int)
        # v10 year-isolation support: calendar year of each game, so a solve can
        # be restricted to its own season. Cheap and always computed; only read
        # when cfg["year_isolated"] is on.
        self.g_year = self.g_date.astype("U4").astype(int)
        # Exact-shape predicate, not substring: the year-end Champions is
        # "YYYY_champions" exactly. Off-season ids like 2025_super_champions_cup
        # and 2023_china_champions_qual (corpus backfill, 2026-07-28) would
        # otherwise silently get the x2 Champions solve weight.
        import re as _re
        self.champ = np.array([_re.fullmatch(r"\d{4}_champions", g["event_id"])
                               is not None for g in self.games])
        # prediction days = unique series dates
        self.pred_days = sorted(self.series.date.unique())
        # roster structures
        self._build_roster_structures()
        # region index per team (for region-prior ridge)
        sys.path.insert(0, "/Users/benny_es1/VCTMM")
        from vctmm.benpom.teams import ORG_REGIONS as _OR
        _regs = ["Americas", "EMEA", "Pacific", "CN"]
        self.team_region_idx = np.array(
            [_regs.index(_OR[t]) if _OR.get(t) in _regs else -1 for t in self.teams])
        # per-team game sequences for games-based decay + break boundaries
        self.team_game_rows = defaultdict(list)   # org -> [game_row_idx] date order
        for i, g in enumerate(self.games):
            self.team_game_rows[g["winner"]].append(i)
            self.team_game_rows[g["loser"]].append(i)
        self.team_break_dates = {}  # org -> sorted ['YYYY-MM-DD'] break-return days
        for org, rows_ in self.team_game_rows.items():
            ds = sorted({self.g_date[i] for i in rows_})
            brk = []
            for a, b in zip(ds, ds[1:]):
                if (np.datetime64(b, "D") - np.datetime64(a, "D")).astype(int) > 45:
                    brk.append(b)
            self.team_break_dates[org] = brk

    # -- roster continuity ---------------------------------------------------
    def _build_roster_structures(self):
        rosters, EVENT_DATES = load_event_rosters()
        # production replica: per (team, calendar year) boundary factor
        hist = defaultdict(list)
        for (org, eid), r in rosters.items():
            end = EVENT_DATES.get(eid, (None, None))[1]
            if end:
                hist[org].append((end, eid, r))
        for org in hist:
            hist[org].sort()
        self.year_cont = {}
        for org, evs in hist.items():
            by_year = defaultdict(list)
            for end, eid, r in evs:
                by_year[int(end[:4])].append((end, eid, r))
            ys = sorted(by_year)
            for i in range(1, len(ys)):
                prev = max(by_year[ys[i - 1]])
                curr = min(by_year[ys[i]])
                self.year_cont[(org, ys[i])] = min(len(prev[2] & curr[2]) / 5.0, 1.0)
        # event-boundary continuity: org -> ordered [(end_date, roster)]
        self.event_hist = {org: [(e, r) for e, _, r in evs] for org, evs in hist.items()}
        # per-match lineups + per-team lineup ids for the continuous mode
        self.lineups = load_match_lineups()
        self.lineups.update(self.patch_lineups)
        self.team_match_seq = defaultdict(list)  # org -> [(date_s, match_id)]
        seen = set()
        for g in self.games:
            for org in (g["winner"], g["loser"]):
                k = (org, g["match_id"])
                if k not in seen:
                    seen.add(k)
                    self.team_match_seq[org].append((g["date_s"], g["match_id"]))
        for org in self.team_match_seq:
            self.team_match_seq[org].sort()

    def _continuity_vec(self, ref_date_s, mode, persistence):
        """Per-game (cont_factor_winner, cont_factor_loser) plus the
        effective-age multiplier used by 'persistence'. Returns
        (cont_w, cont_l) arrays in [0,1]."""
        n = len(self.games)
        if mode == "none":
            return np.ones(n), np.ones(n)
        ref_year = int(ref_date_s[:4])
        if mode == "year":
            cw = np.ones(n)
            cl = np.ones(n)
            cache = {}

            def f(org, gyear):
                key = (org, gyear)
                if key not in cache:
                    v = 1.0
                    for by in range(gyear + 1, ref_year + 1):
                        c = self.year_cont.get((org, by))
                        if c is not None:
                            v *= c
                    cache[key] = v
                return cache[key]
            gyears = self.g_date.astype("U4").astype(int)
            for i in range(n):
                cw[i] = f(self.games[i]["winner"], gyears[i])
                cl[i] = f(self.games[i]["loser"], gyears[i])
            return cw, cl
        if mode == "lineup":
            # overlap between the lineup that played game i and the team's
            # latest lineup before ref_date. cont = (overlap/5)^gamma
            gamma = persistence  # reuse param as sharpness here
            cur = {}
            for org, seq in self.team_match_seq.items():
                latest = None
                for ds, mid in seq:
                    if ds < ref_date_s:
                        latest = self.lineups.get((org, mid), None)
                    else:
                        break
                cur[org] = latest
            cw = np.ones(n)
            cl = np.ones(n)
            for i, g in enumerate(self.games):
                for org, arr in ((g["winner"], cw), (g["loser"], cl)):
                    cur_l = cur.get(org)
                    then_l = self.lineups.get((org, g["match_id"]))
                    if cur_l and then_l:
                        ov = len(cur_l & then_l) / max(len(cur_l), len(then_l), 5)
                        arr[i] = max(ov, 0.04) ** gamma
            return cw, cl
        raise ValueError(mode)

    # -- one full walk-forward run --------------------------------------------
    def run(self, cfg):
        """cfg keys:
          decay: dict(kind=..., params...)
          rd: dict(power=0.5, scale=2.5) or dict(mode='win', const=..) or
              dict(mode='blend', power, scale, win_const, w)
          roster_mode: 'none'|'year'|'lineup'; roster_persistence: float
          ridge: float; champ_mult: float; beta: None => fit on train
        Returns dict with metrics + per-series probs.
        """
        rd = cfg.get("rd", {"power": 0.5, "scale": 2.5})
        if "rd_custom" in cfg:
            rd = {"power": 0.0, "scale": 0.0}  # ignored below
        if rd.get("mode") == "win":
            rd_t = np.full_like(self.rd_raw, rd.get("const", 3.0))
        elif rd.get("mode") == "blend":
            base = np.abs(self.rd_raw) ** rd["power"] * rd["scale"]
            rd_t = (1 - rd["w"]) * base + rd["w"] * rd.get("win_const", 3.0)
        else:
            rd_t = np.abs(self.rd_raw) ** rd.get("power", 0.5) * rd.get("scale", 2.5)
        rd_t = np.copysign(rd_t, self.rd_raw)
        if "rd_custom" in cfg:
            rd_t = np.asarray(cfg["rd_custom"], dtype=float)

        champ_mult = cfg.get("champ_mult", 2.0)
        ridge = cfg.get("ridge", 0.5)
        n_t = len(self.teams)
        dcfg = dict(cfg.get("decay", {"kind": "exp", "hl": 6.0}))
        dkind = dcfg.pop("kind")
        r_mode = cfg.get("roster_mode", "year")
        r_pers = cfg.get("roster_persistence", 0.3)

        # per-day ratings for the series predictions
        s = self.series
        daily_r = {} if cfg.get("daily_out") else None
        preds = np.full(len(s), np.nan)
        rat_w = np.full(len(s), np.nan)
        rat_l = np.full(len(s), np.nan)
        s_by_day = defaultdict(list)
        for i, r in enumerate(s.itertuples(index=False)):
            s_by_day[r.date].append(i)

        year_iso = bool(cfg.get("year_isolated", False))
        _iso_prev_year = None
        for day in self.pred_days:
            day_num = int(np.datetime64(day, "D").astype(int))
            m_hist = self.g_dnum < day_num
            if year_iso:
                # v10: a solve sees only games from its own calendar year. This
                # also makes the >=30 gate below mean "30 IN-YEAR games" rather
                # than 30 global ones, which is the whole point — otherwise
                # every January day passes on ~4000 prior-season games while
                # having ~0 of its own.
                m_hist = m_hist & (self.g_year == int(day[:4]))
                # channel B: the region-prior chain is the dominant cross-year
                # carrier (a team with no in-year data solves to 0.75x last
                # year's regional mean). Isolation is not isolation unless the
                # chain resets at the boundary too.
                if int(day[:4]) != _iso_prev_year:
                    self._prev_rvec = None
                    _iso_prev_year = int(day[:4])
            if m_hist.sum() < 30:
                continue
            weeks = (day_num - self.g_dnum[m_hist]) / 7.0
            if dkind == "games":
                # decay by how many games EACH team has played since game i
                # (information-replacement decay: breaks don't burn weight)
                form = dcfg.get("form", "exp")  # exp | power | boxexp
                hl_g = dcfg.get("hl_games", 12.0)
                lam_g = math.log(2) / hl_g
                hl_loss = dcfg.get("hl_games_loss")  # asymmetric option
                lam_loss = math.log(2) / hl_loss if hl_loss else lam_g
                consist = dcfg.get("consistency")  # (hl_consistent, hl_anomaly)
                if consist and not hasattr(self, "_game_consist"):
                    # per (org, game_row): was the result consistent with the
                    # team's level AT THE TIME (decayed map winrate, HL16)?
                    self._game_consist = {}
                    lam_wr = math.log(2) / 16.0
                    for org, rows_all in self.team_game_rows.items():
                        wr_num = wr_den = 0.0
                        for ri in rows_all:
                            won = self.games[ri]["winner"] == org
                            wr = wr_num / wr_den if wr_den > 0.5 else 0.5
                            self._game_consist[(org, ri)] = (
                                (won and wr >= 0.5) or ((not won) and wr < 0.5))
                            wr_num = wr_num * math.exp(-lam_wr) + (1.0 if won else 0.0)
                            wr_den = wr_den * math.exp(-lam_wr) + 1.0
                tau_g = dcfg.get("tau", 8.0)
                alpha_g = dcfg.get("alpha", 1.5)
                c_g = dcfg.get("c", 5.0)
                cal_env_hl = dcfg.get("cal_env_hl")  # calendar envelope (weeks)

                def gdecay(ago, is_win, org=None, ri=None):
                    if consist:
                        hl_c, hl_a = consist
                        ok = self._game_consist.get((org, ri), True)
                        lam = math.log(2) / (hl_c if ok else hl_a)
                        return math.exp(-lam * ago)
                    if form == "power":
                        return (1.0 + ago / tau_g) ** (-alpha_g)
                    if form == "boxexp":
                        return 1.0 if ago <= c_g else math.exp(-lam_g * (ago - c_g))
                    return math.exp(-(lam_g if is_win else lam_loss) * ago)

                count_series = dcfg.get("count", "maps") == "series"
                hist_idx = np.where(m_hist)[0]
                pos_in_hist = {gi: k for k, gi in enumerate(hist_idx)}
                w_w = np.ones(len(hist_idx))
                w_l = np.ones(len(hist_idx))
                for org, rows_ in self.team_game_rows.items():
                    played = [ri for ri in rows_ if ri in pos_in_hist]
                    if count_series:
                        # age in SERIES: all maps of the same match share an age
                        mids = []
                        for ri in played:
                            mid = self.games[ri]["match_id"]
                            if not mids or mids[-1] != mid:
                                mids.append(mid)
                        n_s = len(mids)
                        s_ord = {mid: k for k, mid in enumerate(mids)}
                        for ri in played:
                            ago = n_s - 1 - s_ord[self.games[ri]["match_id"]]
                            j = pos_in_hist[ri]
                            if self.games[ri]["winner"] == org:
                                w_w[j] = gdecay(ago, True, org, ri)
                            else:
                                w_l[j] = gdecay(ago, False, org, ri)
                    else:
                        n_p = len(played)
                        for k, ri in enumerate(played):
                            ago = n_p - 1 - k
                            j = pos_in_hist[ri]
                            if self.games[ri]["winner"] == org:
                                w_w[j] = gdecay(ago, True, org, ri)
                            else:
                                w_l[j] = gdecay(ago, False, org, ri)
                base = np.sqrt(w_w * w_l)
                if cal_env_hl:
                    lam_c = math.log(2) / cal_env_hl
                    base = base * np.exp(-lam_c * weeks)
                if r_mode == "year":
                    cw, cl = self._continuity_vec(day, "year", 1.0)
                    base = base * np.sqrt(cw[m_hist] * cl[m_hist])
            elif r_mode == "year" and r_pers > 0:
                cw, cl = self._continuity_vec(day, "year", r_pers)
                cw, cl = cw[m_hist], cl[m_hist]
                eff_w = weeks * (1 - r_pers * cw)
                eff_l = weeks * (1 - r_pers * cl)
                w_w = decay_weight(eff_w, dkind, **dcfg)
                w_l = decay_weight(eff_l, dkind, **dcfg)
                base = np.sqrt(w_w * w_l) * np.sqrt(cw * cl)
            elif r_mode == "none":
                base = decay_weight(weeks, dkind, **dcfg)
            else:  # lineup mode: continuity as direct weight, plain decay age
                cw, cl = self._continuity_vec(day, "lineup", r_pers)
                cw, cl = cw[m_hist], cl[m_hist]
                base = decay_weight(weeks, dkind, **dcfg) * np.sqrt(cw * cl)
            bg = cfg.get("break_gamma")
            if bg is not None:
                # soft boundary at every >45d break: weight *= gamma^breaks
                hist_idx = np.where(m_hist)[0]
                bw = np.ones(len(hist_idx))
                for j, gi in enumerate(hist_idx):
                    g = self.games[gi]
                    nb = 0
                    for org in (g["winner"], g["loser"]):
                        for bd in self.team_break_dates.get(org, ()):
                            if g["date_s"] < bd <= day:
                                nb += 1
                    bw[j] = bg ** (nb / 2.0)  # geometric mean of the two sides
                base = base * bw
            mult = np.where(self.champ[m_hist], champ_mult, 1.0)
            if "w_custom" in cfg:
                mult = mult * np.asarray(cfg["w_custom"])[m_hist]
            w = base * mult

            wi, li = self.wi[m_hist], self.li[m_hist]
            rdv = rd_t[m_hist]
            M = np.zeros((n_t, n_t))
            p = np.zeros(n_t)
            np.add.at(M, (wi, wi), w)
            np.add.at(M, (li, li), w)
            np.add.at(M, (wi, li), -w)
            np.add.at(M, (li, wi), -w)
            np.add.at(p, wi, w * rdv)
            np.add.at(p, li, -w * rdv)
            M[np.diag_indices(n_t)] += ridge
            rpr = cfg.get("region_prior_ridge", 0.0)
            if rpr > 0 and hasattr(self, "team_region_idx"):
                # second ridge pulling each team toward its region's trailing
                # mean (previous day's solve) instead of the global 0
                prior = np.zeros(n_t)
                prev = getattr(self, "_prev_rvec", None)
                if prev is not None:
                    for ri_ in range(4):
                        m_reg = self.team_region_idx == ri_
                        if m_reg.sum() >= 4:
                            prior[m_reg] = prev[m_reg].mean()
                M[np.diag_indices(n_t)] += rpr
                p += rpr * prior
            M[-1, :] = 1.0
            p[-1] = 0.0
            try:
                r_vec = np.linalg.solve(M, p)
            except np.linalg.LinAlgError:
                r_vec, *_ = np.linalg.lstsq(M, p, rcond=None)
            self._prev_rvec = r_vec.copy()
            if daily_r is not None:
                daily_r[day] = r_vec.copy()

            for i in s_by_day[day]:
                row = s.iloc[i]
                rat_w[i] = r_vec[self.tidx.get(row.winner, -1)] if row.winner in self.tidx else 0.0
                rat_l[i] = r_vec[self.tidx.get(row.loser, -1)] if row.loser in self.tidx else 0.0

        valid = ~np.isnan(rat_w)
        rdiff = rat_w - rat_l
        train = valid & (s.date <= BETA_TRAIN_END).values
        test = valid & (s.date > BETA_TRAIN_END).values

        fmts = s.fmt.values

        def p_series(beta, mask):
            pm = 1 / (1 + np.exp(-beta * rdiff[mask]))
            fm = fmts[mask]
            return np.where(np.isin(fm, ("bo5", "bo5_gf")),
                            pm ** 3 * (1 + 3 * (1 - pm) + 6 * (1 - pm) ** 2),
                            np.where(fm == "bo1", pm, pm ** 2 * (3 - 2 * pm)))

        from scipy.optimize import minimize_scalar

        def nll(beta, mask):
            return -np.mean(np.log(np.clip(p_series(beta, mask), 1e-9, 1)))

        beta = cfg.get("beta")
        if beta is None:
            beta = float(minimize_scalar(lambda b: nll(b, train),
                                         bounds=(0.03, 0.6), method="bounded").x)
        ll_test = float(nll(beta, test))
        ll_train = float(nll(beta, train))
        p_test = p_series(beta, test)
        return {"beta": round(beta, 4), "ll_train": round(ll_train, 5),
                "ll_test": round(ll_test, 5),
                "brier_test": round(float(np.mean((1 - p_test) ** 2)), 5),
                "n_test": int(test.sum()), "n_train": int(train.sum()),
                "p_test": p_test, "test_mask": test, "rdiff": rdiff,
                "rat_w": rat_w, "rat_l": rat_l, "daily_r": daily_r}
