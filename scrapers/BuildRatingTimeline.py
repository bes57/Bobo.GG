"""
BuildRatingTimeline.py — builds the PUBLIC BenPom v6 rating timeline.

OPERATOR DIRECTIVE 2026-07-30: "deploy v6 to BenPom in BoboGG on all fronts."
This reverses the earlier "v6 stays private" decision. As of that date this
file no longer runs the calendar-decay Massey + intl-weights + CN-shrinkage
pipeline; it runs the exact v6 champion solve (testing_lab v9 protocol
winner), ported NATIVELY — scrapers must not import testing_lab (the lab
harness drags the VCTMM sys.path into the process).

The v6 spec (walk-forward Massey; ratings used for day D are solved from
games with date strictly before D):
  - games-counted consistency decay: each game's weight decays by how many
    games THAT team has played since (information replacement — breaks don't
    burn weight). Results consistent with the team's decayed map winrate
    (HL 16 games) persist with HL 20 games; anomalies decay with HL 12.
    Per-side ages, combined as sqrt(w_winner_side * w_loser_side).
  - margin target sign(rd) * |rd|^0.75 * 2.5
  - year-boundary roster continuity 0.3-style factor: per (team, year)
    carryover = min(overlap_of_5 / 5, 1) between the last event roster of the
    previous active year and the first of the next; applied as
    sqrt(cont_w * cont_l) on prior-year games.
  - ridge 0.5 toward 0 PLUS region-prior ridge 1.5 toward each region's
    PREVIOUS-day solved mean (>=4 rated teams per region group).
  - playoff/grand-final solve weight x1.6; Champions x2.0 with the
    EXACT-SHAPE id guard: only "YYYY_champions" (fullmatch) counts — the
    2026-07-28 corpus backfill added off-season ids (2025_super_champions_cup,
    2023_china_champions_qual) that substring tests would wrongly boost.

CN shrinkage: GONE from the timeline. v6's region-prior ridge (solve-side)
plus the prediction-layer cross-region offsets in data/site_model.json
replace the v10 cluster-offset pipeline entirely.

Outputs (schema unchanged — site pages + the testing-lab harness read these):
  data/rating_timeline.json        (2026, live)
  data/rating_timeline_<year>.json (2023/2024/2025)
    { year, lambda_decay, checkpoints:[{date, ratings:{org:val}}],
      match_events:[{match_id, date, event_id, winner, loser, series_score,
                     maps:[...], winner_before/after/delta,
                     loser_before/after/delta}], generated }
  data/site_model.json — the single source stage 2 wires every prediction
    surface to: {model_version, generated_utc, beta, xregion_offsets,
    region_priors, gf_upper_logit, b_pick, ratings_as_of}.

Semantics notes vs the pre-v6 file:
  - checkpoints[D].ratings = solve INCLUDING day D's games (ref date D);
    match_events[*]_before = the leak-free pre-day solve (games < D) — the
    exact engine daily_r the harness scores against. Within a season the
    pre-day solve at the next match day equals the previous checkpoint, so
    chart dots still sit on the line.
  - checkpoint membership: orgs with >=1 map game in that calendar year on or
    before D (the historical-rankings page drops teams the timeline doesn't
    rate at a cutoff; ghost ratings for long-dead orgs would pollute it).
  - incremental checkpoint reuse is deleted: the vectorized solve rebuilds
    all four years in seconds, and stale checkpoints would fight the
    region-prior chain (each day's solve depends on the previous day's).

Parity gates (run for the deploy; see testing_lab/v9/briefs/deploy_solve.md):
  gate A (--verify-parity): candidate timelines are written to a temp dir,
    testing_lab/engine.py is run in a SUBPROCESS (no import) against those
    candidates, and 20 sampled solved days across 2023-2026 must match this
    file's ratings to <= 1e-9 before anything is promoted into data/.
    Evidence: testing_lab/v9/stats/deploy_solve_parity.json.
  gate B (always on): the beta refit (minimize_scalar, bounded 0.03-0.6, all
    valid completed series to date, engine-parity math) must land in
    [0.115, 0.145] (v9 protocol refit was 0.128512 through 2026-07-28) or the
    build aborts with no files written.

Requires: data/match_dates.json (from ScrapeMatchDates.py)
Usage: python scrapers/BuildRatingTimeline.py [--live] [--verify-parity]
                                              [--dry-run]
"""

import os, sys, json, math, re, subprocess, tempfile
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scrapers"))
from MoreTestingMaybeFiles import ALL_EVENTS

# TEAM_REGIONS is the site's own org->region map. It is byte-identical to
# vctmm.benpom.teams.ORG_REGIONS (which is just a re-export of the vendored
# copy of this dict), so the region-prior ridge here matches the lab engine
# exactly; gate A verifies that numerically at deploy time.
# BMR_EVENT_DATES (not the local EVENT_DATES below) drives roster-continuity
# event ordering — same source the lab engine uses.
from BuildMapRatings import TEAM_REGIONS as ORG_REGIONS
from BuildMapRatings import EVENT_DATES as BMR_EVENT_DATES

DATA_DIR = os.path.join(ROOT, "data")
OUT_PATH  = os.path.join(DATA_DIR, "rating_timeline.json")  # 2026 (live; used by Modern Hub)
SITE_MODEL_PATH = os.path.join(DATA_DIR, "site_model.json")

def out_path_for_year(year):
    """Historical years write to rating_timeline_<year>.json. 2026 is live and
    keeps the bare filename so the Modern Hub doesn't need to change."""
    if year == 2026:
        return OUT_PATH
    return os.path.join(DATA_DIR, f"rating_timeline_{year}.json")

# ── v6 champion constants (do not tune here; retune via the testing lab) ──────
REGS               = ["Americas", "EMEA", "Pacific", "CN"]
HL_CONSISTENT      = 20.0   # games half-life, result consistent with team level
HL_ANOMALY         = 12.0   # games half-life, anomalous result
WR_HALF_LIFE       = 16.0   # games half-life of the running map-winrate used
                            # to classify consistency
RD_POWER           = 0.75
RD_SCALE           = 2.5
RIDGE              = 0.5
REGION_PRIOR_RIDGE = 1.5
CHAMP_MULT         = 2.0
PLAYOFF_MULT       = 1.6
ROSTER_CONT        = 0.3    # documented v6 knob; year-mode continuity factors
                            # are the measured roster carryover, applied as
                            # sqrt(cw*cl) exactly like the engine's games branch
MIN_HIST_GAMES     = 30     # days with fewer prior games get no chain solve
BETA_FIT_BOUNDS    = (0.03, 0.6)
BETA_GATE          = (0.115, 0.145)   # deploy gate B (v9 refit: 0.128512)
GF_UPPER_LOGIT     = 0.25
PARITY_TOL         = 1e-9
N_PARITY_DAYS      = 20
ENGINE_PROBE = os.path.join(ROOT, "testing_lab", "v9", "scratch", "deploy",
                            "engine_probe.py")

# lambda_decay is kept as a top-level key for schema stability. Under v6 there
# is no calendar decay; this reports the consistent-result games half-life.
LAMBDA_DECAY = math.log(2) / HL_CONSISTENT
MIN_GAMES    = 1   # retained for API compatibility; v6 emits a rating for any
                   # org that has played a game in the checkpoint's year

# Keep these dates in sync with BuildMapRatings._HISTORICAL_EVENT_DATES —
# they're the real first/last match days per event (from match_dates.json).
# They gate + date-interpolate load_all_games() and MUST NOT change meaning:
# testing_lab/engine.py imports load_all_games() for its frozen frame.
EVENT_DATES = {
    "2023_lock_in":          ("2023-02-13", "2023-03-04"),
    "2023_league":           ("2023-03-25", "2023-05-28"),
    "2023_masters_tokyo":    ("2023-06-10", "2023-06-25"),
    "2023_champions":        ("2023-08-06", "2023-08-26"),
    "2024_kickoff":          ("2024-02-16", "2024-03-03"),
    "2024_china_kickoff":    ("2024-02-22", "2024-03-02"),
    "2024_masters_madrid":   ("2024-03-14", "2024-03-24"),
    "2024_stage1":           ("2024-04-03", "2024-05-12"),
    "2024_china_stage1":     ("2024-04-05", "2024-05-12"),
    "2024_masters_shanghai": ("2024-05-23", "2024-06-09"),
    "2024_stage2":           ("2024-06-15", "2024-07-21"),
    "2024_china_stage2":     ("2024-06-15", "2024-07-20"),
    "2024_champions":        ("2024-08-01", "2024-08-25"),
    "2025_kickoff":          ("2025-01-15", "2025-02-09"),
    "2025_china_kickoff":    ("2025-01-11", "2025-01-25"),
    "2025_masters_bangkok":  ("2025-02-20", "2025-03-02"),
    "2025_stage1":           ("2025-03-21", "2025-05-18"),
    "2025_china_stage1":     ("2025-03-13", "2025-05-04"),
    "2025_masters_toronto":  ("2025-06-07", "2025-06-22"),
    "2025_stage2":           ("2025-07-15", "2025-08-31"),
    "2025_china_stage2":     ("2025-07-03", "2025-08-24"),
    "2025_champions":        ("2025-09-12", "2025-10-05"),
    "2026_kickoff":          ("2026-01-15", "2026-02-16"),
    "2026_china_kickoff":    ("2026-01-21", "2026-02-09"),
    "2026_masters_santiago": ("2026-02-28", "2026-03-15"),
    "2026_stage1":           ("2026-04-01", "2026-05-25"),
}

# Auto-extend with any ALL_EVENTS entry not already covered. Hardcoded entries
# above use real first/last match dates (better date interpolation when
# match_dates.json is missing IDs); new events fall back to the declared
# start/end window from ALL_EVENTS so they aren't silently dropped from the
# timeline by the `eid not in EVENT_DATES` gate in load_all_games().
for _e in ALL_EVENTS:
    _eid = _e.get("id")
    if _eid and _e.get("start") and _e.get("end"):
        EVENT_DATES.setdefault(_eid, (_e["start"], _e["end"]))


# ── Data loading ───────────────────────────────────────────────────────────────

def load_all_games():
    """
    Load every scraped map-level game and attach an actual date.
    Falls back to match_id-rank interpolation within an event if the date
    is not in match_dates.json.
    Returns list of game dicts.
    """
    dates_path = os.path.join(DATA_DIR, "match_dates.json")
    match_dates = {}
    if os.path.exists(dates_path):
        with open(dates_path) as f:
            match_dates = json.load(f)
        print(f"  Loaded {len(match_dates)} match dates from match_dates.json")
    else:
        print("  WARNING: match_dates.json not found — dates will be interpolated")

    mr = pd.read_csv(os.path.join(DATA_DIR, "match_results.csv"))
    mr = mr[mr["MapNum"] != "all"].copy()
    mr["MapNum"] = mr["MapNum"].astype(str)
    mr_idx = mr.set_index(["MatchID", "MapNum"])

    games = []

    for event in ALL_EVENTS:
        eid  = event["id"]
        path = os.path.join(DATA_DIR, "maps", f"{eid}.csv")
        if not os.path.exists(path):
            continue
        if eid not in EVENT_DATES:
            continue

        df = pd.read_csv(path)
        df["event_id"] = eid
        df["MapNum"]   = df["MapNum"].astype(str)
        df["MapName"]  = df["MapName"].str.replace("PICK", "", regex=False).str.strip()

        meta = df.groupby(["MatchID", "MapNum"]).agg(
            orgs=("Org",     lambda x: list(x.unique())),
            map_name=("MapName",  "first"),
            event_id=("event_id", "first"),
        ).reset_index()

        for _, row in meta.iterrows():
            key = (int(row["MatchID"]), row["MapNum"])
            if key not in mr_idx.index:
                continue
            mr_row = mr_idx.loc[key]
            winner = mr_row["WinnerOrg"]
            losers = [o for o in row["orgs"] if o != winner]
            if not losers:
                continue
            try:
                wr, lr = map(int, str(mr_row["Score"]).split("-"))
            except Exception:
                continue

            mid_str = str(int(row["MatchID"]))
            date_str = match_dates.get(mid_str)  # may be None

            games.append({
                "match_id":   int(row["MatchID"]),
                "event_id":   eid,
                "map_name":   row["map_name"],
                "winner":     winner,
                "loser":      losers[0],
                "wr":         wr,
                "lr":         lr,
                "date":       datetime.strptime(date_str, "%Y-%m-%d") if date_str else None,
                "_date_known": date_str is not None,
            })

    # Fill interpolated dates for games without real dates
    gdf = pd.DataFrame(games)
    for eid, (start_str, end_str) in EVENT_DATES.items():
        mask = (gdf["event_id"] == eid) & gdf["date"].isna()
        if not mask.any():
            continue
        start_dt = datetime.strptime(start_str, "%Y-%m-%d")
        end_dt   = datetime.strptime(end_str,   "%Y-%m-%d")
        span     = max(1, (end_dt - start_dt).days)

        mids         = gdf.loc[mask, "match_id"].values
        sorted_uniq  = sorted(set(mids))
        rank_map     = {mid: i for i, mid in enumerate(sorted_uniq)}
        max_rank     = max(rank_map.values()) if rank_map else 1

        for i_row, mid in zip(gdf.index[mask], mids):
            frac = rank_map[mid] / max_rank
            gdf.at[i_row, "date"] = start_dt + timedelta(days=int(frac * span))

    gdf = gdf.dropna(subset=["date"])
    return gdf.to_dict("records")


# ── series enrichment (native ports of the harness helpers) ────────────────────

def _stage(match_name):
    """Verbatim port of testing_lab/harness._stage (drives the playoff x1.6)."""
    s = (match_name or "").lower()
    if "grand final" in s:
        return "grand_final"
    if re.search(r"playoff|bracket|upper|lower|semifinal|quarterfinal|round of|"
                 r"knockout|final", s):
        return "playoffs"
    if re.search(r"group|swiss|league|regular|week ", s):
        return "groups"
    return "other"


def _match_names():
    mr = pd.read_csv(os.path.join(DATA_DIR, "match_results.csv"),
                     usecols=["MatchID", "MatchName"])
    return dict(mr.drop_duplicates("MatchID").values)


def _derive_day_matches(games_sorted):
    """(date_s, match_id) -> [game rows], insertion order = solve order."""
    by_day_match = {}
    for g in games_sorted:
        by_day_match.setdefault((g["date_s"], g["match_id"]), []).append(g)
    return by_day_match


def _series_outcome(maps):
    """Winner/loser/map counts for one match, exactly as the old timeline
    derived match_events (harness re-derives validity from these fields)."""
    map_wins, teams_seen = {}, set()
    for g in maps:
        map_wins[g["winner"]] = map_wins.get(g["winner"], 0) + 1
        teams_seen.add(g["winner"])
        teams_seen.add(g["loser"])
    if len(teams_seen) < 2:
        return None
    teams  = list(teams_seen)
    winner = max(teams, key=lambda t: map_wins.get(t, 0))
    loser  = min(teams, key=lambda t: map_wins.get(t, 0))
    return winner, loser, map_wins.get(winner, 0), map_wins.get(loser, 0)


# ── the v6 solver ──────────────────────────────────────────────────────────────

class V6Solver:
    """Native, vectorized replica of testing_lab/engine.py run() games-decay
    branch under the v6 champion config. Gate A checks it against the real
    engine to <= 1e-9 on sampled days at deploy time."""

    def __init__(self, games):
        for g in games:
            g["date_s"] = g["date"].strftime("%Y-%m-%d")
        games.sort(key=lambda g: (g["date_s"], g["match_id"]))
        self.games = games
        n_g = len(games)

        teams = sorted({g["winner"] for g in games} | {g["loser"] for g in games})
        self.teams = teams
        self.tidx = {t: i for i, t in enumerate(teams)}
        self.n_t = len(teams)
        self.wi = np.array([self.tidx[g["winner"]] for g in games])
        self.li = np.array([self.tidx[g["loser"]] for g in games])
        rd_raw = np.array([g["wr"] - g["lr"] for g in games], dtype=float)
        rd_t = np.abs(rd_raw) ** RD_POWER * RD_SCALE
        self.rd_t = np.copysign(rd_t, rd_raw)
        self.g_date = np.array([g["date_s"] for g in games])
        self.g_dnum = pd.to_datetime(self.g_date).values.astype("datetime64[D]").astype(int)
        # exact-shape Champions guard (keep: off-season ids must NOT match)
        self.champ = np.array([re.fullmatch(r"\d{4}_champions", g["event_id"])
                               is not None for g in games])
        self.region_idx = np.array(
            [REGS.index(ORG_REGIONS[t]) if ORG_REGIONS.get(t) in REGS else -1
             for t in teams])

        # per-team game rows (chronological because games are sorted)
        self.team_game_rows = defaultdict(list)
        for i, g in enumerate(games):
            self.team_game_rows[g["winner"]].append(i)
            self.team_game_rows[g["loser"]].append(i)

        self._build_series()
        self._build_po()
        self._build_consistency()
        self._build_year_continuity()

        self.daily_before = {}       # date_s -> r_vec (chain-solved days only)
        self._chain = None

    # -- series table (native port of harness.load_series validity) ----------
    def _build_series(self):
        names = _match_names()
        self.by_day_match = _derive_day_matches(self.games)
        rows, seen_mid = [], set()
        for (ds, mid), maps in sorted(self.by_day_match.items()):
            out = _series_outcome(maps)
            if out is None:
                continue
            w, l, wm, lm = out
            if w not in ORG_REGIONS or l not in ORG_REGIONS:
                continue
            if wm <= lm or wm not in (1, 2, 3):
                continue
            if mid in seen_mid:
                continue
            seen_mid.add(mid)
            mname = names.get(mid, "")
            fmt = {1: "bo1", 2: "bo3", 3: "bo5"}[wm]
            stage = _stage(mname)
            if fmt == "bo5" and stage == "grand_final":
                fmt = "bo5_gf"
            rows.append({"match_id": mid, "date": ds, "winner": w, "loser": l,
                         "fmt": fmt, "stage": stage,
                         "reg_w": ORG_REGIONS[w], "reg_l": ORG_REGIONS[l]})
        self.series = rows
        self.grid_days = sorted({r["date"] for r in rows})

    def _build_po(self):
        stage_by_mid = {r["match_id"]: r["stage"] for r in self.series}
        g_stage = np.array([stage_by_mid.get(g["match_id"], "groups")
                            for g in self.games])
        self.po = np.where(np.isin(g_stage, ("playoffs", "grand_final")),
                           PLAYOFF_MULT, 1.0)

    # -- consistency-conditioned games decay ----------------------------------
    def _build_consistency(self):
        """Per (org, game): was the result consistent with the org's decayed
        map winrate AT THE TIME (HL 16)? Fixes floor-team inflation: wins by
        weak teams (anomalies) age out faster than their losses."""
        lam20 = math.log(2) / HL_CONSISTENT
        lam12 = math.log(2) / HL_ANOMALY
        dec = math.exp(-(math.log(2) / WR_HALF_LIFE))
        self.org_rows, self.org_iswin, self.org_lam = {}, {}, {}
        for org, rows_all in self.team_game_rows.items():
            wr_num = wr_den = 0.0
            lam_arr = np.empty(len(rows_all))
            iswin = np.empty(len(rows_all), dtype=bool)
            for k, ri in enumerate(rows_all):
                won = self.games[ri]["winner"] == org
                wr = wr_num / wr_den if wr_den > 0.5 else 0.5
                ok = (won and wr >= 0.5) or ((not won) and wr < 0.5)
                lam_arr[k] = lam20 if ok else lam12
                iswin[k] = won
                wr_num = wr_num * dec + (1.0 if won else 0.0)
                wr_den = wr_den * dec + 1.0
            self.org_rows[org] = np.array(rows_all)
            self.org_iswin[org] = iswin
            self.org_lam[org] = lam_arr

    # -- year-boundary roster continuity --------------------------------------
    def _build_year_continuity(self):
        """min(roster_overlap/5, 1) between the last event of a team's previous
        active year and the first of the next; multiplies through skipped
        boundaries. Same roster source as the lab engine (data/<eid>.csv +
        BuildMapRatings.EVENT_DATES)."""
        hist = defaultdict(list)
        for ev in ALL_EVENTS:
            eid = ev["id"]
            path = os.path.join(DATA_DIR, f"{eid}.csv")
            if not os.path.exists(path):
                continue
            try:
                df = pd.read_csv(path, usecols=["Org", "ProfileURL"])
            except Exception:
                continue
            end = BMR_EVENT_DATES.get(eid, (None, None))[1]
            if not end:
                continue
            for org, grp in df.groupby("Org"):
                urls = frozenset(grp["ProfileURL"].dropna().unique())
                if urls:
                    hist[org].append((end, eid, urls))
        year_cont = {}
        for org, evs in hist.items():
            evs.sort()
            by_year = defaultdict(list)
            for end, eid, r in evs:
                by_year[int(end[:4])].append((end, eid, r))
            ys = sorted(by_year)
            for i in range(1, len(ys)):
                prev = max(by_year[ys[i - 1]])
                curr = min(by_year[ys[i]])
                year_cont[(org, ys[i])] = min(len(prev[2] & curr[2]) / 5.0, 1.0)
        self.year_cont = year_cont

        # continuity factor per game per possible ref-year, precomputed
        gyears = self.g_date.astype("U4").astype(int)
        ref_years = sorted({int(ds[:4]) for ds in self.g_date} |
                           {datetime.now().year})
        cache = {}

        def f(org, gyear, ref_year):
            key = (org, gyear, ref_year)
            if key not in cache:
                v = 1.0
                for by in range(gyear + 1, ref_year + 1):
                    c = self.year_cont.get((org, by))
                    if c is not None:
                        v *= c
                cache[key] = v
            return cache[key]

        self.cw_by_ref, self.cl_by_ref = {}, {}
        for ry in ref_years:
            cw = np.ones(len(self.games))
            cl = np.ones(len(self.games))
            for i, g in enumerate(self.games):
                cw[i] = f(g["winner"], gyears[i], ry)
                cl[i] = f(g["loser"], gyears[i], ry)
            self.cw_by_ref[ry] = cw
            self.cl_by_ref[ry] = cl

    # -- one solve -------------------------------------------------------------
    def solve(self, n_hist, ref_year, chain):
        """Massey solve over the first n_hist games (chronological prefix),
        viewed from a day in ref_year, with region priors from `chain` (the
        previous chain day's solve). Mirrors engine.run()'s inner loop."""
        n_t = self.n_t
        w_w = np.ones(n_hist)
        w_l = np.ones(n_hist)
        for org, rows_arr in self.org_rows.items():
            m = int(np.searchsorted(rows_arr, n_hist, side="left"))
            if m == 0:
                continue
            ago = np.arange(m - 1, -1, -1, dtype=float)
            vals = np.exp(-(self.org_lam[org][:m] * ago))
            rows = rows_arr[:m]
            iw = self.org_iswin[org][:m]
            w_w[rows[iw]] = vals[iw]
            w_l[rows[~iw]] = vals[~iw]
        base = np.sqrt(w_w * w_l)
        if ref_year in self.cw_by_ref:
            base = base * np.sqrt(self.cw_by_ref[ref_year][:n_hist] *
                                  self.cl_by_ref[ref_year][:n_hist])
        mult = np.where(self.champ[:n_hist], CHAMP_MULT, 1.0)
        mult = mult * self.po[:n_hist]
        w = base * mult

        wi, li, rdv = self.wi[:n_hist], self.li[:n_hist], self.rd_t[:n_hist]
        M = np.zeros((n_t, n_t))
        p = np.zeros(n_t)
        np.add.at(M, (wi, wi), w)
        np.add.at(M, (li, li), w)
        np.add.at(M, (wi, li), -w)
        np.add.at(M, (li, wi), -w)
        np.add.at(p, wi, w * rdv)
        np.add.at(p, li, -w * rdv)
        M[np.diag_indices(n_t)] += RIDGE
        prior = np.zeros(n_t)
        if chain is not None:
            for ri_ in range(4):
                m_reg = self.region_idx == ri_
                if m_reg.sum() >= 4:
                    prior[m_reg] = chain[m_reg].mean()
        M[np.diag_indices(n_t)] += REGION_PRIOR_RIDGE
        p += REGION_PRIOR_RIDGE * prior
        M[-1, :] = 1.0
        p[-1] = 0.0
        try:
            return np.linalg.solve(M, p)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(M, p, rcond=None)[0]

    def _n_hist(self, dnum, inclusive):
        return int(np.searchsorted(self.g_dnum, dnum,
                                   side="right" if inclusive else "left"))

    # -- full walk -------------------------------------------------------------
    def walk(self, today_s):
        """One chronological pass over every game day 2023->now. Chain solves
        (region-prior carrier + parity surface) happen only on valid-series
        days with >= MIN_HIST_GAMES prior games — exactly the engine's rule.
        Returns {year: (checkpoints, match_events)}."""
        grid = set(self.grid_days)
        all_days = sorted({g["date_s"] for g in self.games})
        # first game dnum per (org, year) for checkpoint membership
        first_in_year = defaultdict(dict)
        for i, g in enumerate(self.games):
            yr = int(g["date_s"][:4])
            for org in (g["winner"], g["loser"]):
                if yr not in first_in_year[org]:
                    first_in_year[org][yr] = self.g_dnum[i]

        out = defaultdict(lambda: ([], []))
        for di, day_s in enumerate(all_days):
            dnum = int(np.datetime64(day_s, "D").astype(int))
            ry = int(day_s[:4])
            n_before = self._n_hist(dnum, inclusive=False)
            if day_s in grid and n_before >= MIN_HIST_GAMES:
                r_before = self.solve(n_before, ry, self._chain)
                self._chain = r_before
                self.daily_before[day_s] = r_before
            else:
                r_before = self.solve(n_before, ry, self._chain)
            r_after = self.solve(self._n_hist(dnum, inclusive=True), ry,
                                 self._chain)

            checkpoints, match_events = out[ry]
            ratings = {}
            for org, fy in first_in_year.items():
                d0 = fy.get(ry)
                if d0 is not None and d0 <= dnum:
                    ratings[org] = round(float(r_after[self.tidx[org]]), 4)
            checkpoints.append({"date": day_s, "ratings": ratings})

            for (ds, mid), maps in self.by_day_match.items():
                if ds != day_s:
                    continue
                oc = _series_outcome(maps)
                if oc is None:
                    continue
                winner, loser, w_maps, l_maps = oc
                wb = float(r_before[self.tidx[winner]])
                wa = float(r_after[self.tidx[winner]])
                lb = float(r_before[self.tidx[loser]])
                la = float(r_after[self.tidx[loser]])
                match_events.append({
                    "match_id":      mid,
                    "date":          day_s,
                    "event_id":      maps[0]["event_id"],
                    "winner":        winner,
                    "loser":         loser,
                    "series_score":  f"{w_maps}-{l_maps}",
                    "maps":          [{"map": g["map_name"], "wr": g["wr"],
                                       "lr": g["lr"], "winner": g["winner"]}
                                      for g in maps],
                    "winner_before": round(wb, 4),
                    "winner_after":  round(wa, 4),
                    "winner_delta":  round(wa - wb, 4),
                    "loser_before":  round(lb, 4),
                    "loser_after":   round(la, 4),
                    "loser_delta":   round(la - lb, 4),
                })
            if (di + 1) % 100 == 0 or (di + 1) == len(all_days):
                print(f"  Day {di+1}/{len(all_days)}: {day_s} — "
                      f"{len(out[ry][0][-1]['ratings'])} teams rated")

        # as-of-now chain solve (games strictly before today), like the lab
        # snapshot builder appending `today` to pred_days
        if today_s not in self.daily_before:
            dnum = int(np.datetime64(today_s, "D").astype(int))
            n_b = self._n_hist(dnum, inclusive=False)
            if n_b >= MIN_HIST_GAMES:
                r = self.solve(n_b, int(today_s[:4]), self._chain)
                self._chain = r
                self.daily_before[today_s] = r
        self.asof_day = max(self.daily_before) if self.daily_before else None
        return dict(out)

    # -- prediction-layer fits (engine-parity math) ---------------------------
    def series_arrays(self):
        rat_w = np.full(len(self.series), np.nan)
        rat_l = np.full(len(self.series), np.nan)
        for i, r in enumerate(self.series):
            vec = self.daily_before.get(r["date"])
            if vec is not None:
                rat_w[i] = vec[self.tidx[r["winner"]]]
                rat_l[i] = vec[self.tidx[r["loser"]]]
        fmts = np.array([r["fmt"] for r in self.series])
        return rat_w, rat_l, fmts

    @staticmethod
    def _p_series(pm, fm):
        return np.where(np.isin(fm, ("bo5", "bo5_gf")),
                        pm ** 3 * (1 + 3 * (1 - pm) + 6 * (1 - pm) ** 2),
                        np.where(fm == "bo1", pm, pm ** 2 * (3 - 2 * pm)))

    def refit_beta(self):
        """minimize_scalar bounded fit on ALL valid completed series to date."""
        from scipy.optimize import minimize_scalar
        rat_w, rat_l, fmts = self.series_arrays()
        rdiff = rat_w - rat_l
        valid = ~np.isnan(rdiff)

        def nll(beta):
            pm = 1 / (1 + np.exp(-beta * rdiff[valid]))
            p = self._p_series(pm, fmts[valid])
            return -np.mean(np.log(np.clip(p, 1e-9, 1)))

        beta = float(minimize_scalar(nll, bounds=BETA_FIT_BOUNDS,
                                     method="bounded").x)
        return beta, int(valid.sum())

    def fit_xregion(self, beta):
        """Cross-region additive offsets, CN pinned 0 — the exact
        trading_model/build_model_snapshot.py method."""
        from scipy.optimize import minimize
        rat_w, rat_l, fmts = self.series_arrays()
        rdiff = rat_w - rat_l
        valid = ~np.isnan(rdiff)
        reg_w = np.array([REGS.index(r["reg_w"]) for r in self.series])
        reg_l = np.array([REGS.index(r["reg_l"]) for r in self.series])
        cross = (reg_w != reg_l) & valid

        def nll_off(d3):
            d4 = np.append(d3, 0.0)
            adj = rdiff[cross] + d4[reg_w[cross]] - d4[reg_l[cross]]
            p = self._p_series(1 / (1 + np.exp(-beta * adj)), fmts[cross])
            return -np.mean(np.log(np.clip(p, 1e-9, 1)))

        res = minimize(nll_off, np.zeros(3), method="Nelder-Mead")
        return {reg: round(float(v), 4)
                for reg, v in zip(REGS, np.append(res.x, 0.0))}, int(cross.sum())

    def latest_ratings(self):
        """org -> rating as of self.asof_day, only orgs with game history."""
        vec = self.daily_before[self.asof_day]
        return {t: round(float(vec[self.tidx[t]]), 4)
                for t in self.teams if t in self.team_game_rows}

    def region_priors(self):
        ratings = self.latest_ratings()
        priors = {}
        for reg in REGS:
            vals = [ratings[t] for t in ratings if ORG_REGIONS.get(t) == reg]
            if len(vals) >= 6:
                priors[reg] = round(float(np.percentile(vals, 25)), 4)
        return priors

    def b_pick(self):
        """Picker-winrate logit over all vetoes to date (snapshot method)."""
        v = pd.read_csv(os.path.join(DATA_DIR, "map_vetos.csv"))
        picker = {(int(r.MatchID), str(r.map).strip()): r.team
                  for r in v.itertuples(index=False) if r.action == "pick"}
        n_pick = n_win = 0
        for g in self.games:
            pk = picker.get((g["match_id"], g["map_name"]))
            if pk == g["winner"]:
                n_pick += 1
                n_win += 1
            elif pk == g["loser"]:
                n_pick += 1
        pw = n_win / max(n_pick, 1)
        return round(math.log(pw / (1 - pw)), 4)


# ── parity gate A (engine subprocess on candidate files) ──────────────────────

def verify_parity(solver, candidate_paths, today_s, evidence_path):
    """Run testing_lab/engine.py in a subprocess against the CANDIDATE
    timeline files and require <= PARITY_TOL agreement on sampled solved days.
    Subprocess (not import): the lab harness drags the VCTMM sys.path."""
    solved_days = sorted(solver.daily_before)
    idx = [round(i * (len(solved_days) - 1) / (N_PARITY_DAYS - 1))
           for i in range(N_PARITY_DAYS)]
    sample_days = sorted({solved_days[i] for i in idx})

    workdir = tempfile.mkdtemp(prefix="benpom_parity_")
    req_path = os.path.join(workdir, "request.json")
    out_path = os.path.join(workdir, "engine_out.json")
    with open(req_path, "w") as f:
        json.dump({"days": sample_days, "today": today_s}, f)
    cmd = [sys.executable, ENGINE_PROBE,
           "--timelines", ",".join(candidate_paths),
           "--days", req_path, "--out", out_path]
    print(f"  [gate A] running engine probe: {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if r.returncode != 0:
        print(r.stdout[-2000:])
        print(r.stderr[-4000:])
        raise RuntimeError("engine probe subprocess failed")
    with open(out_path) as f:
        probe = json.load(f)

    ok = True
    detail = []
    if probe.get("n_games") != len(solver.games):
        ok = False
        detail.append({"error": "game-frame mismatch",
                       "engine_n_games": probe.get("n_games"),
                       "native_n_games": len(solver.games)})
    if probe["teams"] != solver.teams:
        ok = False
        detail.append({"error": "team universe mismatch"})
    eng_days = probe["pred_days_solved"]
    if eng_days != solved_days:
        ok = False
        detail.append({"error": "solved-day grid mismatch",
                       "only_native": sorted(set(solved_days) - set(eng_days))[:5],
                       "only_engine": sorted(set(eng_days) - set(solved_days))[:5]})
    global_max = 0.0
    for d in sample_days:
        if d not in probe["daily"]:
            ok = False
            detail.append({"date": d, "error": "missing from engine daily_r"})
            continue
        diff = float(np.max(np.abs(np.array(probe["daily"][d]) -
                                   solver.daily_before[d])))
        global_max = max(global_max, diff)
        detail.append({"date": d, "max_abs_diff": diff})
        if diff > PARITY_TOL:
            ok = False
    evidence = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "method": ("testing_lab/engine.py run in a subprocess with "
                   "harness.TIMELINE_FILES pointed at the candidate timeline "
                   "JSONs (pre-promotion), v6 champion config, daily_out; "
                   "compared to the native solver's chain solves"),
        "frame": {"engine_file": probe.get("engine_file"),
                  "n_games": len(solver.games),
                  "engine_n_games": probe.get("n_games"),
                  "n_valid_series": len(solver.series),
                  "n_grid_days": len(solver.grid_days),
                  "n_solved_days": len(solved_days),
                  "engine_n_valid_series": probe.get("n_series"),
                  "engine_beta_all_refit": probe.get("beta_all_refit")},
        "gate_a": {"tolerance": PARITY_TOL, "n_sampled_days": len(sample_days),
                   "global_max_abs_diff": global_max, "days": detail,
                   "pass": bool(ok)},
    }
    os.makedirs(os.path.dirname(evidence_path), exist_ok=True)
    return ok, evidence, global_max


# ── compat wrappers (old public API) ──────────────────────────────────────────

def build_year_timeline(all_games, year, existing=None):
    """Full v6 walk, returning (checkpoints, match_events) for one year.
    `existing` is accepted and ignored — incremental reuse is gone (the chain
    solve is seconds, and reused checkpoints would fight the region prior)."""
    solver = V6Solver(list(all_games))
    per_year = solver.walk(datetime.now().strftime("%Y-%m-%d"))
    return per_year.get(year, ([], []))


def build_2026_timeline(all_games, existing=None):
    return build_year_timeline(all_games, 2026, existing=existing)


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    do_parity = "--verify-parity" in sys.argv
    dry_run = "--dry-run" in sys.argv
    if "--live" in sys.argv:
        print("[--live] note: v6 always rebuilds all years (chain solve)")
    today_s = datetime.now().strftime("%Y-%m-%d")

    print("Loading all scraped games with actual dates...")
    all_games = load_all_games()
    print(f"Loaded {len(all_games)} map games across all events\n")

    print("Building v6 walk-forward solve (2023-now)...")
    solver = V6Solver(all_games)
    print(f"  {solver.n_t} teams, {len(solver.series)} valid series, "
          f"{len(solver.grid_days)} grid days")
    per_year = solver.walk(today_s)

    # ── prediction-layer fits ────────────────────────────────────────────────
    beta, n_valid = solver.refit_beta()
    print(f"\nbeta refit (all {n_valid} valid series): {beta:.6f}")

    # gate B — always on. Outside the window means the solve or the data
    # changed in a way v9 never validated: investigate, don't ship.
    if not (BETA_GATE[0] <= beta <= BETA_GATE[1]):
        print(f"GATE B FAILED: beta {beta:.6f} outside {BETA_GATE} — "
              f"no outputs written.")
        sys.exit(1)
    print(f"  gate B pass: {BETA_GATE[0]} <= {beta:.6f} <= {BETA_GATE[1]}")

    xregion, n_cross = solver.fit_xregion(beta)
    print(f"cross-region offsets ({n_cross} series): {xregion}")
    priors = solver.region_priors()
    print(f"region priors (25th pct): {priors}")
    b_pick = solver.b_pick()
    print(f"b_pick: {b_pick}")

    # ── write candidates, then gate A, then promote ─────────────────────────
    years = sorted(per_year)
    cand_dir = os.path.join(DATA_DIR, ".tl_candidates_tmp")
    os.makedirs(cand_dir, exist_ok=True)
    cand_paths = []
    for year in years:
        checkpoints, match_events = per_year[year]
        out = {
            "year":         year,
            "lambda_decay": round(LAMBDA_DECAY, 6),
            "checkpoints":  checkpoints,
            "match_events": match_events,
            "generated":    today_s,
        }
        cp = os.path.join(cand_dir, os.path.basename(out_path_for_year(year)))
        with open(cp, "w") as f:
            json.dump(out, f, separators=(",", ":"))
        cand_paths.append(cp)
        print(f"  candidate {os.path.basename(cp)}: {len(checkpoints)} "
              f"checkpoints, {len(match_events)} match events")

    evidence = None
    if do_parity:
        evidence_path = os.path.join(ROOT, "testing_lab", "v9", "stats",
                                     "deploy_solve_parity.json")
        ok, evidence, gmax = verify_parity(solver, cand_paths, today_s,
                                           evidence_path)
        evidence["gate_b"] = {"beta": beta, "bounds": list(BETA_GATE),
                              "v9_reference_refit": 0.128512, "pass": True}
        with open(evidence_path, "w") as f:
            json.dump(evidence, f, indent=1)
        print(f"  [gate A] max |native - engine| over sampled days: {gmax:.3e} "
              f"(tol {PARITY_TOL:.0e}) -> {'PASS' if ok else 'FAIL'}")
        print(f"  evidence: {evidence_path}")
        if not ok:
            print(f"GATE A FAILED — candidates left in {cand_dir}, "
                  f"no outputs promoted.")
            sys.exit(1)

    if dry_run:
        print(f"\n[--dry-run] gates passed; candidates in {cand_dir}, "
              f"data/ untouched.")
        return

    for year, cp in zip(years, cand_paths):
        dest = out_path_for_year(year)
        os.replace(cp, dest)
        print(f"  Saved {os.path.basename(dest)}")
    try:
        os.rmdir(cand_dir)
    except OSError:
        pass

    site_model = {
        "model_version": f"benpom-v6-site-{today_s}",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "beta":            round(beta, 6),
        "xregion_offsets": xregion,
        "region_priors":   priors,
        "gf_upper_logit":  GF_UPPER_LOGIT,
        "b_pick":          b_pick,
        "ratings_as_of":   solver.asof_day,
    }
    with open(SITE_MODEL_PATH, "w") as f:
        json.dump(site_model, f, indent=1)
    print(f"  Saved site_model.json (beta {beta:.6f}, as of {solver.asof_day})")


if __name__ == "__main__":
    main()
