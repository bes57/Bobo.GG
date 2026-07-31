"""Player-carryover ratings + post-break dampener, on the v3 base.

r_adj(team, date) = mean over the FIELDED five of each player's source rating:
  - player already on team (>=M matches together): the org rating
  - recent joiner (n<M matches with team): fade prev-org rating -> org rating
  - rookie (no prior org): region 25th-percentile prior, fading in
Then optional break dampener: shrink rating gap by gamma when either side
returns from a 45+ day break (gamma fit on train breaks only).
Writes out/carryover.json."""
import bisect
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/Users/benny_es1/VCTMM")
from engine import Engine
from harness import paired_bootstrap
from scipy.optimize import minimize, minimize_scalar
from vctmm.benpom.teams import ORG_REGIONS

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

eng = Engine()
s = eng.series.reset_index(drop=True)
fmts = s.fmt.values
train_v = (s.date <= "2024-12-31").values
test_v = (s.date > "2024-12-31").values

stage_by_mid = dict(zip(s.match_id, s.stage))
g_stage = np.array([stage_by_mid.get(g["match_id"], "groups") for g in eng.games])
wc = np.where(np.isin(g_stage, ("playoffs", "grand_final")), 1.6, 1.0)
V3 = {"decay": {"kind": "games", "hl_games": 20.0, "hl_games_loss": 12.0},
      "rd": {"power": 0.75, "scale": 2.5}, "roster_mode": "year",
      "roster_persistence": 0.3, "ridge": 0.5, "champ_mult": 2.0,
      "w_custom": wc, "daily_out": True}
out = eng.run(V3)
daily = out["daily_r"]
days_sorted = sorted(daily.keys())
rd_base = out["rdiff"]

# player histories: ProfileURL -> [(date, org, match_id)]
player_hist = defaultdict(list)
for (org, mid), urls in eng.lineups.items():
    # date of that match
    pass
mid_date = {}
for g in eng.games:
    mid_date[(g["winner"], g["match_id"])] = g["date_s"]
    mid_date[(g["loser"], g["match_id"])] = g["date_s"]
for (org, mid), urls in eng.lineups.items():
    d = mid_date.get((org, mid))
    if d is None:
        continue
    for u in urls:
        player_hist[u].append((d, org, mid))
for u in player_hist:
    player_hist[u].sort()

# per-team fielded lineup per series match (knowable pre-match)
series_lineup = {}
for row in s.itertuples(index=False):
    for org in (row.winner, row.loser):
        lu = eng.lineups.get((org, row.match_id))
        if lu:
            series_lineup[(org, row.match_id)] = lu

# rest days per team per series
def rest_days(org, date):
    seq = eng.team_match_seq.get(org, [])
    prior = [d for d, _ in seq if d < date]
    if not prior:
        return 999
    return (pd.Timestamp(date) - pd.Timestamp(prior[-1])).days


def daily_vec(date):
    j = bisect.bisect_left(days_sorted, date) - 1
    return daily[days_sorted[j]] if j >= 0 else None


first_game = {org: min(eng.g_date[i] for i in rows_)
              for org, rows_ in eng.team_game_rows.items()}


def region_prior(date, reg, rv):
    regs = [rv[eng.tidx[t]] for t in eng.teams
            if ORG_REGIONS.get(t) == reg and first_game.get(t, "9999") < date]
    return float(np.percentile(regs, 25)) if len(regs) >= 6 else 0.0


def n_matches_with(org, url, before_date):
    return sum(1 for d, o, _ in player_hist.get(url, ())
               if o == org and d < before_date)


def prev_org_rating(url, org, date, rv):
    """Most recent other-org appearance within 400 days."""
    for d, o, _ in reversed(player_hist.get(url, ())):
        if d >= date:
            continue
        if o != org:
            if (pd.Timestamp(date) - pd.Timestamp(d)).days > 400:
                return None
            return float(rv[eng.tidx[o]])
    return None


def r_adjusted(org, mid, date, rv, M):
    r_org = float(rv[eng.tidx[org]])
    lu = series_lineup.get((org, mid))
    if not lu:
        return r_org
    reg_pr = None
    vals = []
    for u in lu:
        n_together = n_matches_with(org, u, date)
        if n_together >= M:
            vals.append(r_org)
            continue
        w_new = max(0.0, (M - n_together) / M)
        prev = prev_org_rating(u, org, date, rv)
        if prev is None:
            if reg_pr is None:
                reg_pr = region_prior(date, ORG_REGIONS.get(org, ""), rv)
            src = reg_pr
        else:
            src = prev
        vals.append(w_new * src + (1 - w_new) * r_org)
    return float(np.mean(vals))


def build_rdiff(M):
    rd_adj = np.full(len(s), np.nan)
    for i, row in enumerate(s.itertuples(index=False)):
        rv = daily_vec(row.date)
        if rv is None or np.isnan(rd_base[i]):
            continue
        rw = r_adjusted(row.winner, row.match_id, row.date, rv, M)
        rl = r_adjusted(row.loser, row.match_id, row.date, rv, M)
        rd_adj[i] = rw - rl
    return rd_adj


def series_pv(b, rdv, mask):
    pm = 1 / (1 + np.exp(-b * rdv[mask]))
    fm = fmts[mask]
    return np.where(np.isin(fm, ("bo5", "bo5_gf")),
                    pm ** 3 * (1 + 3 * (1 - pm) + 6 * (1 - pm) ** 2),
                    np.where(fm == "bo1", pm, pm ** 2 * (3 - 2 * pm)))


def nll(b, rdv, mask):
    return -np.mean(np.log(np.clip(series_pv(b, rdv, mask), 1e-9, 1)))


def score(name, rdv, extra=None):
    v = ~np.isnan(rdv)
    b = float(minimize_scalar(lambda x: nll(x, rdv, v & train_v),
                              bounds=(0.02, 0.6), method="bounded").x)
    r = {"beta": round(b, 4), "ll_test": round(float(nll(b, rdv, v & test_v)), 5)}
    if extra:
        r.update(extra)
    print(f"{name:<26} b={b:.3f} ll={r['ll_test']:.5f}", flush=True)
    return r, b, v


res = {}
res["v3_base"], b_v3, v_v3 = score("v3_base", rd_base)

rest_max = np.array([max(rest_days(r.winner, r.date), rest_days(r.loser, r.date))
                     for r in s.itertuples(index=False)])
res["n_postbreak_test"] = int(((rest_max > 45) & (rest_max < 900) & test_v).sum())
pb_mask = (rest_max > 45) & (rest_max < 900)

for M in (6, 10, 14):
    rdM = build_rdiff(M)
    res[f"carry_M{M}"], bM, vM = score(f"carry_M{M}", rdM)
    # post-break subset scores
    res[f"carry_M{M}"]["ll_postbreak"] = round(
        float(nll(bM, rdM, vM & test_v & pb_mask)), 5)
    if M == 10:
        rd_c10, b_c10, v_c10 = rdM, bM, vM
res["v3_base"]["ll_postbreak"] = round(
    float(nll(b_v3, rd_base, v_v3 & test_v & pb_mask)), 5)
print("post-break test n:", res["n_postbreak_test"],
      "| v3:", res["v3_base"]["ll_postbreak"],
      "| carry M10:", res["carry_M10"]["ll_postbreak"])

# break dampener on top of carry_M10: shrink gap by gamma for post-break rows
def fit_gamma(rdv, bfix):
    def g_nll(gm):
        rd2 = rdv.copy()
        rd2[pb_mask] = gm * rdv[pb_mask]
        return nll(bfix, rd2, ~np.isnan(rd2) & train_v & pb_mask)
    r = minimize_scalar(g_nll, bounds=(0.4, 1.4), method="bounded")
    return float(r.x)


gm = fit_gamma(rd_c10, b_c10)
rd_g = rd_c10.copy()
rd_g[pb_mask] = gm * rd_c10[pb_mask]
res["gamma_fit"] = round(gm, 4)
res["carry_damp"], b_g, v_g = score(f"carry+damp(g={gm:.2f})", rd_g)
res["carry_damp"]["ll_postbreak"] = round(
    float(nll(b_g, rd_g, v_g & test_v & pb_mask)), 5)
print("damp post-break:", res["carry_damp"]["ll_postbreak"])

# bootstraps
vv = v_v3 & v_c10 & test_v
res["boot_carry_vs_v3"] = paired_bootstrap(
    series_pv(b_c10, rd_c10, vv), series_pv(b_v3, rd_base, vv))
print("boot carry vs v3:", res["boot_carry_vs_v3"])

np.save(os.path.join(OUT, "rd_carry_M10.npy"), rd_c10)
with open(os.path.join(OUT, "carryover.json"), "w") as f:
    json.dump(res, f, indent=1, default=str)
print("saved out/carryover.json")
