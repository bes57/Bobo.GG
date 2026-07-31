"""E2 round-level Bradley-Terry machinery.

daily_weights(): VERBATIM replication of engine.run's per-day weight block for
the v6 config path (games/consistency decay, year continuity, champ mult,
w_custom). Validated by massey_parity() reproducing the engine's rdiff to
<1e-8 before any BT run is trusted (preregistered gate)."""
import math
import os
import sys

import numpy as np
from scipy.special import comb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from h1_lib import TRAIN_END, log, t_of_m  # noqa: E402


# ── engine weight replication (v6 path only) ────────────────────────────────

def build_game_consist(eng):
    """Copy of engine.run's lazy _game_consist build (HL16 trailing map wr)."""
    if hasattr(eng, "_game_consist"):
        return eng._game_consist
    gc = {}
    lam_wr = math.log(2) / 16.0
    for org, rows_all in eng.team_game_rows.items():
        wr_num = wr_den = 0.0
        for ri in rows_all:
            won = eng.games[ri]["winner"] == org
            wr = wr_num / wr_den if wr_den > 0.5 else 0.5
            gc[(org, ri)] = ((won and wr >= 0.5) or ((not won) and wr < 0.5))
            wr_num = wr_num * math.exp(-lam_wr) + (1.0 if won else 0.0)
            wr_den = wr_den * math.exp(-lam_wr) + 1.0
    eng._game_consist = gc
    return gc


def daily_weights(eng, day, w_custom, hl_c=20.0, hl_a=12.0, champ_mult=2.0):
    """(m_hist, w) for one day under the v6 config. Returns (None, None) when
    the engine would skip the day (<30 hist games)."""
    day_num = int(np.datetime64(day, "D").astype(int))
    m_hist = eng.g_dnum < day_num
    if m_hist.sum() < 30:
        return None, None
    gc = build_game_consist(eng)

    def gdecay(ago, org, ri):
        ok = gc.get((org, ri), True)
        lam = math.log(2) / (hl_c if ok else hl_a)
        return math.exp(-lam * ago)

    hist_idx = np.where(m_hist)[0]
    pos_in_hist = {gi: k for k, gi in enumerate(hist_idx)}
    w_w = np.ones(len(hist_idx))
    w_l = np.ones(len(hist_idx))
    for org, rows_ in eng.team_game_rows.items():
        played = [ri for ri in rows_ if ri in pos_in_hist]
        n_p = len(played)
        for k, ri in enumerate(played):
            ago = n_p - 1 - k
            j = pos_in_hist[ri]
            if eng.games[ri]["winner"] == org:
                w_w[j] = gdecay(ago, org, ri)
            else:
                w_l[j] = gdecay(ago, org, ri)
    base = np.sqrt(w_w * w_l)
    cw, cl = eng._continuity_vec(day, "year", 1.0)
    base = base * np.sqrt(cw[m_hist] * cl[m_hist])
    mult = np.where(eng.champ[m_hist], champ_mult, 1.0)
    mult = mult * np.asarray(w_custom)[m_hist]
    return m_hist, base * mult


def massey_parity(eng, frame, w_custom, ref_rdiff, ridge=0.5, rpr=1.5):
    """Replicate engine.run's Massey solve with daily_weights; compare rdiff."""
    n_t = len(eng.teams)
    rd_t = t_of_m(np.array([g["wr"] - g["lr"] for g in eng.games], dtype=float))
    s = frame
    rat_w = np.full(len(s), np.nan)
    rat_l = np.full(len(s), np.nan)
    from collections import defaultdict
    s_by_day = defaultdict(list)
    for i, r in enumerate(s.itertuples(index=False)):
        s_by_day[r.date].append(i)
    prev_rvec = None
    for day in eng.pred_days:
        m_hist, w = daily_weights(eng, day, w_custom)
        if m_hist is None:
            continue
        wi, li = eng.wi[m_hist], eng.li[m_hist]
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
        prior = np.zeros(n_t)
        if prev_rvec is not None:
            for ri_ in range(4):
                m_reg = eng.team_region_idx == ri_
                if m_reg.sum() >= 4:
                    prior[m_reg] = prev_rvec[m_reg].mean()
        M[np.diag_indices(n_t)] += rpr
        p += rpr * prior
        M[-1, :] = 1.0
        p[-1] = 0.0
        r_vec = np.linalg.solve(M, p)
        prev_rvec = r_vec.copy()
        for i in s_by_day[day]:
            row = s.iloc[i]
            rat_w[i] = r_vec[eng.tidx[row.winner]]
            rat_l[i] = r_vec[eng.tidx[row.loser]]
    my_rdiff = rat_w - rat_l
    both = ~np.isnan(my_rdiff) & ~np.isnan(ref_rdiff)
    gap = float(np.max(np.abs(my_rdiff[both] - ref_rdiff[both])))
    same_valid = bool((np.isnan(my_rdiff) == np.isnan(ref_rdiff)).all())
    return gap, same_valid


# ── race function (first-to-13, OT prob = round prob) on a grid ─────────────

_PG = np.linspace(1e-6, 1 - 1e-6, 8001)


def _race_exact(p):
    ks = np.arange(13, 25)
    P13 = np.zeros_like(p)
    for k in ks:
        P13 += comb(24, k) * p ** k * (1 - p) ** (24 - k)
    P12 = comb(24, 12) * p ** 12 * (1 - p) ** 12
    return P13 + P12 * p


_RACE = _race_exact(_PG)
_DRACE = np.gradient(_RACE, _PG)


def race(p):
    return np.interp(p, _PG, _RACE)


def drace(p):
    return np.interp(p, _PG, _DRACE)


def sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35, 35)))


# ── the joint BT fit for one day ────────────────────────────────────────────

def bt_fit(cells, maps_fb, w_g, m_hist, n_t, lam, region_idx, prior_vec,
           s0, h0, h_from_cells=True, max_iter=60, tol=1e-8):
    """cells: dict(gi, att, dfn, k, n) arrays; maps_fb: dict(gi, i, j) arrays.
    w_g: per-GAME weight array full length (0 where not hist). Returns s, h."""
    cm = w_g[cells["gi"]] > 0
    fm = w_g[maps_fb["gi"]] > 0
    ca, cd = cells["att"][cm], cells["dfn"][cm]
    ck, cn = cells["k"][cm].astype(float), cells["n"][cm].astype(float)
    cw = w_g[cells["gi"]][cm]
    fi, fj = maps_fb["i"][fm], maps_fb["j"][fm]
    fw = w_g[maps_fb["gi"]][fm]
    s = s0.copy()
    h = h0
    lam_reg = 3.0 * lam
    for it in range(max_iter):
        grad = np.zeros(n_t + 1)
        H = np.zeros((n_t + 1, n_t + 1))
        # round cells
        eta = s[ca] - s[cd] + h
        p = sig(eta)
        g_row = cw * (ck - cn * p)
        f_row = cw * cn * p * (1 - p) + 1e-12
        np.add.at(grad, ca, g_row)
        np.add.at(grad, cd, -g_row)
        np.add.at(H, (ca, ca), f_row)
        np.add.at(H, (cd, cd), f_row)
        np.add.at(H, (ca, cd), -f_row)
        np.add.at(H, (cd, ca), -f_row)
        if h_from_cells:
            grad[n_t] += g_row.sum()
            H[n_t, n_t] += f_row.sum()
            np.add.at(H, (ca, np.full(len(ca), n_t)), f_row)
            np.add.at(H, (np.full(len(ca), n_t), ca), f_row)
            np.add.at(H, (cd, np.full(len(cd), n_t)), -f_row)
            np.add.at(H, (np.full(len(cd), n_t), cd), -f_row)
        # map-level fallback rows (winner-referenced y=1)
        if len(fi):
            q = s[fi] - s[fj]
            pb = 0.5 * (sig(q + h) + sig(q - h))
            dpb = 0.5 * (sig(q + h) * (1 - sig(q + h))
                         + sig(q - h) * (1 - sig(q - h)))
            P = np.clip(race(pb), 1e-9, 1 - 1e-9)
            dP = drace(pb) * dpb
            u = fw * dP / P
            f2 = fw * dP * dP / (P * (1 - P)) + 1e-12
            np.add.at(grad, fi, u)
            np.add.at(grad, fj, -u)
            np.add.at(H, (fi, fi), f2)
            np.add.at(H, (fj, fj), f2)
            np.add.at(H, (fi, fj), -f2)
            np.add.at(H, (fj, fi), -f2)
        # priors: ridge to 0 + region ridge to prior_vec; small h ridge
        grad[:n_t] += -lam * s - lam_reg * (s - prior_vec)
        H[np.arange(n_t), np.arange(n_t)] += lam + lam_reg
        grad[n_t] += -1e-4 * h
        H[n_t, n_t] += 1e-4 + 1e-9
        step = np.linalg.solve(H, grad)
        nrm = np.max(np.abs(step))
        if nrm > 5.0:
            step *= 5.0 / nrm
        s += step[:n_t]
        h += step[n_t]
        if nrm < tol:
            break
    return s, h, it + 1
