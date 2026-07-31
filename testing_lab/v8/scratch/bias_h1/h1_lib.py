"""bias_h1 shared library: frame/engine setup, v6 config, as-of rating lookup,
map-class taxonomy, Tobit imputation. Preregistered in preregister.bias_h1.md."""
import hashlib
import json
import math
import os
import sys

import numpy as np
import pandas as pd

TL = "/Users/benny_es1/PythonTest/testing_lab"
V8 = os.path.join(TL, "v8")
SCRATCH = os.path.join(V8, "scratch", "bias_h1")
STATS = os.path.join(V8, "stats")
LOG = os.path.join(V8, "logs", "bias_h1.log")
for p in (TL, V8):
    if p not in sys.path:
        sys.path.insert(0, p)

FRAME_CSV = os.path.join(V8, "data", "frame_expanded", "series.csv")
FRAME_SHA = "ff772d417b714844aa1b11426d8c04917e1b2848c9c8df811cd9988694d55142"
TRAIN_END = "2024-12-31"

# v6 target transform + censoring constants (preregistered)
RD_POWER, RD_SCALE = 0.75, 2.5
C13 = 13.0 ** RD_POWER * RD_SCALE      # 17.1130
C_OT = 2.0 ** RD_POWER * RD_SCALE      # 4.2045


def log(msg):
    from datetime import datetime
    line = f"{datetime.now():%F %T} [bias_h1] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def t_of_m(m):
    return np.abs(m) ** RD_POWER * RD_SCALE


def load_frame():
    h = hashlib.sha256(open(FRAME_CSV, "rb").read()).hexdigest()
    if h != FRAME_SHA:
        raise RuntimeError(f"frame_expanded sha256 mismatch: {h}")
    f = pd.read_csv(FRAME_CSV)
    assert len(f) == 2058 and (f.date > TRAIN_END).sum() == 1217
    return f.reset_index(drop=True)


def make_engine(frame):
    from engine import Engine
    eng = Engine()
    eng.series = frame.reset_index(drop=True)
    eng.pred_days = sorted(frame.date.unique())
    return eng


def v6_cfg(eng, frame, daily_out=False):
    stage_by_mid = dict(zip(frame.match_id, frame.stage))
    g_stage = np.array([stage_by_mid.get(g["match_id"], "groups")
                        for g in eng.games])
    PO = np.where(np.isin(g_stage, ("playoffs", "grand_final")), 1.6, 1.0)
    cfg = {"rd": {"power": RD_POWER, "scale": RD_SCALE},
           "decay": {"kind": "games", "consistency": (20.0, 12.0)},
           "roster_mode": "year", "roster_persistence": 0.3,
           "ridge": 0.5, "champ_mult": 2.0, "region_prior_ridge": 1.5,
           "w_custom": PO}
    if daily_out:
        cfg["daily_out"] = True
    return cfg


def game_class(g):
    """CAP / OT / REG / JUNK from rounds won (wr, lr)."""
    m = g["wr"] - g["lr"]
    if g["lr"] >= 12 and m == 2:
        return "OT"
    if m == 13:
        return "CAP"
    if 2 <= m <= 12:
        return "REG"
    return "JUNK"   # forfeits / weird margins (reported, kept at raw target)


def asof_lookup(daily_r, pred_days):
    """Return f(date_s) -> r_vec for the latest solved day <= date_s (games on
    day D use day-D ratings, solved from games < D). None if no earlier day."""
    days = sorted(daily_r.keys())
    days_np = np.array(days)

    def f(date_s):
        i = np.searchsorted(days_np, date_s, side="right") - 1
        if i < 0:
            return None
        return daily_r[days[i]]
    return f


def series_prob(beta, rdiff, fmts):
    pm = 1 / (1 + np.exp(-beta * rdiff))
    return np.where(np.isin(fmts, ("bo5", "bo5_gf")),
                    pm ** 3 * (1 + 3 * (1 - pm) + 6 * (1 - pm) ** 2),
                    np.where(fmts == "bo1", pm, pm ** 2 * (3 - 2 * pm)))


def fit_beta(rdiff, fmts, mask, bounds=(0.03, 0.6)):
    """Engine-default bounds for Massey-scale rdiffs; round-BT logit-scale
    rdiffs need wide bounds (scale-bound refit, wave2_common law)."""
    from scipy.optimize import minimize_scalar
    def nll(b):
        p = series_prob(b, rdiff[mask], fmts[mask])
        return -np.mean(np.log(np.clip(p, 1e-9, 1)))
    return float(minimize_scalar(nll, bounds=bounds, method="bounded").x)


# ── Tobit imputation (preregistered formulas) ───────────────────────────────

def impute_targets(games, mu, sigma, base_target):
    """Per-game imputed rd_custom. mu: winner-referenced latent mean
    (r_w - r_l as-of the game's date; np.nan -> 0). base_target: v6 t(m)
    signed +(winner-ref) — engine wants winner-referenced positive targets
    (rd_raw>0 always; copysign keeps them positive)."""
    from scipy.stats import norm
    out = base_target.copy()
    mu = np.where(np.isnan(mu), 0.0, mu)
    for i, g in enumerate(games):
        c = game_class(g)
        if c == "CAP":
            a = (C13 - mu[i]) / sigma
            lam = norm.pdf(a) / max(norm.sf(a), 1e-12)
            out[i] = mu[i] + sigma * lam
        elif c == "OT":
            a = (0.0 - mu[i]) / sigma
            b = (C_OT - mu[i]) / sigma
            den = max(norm.cdf(b) - norm.cdf(a), 1e-12)
            out[i] = mu[i] + sigma * (norm.pdf(a) - norm.pdf(b)) / den
        # REG / JUNK: keep base target
    return out
