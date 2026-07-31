"""agent:bias-h2 shared machinery. Frame verify, engine setup, v6 config,
prob helpers. Preregistered in v8/preregister.bias_h2.md."""
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
V8 = os.path.dirname(os.path.dirname(HERE))
TL = os.path.dirname(V8)
STATS = os.path.join(V8, "stats")
LOG = os.path.join(V8, "logs", "bias_h2.log")
FRAME = os.path.join(V8, "data", "frame_expanded", "series.csv")
CRN = os.path.join(V8, "crn.json")
for p in (TL, V8):
    if p not in sys.path:
        sys.path.insert(0, p)

BETA_TRAIN_END = "2024-12-31"
IT_SPLIT = "2024-06-30"          # internal walk-forward split inside train
ELITE = ["T1", "PRX", "100T", "NRG", "TL"]
FLOOR = ["TS", "JDG", "TE", "C9"]


def log(msg):
    with open(LOG, "a") as f:
        f.write(msg.rstrip() + "\n")
    print(msg, flush=True)


def load_frame():
    """Load the canonical frame, verifying sha256 against crn.json. Abort
    loudly on mismatch (wave2_common law)."""
    h = hashlib.sha256(open(FRAME, "rb").read()).hexdigest()
    want = json.load(open(CRN))["frame_expanded"]["series_csv_sha256"]
    if h != want:
        raise RuntimeError(f"FRAME SHA MISMATCH: {h} != {want} — aborting")
    df = pd.read_csv(FRAME)
    df = df.sort_values(["date", "match_id"]).reset_index(drop=True)
    return df


def build_engine(frame):
    """Engine on the expanded corpus with the frame as evaluation series.
    Returns (eng, PO) where PO is the playoff w_custom vector (per game)."""
    from engine import Engine
    eng = Engine()
    eng.series = frame.reset_index(drop=True)
    eng.pred_days = sorted(frame.date.unique())
    stage_by_mid = dict(zip(frame.match_id, frame.stage))
    g_stage = np.array([stage_by_mid.get(g["match_id"], "groups")
                        for g in eng.games])
    PO = np.where(np.isin(g_stage, ("playoffs", "grand_final")), 1.6, 1.0)
    return eng, PO


def v6_cfg(PO, **over):
    cfg = {"decay": {"kind": "games", "consistency": (20.0, 12.0)},
           "rd": {"power": 0.75, "scale": 2.5},
           "roster_mode": "year", "roster_persistence": 0.3,
           "ridge": 0.5, "champ_mult": 2.0,
           "region_prior_ridge": 1.5, "w_custom": PO}
    cfg.update(over)
    return cfg


def run_cfg(eng, cfg):
    """Deterministic self-contained run: reset cross-run engine state first."""
    eng._prev_rvec = None
    return eng.run(cfg)


def sp(pm, fm):
    """Series prob from map prob, engine closed form."""
    return np.where(np.isin(fm, ("bo5", "bo5_gf")),
                    pm ** 3 * (1 + 3 * (1 - pm) + 6 * (1 - pm) ** 2),
                    np.where(fm == "bo1", pm, pm ** 2 * (3 - 2 * pm)))


def fit_beta(rdiff, fmts, mask):
    from scipy.optimize import minimize_scalar
    def nll(b):
        p = sp(1 / (1 + np.exp(-b * rdiff[mask])), fmts[mask])
        return -np.mean(np.log(np.clip(p, 1e-9, 1)))
    return float(minimize_scalar(nll, bounds=(0.03, 0.6), method="bounded").x)


def probs_full(rdiff, fmts, beta):
    return sp(1 / (1 + np.exp(-beta * rdiff)), fmts)


def ll(p, mask):
    return float(-np.mean(np.log(np.clip(p[mask], 1e-9, 1))))


def masks(frame, rdiff):
    valid = ~np.isnan(rdiff)
    train = valid & (frame.date <= BETA_TRAIN_END).values
    hold = valid & (frame.date > BETA_TRAIN_END).values
    return valid, train, hold
