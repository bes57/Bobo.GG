"""SPEC RUN engine wrapper: v6 + roster subsystem via the per-side hook.

Look hygiene: run_config() pops every holdout metric from the engine return
unseen. Holdout probabilities exist only inside the returned arrays and are
never aggregated here; ONLY read_corpus.py computes a holdout number, once,
for the preregistered read(s).

Nesting: with plan=None (or a=s=0) the _continuity_vec override falls through
to the parent path untouched — asserted bit-identical by sha256 checksum in
train_stages.py step 0.
"""
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

TL = "/Users/benny_es1/PythonTest/testing_lab"
V8 = os.path.join(TL, "v8")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TL)
sys.path.insert(0, V8)
sys.path.insert(0, HERE)

from engine import Engine  # noqa: E402

FRAME_PATH = os.path.join(V8, "data", "frame_expanded", "series.csv")
BETA_TRAIN_END = "2024-12-31"


def load_frame():
    crn = json.load(open(os.path.join(V8, "crn.json")))
    sha = hashlib.sha256(open(FRAME_PATH, "rb").read()).hexdigest()
    assert sha == crn["frame_expanded"]["series_csv_sha256"], \
        f"FRAME SHA MISMATCH {sha}"
    return pd.read_csv(FRAME_PATH).reset_index(drop=True)


def p_series_closed(beta, rdiff, fm):
    pm = 1.0 / (1.0 + np.exp(-beta * rdiff))
    return np.where(np.isin(fm, ("bo5", "bo5_gf")),
                    pm ** 3 * (1 + 3 * (1 - pm) + 6 * (1 - pm) ** 2),
                    np.where(fm == "bo1", pm, pm ** 2 * (3 - 2 * pm)))


class EngineSpec(Engine):
    """v6 + spec subsystem. Modifiers enter through the same per-side
    continuity-vector hook treatment (e) used: x mult^2 on the affected
    side's factor pre-sqrt => x mult per side on the game weight."""

    def prepare(self):
        """Static org -> (winner rows, loser rows, match positions)."""
        self._org_rows = {}
        for org, ti in self.tidx.items():
            seq = self.team_match_seq.get(org, [])
            pos_of = {mid: i for i, (d, mid) in enumerate(seq)}
            rw = np.where(self.wi == ti)[0]
            rl = np.where(self.li == ti)[0]
            pw = np.array([pos_of[self.games[r]["match_id"]] for r in rw],
                          dtype=np.int64)
            pl = np.array([pos_of[self.games[r]["match_id"]] for r in rl],
                          dtype=np.int64)
            self._org_rows[org] = (rw, pw, rl, pl)
        self._spec = None

    def enable_spec(self, plan, a, tau, s, n_min=3, cap=None,
                    team_filter=None, boost_only=False,
                    min_boundary_date=None):
        self._spec = {"plan": plan, "a": a, "tau": tau, "s": s,
                      "n_min": n_min, "cap": cap, "team_filter": team_filter,
                      "boost_only": boost_only,
                      "min_boundary_date": min_boundary_date}

    def disable_spec(self):
        self._spec = None

    def _continuity_vec(self, ref_date_s, mode, persistence):
        cw, cl = super()._continuity_vec(ref_date_s, mode, persistence)
        sp = self._spec
        if sp is None or (sp["a"] == 0 and sp["s"] == 0):
            return cw, cl
        plan = sp["plan"]
        orgs = sp["team_filter"] if sp["team_filter"] is not None \
            else self._org_rows.keys()
        for org in orgs:
            if org not in self._org_rows:
                continue
            ver = plan.version_asof(org, ref_date_s)
            if ver is None:
                continue
            mult = plan.multipliers(ver, sp["a"], sp["tau"], sp["s"],
                                    n_min=sp["n_min"], cap=sp["cap"],
                                    boost_only=sp["boost_only"],
                                    min_boundary_date=sp["min_boundary_date"])
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
        return cw, cl


_ENGINE = None
_FRAME = None


def get_engine():
    """One engine instance per process; configs must call enable/disable
    around each run (state is only the _spec dict; run() is stateless apart
    from _prev_rvec which run() re-seeds day by day)."""
    global _ENGINE, _FRAME
    if _ENGINE is None:
        _FRAME = load_frame()
        _ENGINE = EngineSpec()
        _ENGINE.series = _FRAME.copy().reset_index(drop=True)
        _ENGINE.pred_days = sorted(_FRAME.date.unique())
        _ENGINE.prepare()
        stage_by_mid = dict(zip(_FRAME.match_id, _FRAME.stage))
        g_stage = np.array([stage_by_mid.get(g["match_id"], "groups")
                            for g in _ENGINE.games])
        _ENGINE._w_custom = np.where(
            np.isin(g_stage, ("playoffs", "grand_final")), 1.6, 1.0)
    return _ENGINE, _FRAME


def v6_cfg(eng, daily=False):
    cfg = {"rd": {"power": 0.75, "scale": 2.5}, "roster_mode": "year",
           "roster_persistence": 0.3, "ridge": 0.5, "champ_mult": 2.0,
           "region_prior_ridge": 1.5,
           "decay": {"kind": "games", "consistency": (20.0, 12.0)},
           "w_custom": eng._w_custom}
    if daily:
        cfg["daily_out"] = True
    return cfg


def run_config(plan, a, tau, s, n_min=3, cap=None, team_filter=None,
               boost_only=False, min_boundary_date=None, daily=False,
               cfg_override=None):
    """One full walk-forward solve. Returns train-side info + rdiff/p_all.
    HOLDOUT METRICS ARE POPPED UNSEEN. cfg_override: diagnostic-only tweaks
    (e.g. region_prior_ridge=0 for the P2 root-cause fixture) — never used
    for any scored configuration."""
    eng, frame = get_engine()
    eng._prev_rvec = None                       # clean region-prior chain
    if plan is None or (a == 0 and s == 0):
        eng.disable_spec()
    else:
        eng.enable_spec(plan, a, tau, s, n_min=n_min, cap=cap,
                        team_filter=team_filter, boost_only=boost_only,
                        min_boundary_date=min_boundary_date)
    cfg = v6_cfg(eng, daily=daily)
    if cfg_override:
        cfg.update(cfg_override)
    out = eng.run(cfg)
    eng.disable_spec()
    for kk in ("ll_test", "brier_test", "p_test", "test_mask"):
        out.pop(kk, None)                       # LOOK HYGIENE
    rd = out["rdiff"]
    valid = ~np.isnan(rd)
    fmts = frame.fmt.values
    p_all = np.full(len(frame), np.nan)
    p_all[valid] = p_series_closed(out["beta"], rd[valid], fmts[valid])
    train_m = valid & (frame.date <= BETA_TRAIN_END).values
    nll = np.full(len(frame), np.nan)
    nll[valid] = -np.log(np.clip(p_all[valid], 1e-9, 1.0))
    return {"beta": out["beta"], "ll_train": out["ll_train"],
            "rdiff": rd, "p_all": p_all, "valid": valid,
            "train_mask": train_m, "nll_rows": nll,
            "daily_r": out.get("daily_r"), "tidx": eng.tidx}


def rd_checksum(rdiff):
    return hashlib.sha256(np.ascontiguousarray(rdiff).tobytes()).hexdigest()


def daily_checksum(daily_r):
    days = sorted(daily_r.keys())
    R = np.stack([daily_r[d] for d in days])
    return hashlib.sha256(np.ascontiguousarray(R).tobytes()).hexdigest()
