"""SPEC RUN prep: dump engine corpus structures + verify named cases from data.

Pre-registration data verification (lineups only; NO outcome/holdout looks).
Dumps corpus.pkl with games, lineups, per-org match sequences for the
classifier layer, and prints the SEN/ENVY/LEV lineup facts the fixtures need.
"""
import os
import pickle
import sys

import pandas as pd

TL = "/Users/benny_es1/PythonTest/testing_lab"
V8 = os.path.join(TL, "v8")
HERE = os.path.join(V8, "scratch", "roster", "spec_run")
sys.path.insert(0, TL)
sys.path.insert(0, V8)

from engine import Engine  # noqa: E402

eng = Engine()
print(f"games={len(eng.games)} teams={len(eng.teams)}")

# per-org ordered match sequence [(date_s, match_id)] + lineups
seq = {o: list(v) for o, v in eng.team_match_seq.items()}
lus = dict(eng.lineups)

# cross-check vs v8/data/lineups.csv (same maps-CSV provenance)
lcsv = pd.read_csv(os.path.join(V8, "data", "lineups.csv"))
csv_l = {(r.org, int(r.match_id)): frozenset(str(r.players).split(";"))
         for r in lcsv.itertuples(index=False)}
common = set(lus) & set(csv_l)
mism = [k for k in common if lus[k] != csv_l[k]]
print(f"lineups: engine={len(lus)} csv={len(csv_l)} common={len(common)} "
      f"mismatch={len(mism)}")
if mism[:3]:
    for k in mism[:3]:
        print("  MISMATCH", k, lus[k] ^ csv_l[k])
eng_only = set(lus) - set(csv_l)
print(f"engine-only (corpus additions not in csv): {len(eng_only)}")

# event_class per event (for EWC-class census tagging)
ev_class = dict(lcsv.drop_duplicates("event_id")[["event_id", "event_class"]]
                .itertuples(index=False))
print("event_class values:", lcsv.event_class.value_counts().to_dict())
missing_ev = sorted({g["event_id"] for g in eng.games} - set(ev_class))
print("events lacking class tag:", missing_ev)

games_slim = [{"match_id": g["match_id"], "event_id": g["event_id"],
               "winner": g["winner"], "loser": g["loser"],
               "date_s": g["date_s"]} for g in eng.games]
with open(os.path.join(HERE, "corpus.pkl"), "wb") as f:
    pickle.dump({"games": games_slim, "lineups": lus, "team_match_seq": seq,
                 "ev_class": ev_class, "teams": eng.teams}, f)
print("corpus.pkl written")


def show(org, d0, d1, hi=()):
    print(f"--- {org} {d0}..{d1} ---")
    for ds, mid in seq.get(org, []):
        if d0 <= ds <= d1:
            lu = lus.get((org, mid))
            names = sorted(u.rsplit("/", 1)[-1] for u in lu) if lu else None
            mark = ""
            if names and any(any(h in n for n in names) for h in hi):
                mark = "   <<<"
            print(f"  {ds} m{mid} {names}{mark}")


# SEN: the spec's named case — find the Marved week (2026)
show("SEN", "2026-04-01", "2026-12-31", hi=("marved",))
# ENVY chain (2026) and LEV/Neon (2025-11) context windows
show("ENVY", "2026-01-15", "2026-03-01")
show("LEV", "2025-11-01", "2025-12-15", hi=("neon",))
