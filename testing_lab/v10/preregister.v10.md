# v10 PREREGISTRATION — unconditional year isolation

Written before any VAL row was scored for this program. Governing law:
`testing_lab/v9/stats/v9_transfer_protocol.json` and
`v9_prospective_protocol.json`, unchanged. This document is bound by them; it
does not amend them.

Author: agent:v10-yeariso. Frozen 2026-08-12.

---

## 1. The operator's hypothesis

> "Treat each year as a new year where you don't look back at match data from
> previous years. Previous years are often with old rosters that shouldn't
> affect current ratings."

Formally: at every calendar-year boundary, set the year-continuity factor to
zero for **every** team, unconditionally — so a solve on day D in year Y uses
only games played in year Y.

## 2. What the champion already does (why this is an endpoint, not a new idea)

v6 already implements year-boundary discounting, but **conditional on measured
roster carryover**:

    year_cont[(org, Y)] = min(|roster(last event Y-1) ∩ roster(first event Y)| / 5, 1)

applied to each game as `sqrt(cw*cl)` (engine.py L408-420). So:

- a team that kept **0/5** players already has its prior-year games **zeroed**;
- a team that kept **5/5** keeps them at full weight;
- everything between is interpolated from the data.

The proposal is therefore the `year_cont ≡ 0` endpoint of a dial v6 already
fits per team. Measured on this corpus (this program, §6): teams retain
**3.15/5 players** on average across a boundary; **24%** keep all five and
**32%** keep two or fewer. The hypothesis is right about the 32% — and v6
already handles them. The question is what happens to the 24% whose history is
genuinely continuous.

## 3. Prior evidence (disclosed before running, because it is refuting)

Recorded in earlier labs, all on the same frame:

| prior experiment | what it did | result |
| --- | --- | --- |
| `g16_roster_none` (run_experiments7) | year_cont ≡ 1 (full carryover, the OTHER endpoint) | **−2.97 milli-LL** |
| v8 Read 1(b) change-point continuity | replace year mode with graded change-points | **−3.889m**, falsifier fired |
| v8 Read (d) phase-reset filter | hard reset at each roster change | **−19.275m**, falsifier fired |
| `run_carryover.py` | import joining players' prior-org rating | −3.3m; fit ran to bound wanting **expansion** |
| v8 decay lab | shorter calendar half-lives | worse everywhere; "memory was too short, not too long" |

Standing ledger entry (`gen_final_model.py`): *"Lineup-overlap roster
reweighting (3 schemes) — year-boundary continuity already covers real
rebuilds."*

Every coarser override of the measured overlap has lost. This program tests
the one endpoint with no record of having been run. **The prior is that it
loses**; it is run because it is cheap, it is the operator's question, and the
endpoint is genuinely unmeasured.

## 4. Arms (frozen)

### Correction recorded before running

`ROSTER_CONT = 0.3` (BuildRatingTimeline.py:132) is **dead code** — grep shows
it is referenced only at its own definition, and mode `"year"` in the lab
engine ignores the `persistence` argument entirely (engine.py:409 passes 1.0).
The factor actually applied is the measured `min(overlap/5, 1)`, whose mean on
this corpus is **0.62**, not 0.3. Any prose describing v6 as "0.3 continuity"
is wrong, including in earlier lab documents.

Consequently v6's real year boundary is a ~38% haircut per crossing, ~22% of
org-boundaries are full carryover (factor 1.0), and only 10 of 130 are hard
zeros.

### The three cross-year channels

Isolation must cut all three or it is not isolation:

| # | channel | strength |
| --- | --- | --- |
| A | prior-year games still in the fit | ~11% weight at 1 year, ~1% at 2 |
| B | **region-prior chain** (`self._chain`, never reset) | **dominant** — a team with no in-year data solves to 0.75 × last year's regional mean |
| C | consistency classifier carries last year's winrate | weak (selects HL 20 vs 12) |

| arm | definition |
| --- | --- |
| `v6` | deployed champion, `year_cont = min(overlap/5, 1)` measured per team |
| `A_iso_hard` | `year_cont ≡ 0` at every boundary, all teams |
| `A_iso_50` | `year_cont ≡ 0.5` — unconditional half-discount (spans the dial) |
| `A_iso_floor` | `year_cont = min(overlap/5, 0.5)` — v6, capped so no team keeps full history |

`A_iso_50` and `A_iso_floor` exist so the result is a **curve**, not a single
point: if the optimum sits at v6's fitted position, that is a stronger and more
informative answer than a lone endpoint failure.

## 5. Evaluation plan (bound by v9 law)

- Hyperparameters: **none are tuned.** Every arm is a re-parameterisation of an
  existing mechanism; no grid, no search. This removes the FIT1 search
  obligation entirely — there is nothing to fit.
- β: refit per arm per fit-window, `minimize_scalar` bounded (0.001, 1.0),
  xatol 1e-6, mean series-NLL, paired against v6 refit identically.
  **Fixture: β(FIT1, v6) must equal 0.1152 ± 1e-3 or the run aborts.**
- Frame: `testing_lab/v8/data/frame_expanded/series.csv`, sha256 verified
  against `ff772d417b714844aa1b11426d8c04917e1b2848c9c8df811cd9988694d55142`;
  abort loudly on mismatch.
- Transfer: T1 (β on FIT1 → score VAL1) and T2 (β on FIT2 → score VAL2).
  **ONE ledgered transfer evaluation per arm**, written to
  `testing_lab/v9/stats/v9_looks.json` → `selection_reads` at run time.
- Advance rule A1–A5 exactly as written in `v9_transfer_protocol.json`. This is
  a **cross-family** change (engine state, not a nested v6 hyperparameter), so
  cross-family MDE context of **5.889m at n=1217** is quoted next to every
  number; the A4 gate floor stays **1.773m** as the protocol specifies.
- Prospective: nothing. The virgin window holds **53 settled series** against a
  first checkpoint of n=100, so no confirmatory read is available or attempted.
  Any v10 result is **selection-grade only** and cannot promote anything.

## 6. Disclosed prior look (exploratory)

Before this preregistration was written, one implementation sanity check was
run and is disclosed here rather than concealed: a 2026-only corpus was solved
end-to-end with the production builder to check the variant was implementable
and sign-plausible. It produced **β = 0.090819**, which fails the builder's own
Gate B sanity band (0.115, 0.145), and the builder refused to write outputs.

This scored **no VAL or holdout row** and ranked nothing; it is an
implementation plausibility check of exactly the kind
`exploratory_budget` describes. Logged against that budget (1 of 3).

## 7. Predictions (recorded before scoring)

1. `A_iso_hard` loses to v6 on both VAL1 and VAL2, by more than the 1.773m
   floor. Stated because the prior evidence is one-directional and it should
   cost something to be wrong.
2. The loss is **concentrated in January–March** of each season, where an
   isolated solve has the least history.
3. `A_iso_floor` loses by less than `A_iso_hard`; the dial is monotone with the
   optimum at or near v6's fitted position, not at either endpoint.

## 8. Standing law restated

Promotion means promotion **within the lab's flow only** — nothing on the
public site regardless of outcome, and VCTMM remains hands-off. Market data is
never a fitting target or a selection signal.
