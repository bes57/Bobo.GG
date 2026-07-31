# Phase 0 — Power analysis: what this program can actually distinguish

agent:power · 2026-07-28 · pre-registered in `preregister.power.md` BEFORE computation ·
data: frozen per-series probabilities `out/v7_probs.npz` / `out/v7_probs2.npz` (holdout n=1007,
2025-01-11 → 2026-07-23, 18 events) · randomness: `crn.json` · numbers: `stats/power_mde.json`,
`stats/v7_reclass.json`, `stats/ledger_reclass.json`, `stats/variance_reduction.json`.

## 0. Reproduction gate (passed)

Test mask reconstructed from `harness.load_series()` (rows 0..1694, date > 2024-12-31) matches
the npz masks element-for-element (1007/688/1695). All 24 published v7 `ll_test` values
reproduce to 5 dp with the reconstructed mask — including the champion's **0.64095**. The two
motivating published bootstraps reproduce to machine precision under legacy seed 7:
sym_20 −1.6541m / p 0.142 and consist_16_10 −0.3922m / p 0.285. All 23 pair records
re-verified (`legacy_seed7_reproduced: true` on every row of v7_reclass.json).

## 1. The instrument's resolution (MDE, 80% power @ two-sided α=0.05)

MDE₈₀ = 2.8016·σ_d/√n on per-series loss differences d = l_v6 − l_cand. σ_d depends
overwhelmingly on how *related* the two models are:

| Regime | σ_d median (range) | **MDE at n=1007** |
|---|---|---|
| Two decay variants of the same core (within-family: v5_asym, consist_*, v6+form*) | 0.022 (0.001–0.054) | **1.92 milli** |
| Two structurally different models (cross-family: sym_*, surprise, boxexp, power) | 0.072 (0.050–0.141) | **6.38 milli** |

Simulation check (the actual paired-bootstrap decision, 400 sims × 2000 resamples, CRN
mc_seeds): power at μ = analytic MDE = **0.795 / 0.795** (accept band 0.74–0.86); size at
μ=0 = 0.062 / 0.058. The analytic estimator stands.

**Plainly: at n=1007 this program cannot 80%-reliably distinguish two structurally different
models closer than ~6 milli, or two same-family variants closer than ~2 milli.** Every effect
size the v7 ladder chased (0.02–2.1 milli among candidates worth considering) was below the
instrument's floor.

## 2. Per-bucket resolution (the promotion rule's blind spots)

MDE₈₀ in milli-LL, using regime-median bucket σ (full table in stats/power_mde.json):

| Bucket | n | within | cross |
|---|---|---|---|
| format bo3 | 939 | 2.0 | 6.4 |
| format bo5 | 45 | 11.2 | **45.1** |
| format bo5_gf | 23 | 11.9 | 36.6 |
| format bo1 | 0 | — | — (no bo1 in holdout) |
| stage groups / playoffs | 502 / 438 | 2.8 / 2.9 | 8.6 / 10.3 |
| stage grand_final | 26 | 11.4 | 33.0 |
| international / domestic | 122 / 885 | 5.9 / 2.0 | 18.3 / 6.8 |
| EWC-class events | 109 | 6.4 | 21.4 |
| cross-region | 131 | 6.2 | 18.8 |
| domestic Am/EMEA/Pac/CN | 205–248 | 3.6–4.2 | 11.7–15.4 |
| fav band [.5,.6) → [.8,.9) | 467→42 | 2.8→5.8 | 9.2→26.8 |
| gap <1.5 → 7+ | 352→44 | 3.3→5.6 | 10.2→25.5 |
| post-break (>45d) | 145 | 4.9 | 15.4 |
| elite vs floor | 26 | 7.9 | **39.5** |
| form shift | 121 | 6.7 | 21.3 |
| roster change ≤3 matches (either team) | 598 | 2.5 | 7.8 |
| roster change 4–10 / stable >10 | 304 / 105 | 3.4 / 6.2 | 12.4 / 21.8 |

(Roster-change recency computed from `v8/data/lineup_features.csv`, which landed mid-run;
coverage 1007/1007.) The buckets that motivated past verdicts — elite-vs-floor (−5.1m in the
v6 profile), bo5, EWC-class — have floors of **20–45 milli**: no bucket-level claim at those
sizes ever had support. "No major bucket regression" in the promotion rule is unenforceable
as stated for any bucket under n≈200.

## 3. v7 ladder re-adjudication: 0 of 23 distinguishable

Verdict rule (pre-registered): DISTINGUISHABLE iff |mean Δ| ≥ that pair's own MDE₈₀.
**Result: 0/23.** The full sortable table is stats/v7_reclass.json (CRN iid + block-by-event
CIs, legacy reproduction flags). The motivating cases, recomputed:

- **sym_20**: Δ = −1.654m, pair MDE 4.37m, iid CI [−4.72, +1.40], P(better) 0.145,
  block CI [−3.77, +1.02] → **INSIDE NOISE FLOOR**. The "−1.65m rejection" was a coin flip.
- **consist_16_10**: Δ = −0.392m, pair MDE 1.92m, iid CI [−1.77, +0.92], P(better) 0.270 →
  **INSIDE NOISE FLOOR**.

Secondary lens (CI excludes 0 at 95%, weaker than the MDE standard): iid — sym_6 and
surprise_12_20 only; block-by-event adds sym_8/10, surprise_16_24, both boxexp. So even the
"clearly worse" tail of the ladder (−4 to −10m) mostly fails the 80%-power standard while
some of it passes bare significance: significant-but-below-MDE means a true effect that size
would be missed ~half the time on a fresh sample. v7's real finding was directional
consistency (every alternative pointed worse), not adjudicated magnitudes.

## 4. Ledger reclassification: 5 REFUTED, 27 UNRESOLVED, 0 CONFIRMED

The published ledger contains **32 entries, not 33** as the v8 brief stated — verified in both
the generator source (`gen_final_model.py` REJECTED list) and the published HTML. Full
sortable table: stats/ledger_reclass.json (per-row: recovered Δ, source file, n at test, MDE
at test, mechanism flag, status).

**REFUTED (stay dead):** v5 asymmetric decay (mechanism: case-verified evidence-accounting
bug — note its statistical margin, −0.50m, was never there); per-map splits + pick bonus
(mechanism: double-count); tanh link (fit-collapse: train-optimal tanh IS the champion);
margin-derived CDF link (−9.0m, map-level, clears any plausible floor); post-break gap
dampener (−3.89m vs within-regime MDE 2.05m; caveat in row — under cross-σ it would be
unresolved).

**UNRESOLVED (verdicts that were coin flips): 27 of 32**, including several the ledger words
as decisive: "reversed asymmetry clearly worse" (−2.92m vs floor 6.78m), "margins matter"
(win-only −5.35m vs floor 6.78m — never actually powered at series level), margin-fitted
cross-region offsets ("−14m" — a bucket number whose bucket floor is ~20m), lineup-overlap
reweighting, carryover priors, rolling β (best variant −1.26m vs floor 1.68m; the −2.93m
rolling_365 variant alone did clear it), H2H/rematch and recency-weighted offsets
(magnitudes UNRECOVERABLE from disk — recorded as such).

**CONFIRMED (rejection overturned): 0.** Nothing in the ledger was actually better beyond
noise. The ledger's *decisions* were all defensible as "no reason to switch"; what was not
defensible is reading the entries as established negatives. **26 of 32 magnitude-based
verdicts were statements about noise.** Standing consequence: a ledger entry may bar
re-testing only with a power annotation attached; UNRESOLVED entries are re-openable by
design (new data, variance reduction, or a bigger claimed effect).

## 5. Variance reduction (effective-n multipliers)

| Design | Multiplier | Note |
|---|---|---|
| Pairing (vs unpaired) | **33× cross / 377× within** | already in use; the program's single biggest asset |
| Control variate l_v6 (CUPED-style) | 1.18× cross / 1.01× within | worth having in referee.py for cross-family tests; nil within-family |
| + multivariate (l_v6, \|rd\|, p_fav) | 1.19× / 1.01× | negligible increment over l_v6 alone |
| Block-by-event vs iid | DEFF median **0.61** (0.45–1.08) | iid CIs are *conservative* here, not understated — the pre-registered DEFF>1.3 alarm did **not** fire; caveat: only 18 events |

Pre-registered triggers that fired: cross-family MDE > 2m (→ all near-tie rejections
unresolved) and ≥⅓ of ledger UNRESOLVED (→ ledger needs power annotations). Triggers that
did not: CV ≥ 1.5× (not achieved — 1.18× is the honest number), DEFF > 1.3 (opposite
direction).

## 6. What changes for the program

1. **The v7 "no promotion" conclusion survives, restated**: not "the alternatives are worse",
   but "nothing was distinguishable and the champion keeps the seat by default".
2. **Effect-size budget**: to adjudicate a 2m cross-family effect at 80% power needs
   n ≈ 10,200 series (≈10× the holdout) — or the equivalent via variance reduction + more
   data (the prefranchise corpus is the only large lever on the table).
3. **Within-family redesigns are the affordable science**: 1.9m floor overall, ~2.5m on the
   roster-change and bo3 buckets. Structure v8 candidates as *nested perturbations of v6*
   wherever possible.
4. **Bucket gates need floors attached**: any future "bucket regression" claim must quote the
   bucket's MDE from stats/power_mde.json; buckets under n≈200 are advisory-only.
5. **Adopt the l_v6 control variate in referee.py** (free 1.18× on cross-family tests);
   keep iid bootstrap as primary (block CIs measured tighter, not wider, so iid is the
   conservative choice at 18 events).

## Answer to the phase question

**Were the near-tie rejections unresolved? Yes.** sym_20 (−1.65m) and consist_16_10
(−0.39m) — and every other v7 config — were inside the noise floor at the measured MDE;
27 of the 32 ledger rejections rest on magnitudes their tests could not resolve. The five
that survive do so on mechanism or on genuinely large margins, and nothing was found that
should have been promoted.

---

## Post-corpus addendum — MDE on the expanded holdout (2026-07-28, later same day)

agent:corpus landed +335 series (26 new registry events). Production rating files are NOT
rebuilt, so the frozen probabilities cannot be re-scored; per the post-corpus addendum in
`preregister.power.md` (written before computing), two estimators, side by side. Counts built
from the raw corpus (`match_results.csv` + `maps/<event>.csv` + `match_dates.json`) mirroring
harness scoreability rules; the pipeline validates by reproducing the frozen npz-era holdout
at exactly **1007** scoreable series, and the corpus agent's own counts exactly (2,068 total,
1,223 raw 2025+, 335 new-event series). Prefranchise verified absent (no 2021–22 dates).

**Scoreability of the additions: 335 of 335 pass.** The corpus's raw failures (10) are all
pre-existing junk in old events (7 non-franchise org parses, 3 bad org counts) — identical to
the 10 series the production frame already drops (1733 raw − 1723 framed).

**Expanded scoreable holdout: n = 1007 (frozen npz era) + 28 (organic post-npz timeline
growth) + 182 (corpus additions, 2025+) = 1217.**

Composition of the 182 additions: 169 ewc_offseason (EWC 2025 chain, CN Evolution family,
RBHG/Ten/Radiant/ACL-type invitationals), 13 "intl" — see hazard note below. Expanded holdout
mix: 804 vct_domestic / 278 ewc_offseason / 135 intl. Class σ (frozen v7 pairs, old holdout):
ewc_offseason is only ~10% noisier than vct_domestic (cross σ 0.0796 vs 0.0713), so the
EWC-heavy skew costs little:

| Regime | MDE old (n=1007) | E1 naive √n-scaled (n=1217) | E2 composition-adjusted | 
|---|---|---|---|
| within-family | 1.92m | 1.75m | **1.77m** |
| cross-family | 6.37m | 5.80m | **5.89m** |

**The checkpoint should quote the composition-adjusted numbers (pre-registered rule):
within 1.77 milli, cross 5.89 milli.** Net: the corpus buys ~8% MDE, not a regime change —
cross-family adjudication still cannot see effects under ~6 milli; the conclusions of §1–§6
stand. (E2 assumptions: class σ transfers within class; new off-season events mapped to the
measured EWC-class σ, likely a mild understatement; possible σ shrink from +2023-24 training
data not counted — direction favorable, unknown until rebuild.)

**Two hazards logged for other agents:**
1. `match_results.csv` carries one `MapNum=="all"` aggregate row per match holding the SERIES
   score — any naive per-map consumer double-counts the winner (it silently corrupted 2-0 →
   "3-0" in my first pass and only bo5s crashed loudly). The production timeline builder
   already filters it; raw-CSV consumers must too.
2. `2024_shanghai_masters`, `2025_shanghai_masters`, `2025_super_champions_cup` are off-season
   invitationals whose ids contain "masters"/"champions": harness `_is_intl_event`'s substring
   rule will class them INTERNATIONAL at rebuild (feeding the intl-attendance gate). Flag for
   the rebuild owner before the timelines are regenerated.
