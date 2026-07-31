# Pre-registration — agent:referee (Phase 6 metric suite)

Written 2026-07-28, BEFORE referee.py is coded. Everything below is frozen:
formulas, acceptance bars, and promotion-gate clauses. Any later deviation gets
written down as a deviation.

## 0. Conventions

- All probabilities are **winner-referenced** unless stated: `p[i]` = the
  probability the model assigned to the eventual winner of series i, so
  per-series loss is `-log(p[i])`. Matches harness/engine/v7 npz convention.
- Paired deltas follow the harness sign convention: `d = loss_B - loss_A`,
  positive = A better.
- "Holdout" = series date > 2024-12-31 (BETA_TRAIN_END), valid = non-NaN
  rating diff.
- House reliability-bin shape everywhere:
  `{bin_lo, bin_hi, pred_mean, emp, n, ci_lo, ci_hi}` with 95% Wilson CI
  (z=1.96), identical math to harness.reliability.

## 1. Canonical baseline (reconciliation, pre-committed)

- **CANONICAL: v6 holdout LL 0.64095, n_test=1007** — artifact
  `out/v7_stage1.json` → `results.v6_consist_20_12.ll_test`, per-series probs
  in `out/v7_probs.npz` (key `v6_consist_20_12`, mask `~isnan(rd__…) & test_v`).
  Jul-23 native data frame (1695 valid), β=0.1294 refit on ≤2024.
- 0.6409 (final_model report) = same champion on the **Jul-22 frame**
  (`out/rd_v6_native.npy`, β=0.12556 from `out/v6_native_beta.json`): holdout
  n=999, pooled LL 0.64085 → displayed "0.6409". This frame is what
  `out/v6_profile.json` (bias table, 23 buckets, 4 bands) was computed on.
- 0.64126 (favorites report) = prose-hardcoded number from the mid-session
  v6-correction build (pre-final frame). No artifact; superseded.
- Frame-prefix condition: both historical frames must be exact prefixes of
  today's `harness.load_series()` ordering (sort: date, match_id). This is
  *verified, not assumed*, by the self-test (bias table 42/42 + 23 buckets +
  bands + cold-start all exact). If a future data rebuild edits pre-2026-07-22
  rows, the self-test fails loudly — by design.

## 2. Metric formulas (exact)

### 2.1 per_series_ll
`per_series_ll(p) = -log(clip(p, 1e-9, 1))`, vector. Aggregate = mean.
`delta_vector(p_a, p_b) = per_series_ll(p_b) - per_series_ll(p_a)` (>0 ⇒ A
better).

### 2.2 paired_bootstrap_crn(d, mode)
- Reads `testing_lab/v8/crn.json` **at call time**; absent ⇒ raise
  (RuntimeError). Never a private seed. Schema per briefs/power.md §6.
- `mode='iid'`: rng = `np.random.Generator(PCG64(crn.bootstrap.seed))`,
  `n_boot = crn.bootstrap.n_boot`; resample b: indices
  `rng.integers(0, n, n)` drawn n_boot times in sequence;
  stat_b = mean(d[idx_b]).
- `mode='block_event'`: requires aligned `event_ids`; unit must equal
  `crn.block_bootstrap.unit == "event_id"`; rng seeded from
  `crn.block_bootstrap.seed`; each resample draws `n_events` events with
  replacement (`rng.integers(0, n_events, n_events)` over the sorted unique
  event list) and concatenates their member rows; stat_b = mean over
  concatenated rows.
- Returns `{mean_delta, ci_lo, ci_hi (percentile 2.5/97.5 of resampled means),
  p_better = mean(stat_b > 0), n, n_boot, seed, generator, mode,
  crn_file, crn_verify}` — full provenance in every result.
- When `len(d) == len(crn.holdout_order)` and the verify block exists, the
  first-100 iid index draws are hashed (sha256 of the concatenated int64
  bytes) and compared; mismatch is reported in `crn_verify` (and treated as a
  blocker in any result that carries it).

### 2.3 bucketed(frame, p, p_ref=None)
Bucket definitions frozen to the recovered v6_profile/final_check set
(evaluated on holdout∧valid rows unless stated; bucket emitted when n ≥ 15):
- year ∈ {2025, 2026}; format ∈ {bo1, bo3, bo5, bo5_gf}; stage ∈ {groups,
  playoffs, grand_final}.
- international = `_is_intl_event(event_id)` (harness: masters/champions/
  lock_in substring); domestic = ¬international.
- CN involved = reg_w=CN ∨ reg_l=CN; cross-region = reg_w≠reg_l;
  domestic {Americas, EMEA, Pacific, CN} = same-region pairs.
- gap bands on |rdiff|: `<1.5`, `[1.5,4)`, `[4,7)`, `≥7`.
- favorite bands on max(p,1−p): `[0.5,0.6) [0.6,0.7) [0.7,0.8) [0.8,0.9)
  [0.9,1]`, excluding exact ties (|p−0.5| < 1e-9).
- EWC-class = event_id startswith ("2026_ewc", "2026_china_evo").
- post-break = days since each team's previous series in the dataset; BOTH
  teams must have a prior series (debuts excluded); bucket = either rest > 45
  (strict).
- cold-start = |r_w| < 5e-4 ∨ |r_l| < 5e-4 (rating exactly 0 ⇒ unrated);
  evaluated over the full frame (deep1 convention), not holdout-only.
- elite vs floor = decayed win-rate machinery from run_v7_stage1 (per-game
  exp decay, HL = 16 team-games, denominator > 3, as-of series date):
  max(wr16_w, wr16_l) ≥ 0.60 ∧ min ≤ 0.40. form-shift = |wr5 − wr16| ≥ 0.15
  for either team (same machinery, HL=5).
- roster-recency hook: when `v8/data/lineup_features.csv` exists, join on
  match_id and bucket by days-since-lineup-change: `≤14d, (14,45], >45d,
  unknown`. Column autodetected from a declared candidate list; absent file ⇒
  buckets reported as PENDING DATA, never silently approximated.
Output per bucket: `{name, n, ll, fav_acc}` (+ `ll_ref, delta_milli` when
p_ref given). delta_milli = (ll_ref − ll)×1000, positive = p better.

### 2.4 per_team_bias(p, winners, losers, min_n=25)
Recovered definition (gen_final_model.py L298-300, verified 42/42 exact):
for each team T over holdout∧valid series it played,
`p_T[i] = p[i]` if T won series i else `1 − p[i]`; `won_T[i] ∈ {1, 0}`;
**`bias(T) = mean(p_T) − mean(won_T)`** — probability points (rendered ×100
as "pts"; negative = model under-rates T). Team LL = `-mean(log(p on T's
realized outcomes))` = plain series LL restricted to T's matches. Teams with
n ≥ 25 only. Returns full sorted table + `max_abs_bias`, `mean_abs_bias`,
`n_teams`. This is a PROMOTION-GATE input.

### 2.5 pnl_weighted_ll(p, won, price)
- price in cents (values ≤ 1 auto-scaled ×100); per-row loss
  `-log(p if won else 1−p)`.
- Weight of row = density mass of the price band containing price, from
  `v8/stats/quote_density.json` (accepted shapes: `{bands:[{lo_cents,
  hi_cents, density}]}`, `{"lo-hi": density}` dict, or a bare list of band
  objects). Weights normalized to mean 1 over scored rows.
- **FALLBACK (until agent:autopsy delivers the file): uniform density over
  the 20–55¢ quoted band, zero outside** — every output carries
  `weights_source: "FALLBACK_uniform_20_55c"` vs `"quote_density.json"`.
- Returns `{ll_weighted, ll_unweighted, n, n_weighted_nonzero,
  weights_source, band_table}`.

### 2.6 expected-ROI translation `expected_roi_of_dll(dll, p_ref)`
Uses `out/quote_margin.json` (read-only) as the ROI ladder:
δ→ROI points = `(0.0, flat_0c.roi)` plus `(δ, logit_δ.roi)` for
δ ∈ {0.1,…,0.6}; linear interpolation, CIs interpolated alongside.
ΔLL→δ equivalence (first-order, pre-committed): a correctly-signed logit
shift δ on the winner side changes per-series log score by ≈ δ·(1−p), so
**δ_equiv = dll / mean(1 − p_ref)** with p_ref = the reference model's
winner-side holdout probs. Headline translation = ROI(δ_op + δ_equiv) −
ROI(δ_op) at operating point δ_op (default 0). Every output echoes the
formula name, δ_equiv, ladder points used, and the ladder CIs. This is a
*translation for quoting, not a claim of realized P&L*.

### 2.7 fill_conditional_calibration(fills_df, …)
Interface agreed to autopsy's brief (step 4): input a DataFrame of FILLED
markets with model prob (`p_col`, default "p_model", prob of the side the
fill references) and settlement (`won_col`, default "won"), plus optional
unconditional arrays `(p_uncond, won_uncond)` for the same window; optional
slice columns (`side_role`, price band, minutes-to-start band) sliced when
present. Output: house-shape reliability bins over [0,1] (default 10 bins)
for filled and unconditional, per-slice bins, and a summary with
`{n, ll, brier, mean_p, emp, gap = emp − mean_p}` each side plus
`adverse_selection = gap_filled − gap_uncond`. Wilson CIs throughout.

### 2.8 margin_mse(pred_margin, realized_margin)
`mse = mean((pred − realized)²)`, `corr` = Pearson, n. House constructions
provided: realized series avg transformed margin/map =
`mean_maps(sign(wr−lr)·|wr−lr|^0.75·2.5)` winner-referenced
(run_margin2.py L51); predicted = `a_slope · rdiff` with a_slope from
`out/margin_link.json` (0.59290, train-fit).

### 2.9 reliability emitters
`reliability_emit(p, y, n_bins, lo, hi)` — harness Wilson math, house JSON
shape, chart-ready. `favorite_reliability(p_win)` = deep1 semantics
(favorite-side, tie-free, bins [0.5,1.0]).

## 3. Promotion gate (code-to-be, clauses frozen)

`promotion_gate(candidate, v6, mde, …)` evaluates, each clause returning
pass/fail + numbers:

- **G1 support**: paired ΔLL (v6→candidate) with `paired_bootstrap_crn`:
  `mean_delta ≥ mde` (mde supplied from Phase-0 power output — never
  invented here) AND `p_better ≥ 0.95` in **iid** mode AND `p_better ≥ 0.95`
  in **block_event** mode.
- **G2 team bias**: `max_abs_bias(candidate) < max_abs_bias(v6)` on the same
  holdout, min_n=25, strict.
- **G3 no major bucket regression** ("major" pre-committed numerically):
  over the frozen bucket set (§2.3, buckets with n ≥ 30, excluding
  roster-recency while PENDING DATA), a major regression =
  `delta_milli(candidate vs v6) ≤ −4.0` for buckets with n ≥ 100, or
  `≤ −8.0` for buckets with 30 ≤ n < 100. G3 passes iff zero major
  regressions. (−4 milli mirrors the historical run_final_check "worse"
  flag; the looser small-n bar acknowledges bucket noise, cf. Phase-0 MDEs.)
- **Verdict**: PROMOTE only if G1 ∧ G2 ∧ G3. Output object carries every
  clause's inputs, numbers, pass/fail, and the CRN provenance.

## 4. Self-test acceptance bars (frozen)

Targets are published artifact values; "exact" = equality after rounding to
the artifact's printed precision; LL bars additionally satisfy |got−target|
≤ 1e-4. ALL must pass or the summary reports the failure as a blocker.

| # | Reproduction | Target | Bar |
|---|---|---|---|
| 1 | npz v6_consist_20_12 holdout LL, n | 0.64095, 1007 | ≤1e-5, n exact |
| 2 | npz v5_asym / consist_16_10 / sym_16 | 0.64145 / 0.64135 / 0.64262 | ≤1e-5 |
| 3 | native year buckets 2025 / 2026 | 0.63036 n=498 / 0.65127 n=501 | ≤1e-5, n exact |
| 4 | native pooled holdout | (498·0.63036+501·0.65127)/999 | ≤1e-4 |
| 5 | per-team bias table | 42/42 rows of v6_profile.teams | bias ≤5e-5, ll ≤5e-5, n exact |
| 6 | max abs bias | 0.1634 (TS) | ≤5e-5 |
| 7 | all 23 v6_profile buckets | each (n, v6 LL) | LL ≤1e-5, n exact |
| 8 | 4 favorite bands | n 466/335/152/43 + pred/emp | n exact, pred/emp ≤5e-4 |
| 9 | cold-start (deep1) | n=57, LL 0.70205 | n exact, ≤1e-5 |
| 10 | reliability emitter vs deep1.reliability_fav_warm | all bins | every field to artifact rounding |
| 11 | margin_mse vs margin2.marginMSE_v5 | 43.095 / 0.148 / 999 | mse ≤1e-3, corr ≤1e-3, n exact |
| 12 | CRN: absent file raises | RuntimeError | must raise |
| 13 | CRN (if crn.json exists): determinism + sha verify | identical repeat call; verify ok | exact |
| 14 | pnl_weighted_ll fallback labeling + toy identity | FALLBACK label; uniform-band = plain mean | exact |
| 15 | expected-ROI ladder endpoints | ROI(0)=0.0021, ROI(0.6)=0.2911 | ≤1e-4 |
| 16 | promotion_gate logic on synthetic vectors | fail-on-self, pass-on-dominating | logic |

Items whose upstream dependency is absent (crn.json present-path, real
quote_density.json, lineup_features.csv) are reported PENDING/SKIPPED with
the reason — skips are named, never silent.

## 5. Non-goals / forbidden

No rating solves; no modification of harness.py, gen_report.py, out/*;
no writes outside referee-owned paths; no network; no private randomness.

## 6. Deviations (recorded after the runs, per the header rule)

**D1 (2026-07-28) — item 10 split into 10a/10b.** Item 10 assumed
`out/deep1.json`'s reliability bins were reproducible from current data
files. They are not: deep1 ran 2026-07-22 01:43 on a data state that
pre-dates the last data commit (dd0d81f, Jul 17 — everything since lives
uncommitted in the working tree) and has since been mutated in place by
production scrapes (some 2025–26 pre-match ratings re-solved). The state
exists nowhere on disk or in git. Resolution: 10a (acceptance, referee-owned)
= emitter identical to harness.reliability on identical input — PASS exact;
10b (reported as an upstream_data_drift finding, not a referee defect) =
deep1-bin comparison with the drift quantified (warm n 1521 and the 57-row
cold set reproduce exactly; 6/10 bins differ, max |Δn| = 3, max |Δemp| =
0.016). Nothing about the metric formulas changed.

**D2 (2026-07-28) — games source for wr/margin machinery.** The plan
implicitly assumed `engine.Engine().games` as the game source. Discovered:
today's `BuildRatingTimeline.load_all_games()` is missing every EWC-class
map (the site registry moved those events to per-region files after Jul 23,
uncommitted), which would have silently broken elite-floor masks and margin
reproduction. Resolution: `referee.load_timeline_games()` reads maps
directly from data/rating_timeline*.json (the artifact the series frame
itself comes from). Verified exact: npz elite_floor/form_shift masks
reproduce, marginMSE_v5 43.095/0.148/n=999 reproduces. Formulas unchanged;
only the data plumbing differs, and the finding is flagged for other agents
in referee_selftest.json.
