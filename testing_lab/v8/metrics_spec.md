# BenPom v8 metric suite — specification (testing_lab/v8/referee.py)

agent:referee, Phase 6. Pre-registered in `preregister.referee.md` (formulas
+ acceptance bars frozen before coding; deviations §6 there). Self-test proof:
`stats/referee_selftest.json` — every number below reproduced from artifacts
on disk, no rating solves. Import as:

```python
import sys; sys.path.insert(0, "testing_lab"); sys.path.insert(0, "testing_lab/v8")
import referee
```

## Conventions

- **Winner-referenced probabilities**: `p[i]` = prob the model gave the
  eventual winner of series i; per-series loss `-log p[i]`.
- **Paired delta**: `d = loss_ref − loss_cand`, positive = candidate better.
- **Holdout**: date > 2024-12-31; valid = non-NaN rating diff.
- **House reliability shape** (chart-ready, used by the Lab pages):
  `{bin_lo, bin_hi, pred_mean, emp, n, ci_lo, ci_hi}`, 95% Wilson.

## Canonical baseline (reconciled)

**v6 holdout LL 0.64095, n_test = 1007** — `out/v7_stage1.json`
(`results.v6_consist_20_12`), probs on disk in `out/v7_probs.npz`. The
"0.6409" in the Final Model report is the same champion on the **Jul-22
frame** (`out/rd_v6_native.npy` + β=0.12556, n=999, pooled 0.64085) — the
frame `out/v6_profile.json` (bias table / 23 buckets / bands) was built on.
"0.64126" (favorites report) is prose from the mid-session v6-correction
build; no artifact. Canonical for every v8 comparison: **0.64095 / n=1007 /
the npz frame** (crn.json holdout_order is aligned to it — verified).

## Functions

### `per_series_ll(p)` / `logloss(p)` / `brier(p)` / `delta_vector(p_cand, p_ref)`
Loss vectors and aligned paired deltas. `delta_vector` raises on unaligned
shapes.

### `paired_bootstrap_crn(d, mode='iid'|'block_event', event_ids=None, crn_path=...)`
Reads `v8/crn.json` **at call time**; **raises RuntimeError if absent** (no
private seeds, standing rule 3). iid: PCG64(bootstrap.seed), full
`(n_boot, n)` index matrix in one draw (power's recipe); block_event:
PCG64(block_bootstrap.seed), events resampled with replacement in
first-appearance order, member rows kept in order. Returns
`{mean_delta, ci_lo, ci_hi, p_better, n, n_boot, seed, generator, mode,
crn_file, crn_verify}` — when the vector is the full 1007-row holdout the
first-100-draw sha256 is checked against crn.verify (self-test: "ok" both
modes). `paired_bootstrap_legacy(p_a, p_b)` = harness seed-7 bootstrap,
ONLY for reproducing published v6/v7 numbers (reproduces the stage-1 boot
JSON bit-for-bit).

### `bucketed(frame, p, p_ref=None, rdiff=None, holdout=None, valid=None, elite_floor=None, form_shift=None, games=None, lineup_path=..., min_n=15)`
Returns `{"buckets": [{name, n, ll, fav_acc[, ll_ref, delta_milli]}],
"pending": [...]}`. Frozen definitions (all reproduce v6_profile exactly —
23/23):
year; format bo1/bo3/bo5/bo5_gf; stage groups/playoffs/grand_final;
international (masters/champions/lock_in substring) / domestic; CN involved;
cross-region; domestic Americas/EMEA/Pacific/CN; rating-gap bands |rd| <1.5 /
[1.5,4) / [4,7) / ≥7; **post-break** = either team's days-since-last-series
> 45, debuts excluded (both teams need a prior series); **elite vs floor** =
decayed map-winrate (HL=16 team-games, denom>3, as-of date) max ≥ 0.60 ∧ min
≤ 0.40; form shift = |wr5 − wr16| ≥ 0.15 either team; **EWC-class** =
event_id startswith 2026_ewc / 2026_china_evo; cold-start = |pre-match r| <
5e-4 either side; favorite bands 0.5–1.0 by 0.1, exact ties excluded;
**roster-recency hook** = joins `v8/data/lineup_features.csv` on match_id
(≤14d / (14,45] / >45d / unknown) and reports PENDING DATA when absent —
never silently approximated. `delta_milli > 0` = p better than p_ref.

### `per_team_bias(p, winners, losers, holdout=None, valid=None, min_n=25)`
**The recovered v6_profile bias** (gen_final_model.py L298-300, verified
42/42 rows exact): per team T over holdout series it played,
`bias(T) = mean(predicted P(T wins)) − mean(T won)` — **probability points**
(reported ×100 as "pts"; NOT rating points), negative = under-rated. Team
ll = series LL restricted to T's matches; teams with ≥25 holdout series.
Returns sorted table + `max_abs_bias` (v6 baseline: **0.1634**, TS) +
`mean_abs_bias`. **Promotion-gate input.**

### `pnl_weighted_ll(p, won, price, density_path=...)`
Loss weighted by where P&L lives: weight = quote-density mass of the row's
price band (mass spread uniformly within band; prices in cents, ≤1
auto-scaled). Density from `v8/stats/quote_density.json` (autopsy's file:
NO-side cents, 5¢ bands; mass field auto-selected dollars→contracts→n,
sides summed). **Fallback when absent: uniform over 20–55¢, zero outside,
labeled `weights_source: "FALLBACK_uniform_20_55c"`** — the live file is
labeled by its path. Output carries ll_weighted, ll_unweighted, source,
convention, band table.

### `expected_roi_of_dll(dll, p_ref, delta_op=0.0)`
Quotes a ΔLL in ROI units via the quote-margin maker sim
(`out/quote_margin.json`): ladder (δ_logit → sim ROI) anchored at flat_0c
(δ=0, roi 0.0021) through logit_0.6 (roi 0.2911); first-order equivalence
**δ_equiv = dll / mean(1 − p_ref)** (a correctly-signed winner-side logit
shift δ changes per-series log score by ≈ δ·(1−p)); reported ROI =
interp(δ_op + δ_equiv) − interp(δ_op) with the sim's CIs interpolated
alongside. A quoting translation under sim assumptions — not realized P&L.

### `fill_conditional_calibration(fills_df, p_uncond=None, won_uncond=None, p_col="p_model", won_col="won", slice_cols=(...), n_bins=10)`
The autopsy interface (its brief step 4): house-shape reliability on filled
markets vs the unconditional window, `gap = emp − mean_p` each side,
`adverse_selection = gap_filled − gap_uncond`; slices by side_role /
price_band / mins_to_start_band when the columns exist (≥5 rows).

### `margin_mse(pred, realized)` + `realized_avg_margin(frame, games)` + `margin_slope()`
The adopted secondary metric: realized = per-map
`sign(wr−lr)·|wr−lr|^0.75·2.5`, winner-referenced, averaged over the series'
maps; predicted = `a_slope·rdiff` (a_slope 0.59290, `out/margin_link.json`).
Reproduces marginMSE_v5 = 43.095 / corr 0.148 / n 999 exactly.

### `reliability_emit(p, y, n_bins, lo, hi)` / `favorite_reliability(p_win)` / `calib_slope(p_win)`
House-shape emitters; `favorite_reliability` = deep1 semantics (favorite
side, exact ties dropped, bins [0.5,1.0]); proven identical to
harness.reliability on identical input. `calib_slope` re-exports harness.

### `promotion_gate(candidate_result, v6_result, mde, frame=..., rdiff_ref=..., ...)`
The v8 bar; `mde` comes from Phase-0 power output (never invented here).
- **G1**: CRN paired bootstrap — `mean ΔLL ≥ mde` AND `P(better) ≥ 0.95` in
  **both** iid and block_event modes.
- **G2**: `max|team bias|` strictly reduced (min_n=25).
- **G3**: no major bucket regression, pre-committed: candidate worse by
  > 4 milli-LL in any bucket with n ≥ 100, or > 8 milli with 30 ≤ n < 100
  (buckets under n=30 noise-exempt; PENDING buckets listed, not gated).
- Verdict `PROMOTE` iff G1∧G2∧G3; the object carries every clause's numbers,
  both bias tables, the full bucket table, CRN provenance, and the
  expected-ROI translation of the mean ΔLL.

### Loaders (shared, artifact-true)
- `load_npz_frame()` — the canonical 1695-row frame + all 18 stage-1
  configs' probs/rdiffs/masks from `out/v7_probs.npz`.
- `load_native_v6()` — the Jul-22 profile frame (1687 rows) + closed-form v6
  probs from `rd_v6_native.npy` (no solving).
- `load_timeline_games()` — map-level games straight from
  data/rating_timeline*.json. **Use this, not Engine().games**: today's
  registry is missing every EWC-class map (per-region file restructure,
  uncommitted) — verified to reproduce the npz elite-floor/form-shift masks
  and marginMSE exactly where Engine().games no longer does.
- `load_series_full()` — cached harness.load_series().

## Self-test (stats/referee_selftest.json)

24 referee-owned items, **ALL PASS**: canonical LL 0.64095/n=1007 + three
sibling configs exact; crn holdout_order alignment; native years/pooled;
bias table 42/42 + max|bias| 0.1634; 23/23 profile buckets; stage-1
elite-floor/form-shift LLs; 4/4 bands; cold-start n=57/0.70205 on the
reconstructed 1578-row deep1 frame; emitter equivalence; margin MSE;
CRN raise-if-absent, sha verify + determinism (iid & block), legacy boot
bit-reproduction; density source labels + fallback identity; ROI ladder
endpoints; gate logic (dominator PROMOTEs, self-copy HOLDs).
One **upstream_data_drift finding** (not a referee defect, preregister D1):
deep1's exact Jul-22 input state no longer exists on disk/git — 6/10
reliability bins differ (max |Δn| = 3) after uncommitted timeline mutations;
plus the Engine games gap (D2) flagged for other agents.
