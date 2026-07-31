# BenPom v6 — how the model works

The complete explanation of the model the bot quotes real money on. Written
2026-07-30, after v6 survived three adversarial research programs and was
deployed to the public site (site and bot now share one solver). The code
that implements everything below: `predict.py` (pricing — the reference),
`build_model_snapshot.py` (packaging), and the site's
`scrapers/BuildRatingTimeline.py` (the solve itself).

---

## 1. The one-paragraph version

Every team gets one number — an opponent-adjusted round-differential rating —
solved daily from every VCT map since 2023 by a weighted least-squares
(Massey) system. A game's weight decays by how many games the team has played
since (not by calendar time), results that fit a team's established level
fade slower than anomalies, and prestige games count more. A match probability
is a logistic function of the rating gap (plus a fitted cross-region
adjustment), converted to a series price by exact Bo3/Bo5 math. That's it —
no momentum, no map-by-map stacking, no market inputs.

## 2. The rating solve

For each solve day D, take every map played before D and solve the Massey
system: find ratings r such that for every game, r_winner − r_loser ≈ the
game's margin target, in the weighted least-squares sense.

- **Margin target:** `sign(rd) · |rd|^0.75 · 2.5` where rd = round diff
  (13-7 → +6). The 0.75 power keeps blowouts informative without letting a
  13-1 count double a 13-7. Margins are used HERE and only here — win
  probability is calibrated on binary outcomes (margins understate win prob:
  better teams win the close maps; every OT margin is 2).
- **Games-counted decay, consistency-conditioned** (the v6 signature): a
  game's age is how many games THAT TEAM has played since it — a 6-week break
  costs no information. Results **consistent** with the team's level at the
  time (their decayed map winrate) fade with a **20-game half-life**;
  **anomalies** fade at **12**. This is deliberately symmetric in win/loss:
  a floor team's routine losses persist (its signal), an elite team's
  routine wins persist, and one-off upsets in either direction fade fast.
  (The naive version — wins slow, losses fast — inflated bad teams and was
  operator-caught; see §6.)
- **Roster continuity:** at each calendar-year boundary, a team's prior-year
  games are down-weighted by how much of its five carried over (overlap/5,
  floor 0.3-scaled). Mid-season changes deliberately do NOT re-weight the
  solve — every finer scheme tested made predictions worse (§6); mid-season
  changes are handled by sizing, not fair value (ROSTER_FLAG.md).
- **Prestige weights:** playoff/grand-final maps ×1.6, Champions maps ×2
  (exact event-id match — off-season events with "champions" in the name
  don't qualify).
- **Regularization:** ridge 0.5 to keep the system solvable, plus a
  region-prior ridge 1.5 pulling each team toward its region's trailing
  mean — thin-schedule teams borrow strength from their region.
- **Corpus:** all franchised VCT 2023→ plus EWC-class and vetted off-season
  events (they feed ratings but are hidden from site player-facing UIs).

## 3. From ratings to prices (what `predict.py` does)

```
p_map    = sigmoid( β · (r_a − r_b + adj) )
adj      = xregion_offsets[region_a] − xregion_offsets[region_b]   (0 if same region)
bo1      = p_map
bo3      = p²(3−2p)
bo5      = p³(1+3q+6q²),  q = 1−p
bo5 GF   = series prob shifted +gf_upper_logit (0.25) toward the upper-bracket team
map pick = z ± b_pick before the sigmoid (map-level quotes only)
```

- **β** is refit at every site build on all completed series to date
  (currently ≈0.128). It is scale-bound to this exact solve config — never
  reuse it with other ratings, never hardcode it.
- **xregion_offsets** are refit each build on all cross-region series, CN
  pinned at 0. They REPLACE the old intl-experience/CN-dog bonuses entirely.
- **Unknown org:** price at its region's cold-start prior (25th percentile of
  the region's ratings). Region unknown too → don't quote.
- **Series price = the closed form.** The veto Monte Carlo adds no
  series-level accuracy (tested; ensembles all lost). Map-level quotes use
  overall rating + b_pick — never per-map rating splits stacked with the
  pick bonus (double count, measured).

## 4. What the numbers mean

Ratings are in transformed-round-margin units; a gap of ~5.4 at β=0.128 ≈
a 2:1 map favorite (p_map=0.667 → bo3 ≈ 0.74). The top of the scale sits
around +6, the floor around −7.

## 5. Validation (why we trust it)

- Walk-forward holdout 2025-26 (n≈1007→1217 series): **LL 0.641 vs 0.653**
  for the previous production model (+11.4 milli, p>0.998).
- Kalshi pre-match market overlap: at/above parity (0.6441 vs 0.6457).
- Then three programs tried to beat it and failed:
  **v7** — 18 decay shapes (recency/symmetry): 4 measurably worse, rest
  indistinguishable. **v8** — full mechanism sweep with preregistration, CRN,
  an independent adversarial reviewer, and a measured noise floor (1.77
  milli within-family): margin censoring, schedule connectivity, series
  dispersion all died at their diagnostics; 0/3 composite stacks promoted.
  **v9** — 3,240-config roster-subsystem search under era-transfer
  validation: zero candidates advanced ("the ladder is v6 alone").
- The **2025-26 holdout is spent** (400+ recorded looks). Anything new is
  adjudicated prospectively: frozen candidates are re-scored on post-07/28
  series with pre-committed rules (site: /testing/report/v9_lab scoreboard).

## 6. Known limits and the do-not-retest ledger (short form)

- **Per-team bias watchlist** (PROBABILITY points = mean predicted P(win) −
  actual win rate, ×100 — not rating points): elites run a few points
  under-priced (PRX ≈ −10, NRG ≈ −9 on the expanded frame), some floor/mid
  teams over-priced (TS ≈ +15). Four mechanisms tested to explain it; all
  died. Band calibration hides it — case-check operator anomaly reports.
- **Roster changes are invisible to fair value** (year boundaries aside) —
  measured cost: LEV under-priced ~15pp/match after adding Neon, ENVY
  over-priced ~24pp after its swaps. EVERY retrospective fix tested (weight
  boosts a=0.5→28, prediction-layer nudges, phase classifiers per the
  operator spec) made overall accuracy worse — early post-change results
  are too noisy to lean on harder than the solve already does. The
  sanctioned response is sizing: quote smaller on fresh low-overlap rosters
  (ROSTER_FLAG.md), and let the prospective scoreboard adjudicate any
  future fair-value treatment.
- **Do not re-try without new data:** shorter/calendar decay, win/loss
  asymmetric decay, lineup re-weighting, player-carryover priors, post-break
  dampeners, tail-shape link mods, margin-as-outcome fits, per-map+pick
  stacking, rolling β refits, learned event weights, patch-boundary fades,
  series dispersion links. Full ledger with kill evidence:
  /testing/report/{final_model,v8_lab,v9_lab}.

## 7. Operating rules for the bot (unchanged since FINDINGS.md, still binding)

1. Fair value = `predict.py` closed form on the snapshot. Parity-test any
   reimplementation against it.
2. Read β/offsets/priors from the snapshot every time; nothing hardcoded.
3. Quote-sizing edge in logit space (+0.5–0.6), not flat cents; skip NO
   quotes on sides the model prices below ~45%.
4. Quarter-size: LANs, first week after 45+ day breaks, 15+pt divergences
   where the market is MORE confident, and fresh low-overlap rosters
   (roster_flags.json).
5. The fair value stays independent of the live Kalshi book (echo-chamber
   guard). Market data is benchmark/diagnostic only.
6. Refresh chain: site scrape → BuildRatingTimeline → build_model_snapshot
   → bot hot-reload. The snapshot embeds its provenance; apply staleness
   rules against `generated_utc`.
