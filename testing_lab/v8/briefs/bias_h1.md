# agent:bias-h1 — Phase 5 H1: bounded margins censor dominant teams

Read briefs/wave2_common.md first; it is law. Scope: does the capped Massey
target (margin pinned at 13, OT pinned at 2) compress the top of the scale
and cause the elite under-rating (T1 −7.6 prob-pts, PRX −7.3, …)?

Bias metric = referee.per_team_bias (probability points ×100; mean predicted
P(win) − actual win rate; negative = under-rated). The v6 caterpillar is the
baseline; your deliverable is the paired caterpillar.

## Experiments (preregister each: mechanism, sign, size, falsifier)
1. **Censored/Tobit likelihood.** Replace the fixed rd transform with a
   censored-margin likelihood: observed rd=13-x margins treated as ≥ latent
   (right-censored at the cap), OT (margin 2 after 12-12) treated as its own
   censoring class. Solve walk-forward (engine-compatible: either an
   iteratively-reweighted rd_custom target or your own solver in scratch —
   preregister which), β refit train-only. Judge: holdout LL, per-team bias
   before/after (did T1/PRX/100T/NRG/TL move toward 0 WITHOUT TS/JDG/TE/C9
   inflating?), max|bias|, buckets.
2. **Round-level Bradley-Terry.** Each round a Bernoulli: P(i beats j | side)
   = sigmoid(s_i − s_j + side_adv). Fit walk-forward on
   data/enriched/round_outcomes.csv. AUDIT COVERAGE FIRST: CN events have no
   round enrichment — preregister the fallback (map-level likelihood for
   uncovered matches in the same joint fit) and report exact coverage. Report:
   holdout LL, per-team bias table, and the MEASURED effective-sample gain
   (per-match Fisher information or bootstrap-SE ratio vs map-level — the
   brief claims ~an order of magnitude; measure it, don't assert it). This is
   also the program's main power play — if the SE ratio is real, say what MDE
   a round-level referee would enjoy.
3. **Censoring diagnostic** (cheap, first): among train matches, the share of
   13-x results by winner trailing-rating quartile, and realized-vs-predicted
   margin residuals at the cap. If elite teams don't actually pile up at the
   cap more than mid teams, H1's premise is weak — report that before
   spending on 1-2.

## Outputs (yours alone)
- stats/h1_censor_diag.json, h1_tobit.json, h1_roundbt.json,
  h1_bias_caterpillar.json (v6 vs each candidate, chart-ready)
- phase5_h1.md (verdict per mechanism, both units, MDE context)
- preregister.bias_h1.md, logs/bias_h1.log, scratch/bias_h1/

Done when: diagnostic answered; both likelihoods measured on holdout with
bias tables; the effective-sample multiplier measured; verdicts stated
plainly (including "premise weak" if the diagnostic kills it).
