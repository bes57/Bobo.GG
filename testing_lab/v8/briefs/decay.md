# agent:decay — Phase 4: recency & asymmetry, tested properly (5a/5b/5c/5e)

Read briefs/wave2_common.md first; it is law. Scope: the operator's two live
hypotheses — more recency, outcome-symmetric decay — re-tested on the expanded
frame with CRN + the control variate. 5d (heterogeneous half-lives) is NOT
yours — it lives with agent:bias-h3 (shared process-noise machinery).

Phase 0 context you must respect: within-family MDE 1.77m at n=1217. Do not
"conclude v6 wins" on sub-MDE deltas; INSIDE NOISE FLOOR is a published
verdict. v7's per-config probs (out/v7_probs*.npz) are the OLD frame — use
them only for continuity checks, never mixed into expanded-frame comparisons.

## Experiments (preregister each)
1. **5a Re-race the near-ties.** v6 consist(20,12) vs consist_16_10 vs sym_20
   vs sym_24, all on the expanded frame, matched pairs, crn bootstrap (iid +
   block), v6-loss control variate applied (report raw and CV-adjusted CIs).
   Publish the new MDE in the table header. If still ties: "ties" is the
   answer, in those words.
2. **5b New conditioning axes** — each an engine decay variant, symmetric in
   win/loss unless stated, preregistered sign/size/falsifier:
   a. LINEUP CONTINUITY: age counted in games; a game's weight additionally
      decays with lineup distance between the lineup that played it and the
      team's current five (lineup_features/lineups tables; walk-forward).
      Outcome-symmetric — the operator's requested axis. Grid the sharpness
      on train only.
   b. OPPONENT QUALITY OF THE ANOMALY: anomaly (vs trailing level, as v6
      defines it) fades slower when the opponent was elite (trailing rating
      top quartile at the time), faster vs floor opponents.
   c. ANOMALY MARGIN: |rd| of the anomalous result scales its persistence.
   d. EVENT CLASS OF THE RESULT: ewc_offseason results fade faster (games-
      counted); vct/intl normal. (Coordinates with agent:context 3a — yours
      is DECAY-side, theirs is solve-WEIGHT-side; keep them distinct.)
   e. PATCH / MAP-POOL BOUNDARY: derive rotation dates from map appearance
      windows in the games list (a map's first/last game date per pool era —
      preregister the derivation); results predating a rotation fade faster.
   For each: holdout LL vs v6, pair-MDE quoted, and whether it is
   outcome-symmetric. If any symmetric axis ≥ v6's consistency conditioning,
   the operator's objection to the asymmetry is vindicated — say so.
3. **5c Performance-based form.** Redefine form from round differential
   (per-map rd from the games list) and side-conditional performance
   (data/enriched/round_outcomes.csv where covered — audit CN gaps first,
   report coverage), optionally per-player rating trajectories
   (player_map_advanced.csv). Refit the b_form term with these definitions on
   train; score holdout. Outcome-based v7 b_form ≈ -0.087 (old frame) is the
   reference. Coordinate note: agent:context tests exposure-controlled b_form;
   you test performance-defined form. Both preregistered, non-overlapping
   outputs.
4. **5e Subpopulation panel** for EVERY test above: post-roster-change (≤3
   matches, lineup_features), post-patch (your 5b-e boundaries), first-match-
   after-45d-break, within-event day 2+, and the 20-55¢ quoted band
   (referee.pnl machinery for band membership from v6 predicted p). Use
   referee.bucketed; report per-bucket Δ with bucket MDEs. No aggregate-only
   verdicts.

## Outputs (yours alone)
- stats/decay_rerace.json (5a table), decay_axes.json (5b, one row per axis),
  decay_form.json (5c), decay_subpops.json (5e panel), decay_curves.json
  (w(g) overlay data: v6, sym20, lineup-conditioned, best new axis)
- phase4_decay.md (prose verdicts, both units, MDE-context on every number)
- preregister.decay.md, logs/decay.log, scratch/decay/

Done when: 5a re-raced with CV-adjusted CIs; all five 5b axes measured;
5c form refit with performance definitions; every result carries its
subpopulation panel; the symmetric-vs-asymmetric question answered plainly.
