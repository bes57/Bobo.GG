# roster_flags.json — the sizing signal for roster changes

The model's fair value deliberately ignores mid-season roster changes (every
retrospective fix tested made accuracy worse — MODEL_EXPLAINED.md §6). The
measured misses are real though (LEV ~15pp under after adding Neon; ENVY
~24pp over after its swaps), so the sanctioned response is SIZING, not price:
quote smaller until a new lineup has a track record.

`build_model_snapshot.py` emits `roster_flags.json` alongside the snapshot
when the research data is present. Per team:

| field | meaning |
|---|---|
| `modal_five` | the established lineup (most frequent five, trailing window, walk-forward) |
| `last_match_deviated` | last fielded five ≠ modal five (could be a sub OR the start of a change) |
| `matches_since_boundary` | matches since the last CONFIRMED sustained change |
| `overlap_k5` | players kept at that boundary (k of 5) — smaller = bigger rebuild |
| `provisional` | a deviation happened but is not yet confirmed as a change or retracted as a sub |

Suggested policy (operator-tunable; none of this changes fair value):
- `provisional == true` → half-size: at quote time a sub and a change are
  indistinguishable; the ambiguity itself is the risk.
- confirmed boundary with `matches_since_boundary < 5` → quarter-size,
  scaled by rebuild size (k≤2 harsher than k=4).
- otherwise → normal sizing.

A sub that reverts (SEN fielding a stand-in for one match) never becomes a
boundary — the classifier is causal (no future peeking) and was fixture-
tested (54/54, incl. the SEN case). Full spec: the operator-authored
roster spec + spec-run report, /testing/report/roster_adaptation.
