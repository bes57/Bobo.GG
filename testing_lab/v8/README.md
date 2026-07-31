# BenPom v8 workspace

Sole-researcher program, 2026-07-28 →. Governing brief: the operator's v8
research brief (epistemic contract §0 applies to every file in here).

## Layout
```
v8/
  README.md                # this file
  briefs/<agent>.md        # spawn briefs (orchestrator-written, audit trail)
  preregister.<agent>.md   # per-agent pre-registrations, written BEFORE runs
  crn.json                 # shared randomness (power agent writes; all read)
  referee.py               # upgraded metric suite (Phase 6)
  data/                    # derived datasets (lineups, prefranchise corpus, db snapshot)
  stats/                   # every number on the v8 Lab page reads from here
  logs/<agent>.log         # append-only agent journals (resumability)
  phase0_power.md          # deliverable 1
  phase7_autopsy.md        # deliverable 3
```

## Standing rules (bind every agent)
1. Walk-forward always: prediction at T uses only data dated < T.
2. Never tune a constant against the holdout (2025+). If it happens, the
   result is invalid and gets written down as such.
3. CRN: all bootstrap/MC randomness comes from `crn.json`. The rating engine
   itself is deterministic (no RNG) — pairing is exact; CRN governs resampling.
4. One writer per artifact. Output paths are disjoint by brief.
5. VCTMM is hands-off: read-only, and only via the autopsy agent's declared
   snapshot path. The frozen v6 `trading_model/model_snapshot.json` is never
   modified.
6. Fail loudly. No silent substitution of samples, metrics, or events.
7. VLR: one scraper at a time (corpus agent owns the host in Wave 1),
   through `scrapers/enriched/vlr_client.fetch` (rate-limited, CF-bypassing).
8. β is scale-bound: any solve-constant change ⇒ refit β on pre-holdout only.
9. **Market data is diagnostic, never a fitting target (operator directive,
   2026-07-28).** The goal is accurate probabilities against settled match
   outcomes. Model selection and the promotion gate run on walk-forward
   holdout metrics (LL, per-team bias, buckets) — NEVER on agreement with
   Kalshi prices, and never steered by the one-week live-fill sample
   (n=28 settled events; Phase 7 is an implementation memo for the operator,
   not an input to the solve). P&L-weighted loss is a REPORTING unit that
   translates ΔLL into what the bot's quoting surface cares about; it is not
   a selection criterion. The autopsy's "model blame" slice (favorite
   overconfidence, n.s. at n=38) may motivate Phase 5 hypotheses, which are
   then tested purely on match outcomes.
