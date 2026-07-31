# INCIDENT — Live bot is quoting an unvalidated model snapshot (2026-07-23)

**To:** the VCTMM-side agent
**From:** PythonTest-side investigation (read-only — nothing in VCTMM was modified)
**Severity:** High — live quotes are coming from a model that was never validated, and it self-reinstalls after every fix attempt.

---

## TL;DR

The operator flagged SEN at 66% vs LOUD (VLR match 706360, Bo3, 2026-07-26). The
validated `benpom-v6-2026-07-22` snapshot prices that match at **59.8%**; the live
bot is quoting **66.8%**. Root cause: a hook in `vctmm/main.py` **rebuilds the
model snapshot on the VM itself** whenever new results are ingested, and the VM's
data directory is **missing ~292 maps of tier-1 data** (the EWC backfill). The
rebuilt snapshot silently replaces the validated one that `push_model.sh` pushed,
while stamping itself with the real v6's version tag and validation numbers.
Every cross-region price is affected, not just SEN–LOUD.

Fix order matters: **disable/gate the on-VM rebuild hook first**, then re-push a
fresh snapshot from the Mac. Re-pushing alone gets overwritten within minutes of
the next results ingest.

---

## Symptom

| | Fair value P(SEN beats LOUD, bo3) |
|---|---|
| Validated v6 snapshot (Mac, `trading_model/model_snapshot.json`) | **0.5982** |
| Live bot (`/api/state`, event `KXVALORANTGAME-26JUL252000LOUDSEN`) | **0.6680** |

The local pricing stack is NOT the problem. `vctmm/fairvalue/predict_v5.py` +
`service.py` reproduce the reference math in `trading_model/predict.py` exactly
(verified end-to-end on the Mac: p_a = 0.5982104739626263, extras show SEN +1.4905,
LOUD −0.5492, β = 0.1299, same-region so no xregion adjustment). The upcoming-match
row is clean: `format: bo3`, no `gf_upper`.

## Root cause chain

1. **`vctmm/main.py` (~lines 173–203)** spawns
   `trading_model/build_model_snapshot.py` on the VM whenever the results hash
   changes (`model_built_for_results` state key). Notification text: *"New results
   ingested — ratings re-solved on the VM; quotes reprice on the fresh snapshot
   automatically."* It fired at **13:07** and **13:24 UTC today** (notifications
   1145 and 1155). This directly contradicts the handoff contract — both
   `scripts/push_model.sh` and the `FairValueService` docstring state the builder
   runs **only on the Mac** and the VM only consumes the pushed
   `model_snapshot.json` (hot-reloaded by mtime).

2. **The VM's data root is missing the EWC backfill.** The VM-rebuilt snapshot has
   `n_games = 4040`; the validated one has `4314`. The missing ~292 maps are the
   six events that exist only in PythonTest (never committed, never synced to the
   VM): `2026_ewc_qual_americas`, `2026_ewc_qual_emea`, `2026_ewc_qual_pacific`,
   `2026_ewc_qual_cn`, `2026_china_evo_2`, `2026_ewc`. SEN (~10 maps) and LOUD
   (~6 maps) both played the Americas qualifier. This backfill was the "+13.5
   milli" data — the mid-break information behind the market's post-break edge —
   so a model built without it is materially worse, not just different.

3. **`build_model_snapshot.py` refits everything on whatever data it sees**, so the
   VM model drifted across the board:

   | Parameter | Validated v6 | VM rebuild (13:24 UTC) |
   |---|---|---|
   | n_games | 4314 | 4040 |
   | xregion Americas | 1.9664 | 1.6303 |
   | xregion EMEA | 1.3288 | 1.0494 |
   | xregion Pacific | 2.1475 | 2.2769 |
   | b_pick | 0.0979 | 0.1082 |
   | region prior Americas | −1.1741 | −0.9041 |
   | region prior Pacific | −2.6604 | −1.9873 |
   | region prior CN | −4.5255 | −3.2891 |

   β is unchanged (0.1299) only because the builder doesn't refit it.

4. **The drift is masked.** The VM snapshot tags itself `benpom-v6-2026-07-22` and
   carries the real v6's validation block (holdout 0.64126, production 0.65262,
   ROI 0.287) as stamped constants. The dashboard therefore reports a validated
   model while serving an unvalidated one.

## Timeline

- 2026-07-22 22:25 UTC — v6 snapshot built on the Mac (PythonTest).
- 2026-07-22 23:03 UTC — copied to VCTMM repo / pushed to the VM (`push_model.sh`).
- 2026-07-23 13:07 & 13:24 UTC — VM rebuild hook fired twice; validated snapshot
  overwritten (`generated_utc: 2026-07-23 13:24:09`, `ratings_as_of: 2026-07-23`).
- 2026-07-23 13:24 UTC — immediately after the rebuild, a role flip on
  BBL–FUT (`KXVALORANTGAME-26JUL241100BBLFUT`): the model flipped which side it
  favors. Treat that flip as suspect.

## Observed exposure

- Bot is **live** (phase 6, trading_enabled, not halted): 15 deployed events,
  34 resting orders at investigation time (13:41 UTC).
- Holding 24 contracts @ 57¢ avg in `KXVALORANTGAME-26JUL252000LOUDSEN-LOUD`.
- All quotes placed since the first on-VM rebuild were priced off drifted ratings.
  (Check `audit_log` for `model_rebuild` entries to find the FIRST occurrence —
  it may predate today.)

## Recommended remediation (in this order)

1. **Disable or gate the rebuild hook in `vctmm/main.py`.** Options: remove it;
   or gate it behind a config flag defaulting off; or make it refuse when the
   rebuilt `n_games` is LOWER than the currently-loaded snapshot's (data-loss
   tripwire). Without this step, any pushed snapshot is overwritten on the next
   results ingest.
2. **Re-push a fresh snapshot from the Mac** (`scripts/push_model.sh`). Note: a
   Mac rebuild folds in the last day of results, and per the final_model
   reconciliation note, **β is expected to move 0.1299 → 0.1256** on the next
   rebuild — so post-push numbers will be close to, not identical to, the 0.5982
   above. That is expected.
3. **Review inventory opened since the first on-VM rebuild** (esp. the BBL–FUT
   flip and the LOUD position) against corrected fair values; decide keep/unwind
   per the playbook, don't auto-unwind.
4. Longer-term, pick ONE of: (a) keep the Mac-push-only contract (simplest,
   matches all documentation), or (b) if on-VM rebuilds are truly wanted, first
   sync the missing event CSVs to the VM data root and add the n_games tripwire
   plus a distinct `model_version` suffix (e.g. `-vmrebuild`) and a cleared
   validation block so a rebuilt snapshot can never impersonate a validated one.

## Post-fix verification checklist

- `/api/state` → `model_version` shows the `generated_utc` of the file the Mac
  pushed (not a later VM timestamp), and it stays that way across two results
  ingests (i.e., the hook no longer fires or no longer overwrites).
- `/api/algorithm` → `model.n_games` ≥ 4314 and `xregion_offsets.Americas` matches
  the pushed snapshot exactly.
- SEN–LOUD (`706360`): `benpom_p` for SEN ≈ 0.60 (exact value from the freshly
  pushed snapshot's ratings).
- No "v5 model snapshot rebuilt" notification after the next data refresh cycle.

---

*Investigation was strictly read-only on both repos and the live API; no VCTMM
code, config, data, or DB was modified. Questions → the operator.*
