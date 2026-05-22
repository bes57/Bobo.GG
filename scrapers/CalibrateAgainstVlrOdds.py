"""
CalibrateAgainstVlrOdds.py
==========================

Reads data/vlr_cn_intl_odds.json (produced by ScrapeVlrCnIntlOdds.py) and
calibrates BenPom's no-cn-debut-boost prediction against Vegas's de-vigged
probability for every CN-involved international match.

Buckets are (CN_is_dog vs CN_is_fav) × (intl_exp_diff sign in {-1, 0, +1}).
For each bucket we:
  - report n, mean p_book, mean p_benpom, mean logit-space gap (book − bp)
  - fit a constant offset (logit(book) = logit(bp) + offset_b) via OLS on the
    per-match logit gap (i.e. report the mean gap as the MLE offset)
  - "ship" recommendation = MLE × shrinkage(n), where shrinkage(n) = n/(n+10)
    if n<15 else 1.0 — honors the "temperance" rule of ~2/3 MLE on tiny n.

Also runs the global Platt-style fit (logit(book) = a + b·logit(bp)) for sanity.

Writes data/vlr_calibration_findings.json and prints a console table.
"""

import json
import math
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_PATH  = os.path.join(ROOT, "data", "vlr_cn_intl_odds.json")
OUT_PATH = os.path.join(ROOT, "data", "vlr_calibration_findings.json")

EPS = 1e-6


def logit(p: float) -> float:
    p = min(1 - EPS, max(EPS, p))
    return math.log(p / (1 - p))


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def cn_side(rec: dict) -> str:
    """'fav' if CN team is the favorite, 'dog' otherwise."""
    fav = rec["fav_org"]
    fav_region = rec["winner_region"] if fav == rec["winner"] else rec["loser_region"]
    return "fav" if fav_region == "CN" else "dog"


def fit_ols(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """Simple OLS: y = a + b·x. Returns (a, b, r2)."""
    n = len(xs)
    if n < 2:
        return (0.0, 0.0, 0.0)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    if den <= 0:
        return (my, 0.0, 0.0)
    b = num / den
    a = my - b * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return (a, b, r2)


def shrink(mle: float, n: int) -> float:
    """Temperance: shrink toward 0 when n is small. n/(n+10) if n<15 else 1.0."""
    if n >= 15:
        return mle
    if n <= 0:
        return 0.0
    factor = n / (n + 10.0)
    return mle * factor


def main():
    with open(IN_PATH) as f:
        doc = json.load(f)
    matches = doc["matches"]
    rows = [m for m in matches if m.get("p_fav_book_devigged") is not None]
    print(f"Loaded {len(matches)} matches; {len(rows)} have Vegas odds\n")

    # Per-row logit gap (book − bp) plus bucket labels.
    enriched = []
    for r in rows:
        lb = logit(r["p_fav_book_devigged"])
        lp = logit(r["p_fav_benpom"])
        enriched.append({
            "match_id": r["match_id"],
            "date": r["date"],
            "event_id": r["event_id"],
            "fav_org": r["fav_org"],
            "dog_org": r["dog_org"],
            "cn_side": cn_side(r),
            "dsign": int(r.get("intl_exp_diff_sign", 0)),
            "p_book": r["p_fav_book_devigged"],
            "p_bp":   r["p_fav_benpom"],
            "logit_book": lb,
            "logit_bp":   lp,
            "gap": lb - lp,
        })

    # Global Platt fit (logit(book) = a + b·logit(bp))
    xs_all = [e["logit_bp"] for e in enriched]
    ys_all = [e["logit_book"] for e in enriched]
    g_a, g_b, g_r2 = fit_ols(xs_all, ys_all)
    print(f"Global Platt fit:  logit(book) = {g_a:+.3f} + {g_b:.3f}·logit(bp)   r²={g_r2:.3f}  (n={len(xs_all)})\n")

    # Bucket: (cn_side, dsign)
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for e in enriched:
        buckets[(e["cn_side"], e["dsign"])].append(e)

    bucket_order = [
        ("dog", +1),  # non-CN fav has intl exp, CN dog does not — the N=7-ish bucket
        ("dog",  0),
        ("dog", -1),
        ("fav", +1),
        ("fav",  0),
        ("fav", -1),
    ]

    header = f"{'cn_side':>8}  {'dsign':>5}  {'n':>3}  {'mean_pb':>8}  {'mean_pbp':>9}  {'mean_gap':>9}  {'mle_offset':>11}  {'ship':>7}"
    print(header)
    print("-" * len(header))

    bucket_findings = []
    for key in bucket_order:
        es = buckets.get(key, [])
        n = len(es)
        if n == 0:
            print(f"{key[0]:>8}  {key[1]:>+5d}  {n:>3}  {'-':>8}  {'-':>9}  {'-':>9}  {'-':>11}  {'-':>7}")
            bucket_findings.append({"cn_side": key[0], "dsign": key[1], "n": 0})
            continue
        mp_b = sum(e["p_book"] for e in es) / n
        mp_p = sum(e["p_bp"]   for e in es) / n
        gaps = [e["gap"] for e in es]
        m_gap = sum(gaps) / n
        sd_gap = math.sqrt(sum((g - m_gap) ** 2 for g in gaps) / n) if n > 1 else 0.0
        se_gap = sd_gap / math.sqrt(n) if n > 1 else 0.0
        mle = m_gap                # constant-offset OLS MLE = mean gap
        ship = shrink(mle, n)
        print(f"{key[0]:>8}  {key[1]:>+5d}  {n:>3}  {mp_b:>8.3f}  {mp_p:>9.3f}  {m_gap:>+9.3f}  {mle:>+11.3f}  {ship:>+7.3f}")
        bucket_findings.append({
            "cn_side": key[0], "dsign": key[1], "n": n,
            "mean_p_book": round(mp_b, 4),
            "mean_p_benpom": round(mp_p, 4),
            "mean_gap_logit": round(m_gap, 4),
            "sd_gap_logit": round(sd_gap, 4),
            "se_gap_logit": round(se_gap, 4),
            "mle_offset": round(mle, 4),
            "shipped_offset": round(ship, 4),
            "match_ids": [e["match_id"] for e in es],
        })

    # ---------- Recommendation comparison ----------
    print("\n--- Comparison to shipped constants ---")
    print("Shipped today: INTL_BONUS=+0.22 (uniform on all intl); CN_INTL_EXP_BOOST=+0.70 (CN-debut bucket only)")

    # Per design, INTL_BONUS already shifts intl matches by 0.22·dsign.
    # So the "extra" CN-specific offset above what INTL_BONUS already provides
    # equals the per-bucket gap (since p_bp already includes the 0.22·dsign).
    # i.e. shipping bucket_offset on top of the current model adds exactly
    # bucket_offset more logit on the fav side.
    print("\nInterpretation: bucket_offset is the EXTRA logit shift that should be applied")
    print("on top of the existing INTL_BONUS=+0.22·dsign (which is already in p_bp).")
    print("dsign=0 buckets currently get no boost; the table tells us whether they should.")
    print("dsign≠0 buckets show whether +0.22 alone is enough vs. needing extra (CN-debut).")

    # Pull the key recommendation rows: CN-dog × dsign=+1 (the n=7-ish target)
    rec_lines = []
    for f in bucket_findings:
        if f["n"] == 0: continue
        # Per-bucket suggested ADDITIONAL boost
        extra = f["shipped_offset"]
        rec_lines.append(
            f"  CN-{f['cn_side']:>3}, dsign={f['dsign']:+d}  (n={f['n']:>2}): "
            f"mean_book−bp gap = {f['mean_gap_logit']:+.3f},  ship extra offset = {extra:+.3f}"
        )

    print("\nPer-bucket extra offset (to add on top of current model):")
    for L in rec_lines:
        print(L)

    # ---------- Specific final recommendations ----------
    # Build a concrete recommendation: what should CN_INTL_EXP_BOOST become?
    cn_dog_plus1 = next((f for f in bucket_findings if f["cn_side"] == "dog" and f["dsign"] == +1), None)
    cn_fav_minus1 = next((f for f in bucket_findings if f["cn_side"] == "fav" and f["dsign"] == -1), None)

    print("\n--- Concrete coefficient recommendations ---")
    if cn_dog_plus1 and cn_dog_plus1["n"] > 0:
        # Current shipped CN_INTL_EXP_BOOST is +0.70 (on top of INTL_BONUS=+0.22).
        # The bucket shows the gap remaining ABOVE INTL_BONUS — so the new
        # CN_INTL_EXP_BOOST recommendation is: current 0 (we set CN boost off
        # in scraper) → mle_offset (per-bucket). Ship a single value averaged
        # across the two CN-specific buckets where dsign != 0.
        v1 = cn_dog_plus1["mle_offset"]
        n1 = cn_dog_plus1["n"]
        print(f"  CN-dog, fav has intl exp (n={n1}):  MLE extra = {v1:+.3f}, "
              f"shipped = {shrink(v1, n1):+.3f}")
    if cn_fav_minus1 and cn_fav_minus1["n"] > 0:
        v2 = cn_fav_minus1["mle_offset"]
        n2 = cn_fav_minus1["n"]
        print(f"  CN-fav, dog has intl exp (n={n2}):  MLE extra = {v2:+.3f}, "
              f"shipped = {shrink(v2, n2):+.3f}")

    # Pooled CN-debut: combine BOTH dsign≠0 CN buckets weighted by n.
    cn_debut_rows = [f for f in bucket_findings
                     if f["dsign"] != 0 and f["n"] > 0]
    if cn_debut_rows:
        total_n = sum(f["n"] for f in cn_debut_rows)
        # For dsign sign-correctness: when CN is the dog with dsign=+1, the
        # fav SHOULD get a positive boost. When CN is the fav with dsign=-1,
        # the dog SHOULD get the boost (i.e. fav gets a negative one). Both
        # mean_gap_logit values are oriented as (book − bp) on the favorite
        # side, so a POSITIVE pooled value means "increase fav prob even more
        # than current model says". Combine by absolute alignment: multiply
        # each gap by sign(dsign · 1) — since INTL_BONUS already pushes the
        # fav side up when dsign=+1 and down when dsign=-1, an additional
        # boost in the SAME direction = a positive gap on dsign=+1 and a
        # negative gap on dsign=-1. So pool gap × dsign to get the symmetric
        # additional shift (in dsign-direction).
        pooled_num = sum(f["mean_gap_logit"] * f["dsign"] * f["n"] for f in cn_debut_rows)
        pooled = pooled_num / total_n
        pooled_ship = shrink(pooled, total_n)
        print(f"\n  POOLED CN-debut bucket (any CN side, dsign≠0, n={total_n}): "
              f"symmetric extra boost MLE = {pooled:+.3f}, shipped = {pooled_ship:+.3f}")
        print(f"  → Recommend CN_INTL_EXP_BOOST ≈ {pooled_ship:+.3f}  (vs. currently shipped +0.70)")
    else:
        pooled = 0.0
        pooled_ship = 0.0
        print("\n  POOLED CN-debut bucket: empty (no rows with dsign != 0)")

    # CN-dog, dsign=0: should INTL_BONUS itself be larger when CN is involved?
    cn_dog_zero = next((f for f in bucket_findings if f["cn_side"] == "dog" and f["dsign"] == 0), None)
    cn_fav_zero = next((f for f in bucket_findings if f["cn_side"] == "fav" and f["dsign"] == 0), None)
    if cn_dog_zero and cn_dog_zero["n"] > 0:
        v = cn_dog_zero["mle_offset"]
        print(f"  CN-dog, dsign=0  (n={cn_dog_zero['n']}): mean gap = {v:+.3f}, "
              f"shipped = {shrink(v, cn_dog_zero['n']):+.3f}  "
              f"(non-zero ⇒ CN-dog is generally underrated by BenPom even w/o intl-exp gap)")
    if cn_fav_zero and cn_fav_zero["n"] > 0:
        v = cn_fav_zero["mle_offset"]
        print(f"  CN-fav, dsign=0  (n={cn_fav_zero['n']}): mean gap = {v:+.3f}, "
              f"shipped = {shrink(v, cn_fav_zero['n']):+.3f}")

    # Save findings
    out_doc = {
        "generated": doc.get("generated"),
        "n_total_matches": len(matches),
        "n_with_odds": len(rows),
        "global_platt": {"a": round(g_a, 4), "b": round(g_b, 4), "r2": round(g_r2, 4)},
        "bucket_findings": bucket_findings,
        "pooled_cn_debut": {
            "n": sum(f["n"] for f in cn_debut_rows) if cn_debut_rows else 0,
            "mle_extra_offset": round(pooled, 4) if cn_debut_rows else None,
            "shipped_extra_offset": round(pooled_ship, 4) if cn_debut_rows else None,
            "current_shipped_value": 0.70,
        },
        "shipped_constants": {"INTL_BONUS": 0.22, "CN_INTL_EXP_BOOST": 0.70, "BETA": 0.154},
        "recommendation_note": (
            "p_fav_benpom already includes INTL_BONUS=+0.22·dsign (but NOT the "
            "CN-debut boost). bucket_findings gaps therefore measure the EXTRA "
            "logit shift Vegas wants on top of the current production model."
        ),
        "rows": enriched,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out_doc, f, indent=2)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
