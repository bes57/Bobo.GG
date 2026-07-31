"""Generate reports/state_of_benpom.html from the out/*.json results."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
RD = os.path.join(OUT, "reports")
os.makedirs(RD, exist_ok=True)


def j(name):
    with open(os.path.join(OUT, name)) as f:
        return json.load(f)


diag = j("diagnosis.json")
deep = j("deep1.json")
e1 = j("experiments1.json")
e2 = j("experiments2.json")
e3 = j("experiments3.json")
e4 = j("experiments4.json")
fc = j("final_check.json")
kc = j("kalshi_compare3.json")
ve = j("veto_eval.json")
ml = j("maplevel.json")
cvk = j("cand_vs_kalshi.json")
tax = j("divergence_taxonomy.json")
smc = j("series_mc.json")
c3 = j("candidate_v3.json")
rv = j("revalidate.json")
ts = j("trade_sim.json")

rel = [dict(r) for r in deep["reliability_fav_warm"]]
# merge sparse right-tail bins (n<15) into their left neighbor so no bin has a
# ballooning CI that dwarfs the chart
import math as _math
while len(rel) > 1 and rel[-1]["n"] < 15:
    a, b_ = rel[-2], rel.pop()
    n = a["n"] + b_["n"]
    k = a["emp"] * a["n"] + b_["emp"] * b_["n"]
    pred = (a["pred_mean"] * a["n"] + b_["pred_mean"] * b_["n"]) / n
    ph = k / n
    z = 1.96
    den = 1 + z * z / n
    half = z * _math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    center = (ph + z * z / (2 * n)) / den
    rel[-1] = {"bin_lo": a["bin_lo"], "bin_hi": b_["bin_hi"], "n": n,
               "pred_mean": round(pred, 4), "emp": round(ph, 4),
               "ci_lo": round(center - half, 4), "ci_hi": round(center + half, 4)}
rel_labels = [f"{r['bin_lo']:.2f}–{r['bin_hi']:.2f}" for r in rel]
rel_pred = [r["pred_mean"] for r in rel]
rel_emp = [r["emp"] for r in rel]
rel_lo = [r["ci_lo"] for r in rel]
rel_hi = [r["ci_hi"] for r in rel]
rel_n = [r["n"] for r in rel]

slopes_prod = [deep[f"calib_slope_warm_{y}"]["b"] for y in (2023, 2024, 2025, 2026)]
cand_s25 = e4["cand_slope_25"]["b"]
cand_s26 = e4["cand_slope_26"]["b"]

hl_map = {"exp_hl4": 4, "exp_hl5": 5, "prod_baseline": 6, "exp_hl8": 8,
          "exp_hl10": 10, "exp_hl13": 13}
hl_pts = sorted((v, e1[k]["ll_test"]) for k, v in hl_map.items())
hl_x = [p[0] for p in hl_pts]
hl_y = [p[1] for p in hl_pts]

bk = fc["buckets"]
bk_names = [b["name"] for b in bk]
bk_delta = [round(b["delta"] * 1000, 2) for b in bk]  # millilogloss

experiments_rows = []
def _row(name, dll, p, verdict, note):
    experiments_rows.append((name, dll, p, verdict, note))


_row("Longer rating memory (HL 6→13w)", "+5.3", "0.968", "ADOPT",
     "Every decay family agrees; plateau 10–16w. Fixes 2026 overconfidence (slope 0.80→1.05).")
_row("Margin transform sqrt→power 0.75", "+0.7 (stacked)", "—", "ADOPT",
     "Consistently positive across grid; raw margins ≈ same, sqrt worst of the three.")
_row("Roster persistence 0.3→0.7 (alone)", "+3.2", "—", "SUPERSEDED",
     "Helps at HL=6 but redundant once HL=13; grid prefers rp0.3 with long memory.")
_row("Roster: lineup-overlap reweighting", "-1.9 to -7.5", "0.20", "REJECT",
     "All variants hurt; year-boundary continuity already covers real rebuilds.")
_row("Roster-instability prediction shrink", "-0.8 to -1.8", "—", "REJECT",
     "Matches with recent roster changes are already predicted BETTER than average.")
_row("Favorite blowout-margin discount ('don't try hard')", "-1.7 to -6.1", "0.04-0.16", "REJECT",
     "Favorites' big margins carry real signal; discounting them loses information.")
_row("Residual margin (credit only surprise)", "-0.5 to -8.1", "0.28", "REJECT",
     "Neutral at k=0.15, worse beyond.")
_row("Upset margin boost", "+0.3 / -0.9", "0.62", "REJECT", "Noise.")
_row("Rolling walk-forward β refit", "-1.3 to -3.1", "0.03-0.10", "REJECT",
     "Trailing fits chase noise; frozen β more robust than any window tried.")
_row("Per-stage (playoffs) β", "-1.1", "0.25", "REJECT",
     "Playoffs are genuinely noisier, not mis-scaled.")
_row("Win-only ratings (no margins)", "-5.4", "—", "REJECT",
     "Margins matter. Blends also negative.")
_row("Veto rate shrinkage toward global", "top1 46.5→44.6%", "—", "REJECT",
     "Raw team-specific rates win at every K; veto habits are idiosyncratic.")
_row("Intl-experience offset (+0.40)", "refits to 0.0", "—", "TRIM (weak)",
     "On long-memory ratings the bonus adds nothing (fit 24-25, test 26; n small).")
_row("CN-dog offset (+0.35)", "refits to ~0.2", "—", "KEEP/TRIM (weak)",
     "Small positive either way; evidence thin (n=48 intl 2026).")

exp_html = "".join(
    f"<tr><td>{n}</td><td class='mono'>{d}</td><td class='mono'>{p}</td>"
    f"<td><span class='verd {v.split()[0].lower()}'>{v}</span></td><td>{note}</td></tr>"
    for n, d, p, v, note in experiments_rows)

fav_rows = "".join(
    f"<tr><td>{r['band']}</td><td class='mono'>{r['n']}</td>"
    f"<td class='mono'>{r['benpom']:.3f}</td><td class='mono'>{r['kalshi']:.3f}</td>"
    f"<td class='mono'>{r['emp']:.3f}</td></tr>" for r in kc["fav_shift"])

div_rows = "".join(
    f"<tr><td class='mono'>{r['date']}</td><td>{r['winner']} bt {r['loser']}</td>"
    f"<td>{r['stage']}</td><td class='mono'>{r['p_benpom']:.2f}</td>"
    f"<td class='mono'>{r['pk_pre']:.2f}</td></tr>"
    for r in kc["top_divergences"][:10])

ml_rows = "".join(
    f"<tr><td class='mono'>{k}</td><td class='mono'>{v['beta_map']}</td>"
    f"<td class='mono'>{v['ll_map_test']:.5f}</td></tr>"
    for k, v in ml.items())

tax_rows = ""
for t in tax:
    ctx_bits = []
    for org, c in t["ctx"].items():
        ctx_bits.append(f"{org}: form {c['recent_form_maps']}, rest {c['days_since_last']}d"
                        + (f", lineup {c['lineup_overlap_last4']}/5" if c["lineup_overlap_last4"] is not None else ""))
    tax_rows += (f"<tr><td class='mono'>{t['date']}</td><td>{t['match']}</td>"
                 f"<td class='mono'>{t['benpom']:.2f}</td><td class='mono'>{t['kalshi']:.2f}</td>"
                 f"<td class='dim' style='font-size:.78rem'>{'; '.join(ctx_bits)}</td></tr>")

veto_steps = ve["K0.0"]["by_step"]
veto_rows = "".join(
    f"<tr><td>{k.replace('_', ' ')}</td><td class='mono'>{v['n']}</td>"
    f"<td class='mono'>{v['top1']:.1%}</td><td class='mono'>{v['top3']:.1%}</td></tr>"
    for k, v in sorted(veto_steps.items()))

html = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex">
<title>State of BenPom — Diagnosis &amp; Optimization (Rounds 1–8, final)</title>
<link rel="icon" href="/logo.svg">
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@700;800&family=DM+Sans:wght@400;500;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.3/dist/chart.umd.min.js"></script>
<style>
 * {{ box-sizing:border-box; margin:0; }}
 :root {{ --ink:#16121d; --dim:#6b6478; --line:#eceef2; --acc:#7c4dd6; --accbg:#f3eefb;
         --good:#1e7a4f; --goodbg:#ecf8f1; --bad:#c0392b; --badbg:#fbeaea;
         --warn:#b3541e; --warnbg:#fdf3ec; }}
 body {{ font-family:'DM Sans',system-ui,sans-serif; background:#faf9fc; color:var(--ink);
        line-height:1.55; padding:34px 18px 90px; }}
 .wrap {{ max-width:920px; margin:0 auto; }}
 h1 {{ font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; font-size:1.55rem;
      text-align:center; margin:6px 0 2px; }}
 .tagline {{ text-align:center; color:var(--dim); font-size:.9rem; margin-bottom:26px; }}
 section {{ background:#fff; border:1px solid var(--line); border-radius:18px;
           padding:24px 28px; margin-bottom:16px; box-shadow:0 3px 14px #00000008; }}
 h2 {{ font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; font-size:1.05rem;
      margin-bottom:12px; }}
 p {{ font-size:.89rem; margin:7px 0; }}
 .dim {{ color:var(--dim); }}
 .mono {{ font-family:'JetBrains Mono',monospace; font-size:.82em; }}
 .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:12px; }}
 .card {{ border:1px solid var(--line); border-radius:14px; padding:14px 16px; }}
 .card .lbl {{ font-size:.7rem; font-weight:700; letter-spacing:.8px; text-transform:uppercase;
              color:var(--dim); }}
 .card .big {{ font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; font-size:1.5rem;
              margin:2px 0; }}
 .card .sub {{ font-size:.78rem; color:var(--dim); }}
 .good {{ color:var(--good); }} .bad {{ color:var(--bad); }}
 table {{ width:100%; border-collapse:collapse; font-size:.84rem; margin:8px 0; }}
 th {{ text-align:left; color:var(--dim); font-size:.7rem; text-transform:uppercase;
      letter-spacing:.5px; padding:6px 9px; border-bottom:2px solid var(--line); }}
 td {{ padding:7px 9px; border-bottom:1px solid var(--line); vertical-align:top; }}
 tr:last-child td {{ border-bottom:0; }}
 .verd {{ font-weight:700; border-radius:999px; padding:2px 9px; font-size:.7rem; white-space:nowrap; }}
 .verd.adopt {{ background:var(--goodbg); color:var(--good); }}
 .verd.reject {{ background:#f1f0f4; color:var(--dim); }}
 .verd.trim, .verd.keep\\/trim {{ background:var(--warnbg); color:var(--warn); }}
 .verd.superseded {{ background:var(--warnbg); color:var(--warn); }}
 .callout {{ border-left:4px solid var(--acc); background:var(--accbg);
            border-radius:0 12px 12px 0; padding:11px 15px; margin:10px 0; font-size:.87rem; }}
 .callout.good {{ border-color:var(--good); background:var(--goodbg); }}
 .callout.warn {{ border-color:var(--warn); background:var(--warnbg); }}
 canvas {{ max-height:340px; }}
 .chartbox {{ margin:14px 0 4px; }}
 .chartbox.tall {{ height:540px; }}
 .chartbox.tall canvas {{ max-height:none; height:100% !important; }}
 code {{ font-family:'JetBrains Mono',monospace; font-size:.8em; background:#f4f2f8;
        border:1px solid var(--line); border-radius:6px; padding:1px 6px; }}
 @media (max-width:640px) {{ section {{ padding:18px 14px; }} }}
</style></head>
<body><div class="wrap">
<h1>State of BenPom</h1>
<style>
 .labtabs {{ display:flex; justify-content:center; gap:6px; margin:14px 0 20px; }}
 .labtabs a {{ font-size:.8rem; font-weight:700; color:var(--dim); text-decoration:none;
   padding:7px 15px; border-radius:999px; border:1px solid var(--line); background:#fff; }}
 .labtabs a:hover {{ color:var(--ink); background:var(--accbg); }}
 .labtabs a.on {{ color:#fff; background:var(--acc); border-color:var(--acc); }}
</style>
<div class="tagline">Diagnosis + optimization rounds 1–8 + production EWC implementation ·
generated 2026-07-22 · {rv['n_series']} walk-forward series 2023–2026 (EWC-class events
now native) · holdout = 2025–26</div>
<div class="labtabs">
<a href="/testing/">Testing Lab</a>
<a href="/testing/report/state_of_benpom" class="on">State of BenPom</a>
<a href="/testing/playbook">Deployment Playbook</a>
<a href="/testing/report/favorites_lab">Favorites Lab</a>
<a href="/testing/report/final_model">Final Model</a>
<a href="/testing/report/v7_lab">v7 Lab</a>
</div>

<section>
<h2>Headline — current state (native rebuilt data)</h2>
<div class="cards">
 <div class="card"><div class="lbl">Production (holdout)</div>
  <div class="big">{rv['prod_hl6_pow05']['ll_test']:.4f}</div>
  <div class="sub">log-loss (lower = better) · coin-flip baseline ≈ 0.693</div></div>
 <div class="card"><div class="lbl">Candidate (full stack)</div>
  <div class="big good">{rv['V4_NATIVE_full_stack']['ll_test']:.4f}</div>
  <div class="sub">asym games decay W20/L12 · margin^0.75 · playoff ×1.6 ·
  cold-start priors · x-region offsets · EWC-class data
  (+{rv['boot_v4_vs_prod']['mean_delta']*1000:.1f}m, p={rv['boot_v4_vs_prod']['p_better']:.3f})</div></div>
 <div class="card"><div class="lbl">Kalshi, VCT matches (n={rv['overlap_vct']['n']})</div>
  <div class="big good">{rv['overlap_vct']['model']:.4f} vs {rv['overlap_vct']['kalshi']:.4f}</div>
  <div class="sub"><b>BenPom ahead</b> (production was 0.6805, −35m behind). VCT Stage 2
  (n={rv['overlap_vct_stage2']['n']}): <b>{rv['overlap_vct_stage2']['model']:.4f} vs
  {rv['overlap_vct_stage2']['kalshi']:.4f}</b>.</div></div>
 <div class="card"><div class="lbl">Not "the same as Kalshi"</div>
  <div class="big good">+{ts['trade_all_5']['roi']*100:.0f}% ROI</div>
  <div class="sub">corr(model, market) = {ts['corr']:.2f} — independent signal. Taking
  BenPom's side at market prices on 5+ pt divergences: n={ts['trade_all_5']['n']},
  CI [{ts['trade_all_5']['ci'][0]*100:+.0f}%, {ts['trade_all_5']['ci'][1]*100:+.0f}%].</div></div>
</div>
<div class="callout" style="border-color:#1e7a4f;background:#ecf8f1">
<b>Companion report:</b> <a href="/testing/report/when_to_deploy"
style="color:#1e7a4f;font-weight:700">When to Deploy the Bot — ROI Playbook &rarr;</a>
— expected ROI by window/region/price-band/divergence (n=90 trades), market microstructure
(when spreads are wide vs sharp), worked example trades, and the deployment checklist.</div>
<div class="callout good"><b>Bottom line (after rounds 5–8 superseded the early answer):</b>
BenPom's core defect was decay measured in <b>calendar weeks</b> — it burned information across
breaks and over-weighted the last few weeks in-season. The fix is <b>decay by games played,
asymmetric</b>: wins fade with a 20-game half-life, losses with 12 (≈6–8 in-season weeks of
effective memory — short, but break-proof). Stacked with playoff up-weighting, cold-start
region priors, and fitted cross-region offsets, candidate v3 scores <b>0.6398 vs production's
0.6502 (+10.4 milli-LL, p=0.999)</b>, fixes the 2026 overconfidence, and closes the
favorite-pricing gap to Kalshi. The earlier "HL 6→13 weeks" candidate (v1, +5.3m) survives only
as the minimal-code fallback. Nothing is deployed: awaiting your go per the promotion rule.</div>
</section>

<section>
<h2>Your hunches — scored against the data</h2>
<table>
<tr><th>Hunch</th><th>Verdict</th><th>Evidence</th></tr>
<tr><td>"BenPom is often stale"</td><td><span class="verd adopt">PARTLY — inverted</span></td>
<td>It's the opposite of stale in the aggregate: calendar decay over-weights the last few weeks
and burns info across breaks. Final answer (round 5+): decay by <b>games played</b>, not weeks —
wins HL 20 games / losses 12. (True staleness = roster-news blindness is real but small.)</td></tr>
<tr><td>"Too close to 50-50"</td><td><span class="verd adopt">CONFIRMED (2026)</span></td>
<td>Kalshi prices BenPom's 0.5–0.7 favorites higher, and results side with the market
(emp {kc['fav_shift'][0]['emp']:.2f} vs BenPom {kc['fav_shift'][0]['benpom']:.2f} in [0.5,0.6)).
The long-memory candidate sharpens exactly this belt.</td></tr>
<tr><td>"Hates heavy favorites"</td><td><span class="verd adopt">CONFIRMED</span></td>
<td>Large-gap bucket improves most under the candidate (+9.6m LL); 0.75–0.80 favorites won
84% vs predicted 77% under production.</td></tr>
<tr><td>"Kalshi often better"</td><td><span class="verd adopt">CONFIRMED (2026)</span></td>
<td>Pre-match market beats production BenPom (LL 0.646 vs 0.681, p≈0.97); on divergences
&gt;15pts the market was right 4/5. In-sample blend weight goes to the market.</td></tr>
<tr><td>"Try non-exponential decay"</td><td><span class="verd adopt">TESTED — settled (rd 8)</span></td>
<td>Exponential is the right family, but in <i>games</i>-space and asymmetric (losses fade
~1.7× faster than wins). Power-law ties; box+exp, linear, calendar envelopes all lose.
Full ladder in §Round 8.</td></tr>
<tr><td>"Roster changes need handling"</td><td><span class="verd reject">ALREADY COVERED</span></td>
<td>Production's year-boundary continuity does the heavy lifting; three finer-grained schemes
(lineup reweighting, step penalties, prediction shrink) all made things worse. Mid-split
change matches are predicted <i>better</i> than average.</td></tr>
<tr><td>"Favorites don't try vs weak teams"</td><td><span class="verd reject">REJECTED</span></td>
<td>Discounting favorites' blowout margins hurts every variant tested — those margins are
signal, not noise.</td></tr>
<tr><td>"Domestic vs intl algorithm split"</td><td><span class="verd trim">PARTIAL</span></td>
<td>No evidence for separate β; the intl-experience offset (+0.40) refits to 0.0 on candidate
ratings (long memory already encodes pedigree). CN-dog stays small (~0.2–0.35).</td></tr>
</table>
</section>

<section>
<h2>Calibration</h2>
<p>Favorite-side reliability of the production model (cold-start ties excluded). Shaded band =
95% Wilson CI; the dashed line is perfect calibration. Sparse right-tail bins (n&lt;15) are
merged into their neighbor so every plotted point has a real sample — hover for each bin's n.</p>
<div class="chartbox"><canvas id="rel"></canvas></div>
<p style="margin-top:14px">Calibration slope by year (1.0 = perfect; &lt;1 = overconfident).
Production drifted overconfident by 2026; the candidate restores it.</p>
<div class="chartbox"><canvas id="slopes"></canvas></div>
</section>

<section>
<h2>The memory curve</h2>
<p>Holdout log-loss vs rating half-life (exponential family; other families overlay the same
plateau). Production sits at 6 weeks, well short of the 10–16 week plateau.</p>
<div class="chartbox"><canvas id="hl"></canvas></div>
</section>

<section>
<h2>Everything tested</h2>
<p class="dim">ΔLL in milli-log-loss per series vs the relevant baseline (positive = better);
p = bootstrap P(better). Walk-forward throughout; β refit on ≤2024 only.</p>
<table>
<tr><th>Idea</th><th>ΔLL (m)</th><th>p</th><th>Verdict</th><th>Note</th></tr>
{exp_html}
</table>
</section>

<section>
<h2>Rounds 5–7: the half-life question answered (and candidate v2)</h2>
<div class="callout good"><b>"13 weeks is too high" — correct, and here's the resolution.</b>
Calendar half-life was the wrong object. Decaying by <b>games played</b> (half-life ≈ 16
team-games ≈ 6–8 in-season weeks) beats calendar HL-13 — because the real defect was calendar
decay burning information across the long breaks when nothing new had happened. In-season
memory stays short; it just stops decaying when no games are played.</div>
<table>
<tr><th>Mechanism test</th><th>Holdout LL</th><th>Verdict</th></tr>
<tr><td>Games-played decay, HL=16 games</td><td class="mono good">0.64383</td><td><span class="verd adopt">WINNER</span></td></tr>
<tr><td>Calendar exp HL=13w (v1)</td><td class="mono">0.64484</td><td>superseded</td></tr>
<tr><td>Two-timescale blend (form+class)</td><td class="mono">0.64506</td><td>ties v1, loses to games</td></tr>
<tr><td>Short HL + soft break boundaries</td><td class="mono">0.64728</td><td>right idea, weaker version</td></tr>
<tr><td>Short HL + heavy ridge (variance story)</td><td class="mono">0.64665</td><td>rejected</td></tr>
</table>
<p style="margin-top:10px"><b>New adds that stack</b> (each walk-forward): playoff/GF games
weighted ×1.6 in the solve (+1.3m); cold-start orgs enter at their region's 25th percentile
instead of league-average 0 (+0.15m, targets the worst bucket); monthly-refit cross-region
offsets replacing hand-tuned constants (cross-region matches 0.672→0.665). Also tested and
rejected in these rounds: saturating tanh link (fit runs to no-op — logistic validated),
intra-series momentum correlation in the MC (hurts), Bo1 down-weighting (no effect),
removing roster continuity under games decay (−3m — it matters <i>more</i> now).</p>
<div class="cards" style="margin-top:10px">
 <div class="card"><div class="lbl">Candidate v2 (stacked)</div>
  <div class="big good">0.64209</div>
  <div class="sub">games-HL16 · margin^0.75 · playoff ×1.6 · cold-start priors ·
  x-region offsets · β≈0.104</div></div>
 <div class="card"><div class="lbl">vs production</div>
  <div class="big good">+8.1m</div><div class="sub">p(better) = 0.988 — passes the
  promotion bar decisively</div></div>
 <div class="card"><div class="lbl">Veto pick model</div>
  <div class="big good">40.1% → 41.5%</div><div class="sub">top-1, adding opponent-weakness
  factor (1.75−opp_win) to pick scores (n=1,872)</div></div>
</div>
<div class="callout warn"><b>Implementation notes for promotion:</b> games-decay needs a small
pipeline change (per-side weight exp(−λ·games_ago) instead of exp(−λ·eff_weeks) in
<code>massey_ratings</code> — the per-side weighting hook already exists). Per-map shrinkage
under games decay needs its own tuning pass (within-map memory wants to be shorter; overall-only
MC ≈ calendar k20 ≈ 0.646, both ≫ production's 0.655). v1 (HL13w/pow0.75/k20) remains the
minimal-change fallback — constants only, +8.6m on the MC surface.</div>
</section>

<section>
<h2>Round 8: the decay function, settled — and the 50-50 belt, resolved</h2>
<h3 style="font-size:.95rem;font-weight:700;margin:10px 0 6px">Why exponential, and in what units</h3>
<p>With games as the clock, every functional form was raced on identical inputs
(v2 base, β refit on ≤2024, scored 2025–26):</p>
<table>
<tr><th>Decay form (games-space)</th><th>Holdout LL</th><th>Read</th></tr>
<tr><td><b>Asymmetric exp: wins HL=20, losses HL=12</b></td><td class="mono good">0.64078</td>
<td><span class="verd adopt">WINNER</span> +1.7m vs symmetric, p=0.968</td></tr>
<tr><td>Asymmetric exp: wins 16 / losses 10 (same ratio)</td><td class="mono">0.64122</td>
<td>confirms the ratio, not a point-estimate fluke</td></tr>
<tr><td>Symmetric exp HL=16 (v2)</td><td class="mono">0.64252</td><td>reference</td></tr>
<tr><td>Power-law (heavy tail), best of 4</td><td class="mono">0.64263</td><td>ties — no tail benefit</td></tr>
<tr><td>+ calendar envelope (65w), best of 3</td><td class="mono">0.64266</td><td>weakest envelope ≈ no envelope: age in weeks adds nothing once games are counted</td></tr>
<tr><td>Box+exp (flat recent window), best of 3</td><td class="mono">0.64327</td><td>rejected</td></tr>
<tr><td>Reversed asymmetry (losses remembered longer)</td><td class="mono">0.64544</td><td>clearly wrong direction — the sign is real</td></tr>
</table>
<div class="callout"><b>Discussion.</b> Exponential is the right family, and <i>games</i> are the
right clock: exponential decay is the unique memoryless forgetting rule, and the data says
information arrives per game played, not per week elapsed — every attempt to reintroduce
calendar time (envelopes, box windows, heavy tails that preserve old eras) scored worse or flat.
The one refinement that beats plain exponential is <b>asymmetric memory: losses fade ~1.7×
faster than wins</b>. A plausible reading: wins against rated opposition are hard evidence about
a team's ceiling, while losses mix in noise (dead maps, experimentation, tilt). Both tested
ratios agree, and the reversed sign is decisively worse.</div>
<h3 style="font-size:.95rem;font-weight:700;margin:14px 0 6px">The 50-50 belt</h3>
<p>On the candidate surface the train-fit calibration slope is <b>0.9989</b> — globally there is
nothing left to expand; the "too close to 50-50" defect was a symptom of the old ratings, not a
missing calibration layer. Confirmations: slope-only Platt, odd-cubic recalibration, and
rolling-12-month slope layers were all fit walk-forward and all <b>lost or tied on test</b>
(the rolling layer keeps estimating slopes >1.25 from trailing data and still loses — trailing
recalibration chases noise, same lesson as rolling β). A power-link that expands small rating
gaps also failed. What remained was one band — 0.70–0.80 favorites winning ~80% — and the
<b>asymmetric decay closes half of it structurally</b> (gap +0.069 → +0.053, ≈1.3σ residual:
monitor, don't curve-fit). Against Kalshi, v3's favorite pricing now sits within ~1.5 points of
the market in every band — the compression story is resolved; the market's remaining edge is
post-break information, not sharpness.</p>
<div class="cards" style="margin-top:10px">
 <div class="card"><div class="lbl">Candidate v3</div>
  <div class="big good">0.63981</div>
  <div class="sub">asym games decay (W20/L12) · margin^0.75 · playoff ×1.6 ·
  cold-start priors · x-region offsets · β≈0.103</div></div>
 <div class="card"><div class="lbl">vs production</div>
  <div class="big good">+10.4m</div><div class="sub">p(better) = 0.999 · improves both
  holdout years (25: 0.630 / 26: 0.652)</div></div>
 <div class="card"><div class="lbl">vs v2</div>
  <div class="big good">+2.3m</div><div class="sub">p(better) = 0.994</div></div>
</div>
</section>

<section>
<h2>Session 9: the margin-outcome proposal ("BenPom v3") — investigated to the bottom</h2>
<p><b>The idea (user's):</b> judge and train the model on <i>average round margin per map</i>
instead of binary outcomes — FNC beating KC 2-1 with maps +2/+2/−7 should count as evidence
FOR a KC-favored model, not against it. Margins carry more information per match than a
coin-flip outcome, so fitting on them should sharpen the model, especially at high
probabilities.</p>
<p><b>Seven constructions, all walk-forward, all scored on binary holdout log-loss (trading
is binary):</b></p>
<table>
<tr><th>Construction</th><th>vs champion (v5)</th><th>Verdict</th></tr>
<tr><td>Empirical-CDF margin link (map level)</td><td class="mono">−9.0m</td><td>rejected</td></tr>
<tr><td>Probit-normal margin link</td><td class="mono">−0.3m</td><td>rejected</td></tr>
<tr><td>Soft-label β (Gaussian margin labels)</td><td class="mono">−1.8m</td><td>rejected</td></tr>
<tr><td>Mixed binary+margin loss (w=0.25/0.5/0.75)</td><td class="mono">−0.5 to −3.0m</td><td>rejected</td></tr>
<tr><td>Margin-residual form feature (momentum, 3 strengths)</td><td class="mono">−0.2 to −3.3m</td><td>rejected</td></tr>
<tr><td>Calibrated retrospective targets (empirical margin→prob map)</td><td class="mono">−0.5m (p=0.24)</td><td>rejected</td></tr>
<tr><td>Margin-fitted cross-region offsets (least squares)</td><td class="mono">−14m on cross-region</td><td>rejected</td></tr>
</table>
<div class="callout"><b>Why it fails — a real domain fact worth keeping:</b> margins
systematically <u>understate</u> win probability. The better team disproportionately wins the
<i>close</i> maps (every overtime win has margin exactly 2), so any margin-derived probability
under-rates favorites — the opposite of the intended sharpening. The empirical margin→win map
is brutally steep (avg margin of just +1/map already implies <b>84%</b>): in the FNC–KC
example the margin ledger says KC "deserved" 84%, far more than the model's 53%. And the
place margins genuinely help — estimating <i>ratings</i> — is already margin-native (the
Massey target is the round margin, transform tuned in round 1). Margins for ratings, binary
for calibration is the optimal division of labor, and it's what the champion already does.</div>
<p><b>Salvaged from the investigation:</b></p>
<ul>
<li><b>The margin-MSE metric is adopted</b> as a permanent secondary evaluation (exactly the
user's scoring idea): predicted vs realized series avg margin/map. The champion beats
production on it too (MSE 43.1 vs 45.4, correlation 0.148 vs 0.104) — the model is better
in margin-space as well, independently confirming the binary result.</li>
<li><b>The premise check came back changed:</b> "the model is scared of high probabilities"
was true of production but is no longer true of the champion — at [0.7,0.8) it now prices
favorites at 0.746 vs the market's 0.718 (n=24, favorites won 0.667), i.e. the model is now
the <i>bolder</i> side. The timidity was cured by the games-decay/asym stack, not by margin
outcomes.</li>
</ul>
</section>

<section>
<h2>Session 7: one more hour of new ideas — one adopted, five retired</h2>
<p><b>Adopted — region-prior ridge (v5):</b> the Massey solve now regularizes each team toward
its <i>region's</i> trailing mean instead of the global zero (second ridge, weight 1.5). New
teams and thin-data teams stop being dragged toward league-average; CN context especially.
Stacked on everything else: <b>0.64132</b> (v4 was 0.64199; +0.7m, p=0.71 — mild but
principled and consistent across the grid 0.8/1.5/2.5).</p>
<p><b>Tested and retired with evidence:</b> reduced OT margins (13-11 vs 14-12 carry the same
info — no-op), rounds-ratio margins (worse), decay counted in <i>series</i> instead of maps
(map-counting confirmed right — a Bo5 really does teach ~5 maps of information), piecewise
β for small vs large gaps (fit converges to a single slope — the link is linear), and a
closed-form × veto-MC ensemble (every blend weight loses to pure closed-form: the MC's map/veto
layer adds quoting granularity but no series-level information — re-confirming the site's
original "per-map MC rejected" finding at a much higher baseline).</p>
</section>

<section>
<h2>Session 6: "same as Kalshi?" — no: the edge lives in the disagreements</h2>
<p>Equal log-loss with an <b>independent</b> model is not "the same as Kalshi" — correlation
between the two is only {ts['corr']:.2f}, and profit comes precisely from where they diverge.
Simulated trading (buy BenPom's side at the pre-match market price — taker, worst case,
no spread capture):</p>
<table>
<tr><th>Divergence filter</th><th>n</th><th>ROI</th><th>95% CI</th></tr>
<tr><td>All joined matches, any divergence</td><td class="mono">{ts['trade_all_0']['n']}</td>
<td class="mono">{ts['trade_all_0']['roi']*100:+.1f}%</td>
<td class="mono">[{ts['trade_all_0']['ci'][0]*100:+.0f}%, {ts['trade_all_0']['ci'][1]*100:+.0f}%]</td></tr>
<tr><td><b>Divergence &gt; 5 pts</b></td><td class="mono">{ts['trade_all_5']['n']}</td>
<td class="mono good"><b>{ts['trade_all_5']['roi']*100:+.1f}%</b></td>
<td class="mono good">[{ts['trade_all_5']['ci'][0]*100:+.0f}%, {ts['trade_all_5']['ci'][1]*100:+.0f}%] — excludes 0</td></tr>
<tr><td>Divergence &gt; 10 pts</td><td class="mono">{ts['trade_all_10']['n']}</td>
<td class="mono">{ts['trade_all_10']['roi']*100:+.1f}%</td>
<td class="mono">[{ts['trade_all_10']['ci'][0]*100:+.0f}%, {ts['trade_all_10']['ci'][1]*100:+.0f}%]</td></tr>
</table>
<p class="dim">Hit rates sit below 50% by design — the model tends to take the cheap
(underdog) side and wins more often than the price implies. The bot's maker-side execution
(spread + rebates) would add to these taker-priced numbers.</p>
<div class="callout"><b>How can log-loss tie while ROI is +31%?</b> They're different
functionals on different samples in different units. Log-loss parity is an average over
<i>all</i> matches; trades exist only on the disagreement subset. And log-loss lives in
probability space while ROI lives in <i>price</i> space: if the market prices an underdog at
20¢ and its true chance is 30%, the log-loss gain from knowing that is tiny (it only registers
the 30% of the time the dog wins) — but the trade's expected ROI is +50% (buy at 0.20 what's
worth 0.30). Conversely, when the model is the wrong one, log-loss punishes it logarithmically
while the dollar loss is capped at the stake. Small probability edges on cheap contracts are
worth little in log space and a lot in dollars.
<br><br><b>Robustness of the +31%</b> (n=88, |div|&gt;5pts): drop the top 1/2/3/5 winning
trades → +28% / +26% / +23% / +19% (not concentration-driven). By window: May/quals +37%,
Stage 2+EWC +45%, London −21% (the known deficit window). By side: 79 underdog-side buys at
+34% (hit 44%), 9 favorite-side at +15% (hit 67%). A 2¢-better maker fill scores ≈ the same
(+30%); real taker slippage would shave a few points the other way.</div>
<p style="margin-top:10px"><b>Traded-surface package (native series veto-MC, n=999):</b>
overall-rating MC + opponent-aware pick scores (×(1.75−opp_win)) + a walk-forward pick-side
bonus (picker's map logit +b_pick; the effect is real and growing — raw picker winrate 52.5%,
b_pick ≈ +0.09 by 2026) scores <b>0.64497</b>, best of seven variants. Per-map rating splits
and the pick bonus <i>double-count</i> each other (picked maps are the good maps) — use the
bonus, drop the splits (SHRINK_K→∞ on this surface).</p>
<p><b>London autopsy</b> (the one remaining deficit window): the gap is NOT cross-region
(model 0.722 vs market 0.718 there) — it's same-region rematches (n=7) plus CN sides that both
model and market underpriced (CN won 50% at London, priced ~33-36%). Tested and rejected:
a head-to-head/rematch term (prior winner rewins 53.2% vs 53.8% model-implied — ratings
already capture it) and recency-weighted region offsets (unstable, hurt London). The residual
is market-side LAN information; accepted and monitored.</p>
<p><b>Built:</b> a prospective prediction logger (<code>testing_lab/log_predictions.py</code>,
idempotent, appends candidate probs for every upcoming match — 71 logged today) so
model-vs-market evidence accumulates forward without reconstruction. Automatable via cron
when desired.</p>
</section>

<section>
<h2>Session 5: EWC implemented in production — native rebuild &amp; final numbers</h2>
<div class="callout good"><b>Done, end to end.</b> Six events are now first-class in the
production pipeline: the four EWC regional qualifiers, China Evolution Series Act 2, and the
Esports World Cup — registered in the event registry as <code>ratings_only</code> (feed BenPom,
hidden from all player-facing UIs, same treatment as CN-only events) and <code>vct_only</code>
(tier-2 guest matches are skipped at scrape time). The whole chain ran: stats + match scrape
(109 matches, 292 maps — more than the 82-match test patch, since full event coverage includes
non-Kalshi-listed games), match results, dates, vetoes, then full rebuilds of the rating
timelines (2026: 392→501 match events), map ratings, and veto model. Production model
constants are unchanged; the site simply knows about more matches now. Site verified healthy.</p></div>
<p>Two engineering fixes landed on the way: <b>(1)</b> <code>ScrapeMatchData</code> still used
the pre-July table selectors — it now delegates to the live pipeline's <code>div.ovw-*</code>
parser (this would have bitten ALL future event scrapes); <b>(2)</b> the testing engine had
been silently importing VCTMM's <i>vendored</i> registry copy (stale data snapshot) whenever
the org-mapping import put VCTMM on the path — now pinned to this repo's modules. Prior
session results stand (the patch supplied what the stale snapshot lacked), but the trap is
fixed and documented.</p>
<table>
<tr><th>Native rebuilt data — holdout 2025-26 (n=997 incl. new events)</th><th>Log-loss</th></tr>
<tr><td>Production constants</td><td class="mono">0.65262</td></tr>
<tr><td>Candidate stack (asym games decay, full)</td><td class="mono good">0.64199
 (+10.6m, p=0.998)</td></tr>
<tr><td>Candidate stack with EWC-class events zero-weighted</td><td class="mono">0.64732
 — <b>the new data alone is worth ~4.5m</b></td></tr>
</table>
<p style="margin-top:10px"><b>Kalshi, native (start−5m anchors everywhere):</b></p>
<table>
<tr><th>Sample</th><th>n</th><th>BenPom candidate</th><th>Market</th><th>Verdict</th></tr>
<tr><td>VCT matches</td><td class="mono">86</td><td class="mono good">0.6441</td>
<td class="mono">0.6457</td><td><b>BenPom ahead</b> (dead heat statistically; production
was 0.6805, decisively behind)</td></tr>
<tr><td>VCT Stage 2 (current window)</td><td class="mono">49</td><td class="mono good">0.6418</td>
<td class="mono">0.6607</td><td><b>BenPom ahead</b></td></tr>
<tr><td>All joined (incl. one-off mixed events)</td><td class="mono">168</td>
<td class="mono">0.6621</td><td class="mono">0.6589</td><td>even (p=0.43)</td></tr>
</table>
<p class="dim">Knob re-sweeps on the complete data confirm plateaus (W20-24/L10-14 and
playoff ×1.4-1.8 all within 0.5m) — spec kept as tested, no re-tuning. Not yet rerun:
BuildIntlCalibration (intl events unchanged) and BuildPythData (cosmetic; picks up new
matches next run).</p>
</section>

<section>
<h2>Session 4: the missing tournaments — and the Kalshi rematch</h2>
<div class="callout warn"><b>Data discovery:</b> BenPom's dataset was missing <b>82 tier-1
matches</b> the market watched: the <b>EWC regional qualifiers</b> in all four regions
(May 16–Jun 1), <b>China Evolution Series Act 2</b>, a few genuine late Stage-1 playoff
matches, and the <b>Esports World Cup itself</b> (Jul 2–12, mid-break). All 82 were scraped
from VLR into a testing-only patch (82/82 found, winners cross-checked 82/82 — production
data untouched, backfill needs your go). These matches feed ratings exactly where the market
had its edge: entering London and entering Stage 2 after the break.</div>
<p><b>Methodology catch along the way:</b> smaller-event Kalshi markets often settle late, so
a "close-time minus 4h" anchor can land <i>post-match</i> and read the settled 0.99 as a
"pre-match" price — this was inflating the market's measured edge. All patched matches now
use real VLR start times (extracted from the scraped pages), and the comparison below is
anchored at start−5m everywhere.</p>
<table>
<tr><th>Surface (holdout, n=890)</th><th>Log-loss</th><th>vs production</th></tr>
<tr><td>Production</td><td class="mono">0.6502</td><td class="mono">—</td></tr>
<tr><td>v1 (constants only)</td><td class="mono">0.6448</td><td class="mono">+5.3m</td></tr>
<tr><td>v3 (games decay stack)</td><td class="mono">0.6398</td><td class="mono">+10.4m</td></tr>
<tr><td><b>v4 = v3 + data patch (EWC weight 1.0)</b></td><td class="mono good">0.6367</td>
<td class="mono good">+13.5m · p&gt;0.9999</td></tr>
</table>
<p style="margin-top:10px"><b>The Kalshi rematch</b> (pre-match market at start−5m,
winner-referenced):</p>
<table>
<tr><th>Sample</th><th>n</th><th>Production</th><th>v4</th><th>Market</th><th>Verdict</th></tr>
<tr><td>VCT matches (original overlap)</td><td class="mono">86</td><td class="mono">0.6805</td>
<td class="mono">0.6493</td><td class="mono">0.6457</td>
<td><b>parity</b> (p=0.42; production was decisively behind)</td></tr>
<tr><td>VCT Stage 2 (current window)</td><td class="mono">49</td><td class="mono">0.7126</td>
<td class="mono good">0.6504</td><td class="mono">0.6607</td>
<td><b>v4 ahead</b></td></tr>
<tr><td>All joined incl. one-off events</td><td class="mono">168</td><td class="mono">—</td>
<td class="mono">0.6704</td><td class="mono">0.6589</td>
<td>market +11.5m, not significant (CI spans 0)</td></tr>
</table>
<p>Remaining market edges are two one-off windows: the May mixed-tier qualifiers (top seeds
resting starters — outcome noise the market reads from lineup news) and London LAN (n=24).
A fixed 50/50 logit blend of v4 + market scores <b>0.6439</b> on the VCT-86 — better than
either alone; that's the sanctioned market-aware surface for display/sizing, never quoting.</p>
<p class="dim">Also tested and rejected this session: player-carryover priors (new signings
import their previous org's rating) — hurt at every fade horizon, overall and post-break; and
a post-break gap dampener — the fit actually wants post-break gaps <i>expanded</i> (γ=1.4),
confirming games-decay already handles breaks structurally.</p>
<div class="callout"><b>Needs your decision:</b> (1) backfill the 82 missing matches into
production data + add EWC-class events to the registry (the site and the bot are currently
blind to them); (2) promote v4 per the promotion flow. Both are staged and reversible;
nothing has been deployed.</div>
</section>

<section>
<h2>Candidate vs production — every bucket</h2>
<p>Positive = candidate better. No bucket regresses.</p>
<div class="chartbox tall"><canvas id="bk"></canvas></div>
</section>

<section>
<h2>Kalshi (2026 overlap, n={kc['n']})</h2>
<p>Pre-match anchors: real VLR start times for {kc['n_real_anchor']} matches, close−4h fallback
for {kc['n_fallback']}. BenPom LL {kc['benpom']['logloss']:.4f} vs market
{kc['kalshi_pre']['logloss']:.4f} (market better, p≈{kc['boot']['p_better']:.2f}).</p>
<table>
<tr><th>BenPom fav band</th><th>n</th><th>BenPom</th><th>Kalshi (same side)</th><th>Empirical</th></tr>
{fav_rows}
</table>
<p style="margin-top:10px"><b>Biggest pre-match divergences</b> (probability of the eventual winner):</p>
<table>
<tr><th>Date</th><th>Match</th><th>Stage</th><th>BenPom</th><th>Kalshi</th></tr>
{div_rows}
</table>
<div class="callout warn">Caveats: 86 matches ≈ 10 weeks of one season; books are thin
(median spread guarded); and since July the bot's own quotes are part of these prices
(echo risk). Treat as directional, re-measure monthly. The quoting fair value stays
book-independent regardless.</div>
</section>

<section>
<h2>Map-level surface (what the MC actually consumes)</h2>
<p>Walk-forward <b>map-level</b> log-loss on 2,287 played maps (2025–26), per-map ratings with
James-Stein shrinkage k toward the overall rating ("kov" = overall only, no per-map splits):</p>
<table>
<tr><th>Config</th><th>β (map)</th><th>Map log-loss</th></tr>
{ml_rows}
</table>
<div class="callout warn"><b>Two takeaways.</b> (1) The candidate wins at map level too
(0.6735 vs production-best 0.6778). (2) <b>Per-map deviations are mostly noise at k=5</b>: on
raw map outcomes the shrinkage optimum is k→∞, and production's SHRINK_K=5 is measurably
worse than overall-only.</div>
<p style="margin-top:12px"><b>Series-level veto-MC check</b> (full simulated surface — veto
sampling from decayed ban/pick rates + per-map Bernoullis; 2,000 sims/match, n=890):</p>
<table>
<tr><th>Config</th><th>Series MC log-loss</th></tr>
<tr><td class="mono">production (HL6, k=5, β .170)</td><td class="mono">{smc['prod_k5']['ll_series_mc']:.5f}</td></tr>
<tr><td class="mono">candidate (HL13, k=5)</td><td class="mono">{smc['cand_k5']['ll_series_mc']:.5f}</td></tr>
<tr><td class="mono">candidate (HL13, k=20)</td><td class="mono good">{smc['cand_k20']['ll_series_mc']:.5f}</td></tr>
<tr><td class="mono">candidate (overall-only maps)</td><td class="mono">{smc['cand_overall']['ll_series_mc']:.5f}</td></tr>
</table>
<div class="callout good">On the traded surface the full candidate package
(<b>HL13 + margin^0.75 + SHRINK_K 5→20</b>) gains <b>+8.6 milli-LL/series</b> over production —
larger than the closed-form gain, because HL-6 per-map spreads were double-injecting noise
into the MC. k=20 beats both k=5 and no-map-splits: a little per-map signal is real, k=5
over-trusts it. (2k-sim MC noise adds ≈0.5–1m here; the deployed 20k-sim surface loses less.)</div>
</section>

<section>
<h2>Where the market wins: the post-break window</h2>
<p><b>Definition — "post-break":</b> a match where at least one team is playing its first
series after <b>45+ days without an official match</b> — in this window, that's mostly teams
returning from the ~10-week gap between the end of Stage 1 (late May) and the start of Stage 2
(mid July), plus teams that missed an event. Rosters, comps, and form shift over the break, but
no official results exist yet — so BenPom's ratings are frozen on pre-break form while the
market prices in scrim reports, roster news, and expectations.</p>
<p>Candidate vs market on the same {cvk['n']} Kalshi matches, split on that definition:</p>
<table>
<tr><th></th><th>n</th><th>BenPom (prod)</th><th>BenPom (candidate)</th><th>Kalshi pre-match</th></tr>
<tr><td>Post-break matches</td><td class="mono">{cvk['post_break']['n']}</td>
<td class="mono">{cvk['post_break']['prod']:.4f}</td>
<td class="mono">{cvk['post_break']['cand']:.4f}</td>
<td class="mono good">{cvk['post_break']['kalshi']:.4f}</td></tr>
<tr><td>Normal weeks</td><td class="mono">{cvk['normal']['n']}</td>
<td class="mono">{cvk['normal']['prod']:.4f}</td>
<td class="mono">{cvk['normal']['cand']:.4f}</td>
<td class="mono">{cvk['normal']['kalshi']:.4f}</td></tr>
</table>
<div class="callout"><b>In normal weeks BenPom is near market parity</b> (0.66 vs 0.65).
The market's entire 2026 edge concentrates in the first matches after the long break —
off-season information (roster news, scrims, expectations) that isn't in any match data yet.
Two actions: (a) trading — quote wider/smaller in the 1–2 weeks after a long break;
(b) data — scrape announced roster moves during breaks (VLR team pages show transfers before
the first match is played), so the model can discount pre-break form for changed rosters
<i>before</i> their first result.</div>
<p style="margin-top:10px"><b>Divergence taxonomy</b> (12 largest, with context):</p>
<table>
<tr><th>Date</th><th>Match (winner first)</th><th>BenPom</th><th>Kalshi</th><th>Context</th></tr>
{tax_rows}
</table>
</section>

<section>
<h2>Veto predictor</h2>
<p>Walk-forward top-1 / top-3 accuracy on 5,322 decisions (2025–26), current-style raw decayed
rates. Shrinkage toward global map rates was tested at K=2/4/8 and <b>lost at every level</b>
(top-1 46.5% → 44.6% at K=8) — team veto habits are idiosyncratic.</p>
<table>
<tr><th>Step</th><th>n</th><th>Top-1</th><th>Top-3</th></tr>
{veto_rows}
</table>
</section>

<section>
<h2>The candidate (awaiting your go — nothing deployed)</h2>
<p><b>Primary: candidate v3</b> — changes to <code>scrapers/BuildMapRatings.py</code> +
prediction layer (PythonTest only; VCTMM untouched until vendor-sync after sign-off):</p>
<table>
<tr><th>Change</th><th>Current</th><th>Candidate v3</th></tr>
<tr><td>Decay clock</td><td class="mono">calendar weeks (HL 6.0)</td>
<td class="mono">games played — wins HL 20, losses HL 12 (per-side weight hook already exists
in massey_ratings)</td></tr>
<tr><td><code>RD_POWER</code></td><td class="mono">0.5</td><td class="mono">0.75</td></tr>
<tr><td>Playoff/GF solve weight</td><td class="mono">1.0</td><td class="mono">1.6</td></tr>
<tr><td>Cold-start orgs</td><td class="mono">enter at 0 (league avg)</td>
<td class="mono">region 25th percentile</td></tr>
<tr><td>Cross-region offsets</td><td class="mono">hand-tuned (intl_exp 0.40 / cn_dog 0.35)</td>
<td class="mono">monthly-refit fitted offsets (intl_exp refits to 0)</td></tr>
<tr><td>Veto pick score</td><td class="mono">(rate+0.02)·(0.3+own)²</td>
<td class="mono">× (1.75−opp_win) — top-1 40.1%→41.5%</td></tr>
<tr><td>β (refit after rebuild)</td><td class="mono">0.170</td><td class="mono">≈0.103 on the new scale</td></tr>
<tr><td><code>SHRINK_K</code> (per-map)</td><td class="mono">5</td>
<td class="mono">needs its own tuning pass under games decay (within-map memory shorter;
overall-only ≈ calendar k20 in the interim)</td></tr>
</table>
<p style="margin-top:8px"><b>Fallback: candidate v1</b> (constants-only, no code change):
<code>HALF_LIFE_WEEKS</code> 6→13, <code>RD_POWER</code> 0.5→0.75, <code>SHRINK_K</code> 5→20,
β≈0.116 — +5.3m closed-form / +8.6m on the MC surface.</p>
<div class="callout">Promotion checklist before this touches the site or (separately, later)
the bot: (1) rebuild ratings + timelines with the new constants through the real pipeline,
(2) re-fit β via the pipeline's own fit, (3) re-run this holdout scoring on pipeline output
(engine is a 0.989-corr replica, not bit-exact; CN shrinkage was held constant), (4) check
per-map ratings + veto MC surface (this round validated the overall-rating surface only),
(5) your sign-off, (6) only then vendor-sync to VCTMM.</div>
<p class="dim">Robustness: the gain is a plateau (HL 10–16w within 0.4m of each other, power-law
family equivalent), not a spike — low overfitting risk. Sensitivity to RD_POWER: 0.75 vs 1.0
within 0.1m; 0.5 (current) is the worst of the three.</p>
</section>

<section>
<h2>Open leads (next session)</h2>
<p>1) Map-level surface: re-validate candidate through per-map ratings + veto MC; SHRINK_K may
want retuning at HL=13. 2) Kalshi divergence taxonomy: the market's wins cluster on teams with
recent form shifts BenPom's 6w window over-reacted to — re-check under candidate. 3) CN offsets:
candidate + cluster-offset interaction (engine held CN shrinkage constant). 4) GF logit +0.25:
n=21 in holdout, untestable — leave. 5) Prospective snapshot logger so next month's eval is
free of reconstruction. 6) Cold-start matches (n=57, LL 0.70): seed new orgs from player
history instead of 0.</p>
</section>

</div>
<script>
const REL = {json.dumps({"labels": rel_labels, "pred": rel_pred, "emp": rel_emp,
                         "lo": rel_lo, "hi": rel_hi, "n": rel_n})};
new Chart(document.getElementById('rel'), {{
 type: 'line',
 data: {{ labels: REL.labels, datasets: [
  {{label:'Predicted', data:REL.pred, borderColor:'#9a93a6', borderDash:[6,4], pointRadius:0}},
  {{label:'Empirical', data:REL.emp, borderColor:'#7c4dd6', backgroundColor:'#7c4dd6',
    pointRadius:4}},
  {{label:'CI low', data:REL.lo, borderColor:'transparent', pointRadius:0, fill:false}},
  {{label:'CI high', data:REL.hi, borderColor:'transparent', pointRadius:0,
    fill:{{target:2, above:'#7c4dd61a'}}}}
 ]}},
 options: {{ plugins: {{ legend: {{ labels: {{ filter: i => i.text.indexOf('CI') < 0 }} }},
   tooltip: {{ callbacks: {{ afterBody: (items) => 'n=' + REL.n[items[0].dataIndex] }} }} }},
  scales: {{ y: {{ min: Math.max(0, Math.floor((Math.min.apply(null, REL.lo) - 0.05) * 10) / 10),
                   max: 1.0 }} }} }}
}});
const SL = {json.dumps({"prod": slopes_prod, "cand": [None, None, cand_s25, cand_s26]})};
new Chart(document.getElementById('slopes'), {{
 type: 'bar',
 data: {{ labels: ['2023','2024','2025','2026'], datasets: [
  {{label:'Production', data:SL.prod, backgroundColor:'#9a93a6'}},
  {{label:'Candidate', data:SL.cand, backgroundColor:'#7c4dd6'}}
 ]}},
 options: {{ scales: {{ y: {{ suggestedMin:0.6, suggestedMax:1.3,
   title:{{display:true,text:'calibration slope b'}} }} }},
  plugins: {{ annotation: undefined }} }}
}});
const HL = {json.dumps({"x": hl_x, "y": hl_y})};
new Chart(document.getElementById('hl'), {{
 type: 'line',
 data: {{ labels: HL.x.map(v => v + 'w'), datasets: [
  {{label:'Holdout log-loss', data:HL.y, borderColor:'#7c4dd6', pointRadius:4,
    pointBackgroundColor: HL.x.map(v => v === 6 ? '#c0392b' : '#7c4dd6')}}
 ]}},
 options: {{ plugins: {{ tooltip: {{ callbacks: {{
     title: items => 'half-life ' + items[0].label }} }} }},
  scales: {{ y: {{ title: {{ display:true, text:'log-loss (lower=better)' }} }} }} }}
}});
const BK = {json.dumps({"labels": bk_names, "delta": bk_delta})};
new Chart(document.getElementById('bk'), {{
 type: 'bar',
 data: {{ labels: BK.labels, datasets: [{{ label: 'ΔLL (milli, cand − prod)',
   data: BK.delta, backgroundColor: BK.delta.map(v => v >= 0 ? '#1e7a4f' : '#c0392b') }}] }},
 options: {{ indexAxis: 'y', maintainAspectRatio: false,
  plugins: {{ legend: {{ display:false }} }},
  scales: {{ x: {{ title: {{ display:true, text:'improvement (milli-LL/series)' }} }},
             y: {{ ticks: {{ autoSkip: false, font: {{ size: 11 }} }} }} }} }}
}});
</script>
</body></html>"""

with open(os.path.join(RD, "state_of_benpom.html"), "w") as f:
    f.write(html)
print(f"report written: {os.path.join(RD, 'state_of_benpom.html')} ({len(html)} bytes)")
