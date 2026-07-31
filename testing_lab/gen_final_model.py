"""Generate reports/final_model.html — the development playbook for the final
model (v6): spec, journey, rejected-ideas ledger, operations manual."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
RD = os.path.join(OUT, "reports")

snap = json.load(open(os.path.join(HERE, "..", "trading_model", "model_snapshot.json")))
prof = json.load(open(os.path.join(OUT, "v6_profile.json")))
bk = sorted(prof["buckets"], key=lambda b: -b["delta"])
bk_names = [b["name"] for b in bk]
bk_delta = [b["delta"] for b in bk]
best5 = sorted(prof["buckets"], key=lambda b: b["v6"])[:5]
worst5 = sorted(prof["buckets"], key=lambda b: -b["v6"])[:5]
best_rows = "".join(f"<tr><td>{b['name']}</td><td class='mono'>{b['n']}</td>"
                    f"<td class='mono'>{b['v6']:.4f}</td></tr>" for b in best5)
worst_rows = "".join(f"<tr><td>{b['name']}</td><td class='mono'>{b['n']}</td>"
                     f"<td class='mono'>{b['v6']:.4f}</td></tr>" for b in worst5)
cal_rows = "".join(f"<tr><td class='mono'>{c['band']}</td><td class='mono'>{c['n']}</td>"
                   f"<td class='mono'>{c['pred']:.3f}</td><td class='mono'><b>{c['emp']:.3f}</b></td>"
                   f"<td class='mono dim'>[{c['ci'][0]:.2f},{c['ci'][1]:.2f}]</td></tr>"
                   for c in prof["bands"])
tm = prof["teams"]
under_rows = "".join(f"<tr><td>{t['team']}</td><td class='mono'>{t['n']}</td>"
                     f"<td class='mono' style='color:#1e7a4f'>{t['bias']*100:+.1f} pts</td></tr>"
                     for t in tm[:6])
over_rows = "".join(f"<tr><td>{t['team']}</td><td class='mono'>{t['n']}</td>"
                    f"<td class='mono' style='color:#c0392b'>{t['bias']*100:+.1f} pts</td></tr>"
                    for t in tm[-6:][::-1])

roi = json.load(open(os.path.join(OUT, "deploy_roi.json")))
mic = json.load(open(os.path.join(OUT, "deploy_micro.json")))
eq = json.load(open(os.path.join(OUT, "equity_curve.json")))
mo = json.load(open(os.path.join(OUT, "monthly_pnl.json")))
ea = json.load(open(os.path.join(OUT, "edge_anatomy.json")))
qm = json.load(open(os.path.join(OUT, "quote_margin.json")))

def roi_row(label, key, note=""):
    r = roi.get(key)
    if not r:
        return ""
    cls = " class='good'" if r["ci"][0] > 0 else ""
    return (f"<tr><td>{label}</td><td class='mono'>{r['n']}</td>"
            f"<td class='mono'{cls}>{r['roi']*100:+.1f}%</td>"
            f"<td class='mono'>[{r['ci'][0]*100:+.0f}%, {r['ci'][1]*100:+.0f}%]</td>"
            f"<td class='dim'>{note}</td></tr>")

micro_rows = "".join(
    f"<tr><td class='mono'>{lab}</td><td class='mono'>{mic[lab]['spread_c']:.1f}¢</td>"
    f"<td class='mono'>{mic[lab]['vol_share_pct']:.1f}%</td>"
    f"<td class='mono'>{mic[lab]['drift_c']:.1f}¢</td></tr>"
    for lab in ("48h-24h", "24h-12h", "12h-6h", "6h-2h", "2h-0h"))
mo_rows = "".join(
    f"<tr><td class='mono'>{m_}</td><td class='mono'>{r['n']}</td>"
    f"<td class='mono {'good' if r['roi']>0 else 'bad'}'>{r['profit']:+.2f}</td>"
    f"<td class='mono {'good' if r['roi']>0 else 'bad'}'>{r['roi']*100:+.1f}%</td></tr>"
    for m_, r in mo.items())

P_ORDER = ["<20c", "20-35c", "35-50c", "50-65c", ">65c"]
D_ORDER = ["10+", "5-10", "2-5", "0-2"]
cell_map = {(c["p"], c["d"]): c for c in ea["cells"]}

def heat_cell(p, d):
    c = cell_map.get((p, d), {"n": 0})
    if c["n"] < 6:
        return ("<div class='hcell hna'><span class='hroi'>—</span>"
                f"<span class='hn'>n={c['n']}</span></div>")
    r = c["roi"]
    t = max(-1.0, min(1.0, r / 0.6))
    if t >= 0:
        bg = f"rgba(30,122,79,{0.12 + 0.55*t})"
        fg = "#0d3a24" if t < 0.7 else "#fff"
    else:
        bg = f"rgba(192,57,43,{0.12 + 0.55*(-t)})"
        fg = "#5a1610" if -t < 0.7 else "#fff"
    return (f"<div class='hcell' style='background:{bg};color:{fg}' "
            f"title='win {c['win']:.0%} vs implied {c['implied']:.0%} (n={c['n']})'>"
            f"<span class='hroi'>{r*100:+.0f}%</span><span class='hn'>n={c['n']}</span></div>")

heat_rows = ""
for d in D_ORDER:
    heat_rows += f"<div class='hlab'>{d} pts</div>"
    for p in P_ORDER:
        heat_rows += heat_cell(p, d)
heat_cols = "<div></div>" + "".join(f"<div class='hcol'>{p}</div>" for p in P_ORDER)

qm_rows = ""
for k in ("flat_1c", "flat_5c", "flat_11c", "logit_0.6", "logit_0.4"):
    r = qm.get(k)
    if not r:
        continue
    lab = k.replace("flat_", "flat ").replace("logit_", "logit +")
    strong = r.get("ci") and r["ci"][0] is not None and r["ci"][0] > 0
    cls = " class='good'" if strong else ""
    ci = f"[{r['ci'][0]*100:+.0f}%, {r['ci'][1]*100:+.0f}%]" if r.get("ci") and r["ci"][0] is not None else "—"
    qm_rows += (f"<tr><td class='mono'>{lab}</td><td class='mono'>{r['fills']}</td>"
                f"<td class='mono'>{r['fill_rate']*100:.0f}%</td>"
                f"<td class='mono'{cls}>{r['profit']:+.2f}u</td>"
                f"<td class='mono'{cls}>{r['roi']*100:+.1f}%</td>"
                f"<td class='mono dim'>{ci}</td></tr>")

NAV = """<div class="labtabs">
<a href="/testing/">Testing Lab</a>
<a href="/testing/report/state_of_benpom">State of BenPom</a>
<a href="/testing/playbook">Deployment Playbook</a>
<a href="/testing/report/favorites_lab">Favorites Lab</a>
<a href="/testing/report/final_model" class="on">Final Model</a>
<a href="/testing/report/v7_lab">v7 Lab</a>
</div>"""

REJECTED = [
    ("Shorter calendar half-lives (4–5w)", "worse everywhere — memory was too short, not too long"),
    ("Power-law / box+exp / linear calendar decay", "all converge to 'longer memory'; none beat exponential-in-games"),
    ("Calendar envelope over games decay", "age in weeks adds nothing once games are counted"),
    ("Series-counted decay (age in series, not maps)", "a Bo5 teaches ~5 maps of information; map-counting is right"),
    ("Plain win/loss asymmetric decay (W20/L12)", "the v5 bug — erases floor teams' losses; replaced by consistency-conditioning"),
    ("Reversed asymmetry (losses persist)", "clearly worse; established the sign of the consistency effect"),
    ("Lineup-overlap roster reweighting (3 schemes)", "year-boundary continuity already covers real rebuilds"),
    ("Roster-instability prediction shrink", "changed-roster matches predict BETTER than average"),
    ("Player-carryover priors (new signings import prior-org rating)", "hurt at every fade horizon, incl. post-break"),
    ("Post-break gap dampener", "fit wants EXPANSION (γ=1.4) — games decay already handles breaks"),
    ("Favorite blowout-margin discount ('don't try hard')", "favorites' big margins are signal, not noise"),
    ("Residual margin (credit only surprise)", "neutral at best"),
    ("Upset margin boost", "noise"),
    ("Win-only ratings / margin-win blends in the solve", "margins matter; blends all negative"),
    ("Rolling / per-stage β refits", "trailing fits chase noise; frozen β wins"),
    ("Piecewise β (small vs large gaps)", "converges to a single slope"),
    ("tanh saturating link", "fit runs to no-op"),
    ("Convex link (extra slope beyond gap threshold)", "fits train, loses holdout — the anomaly was the ratings bug, not the link"),
    ("Margin-derived probability links (empirical CDF, probit)", "margins UNDERSTATE win prob (better teams win the close maps)"),
    ("Soft/mixed margin-label β fits, Rao-Blackwell retro targets", "all lose to binary fitting; ratings already consume margins"),
    ("Margin-fitted cross-region offsets", "−14m on cross-region test"),
    ("Margin-residual momentum feature", "redundant with games-decay ratings"),
    ("Decayed-winrate additive feature", "redundant once consistency decay fixed evidence accounting"),
    ("Veto rate shrinkage toward global", "team veto habits are idiosyncratic; raw rates win"),
    ("Per-map rating splits stacked with pick bonus", "double-count — picked maps ARE the good maps"),
    ("Closed-form × veto-MC ensemble", "MC adds no series-level information at any blend weight"),
    ("Intra-series momentum correlation in the MC", "hurts"),
    ("Intl-experience offset (+0.40)", "refits to zero on long-memory ratings"),
    ("Recency-weighted cross-region offsets", "unstable small-sample fits; flat cumulative fit wins"),
    ("Head-to-head / rematch term", "prior winner re-wins 53.2% vs 53.8% implied — ratings already carry it"),
    ("Bo1 down-weighting / intl margin reweighting", "no effect / marginal"),
    ("Rolling Platt / cubic recalibration layers", "walk-forward they chase noise; surface is calibrated"),
]

rej_rows = "".join(f"<tr><td>{a}</td><td class='dim'>{b}</td></tr>" for a, b in REJECTED)

journey = [
    ("Production (start)", "0.6526", "calendar HL 6w, sqrt margins, β=0.17",
     "baseline: −35m behind the Kalshi market on favorites-heavy 2026"),
    ("v1", "0.6448", "HL 6→13w calendar + margin^0.75",
     "the 'forgets too fast' discovery; survives only as constants-only fallback"),
    ("v3", "0.6398", "games-counted asym decay + playoff ×1.6 + cold-start priors + fitted x-region offsets",
     "decay by games played, not weeks; hand-tuned intl offsets retired"),
    ("v4", "0.6367*", "+ EWC-class events implemented in production data",
     "the market's post-break edge was missing tournaments, not modeling (*pre-native holdout)"),
    ("v5", "0.6413", "+ region-prior ridge 1.5 (native rebuilt data, n≈1000 holdout)",
     "teams regress to region context, not league average"),
    ("v6 (FINAL)", "0.6409", "asym → consistency-conditioned decay (W-consistent 20 / anomaly 12)",
     "operator-caught floor-team bug fixed; every elite-vs-floor gap corrected 2–7pts"),
]
journey_rows = "".join(
    f"<tr><td><b>{v}</b></td><td class='mono'>{ll}</td><td>{ch}</td><td class='dim'>{why}</td></tr>"
    for v, ll, ch, why in journey)

html = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex">
<title>Final Model — Development Playbook (BenPom v6)</title>
<link rel="icon" href="/logo.svg">
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@700;800&family=DM+Sans:wght@400;500;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.3/dist/chart.umd.min.js"></script>
<style>
 * {{ box-sizing:border-box; margin:0; }}
 :root {{ --ink:#16121d; --dim:#6b6478; --line:#eceef2; --acc:#7c4dd6; --accbg:#f3eefb;
         --good:#1e7a4f; --goodbg:#ecf8f1; --bad:#c0392b; --warn:#b3541e; --warnbg:#fdf3ec; }}
 body {{ font-family:'DM Sans',system-ui,sans-serif; background:#faf9fc; color:var(--ink);
        line-height:1.55; padding:30px 18px 90px; }}
 .wrap {{ max-width:900px; margin:0 auto; }}
 h1 {{ font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; font-size:1.55rem;
      text-align:center; margin:6px 0 2px; }}
 .tagline {{ text-align:center; color:var(--dim); font-size:.9rem; margin-bottom:18px; }}
 .labtabs {{ display:flex; justify-content:center; gap:6px; margin:0 0 24px; flex-wrap:wrap; }}
 .labtabs a {{ font-size:.8rem; font-weight:700; color:var(--dim); text-decoration:none;
   padding:7px 15px; border-radius:999px; border:1px solid var(--line); background:#fff; }}
 .labtabs a:hover {{ color:var(--ink); background:var(--accbg); }}
 .labtabs a.on {{ color:#fff; background:var(--acc); border-color:var(--acc); }}
 section {{ background:#fff; border:1px solid var(--line); border-radius:18px;
           padding:24px 28px; margin-bottom:16px; box-shadow:0 3px 14px #00000008; }}
 h2 {{ font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; font-size:1.08rem;
      margin-bottom:10px; display:flex; align-items:center; gap:10px; }}
 h2 .n {{ background:var(--acc); color:#fff; border-radius:8px; font-size:.78rem; width:24px;
        height:24px; display:inline-flex; align-items:center; justify-content:center; flex-shrink:0; }}
 h3 {{ font-size:.94rem; font-weight:700; margin:13px 0 5px; }}
 p {{ font-size:.9rem; margin:7px 0; }}
 .dim {{ color:var(--dim); }}
 .mono {{ font-family:'JetBrains Mono',monospace; font-size:.82em; }}
 table {{ width:100%; border-collapse:collapse; font-size:.84rem; margin:8px 0; }}
 th {{ text-align:left; color:var(--dim); font-size:.7rem; text-transform:uppercase;
      letter-spacing:.5px; padding:6px 9px; border-bottom:2px solid var(--line); }}
 td {{ padding:6px 9px; border-bottom:1px solid var(--line); vertical-align:top; }}
 tr:last-child td {{ border-bottom:0; }}
 .callout {{ border-left:4px solid var(--acc); background:var(--accbg); border-radius:0 12px 12px 0;
            padding:12px 16px; margin:10px 0; font-size:.89rem; }}
 .callout.good {{ border-color:var(--good); background:var(--goodbg); }}
 .callout.warn {{ border-color:var(--warn); background:var(--warnbg); }}
 canvas {{ max-height:340px; }}
 .chartbox.tall {{ height:560px; }} .chartbox.tall canvas {{ max-height:none; height:100% !important; }}
 .good {{ color:var(--good); font-weight:700; }} .bad {{ color:var(--bad); font-weight:700; }}
 .heat {{ display:grid; grid-template-columns:70px repeat(5,1fr); gap:5px; margin:12px 0; }}
 .hcol {{ font-size:.72rem; font-weight:700; color:var(--dim); text-align:center;
         text-transform:uppercase; letter-spacing:.4px; align-self:end; padding-bottom:2px; }}
 .hlab {{ font-size:.74rem; font-weight:700; color:var(--dim); align-self:center;
         text-align:right; padding-right:6px; }}
 .hcell {{ border-radius:10px; padding:10px 4px; text-align:center; min-height:56px;
          display:flex; flex-direction:column; justify-content:center; }}
 .hcell .hroi {{ font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; font-size:1.02rem; }}
 .hcell .hn {{ font-size:.66rem; opacity:.75; }}
 .hcell.hna {{ background:#f1f0f4; color:var(--dim); }}
 .chartbox {{ height:560px; margin:12px 0 4px; }}
 .chartbox.short {{ height:300px; }}
 code {{ font-family:'JetBrains Mono',monospace; font-size:.8em; background:#f4f2f8;
        border:1px solid var(--line); border-radius:6px; padding:1px 6px; }}
 ol, ul {{ font-size:.9rem; margin:8px 0 8px 20px; }} li {{ margin:5px 0; }}
</style></head>
<body><div class="wrap">
<h1>Final Model — Development Playbook</h1>
<div class="tagline">BenPom v6 · the private trading model: complete spec, how it got here,
what was ruled out, and how to maintain it · generated 2026-07-22</div>
{NAV}

<section>
<h2><span class="n">0</span>The final model at a glance</h2>
<table>
<tr><th>Component</th><th>Value</th><th>Why it exists</th></tr>
<tr><td><b>Rating solve</b></td><td class="mono">Massey, target = sign(rd)·|rd|^0.75 · 2.5</td>
<td>margins carry the signal; ^0.75 beat sqrt and raw in every grid</td></tr>
<tr><td><b>Decay</b></td><td class="mono">games-counted, consistency-conditioned:
HL 20 (result matches team level) / HL 12 (anomaly)</td>
<td>information arrives per game, not per week (break-proof); consistency
conditioning keeps elite wins AND floor losses as signal</td></tr>
<tr><td><b>Solve weights</b></td><td class="mono">playoffs/GF ×1.6 · Champions ×2 ·
ridge 0.5 · region-prior ridge 1.5</td>
<td>high-stakes games more informative; teams regress to region context</td></tr>
<tr><td><b>Roster</b></td><td class="mono">year-boundary continuity, persistence 0.3</td>
<td>covers real rebuilds; every finer-grained scheme tested worse</td></tr>
<tr><td><b>Cold start</b></td><td class="mono">region 25th percentile</td>
<td>new orgs are below-average entrants, not league-average</td></tr>
<tr><td><b>Cross-region</b></td><td class="mono">fitted additive offsets, CN pinned 0,
refit each snapshot</td><td>replaces hand-tuned intl_exp/cn_dog entirely</td></tr>
<tr><td><b>Link</b></td><td class="mono">logistic, β = {snap['beta']} (snapshot);
canonical full-stack refit 0.1256 — reconcile at next rebuild</td>
<td>five tail-modification families tested; the link was never the defect</td></tr>
<tr><td><b>Series aggregation</b></td><td class="mono">closed form (bo1/bo3/bo5);
GF: +0.25 logit to upper bracket</td>
<td>the veto-MC adds no series-level information (tested at every blend)</td></tr>
<tr><td><b>Map-level (optional)</b></td><td class="mono">overall rating + pick bonus
b_pick = {snap['b_pick']} (refit each snapshot)</td>
<td>picker advantage is real and growing; per-map splits double-count it</td></tr>
</table>
<div class="callout good"><b>Validation:</b> holdout 2025–26 log-loss <b>0.6409</b> vs
production 0.6526 (+11.8 milli); at/above Kalshi pre-match parity on VCT matches; +29–31%
simulated ROI on divergences and maker fills (logit+0.6 margin). Deployed as
<code>{snap['model_version']}</code> in <code>trading_model/model_snapshot.json</code> —
the bot reads it live. <b>Never deployed to bobo-gg</b> (operator decision: the site keeps
the public production model).</div>
</section>

<section>
<h2><span class="n">1</span>Where it does best — and worst</h2>
<p>Everything below is the final model on the 2025–26 native holdout (~1,000 series),
walk-forward.</p>
<h3>Improvement over production, every bucket (milli-LL; positive = v6 better)</h3>
<div class="chartbox tall"><canvas id="bk6"></canvas></div>
<p class="dim">v6 improves 21 of 23 buckets. The two negatives: domestic Pacific (−3.1m,
production got lucky on a coin-flip-heavy slate) and the tiny elite-vs-floor bucket (−5.1m,
n=26 — see the team-bias table below for the real story there).</p>
<h3>Easiest and hardest match types (absolute v6 log-loss)</h3>
<table><tr><th colspan="3">BEST — most predictable</th></tr>
<tr><th>Bucket</th><th>n</th><th>v6 LL</th></tr>
{best_rows}
</table>
<table><tr><th colspan="3">WORST — least predictable</th></tr>
<tr><th>Bucket</th><th>n</th><th>v6 LL</th></tr>
{worst_rows}
</table>
<p class="dim">Reading: huge rating gaps and grand finals are where the model is sharpest;
EWC-class one-off events, coin-flip matchups, and playoffs are inherently the noisiest —
size accordingly (this ranking drives the Playbook's sizing rules).</p>
<h3>Self-calibration (favorite bands)</h3>
<table>
<tr><th>Band</th><th>n</th><th>Predicted</th><th>Won</th><th>95% CI</th></tr>
{cal_rows}
</table>
<h3>Per-team bias — the honest watchlist</h3>
<p>Mean predicted probability minus actual win rate per team (holdout, ≥25 matches).
Negative = the model UNDER-rates the team:</p>
<table><tr><th colspan="3">Most under-rated</th></tr><tr><th>Team</th><th>n</th><th>Bias</th></tr>
{under_rows}
</table>
<table><tr><th colspan="3">Most over-rated</th></tr><tr><th>Team</th><th>n</th><th>Bias</th></tr>
{over_rows}
</table>
<div class="callout warn"><b>The residual the operator flagged is visible here:</b> even
after the v6 fix, the elite tier (T1, PRX, 100T, NRG, TL) still runs ~6–8 points under-rated
at the team level, while several mid/floor teams run over-rated — band calibration hides it
because the biases cancel within bands. This is the #1 target for v7 (candidate mechanisms:
per-team evidence-quality weighting, opponent-adjusted consistency thresholds), and until
then it is a quoting note: lean toward the model's number MINUS caution when it fades T1/PRX-
class teams, and don't defend over-rated mid teams at full size.</div>
</section>

<section>
<h2><span class="n">2</span>Trading record &amp; edge anatomy — the full chart set</h2>

<h3>The season ledger — equity curve (90 divergence trades, taker-priced)</h3>
<div class="chartbox"><canvas id="equity"></canvas></div>
<table>
<tr><th>Month</th><th>Trades</th><th>P&amp;L (units)</th><th>ROI</th></tr>
{mo_rows}
</table>

<h3>Where the ROI concentrates</h3>
<div class="chartbox"><canvas id="roibars"></canvas></div>
<table>
<tr><th>Slice</th><th>n</th><th>ROI</th><th>95% CI</th><th></th></tr>
{roi_row("Everything, no filter", "ALL")}
{roi_row("Regular rest (≤45d)", "normal rest", "the default deployment state")}
{roi_row("Buy price 30–45¢", "buy price 30-45c", "the sweet spot")}
{roi_row("Divergence 5–10 pts", "div 5-10", "trust these most")}
{roi_row("Divergence 10–20 pts", "div 10-20", "market-information cases mixed in")}
{roi_row("High volume (≥ median)", "high volume", "edge survives order flow")}
{roi_row("Buy price >55¢ (favorites)", "buy price >55c", "market prices favorites well")}
{roi_row("Masters London (LAN)", "window: London", "the information-risk window")}
</table>

<h3>Market microstructure — when to have quotes up</h3>
<div class="chartbox"><canvas id="micro"></canvas></div>
<table>
<tr><th>Hours before start</th><th>Median spread</th><th>Share of volume</th><th>|Move still to come|</th></tr>
{micro_rows}
</table>

<h3>Edge by pricing — where the model's sides beat their price</h3>
<div class="chartbox"><canvas id="edgecurve"></canvas></div>

<h3>The interaction grid — ROI by price × divergence</h3>
<div class="heat">{heat_cols}{heat_rows}</div>

<h3>Every match as a dot</h3>
<div class="chartbox"><canvas id="edgescatter"></canvas></div>
<p class="dim">Green won (sized by return), red lost; dashed line = the 5-pt trade threshold.
The green cluster at 25–50¢ above the line is the strategy in one picture.</p>

<h3>The quoting margin (maker-fill simulation, 310 markets)</h3>
<table>
<tr><th>Rule</th><th>Fills</th><th>Fill rate</th><th>Profit</th><th>ROI</th><th>95% CI</th></tr>
{qm_rows}
</table>
<p class="dim">logit+0.6 is the winner (only CI above zero); flat 1¢ — the old config —
earned nothing. Cents equivalent: ≈14¢ @ 50¢ → ≈5¢ @ 92¢. Full verification battery in
Deployment Playbook §9.</p>
</section>

<section>
<h2><span class="n">3</span>How it got here — six models in one day of science</h2>
<table>
<tr><th>Version</th><th>Holdout LL</th><th>Change</th><th>The insight</th></tr>
{journey_rows}
</table>
<p class="dim">Log-losses across rows are comparable only within the same data era (v4
onward includes the EWC-class events in both train and holdout). The arc: fix the memory
clock → fix the information set → fix the evidence accounting.</p>
</section>

<section>
<h2><span class="n">4</span>The rejected-ideas ledger — do not re-test without new cause</h2>
<p>Every idea below was implemented, run walk-forward, and lost (or tied) against the
champion of its day. This table is half the project's value: it prevents re-litigating.</p>
<table>
<tr><th>Idea</th><th>Why it died</th></tr>
{rej_rows}
</table>
<div class="callout warn"><b>Standing re-open triggers:</b> a rejected idea may be re-tested
only if (a) the model's own [0.8,0.9) self-band drifts out of calibration as n grows,
(b) a new data source arrives (roster-news timestamps, telemetry markouts), or
(c) an operator anomaly report implicates it — anomaly reports get case-level forensics
first, statistics second (the v6 lesson).</div>
</section>

<section>
<h2><span class="n">5</span>Operations manual</h2>
<h3>Rebuild cadence</h3>
<p><code>python3 trading_model/build_model_snapshot.py</code> after every data refresh
(site scrape). Everything inside — ratings, region priors, cross-region offsets, pick
bonus — refits walk-forward-consistently. The bot applies its own staleness rules against
<code>generated_utc</code> / <code>ratings_as_of</code>.</p>
<h3>β discipline</h3>
<p>β is scale-bound to the exact solve config. If ANY constant changes, refit β on
pre-holdout data before quoting. Current snapshot carries 0.1299 (fit on the core stack);
the canonical full-stack refit is 0.1256 (≤0.5pt difference) — adopt at next rebuild.</p>
<h3>Monitoring tripwires (check monthly)</h3>
<ul>
<li><b>Self-band calibration</b>: model-referenced [0.7,0.8) and [0.8,0.9) bands on
accumulating results — drift outside CIs = investigate.</li>
<li><b>Kalshi re-measure</b>: market-band table (Favorites Lab §3) and divergence-trade
ledger with fresh settles; the market-favorite-bias claim is provisional pending v6-era data.</li>
<li><b>Prediction logger</b> (<code>testing_lab/log_predictions.py</code>): keep it running —
prospective evidence needs no reconstruction arguments.</li>
<li><b>Telemetry spec</b> (Deployment Playbook §8): once the bot logs book snapshots with
own-order flags + fill markouts, quoting-margin and adverse-selection tuning move from
simulation to measurement.</li>
</ul>
<h3>Change control</h3>
<ul>
<li>All experiments in <code>PythonTest/testing_lab/</code>; VCTMM is hands-off.</li>
<li>Model changes reach the bot ONLY via <code>trading_model/model_snapshot.json</code>
rebuilds; <code>predict.py</code> is the reference math — parity-test any reimplementation.</li>
<li>The website never gets this model (operator decision, standing).</li>
<li>Promotion bar for any future v7: beats v6 on walk-forward holdout log-loss with
paired-bootstrap support, no major bucket regression, AND survives case-level review of
operator-flagged matchups.</li>
</ul>
<h3>Trading parameters (from the research pages)</h3>
<ul>
<li><b>Quoting margin</b>: logit-space +0.5 to +0.6 (≈14¢ @ 50¢ → ≈5¢ @ 92¢), not flat
cents; skip NO quotes on sides the model prices below ~45%. (Favorites Lab §7 /
Deployment Playbook §9.)</li>
<li><b>Windows &amp; sizing</b>: Deployment Playbook rules 1–7 (regular-season default,
quarter-size info-risk shapes, quote early / size 12h→2h, expire start−2h).</li>
</ul>
</section>

<section>
<h2><span class="n">6</span>Open leads for v7</h2>
<ul>
<li><b>Roster-news timestamps</b> — weekly off-week scrapes of VLR team pages (telemetry
spec) to price announced-but-unplayed roster changes; the market's one structural edge.</li>
<li><b>Per-map layer under consistency decay</b> — within-map memory wants its own
half-life; currently overall-only + pick bonus is the validated surface.</li>
<li><b>Markout-driven quoting</b> — once fill markouts accumulate, tune margin and expiry
from measured adverse selection instead of candle simulation.</li>
<li><b>EWC-class recurrence</b> — the registry pattern (ratings_only + vct_only) is ready
for future off-season events; add them the week they're announced, not after.</li>
<li><b>Market-blend surface</b> — sanctioned for display/sizing only; revisit when the
prospective ledger reaches ~250 VCT matches.</li>
</ul>
</section>
</div>
<script>
const EQ = {json.dumps(eq)};
new Chart(document.getElementById('equity'), {{
 type: 'line',
 data: {{ labels: EQ.dates, datasets: [{{ label: 'Cumulative P&L (units)',
   data: EQ.cum, borderColor: '#1e7a4f', backgroundColor: '#1e7a4f22', fill: true,
   pointRadius: 0, borderWidth: 2, tension: .15 }}] }},
 options: {{ plugins: {{ tooltip: {{ callbacks: {{
     afterTitle: items => EQ.labels[items[0].dataIndex] }} }} }},
  scales: {{ x: {{ ticks: {{ maxTicksLimit: 8 }} }},
            y: {{ title: {{ display: true, text: 'units' }} }} }} }}
}});
const ROIB = {json.dumps({
  "labels": ["ALL", "regular rest", "30-45c", "div 5-10", "high vol", "Americas", "CN",
              "favorites >55c", "div 10-20", "London"],
  "vals": [roi.get(k, {}).get("roi", 0) * 100 for k in
           ("ALL", "normal rest", "buy price 30-45c", "div 5-10", "high volume",
            "region: Americas", "region: CN", "buy price >55c", "div 10-20",
            "window: London")]})};
new Chart(document.getElementById('roibars'), {{
 type: 'bar',
 data: {{ labels: ROIB.labels, datasets: [{{ label: 'ROI %', data: ROIB.vals,
   backgroundColor: ROIB.vals.map(v => v >= 20 ? '#1e7a4f' : v >= 0 ? '#9ec7ae' : '#c0392b') }}] }},
 options: {{ plugins: {{ legend: {{ display: false }} }},
  scales: {{ y: {{ title: {{ display: true, text: 'ROI %' }} }} }} }}
}});
const MIC = {json.dumps({
  "labels": ["48-24h", "24-12h", "12-6h", "6-2h", "2-0h"],
  "spread": [mic[k]["spread_c"] for k in ("48h-24h", "24h-12h", "12h-6h", "6h-2h", "2h-0h")],
  "vol": [mic[k]["vol_share_pct"] for k in ("48h-24h", "24h-12h", "12h-6h", "6h-2h", "2h-0h")]})};
new Chart(document.getElementById('micro'), {{
 data: {{ labels: MIC.labels, datasets: [
  {{ type: 'bar', label: 'share of volume (%)', data: MIC.vol, backgroundColor: '#7c4dd655', yAxisID: 'y' }},
  {{ type: 'line', label: 'median spread (¢)', data: MIC.spread, borderColor: '#b3541e',
     pointRadius: 4, yAxisID: 'y2' }} ] }},
 options: {{ scales: {{ y: {{ position: 'left', title: {{ display: true, text: 'volume share %' }} }},
   y2: {{ position: 'right', grid: {{ drawOnChartArea: false }},
        title: {{ display: true, text: 'spread ¢' }} }} }} }}
}});
const EA = {json.dumps({"curve": ea["curve"], "scatter": ea["scatter"]})};
new Chart(document.getElementById('edgecurve'), {{
 type: 'line',
 data: {{ labels: EA.curve.map(c => c.lab), datasets: [
  {{ label: 'market-implied win rate', data: EA.curve.map(c => c.implied),
     borderColor: '#9a93a6', borderDash: [6,4], pointRadius: 3 }},
  {{ label: 'realized win rate (model sides)', data: EA.curve.map(c => c.realized),
     borderColor: '#7c4dd6', backgroundColor: '#7c4dd6', pointRadius: 5,
     fill: {{ target: 0, above: 'rgba(30,122,79,.15)', below: 'rgba(192,57,43,.15)' }} }} ] }},
 options: {{ plugins: {{ tooltip: {{ callbacks: {{
     afterBody: items => 'n=' + EA.curve[items[0].dataIndex].n }} }} }},
  scales: {{ y: {{ min: 0, max: 1, title: {{ display: true, text: 'win probability' }} }} }} }}
}});
new Chart(document.getElementById('edgescatter'), {{
 type: 'scatter',
 data: {{ datasets: [
  {{ label: 'won', data: EA.scatter.filter(p => p.won),
     backgroundColor: 'rgba(30,122,79,.75)',
     pointRadius: EA.scatter.filter(p => p.won).map(p => 3 + Math.min(9, Math.sqrt(Math.abs(p.ret)) * 3)) }},
  {{ label: 'lost', data: EA.scatter.filter(p => !p.won),
     backgroundColor: 'rgba(192,57,43,.65)', pointRadius: 4 }},
  {{ label: '5-pt trade threshold', type: 'line', data: [{{x:0.02,y:0.05}},{{x:0.98,y:0.05}}],
     borderColor: '#6b6478', borderDash: [7,5], pointRadius: 0, borderWidth: 1.5 }} ] }},
 options: {{ plugins: {{ tooltip: {{ callbacks: {{
     label: ctx => {{ const p = ctx.raw; return p.lab ? p.lab + ' — price ' +
       Math.round(p.x*100) + 'c, div ' + Math.round(p.y*100) + 'pts, ret ' +
       (p.ret>0?'+':'') + Math.round(p.ret*100) + '%' : ''; }} }} }} }},
  scales: {{ x: {{ min: 0, max: 1, title: {{ display: true, text: "price of the model's side (¢)" }},
              ticks: {{ callback: v => Math.round(v*100) + 'c' }} }},
            y: {{ min: 0, title: {{ display: true, text: 'divergence (model − market)' }},
              ticks: {{ callback: v => Math.round(v*100) + 'pts' }} }} }} }}
}});
const BK6 = {json.dumps({"labels": bk_names, "delta": bk_delta})};
new Chart(document.getElementById('bk6'), {{
 type: 'bar',
 data: {{ labels: BK6.labels, datasets: [{{ label: 'v6 − production (milli-LL)',
   data: BK6.delta,
   backgroundColor: BK6.delta.map(v => v >= 0 ? '#1e7a4f' : '#c0392b') }}] }},
 options: {{ indexAxis: 'y', maintainAspectRatio: false,
  plugins: {{ legend: {{ display: false }} }},
  scales: {{ x: {{ title: {{ display: true, text: 'improvement (milli-LL/series)' }} }},
            y: {{ ticks: {{ autoSkip: false, font: {{ size: 11 }} }} }} }} }}
}});
</script>
</body></html>"""

os.makedirs(RD, exist_ok=True)
with open(os.path.join(RD, "final_model.html"), "w") as f:
    f.write(html)
print(f"written: final_model.html ({len(html)} bytes)")
