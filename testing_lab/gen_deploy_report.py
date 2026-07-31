"""Generate the full Bot Deployment Playbook page (reports/when_to_deploy.html,
also served at /testing/playbook). Expansive: historical evidence, complete
trade analysis with equity curve, market-behavior research, organized advice."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
RD = os.path.join(OUT, "reports")


def j(name):
    with open(os.path.join(OUT, name)) as f:
        return json.load(f)


roi = j("deploy_roi.json")
mic = j("deploy_micro.json")
eq = j("equity_curve.json")
mo = j("monthly_pnl.json")
rv = j("revalidate.json")
ts = j("trade_sim.json")
v5 = j("v5_native.json")
ea = j("edge_anatomy.json")

# heatmap grid: rows = divergence bands (top = biggest), cols = price bands
P_ORDER = ["<20c", "20-35c", "35-50c", "50-65c", ">65c"]
D_ORDER = ["10+", "5-10", "2-5", "0-2"]
cell_map = {(c["p"], c["d"]): c for c in ea["cells"]}


def heat_cell(p, d):
    c = cell_map.get((p, d), {"n": 0})
    if c["n"] < 6:
        return (f"<div class='hcell hna' title='n={c['n']} — too few'>"
                f"<span class='hroi'>—</span><span class='hn'>n={c['n']}</span></div>")
    r = c["roi"]
    # color scale: -50% -> red, 0 -> neutral, +60% -> green
    t = max(-1.0, min(1.0, r / 0.6))
    if t >= 0:
        bg = f"rgba(30,122,79,{0.12 + 0.55*t})"
        fg = "#0d3a24" if t < 0.7 else "#fff"
    else:
        bg = f"rgba(192,57,43,{0.12 + 0.55*(-t)})"
        fg = "#5a1610" if -t < 0.7 else "#fff"
    return (f"<div class='hcell' style='background:{bg};color:{fg}' "
            f"title='win rate {c['win']:.0%} vs implied {c['implied']:.0%} (n={c['n']})'>"
            f"<span class='hroi'>{r*100:+.0f}%</span><span class='hn'>n={c['n']}</span></div>")


heat_rows = ""
for d in D_ORDER:
    heat_rows += f"<div class='hlab'>{d} pts</div>"
    for p in P_ORDER:
        heat_rows += heat_cell(p, d)
heat_cols = "<div></div>" + "".join(f"<div class='hcol'>{p}</div>" for p in P_ORDER)

NAV = """<div class="labtabs">
<a href="/testing/">Testing Lab</a>
<a href="/testing/report/state_of_benpom">State of BenPom</a>
<a href="/testing/playbook" class="on">Deployment Playbook</a>
<a href="/testing/report/favorites_lab">Favorites Lab</a>
<a href="/testing/report/final_model">Final Model</a>
<a href="/testing/report/v7_lab">v7 Lab</a>
</div>"""


def row(label, key, note=""):
    r = roi.get(key)
    if not r:
        return ""
    strong = r["ci"][0] > 0
    cls = " class='good'" if strong else ""
    return (f"<tr><td>{label}</td><td class='mono'>{r['n']}</td>"
            f"<td class='mono'{cls}>{r['roi']*100:+.1f}%</td>"
            f"<td class='mono'>[{r['ci'][0]*100:+.0f}%, {r['ci'][1]*100:+.0f}%]</td>"
            f"<td class='dim'>{note}</td></tr>")


micro_rows = ""
for lab in ("48h-24h", "24h-12h", "12h-6h", "6h-2h", "2h-0h"):
    r = mic[lab]
    micro_rows += (f"<tr><td class='mono'>{lab}</td><td class='mono'>{r['spread_c']:.1f}¢</td>"
                   f"<td class='mono'>{r['vol_share_pct']:.1f}%</td>"
                   f"<td class='mono'>{r['drift_c']:.1f}¢</td></tr>")

mo_rows = ""
for month, r in mo.items():
    cls = " class='good'" if r["roi"] > 0 else " class='bad'"
    mo_rows += (f"<tr><td class='mono'>{month}</td><td class='mono'>{r['n']}</td>"
                f"<td class='mono'{cls}>{r['profit']:+.2f}</td>"
                f"<td class='mono'{cls}>{r['roi']*100:+.1f}%</td></tr>")

html = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex">
<title>Bot Deployment Playbook</title>
<link rel="icon" href="/logo.svg">
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@700;800&family=DM+Sans:wght@400;500;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.3/dist/chart.umd.min.js"></script>
<style>
 * {{ box-sizing:border-box; margin:0; }}
 :root {{ --ink:#16121d; --dim:#6b6478; --line:#eceef2; --acc:#7c4dd6; --accbg:#f3eefb;
         --good:#1e7a4f; --goodbg:#ecf8f1; --bad:#c0392b; --badbg:#fbeaea;
         --warn:#b3541e; --warnbg:#fdf3ec; }}
 body {{ font-family:'DM Sans',system-ui,sans-serif; background:#faf9fc; color:var(--ink);
        line-height:1.55; padding:30px 18px 90px; }}
 .wrap {{ max-width:900px; margin:0 auto; }}
 h1 {{ font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; font-size:1.6rem;
      text-align:center; margin:6px 0 2px; }}
 .tagline {{ text-align:center; color:var(--dim); font-size:.9rem; margin-bottom:18px; }}
 .labtabs {{ display:flex; justify-content:center; gap:6px; margin:0 0 24px; }}
 .labtabs a {{ font-size:.8rem; font-weight:700; color:var(--dim); text-decoration:none;
   padding:7px 15px; border-radius:999px; border:1px solid var(--line); background:#fff; }}
 .labtabs a:hover {{ color:var(--ink); background:var(--accbg); }}
 .labtabs a.on {{ color:#fff; background:var(--acc); border-color:var(--acc); }}
 section {{ background:#fff; border:1px solid var(--line); border-radius:18px;
           padding:24px 28px; margin-bottom:16px; box-shadow:0 3px 14px #00000008; }}
 h2 {{ font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; font-size:1.1rem;
      margin-bottom:10px; display:flex; align-items:center; gap:10px; }}
 h2 .n {{ background:var(--acc); color:#fff; border-radius:8px; font-size:.78rem; width:24px;
        height:24px; display:inline-flex; align-items:center; justify-content:center; flex-shrink:0; }}
 h3 {{ font-size:.95rem; font-weight:700; margin:14px 0 6px; }}
 p {{ font-size:.9rem; margin:7px 0; }}
 .dim {{ color:var(--dim); }} .good {{ color:var(--good); font-weight:700; }}
 .bad {{ color:var(--bad); font-weight:700; }}
 .mono {{ font-family:'JetBrains Mono',monospace; font-size:.82em; }}
 table {{ width:100%; border-collapse:collapse; font-size:.85rem; margin:8px 0; }}
 th {{ text-align:left; color:var(--dim); font-size:.7rem; text-transform:uppercase;
      letter-spacing:.5px; padding:6px 9px; border-bottom:2px solid var(--line); }}
 td {{ padding:7px 9px; border-bottom:1px solid var(--line); }}
 tr:last-child td {{ border-bottom:0; }}
 .callout {{ border-left:4px solid var(--acc); background:var(--accbg); border-radius:0 12px 12px 0;
            padding:12px 16px; margin:10px 0; font-size:.89rem; }}
 .callout.good {{ border-color:var(--good); background:var(--goodbg); }}
 .callout.warn {{ border-color:var(--warn); background:var(--warnbg); }}
 .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:12px; margin:10px 0; }}
 .card {{ border:1px solid var(--line); border-radius:14px; padding:13px 15px; }}
 .card .lbl {{ font-size:.68rem; font-weight:700; letter-spacing:.8px; text-transform:uppercase; color:var(--dim); }}
 .card .big {{ font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; font-size:1.4rem; margin:2px 0; }}
 .card .sub {{ font-size:.76rem; color:var(--dim); }}
 ol, ul {{ font-size:.9rem; margin:8px 0 8px 20px; }}
 li {{ margin:6px 0; }}
 canvas {{ max-height:320px; }}
 .chartbox {{ margin:12px 0 4px; }}
 .rule {{ background:var(--goodbg); border:1px solid #cde8da; border-radius:12px;
         padding:12px 16px; margin:8px 0; font-size:.9rem; }}
 .rule b {{ color:var(--good); }}
 .rule.no {{ background:var(--badbg); border-color:#f0d4d0; }}
 .rule.no b {{ color:var(--bad); }}
 .heat {{ display:grid; grid-template-columns:70px repeat(5,1fr); gap:5px; margin:12px 0; }}
 .hcol {{ font-size:.72rem; font-weight:700; color:var(--dim); text-align:center;
         text-transform:uppercase; letter-spacing:.4px; align-self:end; padding-bottom:2px; }}
 .hlab {{ font-size:.74rem; font-weight:700; color:var(--dim); align-self:center;
         text-align:right; padding-right:6px; }}
 .hcell {{ border-radius:10px; padding:10px 4px; text-align:center; min-height:56px;
          display:flex; flex-direction:column; justify-content:center; cursor:default; }}
 .hcell .hroi {{ font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; font-size:1.02rem; }}
 .hcell .hn {{ font-size:.66rem; opacity:.75; }}
 .hcell.hna {{ background:#f1f0f4; color:var(--dim); }}
 .axnote {{ display:flex; justify-content:space-between; font-size:.72rem; color:var(--dim);
           margin-top:2px; }}
 @media (max-width:640px) {{ section {{ padding:18px 14px; }} }}
</style></head>
<body><div class="wrap">
<h1>Bot Deployment Playbook</h1>
<div class="tagline">When and how VCTMM earns its highest expected ROI — full data analysis,
historical research, and operating rules · candidate v5 vs Kalshi pre-match prices ·
generated 2026-07-22</div>
{NAV}

<section>
<h2><span class="n">0</span>Executive summary</h2>
<div class="cards">
 <div class="card"><div class="lbl">Season record</div><div class="big good">+10.25u</div>
  <div class="sub">on 35.8u staked · 90 trades · 46W/44L · +28.7% ROI [CI +3%, +53%]</div></div>
 <div class="card"><div class="lbl">The edge pocket</div><div class="big">30–45¢ dogs</div>
  <div class="sub">model-backed sides at 30–45¢, 5–10pt divergence, regular season:
  +35–41%, CIs exclude zero</div></div>
 <div class="card"><div class="lbl">The avoid list</div><div class="big bad">LANs</div>
  <div class="sub">London was −12%; post-break weeks barely trigger; 15+pt gaps vs
  favorites = market knows</div></div>
 <div class="card"><div class="lbl">Quote timing</div><div class="big">early + 12h→2h</div>
  <div class="sub">3¢ spreads a day out (5% of volume) · 72% of volume in final 6h at 1¢</div></div>
</div>
<div class="callout good"><b>One-paragraph version:</b> deploy by default during regular-season
VCT windows and EWC-class events; concentrate size where the model backs a 25–45¢ side that the
market prices 5–10 points cheaper; treat very large divergences (15+ pts), LAN internationals,
and the first post-break week as information-risk zones and cut size there; keep quotes up from
listing (wide spreads, drifting prices) and scale through the 12h→2h window where the volume
actually arrives. Every claim below has its data attached.</div>
</section>

<section>
<h2><span class="n">1</span>Historical evidence: why the model has an edge at all</h2>
<p>The prerequisite for profitable quoting is a fair value at least as accurate as the market's
price, produced <b>independently</b> of it. Both halves are now measured:</p>
<table>
<tr><th>Milestone (2026 season, walk-forward)</th><th>BenPom LL</th><th>Market LL</th><th>Meaning</th></tr>
<tr><td>Production model, VCT overlap (n=86)</td><td class="mono">0.6805</td><td class="mono">0.6457</td>
<td class="bad">−35m behind — quoting this loses to the tape</td></tr>
<tr><td>Candidate (games decay + stack + EWC data)</td><td class="mono good">0.6441</td>
<td class="mono">0.6457</td><td class="good">ahead — parity or better</td></tr>
<tr><td>VCT Stage 2 window (n=49, most recent)</td><td class="mono good">0.6418</td>
<td class="mono">0.6607</td><td class="good">clearly ahead in the current era</td></tr>
</table>
<p>Independence: correlation between model and market probabilities is <b>{ts['corr']:.2f}</b>.
The model reads only match results; the market reads news, lineups, and sentiment. When two
equally-accurate but different signals disagree, the disagreement itself is the alpha — that's
what the rest of this page quantifies.</p>
<p class="dim">Provenance for the accuracy claims (8 optimization rounds, 1,687 walk-forward
series 2023–2026, every hypothesis' fate): see the State of BenPom report via the nav above.</p>
</section>

<section>
<h2><span class="n">2</span>The season ledger — every divergence trade</h2>
<p>Rules of the ledger: whenever model and pre-match market disagreed by &gt;5 points, buy one
contract of the model's side at the market price (taker — worst case; maker fills only improve
this). {roi['ALL']['n'] if 'ALL' in roi else 90} trades, May 16 – Jul 21.</p>
<div class="chartbox"><canvas id="equity"></canvas></div>
<table>
<tr><th>Month</th><th>Trades</th><th>P&amp;L (units)</th><th>ROI</th></tr>
{mo_rows}
</table>
<p><b>Expectancy anatomy:</b> 46 wins averaging +0.56u vs 44 losses averaging −0.35u — the
edge is payoff asymmetry, not hit rate. That is exactly the shape you want from buying
underpriced underdogs, and exactly the shape that dies if you only back favorites.</p>
<p class="dim">June's drawdown is Masters London — the LAN window where the market holds an
information advantage. It is the deliberate exception in the rules below, identified from the
data, not assumed.</p>
</section>

<section>
<h2><span class="n">3</span>Where the ROI concentrates (full slicing)</h2>
<div class="chartbox"><canvas id="roibars"></canvas></div>
<h3>By calendar window</h3>
<table>
<tr><th>Slice</th><th>n</th><th>ROI</th><th>95% CI</th><th></th></tr>
{row("May qualifiers (EWC quals + CES)", "window: May/quals", "one-off events, still positive")}
{row("Masters London (LAN)", "window: London", "the information-risk window")}
{row("Stage 2 + EWC (current era)", "window: Stage2+EWC", "the going-forward regime")}
{row("Regular rest (≤45d)", "normal rest", "the default deployment state")}
</table>
<h3>By region</h3>
<table>
<tr><th>Slice</th><th>n</th><th>ROI</th><th>95% CI</th><th></th></tr>
{row("Americas", "region: Americas", "strongest slice on the record")}
{row("CN", "region: CN", "EWC-qualifier data gave the model an information edge here")}
{row("EMEA", "region: EMEA", "")}
{row("Cross-region", "region: cross", "")}
{row("Pacific", "region: Pacific", "only negative region; overlaps the London losses")}
</table>
<h3>By price of the side bought</h3>
<table>
<tr><th>Slice</th><th>n</th><th>ROI</th><th>95% CI</th><th></th></tr>
{row("&lt;30¢ (deep dogs)", "buy price <30c", "huge variance — wins pay 3-6x")}
{row("30–45¢ (live dogs)", "buy price 30-45c", "the sweet spot — CI excludes zero")}
{row("45–55¢ (coin flips)", "buy price 45-55c", "")}
{row("&gt;55¢ (favorites)", "buy price >55c", "market prices favorites well — little edge")}
</table>
<h3>By size of the disagreement</h3>
<table>
<tr><th>Slice</th><th>n</th><th>ROI</th><th>95% CI</th><th></th></tr>
{row("5–10 pts", "div 5-10", "trust these most")}
{row("10–20 pts", "div 10-20", "mixed — includes market-information cases")}
</table>
<h3>By liquidity</h3>
<table>
<tr><th>Slice</th><th>n</th><th>ROI</th><th>95% CI</th><th></th></tr>
{row("High volume (≥ median)", "high volume", "edge survives real order flow")}
{row("Low volume", "low volume", "")}
</table>
<div class="callout"><b>The three findings that should drive sizing:</b> (1) favorites are
near-fair — the money is in dogs; (2) <b>moderate</b> disagreement beats extreme disagreement —
when the market is 15+ points away from the model, it too often has a reason (the four −100%
trades of May were all fades of surging teams the market had already re-priced); (3) liquidity
is not the enemy — the biggest ROI came where volume was highest.</div>
</section>

<section>
<h2><span class="n">4</span>Market behavior research — when to have quotes up</h2>
<p>Minute-candle microstructure across {mic['n_markets']} tier-1 markets, aligned to scheduled
start times:</p>
<table>
<tr><th>Hours before start</th><th>Median spread</th><th>Share of pre-match volume</th>
<th>Median |price move still to come|</th></tr>
{micro_rows}
</table>
<div class="chartbox"><canvas id="micro"></canvas></div>
<p><b>Reading:</b> a day out, the book is soft — 3¢ spreads and prices still a median 3¢ from
their eventual pre-match value. That is maker heaven (edge per fill), but almost nobody is
there (≈5% of volume). The crowd arrives in the last 6 hours (72% of volume) when spreads are
1¢ and prices nearly final. So the bot should <b>quote from listing with small size</b>
(capturing spread + favorable drift), <b>scale size through 12h→2h</b> (44% of volume, 1–1.5¢
of drift left), and let the standing start−2h expiry rule take quotes down.</p>
<h3>When the market is the smart one (stand down)</h3>
<ul>
<li><b>Post-break weeks:</b> the market prices off-season information (roster news, scrims)
the model cannot see. Notably, model-market divergences barely even trigger there — both
compress toward similar numbers — so exposure is naturally low; keep it that way.</li>
<li><b>LAN internationals:</b> London was the season's one negative window (−12%). On-site form
reads, stand-in news and prep intel concentrate in market prices.</li>
<li><b>15+ point gaps where the market is more confident in a favorite:</b> the May −100%
cluster (PRX, XLG fades) is this exact shape. The market had re-priced surging teams faster
than any results-based model could.</li>
</ul>
</section>

<section>
<h2><span class="n">5</span>Operating rules</h2>
<div class="rule"><b>RULE 1 — Default on.</b> Deploy across regular-season VCT windows (group
stages, playoffs) and EWC-class events. Every regular-window slice is ROI-positive.</div>
<div class="rule"><b>RULE 2 — Size the pocket.</b> Largest size where the model's side costs
25–45¢ and the divergence is 5–10 points. This pocket returned +35–41% with CIs above zero.</div>
<div class="rule"><b>RULE 3 — Respect favorites.</b> Above ~55¢ the market is near-fair (+6%).
Quote them for spread, not for edge.</div>
<div class="rule no"><b>RULE 4 — Information-risk zones = quarter size.</b> LAN internationals,
the first week after a 45+ day break, and any 15+ point divergence where the market is the more
confident side. If the model's inputs can't explain the gap, assume the market can.</div>
<div class="rule"><b>RULE 5 — Quote early, size late.</b> Quotes up at listing (3¢ spreads, 3¢
drift remaining, small size), scale through 12h→2h (44% of volume), expire at start−2h.</div>
<div class="rule"><b>RULE 6 — Let liquidity in.</b> No thin-market preference — the edge was
larger with volume, and maker fills are likelier.</div>
<div class="rule"><b>RULE 7 — Re-underwrite monthly.</b> The prediction logger accrues
prospective trades; regenerate this page as n grows. If the 30–45¢ pocket's CI ever includes
zero on 250+ trades, retire Rule 2 and re-slice.</div>
</section>

<section>
<h2><span class="n">6</span>Edge anatomy — pricing × divergence, and how they interact</h2>
<p>This section uses <b>all 168 matches</b> (not just the 5+pt trades), always from the
perspective of the model's preferred side: its market <b>price</b>, the model's
<b>divergence</b> above that price, and what happened.</p>

<h3>1 · Edge by pricing: where the model's sides beat their price</h3>
<p>Two lines — what the market price implied, and what actually happened. The vertical gap
IS the edge:</p>
<div class="chartbox"><canvas id="edgecurve"></canvas></div>
<div class="callout"><b>Read:</b> the model's sides outperform their price most in the
<b>35–45¢ band (priced 39%, won 51% — a 12-point edge)</b> and modestly at 15–35¢. At coin
flips (45–55¢) they <i>under</i>-perform (50% → 44%): when the model can't separate the teams,
its "preference" is noise. From 55¢ up the market is fair. Edge lives in live dogs,
not in favorites and not in toss-ups.</div>

<h3>2 · The interaction grid: ROI by price × divergence</h3>
<p>Columns = price of the model's side; rows = how far the model is above the market.
Green = profitable, red = losing, gray = too few matches (n&lt;6). Hover any cell for win
rate vs implied.</p>
<div class="heat">{heat_cols}{heat_rows}</div>
<div class="axnote"><span>&larr; cheaper sides</span><span>more expensive sides &rarr;</span></div>
<div class="callout good"><b>The three lessons in the grid:</b>
(1) <b>The 2–5pt row is red across every price</b> — small disagreements are model noise, and
trading them just pays the spread; the 5-point trade threshold isn't a convention, it's where
the data says edge switches on. (2) <b>The golden cells are 35–65¢ × 5–10pts</b>
(+51% ROI, winning ~62–88% vs ~42–55% implied) — divergence converts to profit best when the
side is competitively priced. (3) <b>At deep-dog prices (&lt;20¢) only huge divergences pay</b>
(+61% at 10+pts) and they hit only 25% of the time — real edge, lottery variance; size
accordingly.</div>

<h3>3 · Every match as a dot</h3>
<p>x = price of the model's side, y = divergence. Green dots won, red lost. The dashed
line is the 5-pt trade threshold — notice how much greener the picture is above it and to
the middle:</p>
<div class="chartbox"><canvas id="edgescatter"></canvas></div>
<p class="dim">Hover any dot for the match and the return. The cluster of big green dots
between 25–50¢ above the line is the strategy in one picture; the red dots at the top-left
(cheap side, giant divergence) are the "market knew something" fades.</p>
</section>

<section>
<h2><span class="n">9</span>The quoting margin — how much edge to demand (triple-tested)</h2>
<p>Maker-fill simulation over 310 markets / 155 events: rest a NO bid on both sides of every
event at (model NO value − margin), fill only if a real trade printed at/through the level
inside the bot's actual quoting window (listing → start−2h), settle at the result. Margin
families swept: flat cents 0–15, logit-space shifts, sqrt-price-scaled. Model = v6.</p>
<table>
<tr><th>Rule</th><th>Fills</th><th>Fill rate</th><th>Total profit (u)</th><th>ROI</th><th>95% CI</th></tr>
<tr><td class="mono">flat 1¢ (≈ current config)</td><td class="mono">221</td><td class="mono">71%</td>
<td class="mono" style="color:#c0392b">−0.68</td><td class="mono" style="color:#c0392b">−0.6%</td>
<td class="mono dim">[−13%, +12%]</td></tr>
<tr><td class="mono">flat 5¢</td><td class="mono">168</td><td class="mono">54%</td>
<td class="mono">+5.27</td><td class="mono">+6.7%</td><td class="mono dim">[−9%, +23%]</td></tr>
<tr><td class="mono">flat 11¢ (best flat)</td><td class="mono">100</td><td class="mono">32%</td>
<td class="mono">+8.07</td><td class="mono">+19.2%</td><td class="mono dim">[−4%, +44%]</td></tr>
<tr><td class="mono"><b>logit +0.6 (winner)</b></td><td class="mono">74</td><td class="mono">24%</td>
<td class="mono good"><b>+8.12</b></td><td class="mono good"><b>+29.1%</b></td>
<td class="mono good">[+1%, +61%] — only CI &gt; 0</td></tr>
<tr><td class="mono">logit +0.4</td><td class="mono">122</td><td class="mono">39%</td>
<td class="mono">+7.82</td><td class="mono">+15.0%</td><td class="mono dim">[−5%, +36%]</td></tr>
</table>
<p><b>Why logit-space wins:</b> a flat cent margin demands wildly different relative edge at
different prices (10¢ at a 15¢ book is enormous; at 50¢ it is modest). A logit shift
auto-scales: it demands the most cents exactly at coin flips — where the edge-anatomy grid
(§6) showed small divergences are worthless — and fewer at the extremes. At equal fill
counts, logit beats flat every time (122 fills: +7.82 vs +5.86).</p>
<h3>Verification battery (logit +0.6)</h3>
<table>
<tr><th>Check</th><th>Fills</th><th>Profit</th><th>ROI</th></tr>
<tr><td>First half (May 16 – Jun 19)</td><td class="mono">49</td><td class="mono">+1.95</td><td class="mono">+10.8%</td></tr>
<tr><td>Second half (Jun 20 – Jul 21)</td><td class="mono">25</td><td class="mono">+6.17</td><td class="mono">+62.7%</td></tr>
<tr><td>Conservative fills (trade must print 1¢ through)</td><td class="mono">61</td><td class="mono">+5.79</td><td class="mono">+26.1%</td></tr>
<tr><td>Excluding info-risk shapes + London (Playbook rules)</td><td class="mono">52</td><td class="mono good">+10.00</td><td class="mono good">+52.6%</td></tr>
<tr><td>Quote window extended to start−5m</td><td class="mono">79</td><td class="mono">+10.78</td><td class="mono">+36.9%</td></tr>
</table>
<p>Positive under every stress. Fill decomposition re-confirms §6 independently: NO bids
resting at 30–50¢ made +56.5% (n=34); NO bids against model-underdogs (bid ≥50¢) were the one
negative pocket (−5.7%, n=16) — consistent with "favorites near fair, dogs undervalued":
selling the dog side is the wrong side of that asymmetry. Optional refinement: skip NO quotes
on sides the model prices below ~45%.</p>
<div class="callout good"><b>Recommendation for the bot config:</b> replace
<span class="mono">min_edge_cents = 1</span> (which earned ≈ nothing on this history) with a
<b>logit-space edge of +0.5 to +0.6</b>, implemented as a per-price cent table:
≈14¢ at 50¢ · ≈12¢ at 65¢ · ≈10¢ at 75¢ · ≈8¢ at 85¢ · ≈5¢ at 92¢ (per side, applied to the
NO cap). The 0.4–0.6 range is a plateau — the exact value matters less than abandoning
flat-1¢.</div>
<p class="dim">Honest limits: fills inferred from 1-minute trade prints (no queue modeling, no
partial fills, assumes our size doesn't move these books); maker fees assumed zero; 10 weeks
of one season. The telemetry spec's markout logging is what turns this from simulation into
measurement.</p>
</section>

<section>
<h2><span class="n">7</span>Risks and honest limits</h2>
<ul>
<li><b>Sample size:</b> 90 trades, 10 weeks, one season. The headline CI is positive but wide;
slice orderings are directional. Rules 2/4's exact thresholds will move as n grows.</li>
<li><b>Regime change:</b> the market is young (listed May 2026) and sharpening — early-season
softness may fade. The microstructure table is the thing to re-measure first.</li>
<li><b>Echo effect:</b> as VCTMM's own quotes become a larger share of these books, measured
"market" prices partially reflect the model itself — future comparisons must use pre-deploy
snapshots (the logger handles this).</li>
<li><b>Model promotion pending:</b> these numbers use candidate v5. The live site/bot still run
production constants until promotion — deploy sizing decisions built on this page assume the
candidate is what's quoting.</li>
</ul>
</section>

<section>
<h2><span class="n">8</span>Telemetry spec — what the bot should record for future improvement</h2>
<p>Guiding principle: <b>capture what cannot be reconstructed later.</b> Prices, results and
candles can always be re-pulled from Kalshi/VLR; book depth, our own quote provenance, and
"what was knowable at the time" vanish if not logged live. Append-only tables, UTC
timestamps, never pruned (a season ≈ a few MB).</p>
<h3>Tier 1 — irreplaceable</h3>
<table>
<tr><th>Record</th><th>What</th><th>Why it matters later</th></tr>
<tr><td><b>Book snapshots</b></td><td>At T−24h/−12h/−6h/−2h/−30m/−5m vs scheduled start:
best bid/ask both sides, top-3 depth, last trade, cum. volume, <b>own orders flagged</b></td>
<td>Kalshi never publishes historical depth; and only the bot knows which liquidity was its
own. Solves clean benchmarking AND the echo-chamber problem (subtracting ourselves from
"the market") permanently.</td></tr>
<tr><td><b>FV provenance</b></td><td>Every quote decision: FV, model-version tag, ratings
snapshot time, rating gap, per-map ratings, veto distribution, β/offsets, MC sims/seed</td>
<td>Every future model promotion becomes an automatic A/B — accuracy changes attributable
to the model, not the calendar.</td></tr>
<tr><td><b>Quote lifecycle + markouts</b></td><td>Place/amend/cancel with book+FV at that
moment; fills with price, time-to-fill, and market mid at +5m/+30m/+2h after</td>
<td>Decomposes maker P&amp;L into spread capture vs adverse selection; identifies toxic
fills (picked off before line moves). The single best dataset for tuning quoting.</td></tr>
<tr><td><b>Deployment snapshots</b></td><td>Every deploy/undeploy of a market from the
dashboard: timestamp, minutes-to-scheduled-start, full book state, FV + provenance, sizing
config (max contracts, quote size, min edge), volume traded so far, context flags</td>
<td>Makes the operator's own deployment-timing decisions analyzable — e.g. "was arming
markets 24h out better than 4h out?" — and ties every episode's outcome to the exact
conditions at arm time.</td></tr>
</table>
<p class="dim"><b>Universal convention:</b> every row in every table carries BOTH
<span class="mono">ts_utc</span> and <span class="mono">mins_to_start</span> (plus the
<span class="mono">scheduled_start</span> it was computed against, since schedules shift) —
so every analysis can slice by match phase without joins or reconstruction.</p>
<h3>Tier 2 — closes known blind spots</h3>
<table>
<tr><th>Record</th><th>What</th><th>Why</th></tr>
<tr><td><b>Roster observations</b></td><td>Lineups seen at discovery time + weekly off-week
scrape of VLR team pages, timestamped when each change became public</td>
<td>The post-break window was the market's whole 2026 edge — announced-but-unplayed roster
changes. This log enables the roster-aware pre-break discount that couldn't be built
retroactively.</td></tr>
<tr><td><b>Timing truth</b></td><td>Listing time, scheduled start at listing, actual start,
settlement time vs match end, delays</td>
<td>Anchor correctness (bit us 3×) + tunes the start−2h expiry rule with real delay
statistics.</td></tr>
<tr><td><b>Context flags</b></td><td>LAN/online, stage, dead-rubber (seeding locked),
bracket rematch, days-since-last-match per team</td>
<td>"Was this match meaningful" is trivial to tag live, painful to reconstruct.</td></tr>
</table>
<h3>Tier 3 — cheap options on the future</h3>
<ul>
<li><b>Trade tape with aggressor side</b> from the WS feed (the public trades API lacks it) —
raw material for sharp-flow detection.</li>
<li><b>Veto auto-scoring</b>: predicted top-3 bans/picks vs the actual veto each match — the
veto model accrues its own prospective track record.</li>
<li><b>n_eff per team</b> alongside each FV — enables confidence-aware sizing when there's
enough history to test it.</li>
</ul>
<div class="callout warn">Implementation note: this is a spec to hand to the VCTMM side —
per the working rule, nothing here touches the live trading environment from the testing lab.
Suggested homes: new append-only tables in the bot's SQLite (<span class="mono">book_snaps,
fv_log, quote_events, fill_markouts, roster_obs, market_meta</span>).</div>
</section>
</div>
<script>
const EQ = {json.dumps(eq)};
new Chart(document.getElementById('equity'), {{
 type: 'line',
 data: {{ labels: EQ.dates, datasets: [{{ label: 'Cumulative P&L (units, 1 contract/trade)',
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
new Chart(document.getElementById('micro'), {{
 data: {{ labels: MIC.labels, datasets: [
  {{ type: 'bar', label: 'share of volume (%)', data: MIC.vol, backgroundColor: '#7c4dd655',
     yAxisID: 'y' }},
  {{ type: 'line', label: 'median spread (¢)', data: MIC.spread, borderColor: '#b3541e',
     pointRadius: 4, yAxisID: 'y2' }} ] }},
 options: {{ scales: {{ y: {{ position: 'left', title: {{ display: true, text: 'volume share %' }} }},
   y2: {{ position: 'right', grid: {{ drawOnChartArea: false }},
        title: {{ display: true, text: 'spread ¢' }} }} }} }}
}});
</script>
</body></html>"""

os.makedirs(RD, exist_ok=True)
with open(os.path.join(RD, "when_to_deploy.html"), "w") as f:
    f.write(html)
print(f"written: when_to_deploy.html ({len(html)} bytes)")
