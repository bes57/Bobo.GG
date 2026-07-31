"""Generate reports/favorites_lab.html — deep research on the favorite-pricing
gap between the model and the market."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
RD = os.path.join(OUT, "reports")


def j(name):
    with open(os.path.join(OUT, name)) as f:
        return json.load(f)


slate = j("slate_favorites.json")
cal = j("favorites_calib.json")
fix = j("favorites_fix.json")

NAV = """<div class="labtabs">
<a href="/testing/">Testing Lab</a>
<a href="/testing/report/state_of_benpom">State of BenPom</a>
<a href="/testing/playbook">Deployment Playbook</a>
<a href="/testing/report/favorites_lab" class="on">Favorites Lab</a>
<a href="/testing/report/final_model">Final Model</a>
<a href="/testing/report/v7_lab">v7 Lab</a>
</div>"""

slate_rows = ""
for r in sorted(slate, key=lambda x: -(x["market"] - x.get("v6", x["v5"]))):
    v6 = r.get("v6", r["v5"])
    gap = r["market"] - v6
    danger = gap >= 0.15
    cls = " style='background:#fdf3ec'" if danger else ""
    tag = " ⚠" if danger else ""
    moved = v6 - r["v5"]
    slate_rows += (f"<tr{cls}><td>{r['a']} vs {r['b']}{tag}</td>"
                   f"<td class='mono'>{r['market']*100:.0f}¢</td>"
                   f"<td class='mono'>{r['v5']*100:.1f}%</td>"
                   f"<td class='mono good'>{v6*100:.1f}%</td>"
                   f"<td class='mono'>{moved*100:+.1f}</td>"
                   f"<td class='mono'>{gap*100:+.1f}</td></tr>")

mb_rows = ""
for b in cal["market_bands"]:
    mb_rows += (f"<tr><td class='mono'>{b['band']}</td><td class='mono'>{b['n']}</td>"
                f"<td class='mono'>{b['market']:.3f}</td><td class='mono'>{b['model']:.3f}</td>"
                f"<td class='mono'><b>{b['emp']:.3f}</b></td>"
                f"<td class='mono dim'>[{b['ci'][0]:.2f},{b['ci'][1]:.2f}]</td></tr>")

sb_rows = ""
for b in cal["self_bands"]:
    sb_rows += (f"<tr><td class='mono'>{b['band']}</td><td class='mono'>{b['n']}</td>"
                f"<td class='mono'>{b['pred']:.3f}</td><td class='mono'><b>{b['emp']:.3f}</b></td>"
                f"<td class='mono dim'>[{b['ci'][0]:.2f},{b['ci'][1]:.2f}]</td></tr>")

gl_rows = ""
for b in cal["gap_link"]:
    flag = " ⚠" if b["pred"] < b["ci"][0] or b["pred"] > b["ci"][1] else ""
    gl_rows += (f"<tr><td class='mono'>{b['gap']}</td><td class='mono'>{b['n']}</td>"
                f"<td class='mono'>{b['pred']:.3f}</td><td class='mono'><b>{b['emp']:.3f}</b>{flag}</td>"
                f"<td class='mono dim'>[{b['ci'][0]:.2f},{b['ci'][1]:.2f}]</td></tr>")

fade_rows = ""
for k, v in fix.items():
    if not k.startswith("fade_"):
        continue
    lab = k.replace("fade_", "div≥").replace("_mkt_", "pts · mkt ").replace("_", " ")
    fade_rows += (f"<tr><td class='mono'>{lab}</td><td class='mono'>{v['n']}</td>"
                  f"<td class='mono'>{v['roi']*100:+.1f}%</td>"
                  f"<td class='mono dim'>[{v['ci'][0]*100:+.0f}%, {v['ci'][1]*100:+.0f}%]</td></tr>")

html = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex">
<title>Favorites Lab — who's right at 75–90¢?</title>
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
 .wrap {{ max-width:880px; margin:0 auto; }}
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
 p {{ font-size:.9rem; margin:7px 0; }}
 .dim {{ color:var(--dim); }}
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
 canvas {{ max-height:320px; }}
 .chartbox {{ margin:12px 0 4px; }}
 ul {{ font-size:.9rem; margin:8px 0 8px 20px; }} li {{ margin:5px 0; }}
</style></head>
<body><div class="wrap">
<h1>Favorites Lab</h1>
<div class="tagline">Deep research: the market priced clear favorites 10–16 points above
the model. Answer: partly both sides — a real model bug (found &amp; fixed → v6) plus residual
market favorite-bias · updated 2026-07-22 after the v6 correction</div>
{NAV}

<section>
<h2><span class="n">0</span>Verdict up front</h2>
<div class="callout warn"><b>Correction — the operator was right, and the first version of
this page was wrong.</b> The initial verdict leaned on band calibration, which cannot see a
subpopulation bias. Case forensics found a real defect: the asymmetric decay (losses fade
faster than wins) systematically <u>erased bad teams' evidence of badness</u> — EG, a 14-20
team with one series win, carried MORE decayed win-evidence than loss-evidence (ratio 1.38
vs a true 0.83) and rated as league-average. Every elite-vs-floor gap was compressed.
Aggregate metrics hid it because mid-table matches dominate every band.</div>
<div class="callout good"><b>The fix — consistency-conditioned decay — is deployed (v6).</b>
Principle: results consistent with a team's level persist (HL 20 games: a good team's wins,
a bad team's losses); anomalies fade (HL 12: a stumble, a fluke win). It beats the flawed
asymmetry on overall holdout LL (0.64126 vs 0.64182) AND moves every slate case toward the
market by 2–7 points. The remaining model-vs-market gap (~5–11 pts on these matches) is the
part the historical evidence still assigns to market favorite-bias: at 75–85¢, market
favorites won 72.7% (n=22), and fading them was ROI-positive — but with the floor-team bug
now fixed, that residual claim should be re-measured as new matches settle.</div>
<div class="callout warn"><b>But respect the exception:</b> divergences of <b>15+ points where
the market is the confident side</b> are the one shape where the market has repeatedly known
something (the May fades: roster surges, stand-ins). Two matches on today's slate are in that
zone — see §1. Quarter-size there per Playbook Rule 4, and check rosters manually before
trusting the model's number.</div>
</section>

<section>
<h2><span class="n">1</span>The trigger: today's slate (bot is already serving the new model)</h2>
<p>Verified: the terminal's BenPom numbers match the private v5 snapshot to the decimal —
the bot is on the new model. These gaps are v5-vs-market, not old-model artifacts:</p>
<table>
<tr><th>Match</th><th>Market</th><th>v5 (buggy)</th><th>v6 (fixed)</th><th>Fix moved</th><th>Gap to market</th></tr>
{slate_rows}
</table>
<p class="dim">⚠ = 15+ point divergence with the market more confident — the information-risk
shape. T1–VARREL and EDG–FunPlus Phoenix qualify today: verify lineups/stand-ins before full
size; the other favorites (LEV, MIBR, RRQ, NRG, PRX at 9–14 pts) sit in the historically
profitable fade zone.</p>
</section>

<section>
<h2><span class="n">1.5</span>The bug the aggregates hid — anatomy</h2>
<p>Decayed evidence mass per team (sum of decay weights on wins vs losses). Under the old
asymmetric decay, chronic losers' losses — their true signal — evaporated:</p>
<table>
<tr><th>Team (2026 maps W-L)</th><th>asym W-mass</th><th>asym L-mass</th><th>ratio</th>
<th>symmetric ratio</th></tr>
<tr><td>EG (14-20, one series win)</td><td class="mono">13.2</td><td class="mono">9.5</td>
<td class="mono" style="color:#c0392b"><b>1.38</b></td><td class="mono">0.83</td></tr>
<tr><td>ZETA (7-22)</td><td class="mono">8.4</td><td class="mono">13.3</td>
<td class="mono" style="color:#c0392b"><b>0.63</b></td><td class="mono">0.38</td></tr>
<tr><td>ENVY (13-22)</td><td class="mono">7.2</td><td class="mono">10.3</td>
<td class="mono" style="color:#c0392b"><b>0.70</b></td><td class="mono">0.52</td></tr>
<tr><td>LEV (40-31, Masters winners)</td><td class="mono">17.7</td><td class="mono">6.3</td>
<td class="mono">2.80</td><td class="mono">1.63</td></tr>
</table>
<p>A 14-20 team presenting a winning evidence profile is how EG reached a −0.12 rating
(league average) and how LEV−EG priced at 69.6%. The consistency rule keeps the top-team
benefit that made asymmetric decay win originally (LEV's rare losses still fade) while
letting a floor team's losses persist as the signal they are. Also tested and rejected here:
a decayed-winrate additive feature (redundant once decay is fixed) and removing the
region-prior ridge (it was not the culprit — best at 1.5 under the new decay too).</p>
</section>

<section>
<h2><span class="n">2</span>Measurement 1 — band calibration (v5), and why it fooled this page</h2>
<p>Full 2025–26 holdout (~1,000 series), model-referenced favorite bands, computed under the
<b>buggy v5</b>. Kept for the record as the exhibit of the failure mode:</p>
<table>
<tr><th>Model band</th><th>n</th><th>Predicted</th><th>Won</th><th>95% CI</th></tr>
{sb_rows}
</table>
<p><b>These numbers look perfect — and were still hiding the bug.</b> Band averages mix
heterogeneous matchups: over-priced coin flips and under-priced elite-vs-floor stomps cancel
inside the same band. Lesson now encoded in lab practice: aggregate calibration can never
clear a model of a subpopulation bias — concrete case anomalies (the operator's LEV–EG
example) get case-level forensics first. v6's own bands will be re-measured as results
settle.</p>
</section>

<section>
<h2><span class="n">3</span>Measurement 2 — is the MARKET calibrated at its price bands?</h2>
<p>Market-referenced favorite bands, clean pre-match prices (n=168 overlap):</p>
<div class="chartbox"><canvas id="mktcal"></canvas></div>
<table>
<tr><th>Market band</th><th>n</th><th>Market said</th><th>Model said</th><th>Favorite won</th><th>95% CI</th></tr>
{mb_rows}
</table>
<p><b>The market shows favorite-bias on this sample</b> — at 75–85¢ its favorites won
72.7% (overpaying ~7 points); at 55–65¢ they were coin flips. <b>But read the model column
as v5</b>: part of its apparent accuracy came from the floor bug dragging favorite prices
down into what happened to be the right neighborhood. Under v6 these model numbers rise 2–7
points; whether the market remains the overconfident side is an open question this page
re-measures as matches settle — tonight's slate alone adds seven cases.</p>
</section>

<section>
<h2><span class="n">4</span>Measurement 3 — the large-gap link, and the fix that failed</h2>
<table>
<tr><th>Rating gap</th><th>n</th><th>Model link</th><th>Higher-rated won</th><th>95% CI</th></tr>
{gl_rows}
</table>
<p>One band — gap [6,8) — showed empirical above the link (0.866 vs 0.789). A convex
link correction was fit and <b>rejected on holdout</b> — correctly so, because the anomaly
was never a link problem: <b>the floor bug was compressing measured gaps</b>, so true
8-gap matchups sat in the [6,8) bucket outperforming it. The consistency-decay fix widens
those gaps at the source; the link stays logistic-linear (fifth tail family tested, all
five rejected — the defect was in the ratings' evidence accounting, never the link).</p>
</section>

<section>
<h2><span class="n">5</span>The fade ledger — was disagreeing with big favorites profitable?</h2>
<p>Historical rule: when the model sits ≥X points under the market on a favorite, buy the
underdog at the market price:</p>
<table>
<tr><th>Rule</th><th>n</th><th>ROI</th><th>95% CI</th></tr>
{fade_rows}
</table>
<p class="dim">Every point estimate positive; every CI wide (n=13–34) — <b>and every
trade was selected using buggy-v5 divergences</b>, which over-triggered against
elite-vs-floor favorites. Treat these ROIs as provisional upper bounds; v6 will generate
fewer and smaller favorite-fades, which is exactly the correction the operator called
for.</p>
</section>

<section>
<h2><span class="n">6</span>Findings &amp; operational conclusions (post-correction)</h2>
<ul>
<li><b>Finding 1 — a real favorite-suppressing bug, now fixed.</b> Asymmetric decay let
losing teams' losses evaporate; floor teams rated far too high; elite-vs-floor favorites
were underpriced 2–7 points. v6 (consistency-conditioned decay) fixes the mechanism AND
improves overall holdout accuracy (0.64126 vs 0.64182). Deployed to the trading snapshot —
the bot's caps on these favorites move up automatically.</li>
<li><b>Finding 2 — the market still shows favorite-bias on the historical sample</b>
(75–85¢ favorites won 72.7%), but that evidence is entangled with the bug and downgraded to
provisional. Re-measure monthly; tonight's seven favorites are the first clean v6 test.</li>
<li><b>Finding 3 — the exception rule survives both stories:</b> 15+ point divergences with
the market confident remain information-risk (roster news, stand-ins, surges) —
quarter-size and check lineups. Under v6 far fewer matches trigger it.</li>
<li><b>Finding 4 — five tail-modification families are tested and dead</b> (tanh,
piecewise β, margin links, convex hinge, soft labels). The favorite problem was never in
the link. Stop bending the link.</li>
<li><b>Process finding — operator anomaly reports beat aggregate dashboards.</b> Band
calibration validated a broken model; three named matchups exposed it in twenty minutes.
Standing practice: every concrete "this number looks wrong" gets evidence-mass forensics
before any statistical defense.</li>
</ul>
</section>

<section>
<h2><span class="n">7</span>Latest findings — the quoting margin (how much edge to demand)</h2>
<p>Follow-up research (full detail: Playbook §9). Maker-fill simulation of the bot's actual
mechanism — NO bids resting on both sides of all 155 events at (model NO value − margin),
filled only when a real trade printed at/through the level inside the true quoting window
(listing → start−2h), settled at results, priced from <b>v6</b>:</p>
<table>
<tr><th>Margin rule</th><th>Fills</th><th>Total profit</th><th>ROI</th><th>95% CI</th></tr>
<tr><td class="mono">flat 1¢ (≈ current bot config)</td><td class="mono">221</td>
<td class="mono" style="color:#c0392b">−0.68u</td>
<td class="mono" style="color:#c0392b">−0.6%</td><td class="mono dim">[−13%, +12%]</td></tr>
<tr><td class="mono">flat 5¢</td><td class="mono">168</td><td class="mono">+5.27u</td>
<td class="mono">+6.7%</td><td class="mono dim">[−9%, +23%]</td></tr>
<tr><td class="mono">flat 11¢ (best flat)</td><td class="mono">100</td><td class="mono">+8.07u</td>
<td class="mono">+19.2%</td><td class="mono dim">[−4%, +44%]</td></tr>
<tr><td class="mono"><b>logit +0.6 — winner</b></td><td class="mono">74</td>
<td class="mono" style="color:#1e7a4f"><b>+8.12u</b></td>
<td class="mono" style="color:#1e7a4f"><b>+29.1%</b></td>
<td class="mono" style="color:#1e7a4f">[+1%, +61%] — only CI above zero</td></tr>
</table>
<div class="callout good"><b>The margin should live in logit space, not cents.</b> A logit
shift of +0.5–0.6 auto-scales the demanded edge to the price: ≈14¢ at a 50¢ book, ≈12¢ at
65¢, ≈10¢ at 75¢, ≈8¢ at 85¢, ≈5¢ at 92¢. It demands the most where small divergences are
noise (coin flips — see the edge grid in the Playbook) and less at the extremes where every
cent is worth more. At equal fill counts, logit beat flat cents in every comparison. The
bot's current flat-1¢ edge earned nothing over the whole history.</div>
<p><b>Triple-tested:</b> positive in both time halves (+10.8% / +62.7%), under conservative
1¢-trade-through fills (+26.1%), with this page's exception rules applied (<b>+52.6%</b> —
its best configuration), and with the quote window extended to start−5m (+36.9%). The fill
breakdown independently reproduced this page's central result: NO bids resting at 30–50¢
(selling modestly overpriced favorites) made <b>+56.5%</b>; the one negative pocket was
NO quotes against teams the model prices below ~45% (−5.7%) — the optional refinement is to
skip posting that side entirely.</p>
<p class="dim">Caveats: fills inferred from 1-minute trade prints (no queue position or
partial fills modeled), maker fees assumed zero, ten weeks of one season. The telemetry
spec's markout logging converts this simulation into live measurement.</p>
</section>
</div>
<script>
const MB = {json.dumps(cal["market_bands"])};
new Chart(document.getElementById('mktcal'), {{
 type: 'bar',
 data: {{ labels: MB.map(b => 'market ' + b.band), datasets: [
  {{ label: 'market price', data: MB.map(b => b.market), backgroundColor: '#9a93a655' }},
  {{ label: 'model', data: MB.map(b => b.model), backgroundColor: '#7c4dd6aa' }},
  {{ label: 'actually won', data: MB.map(b => b.emp), backgroundColor: '#1e7a4fcc' }} ] }},
 options: {{ scales: {{ y: {{ min: 0.4, max: 1.0,
    title: {{ display: true, text: 'favorite win probability' }} }} }},
  plugins: {{ tooltip: {{ callbacks: {{
    afterBody: items => 'n=' + MB[items[0].dataIndex].n }} }} }} }}
}});
</script>
</body></html>"""

os.makedirs(RD, exist_ok=True)
with open(os.path.join(RD, "favorites_lab.html"), "w") as f:
    f.write(html)
print(f"written: favorites_lab.html ({len(html)} bytes)")
