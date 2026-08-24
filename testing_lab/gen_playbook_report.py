"""Build /testing/report/playbook_bt — the playbook, exactly as specified.

    Take the YES price an hour or two before the match starts.
    Bet the side where BenPom differs from it by 5 percentage points or more.
    No fees.

Reads testing_lab/v10/stats/v10_simple.json (run_simple.py). NO FEE is charged
anywhere on this page. The threshold is GIVEN, not fitted, so the whole sample
is usable and there is no train/test split to argue about.

Run: python3 testing_lab/gen_playbook_report.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
V10S = os.path.join(HERE, "v10", "stats")
RD = os.path.join(HERE, "out", "reports")

SP = json.load(open(os.path.join(V10S, "v10_simple.json")))
LA = json.load(open(os.path.join(V10S, "v10_live_autopsy.json")))
with open(os.path.join(HERE, "gen_v9_report.py")) as f:
    CSS = f.read().split('CSS = """', 1)[1].split('"""', 1)[0]

C = {"s1": "#7c4dd6", "d5": "#c96a2a", "a1": "#3a90cc", "gray": "#9a93a6",
     "good": "#1e7a4f", "bad": "#c0392b"}

NAV = """<div class="labtabs">
<a href="/testing/">Testing Lab</a>
<a href="/testing/report/state_of_benpom">State of BenPom</a>
<a href="/testing/playbook">Deployment Playbook</a>
<a href="/testing/report/favorites_lab">Favorites Lab</a>
<a href="/testing/report/final_model">Final Model</a>
<a href="/testing/report/v7_lab">v7 Lab</a><span class="brk"></span>
<a href="/testing/report/v8_lab">v8 Lab</a>
<a href="/testing/report/roster_adaptation">Roster</a>
<a href="/testing/report/v9_lab">v9 Lab</a>
<a href="/testing/report/v10_lab">v10 Lab</a>
<a href="/testing/report/edge_lab">Edge vs Market</a>
<a href="/testing/report/playbook_bt" class="on">Playbook (backtested)</a>
</div>"""


def dl(n):
    return (f'<div class="dl"><a href="/testing/edge/stats/{n}" download>'
            f'&#8681; {n}</a></div>')


def sg(x, nd=2):
    return f"{x:+.{nd}f}"


t2, t1 = SP["T2h"], SP["T1h"]
h1, h2 = SP["halves"]["first"], SP["halves"]["second"]
cd, cf = SP["control_blind_underdog"], SP["control_blind_favourite"]
od, of = SP["on_underdogs"], SP["on_favourites"]
gap = t2["roi_pct"] - cd["roi_pct"]

page_bars = {"labels": ["The rule\\n(T-1h)", "The rule\\n(T-2h)",
                        "Back every underdog\\n(no model)",
                        "Back every favourite\\n(no model)"],
             "roi": [t1["roi_pct"], t2["roi_pct"], cd["roi_pct"], cf["roi_pct"]]}
page_month = {"labels": [r["month"] for r in SP["by_month"]],
              "roi": [r["roi_pct"] for r in SP["by_month"]],
              "n": [r["n_bets"] for r in SP["by_month"]]}
json.dump(page_bars, open(os.path.join(V10S, "v10_page_pb_bars.json"), "w"), indent=1)
json.dump(page_month, open(os.path.join(V10S, "v10_page_pb_month.json"), "w"), indent=1)

monrows = "".join(
    f'<tr><td class="mono">{r["month"]}</td><td class="mono">{r["n_bets"]}</td>'
    f'<td class="mono">{r["hit_pct"]}%</td>'
    f'<td class="mono {"good" if r["roi_pct"] > 0 else "bad"}">{sg(r["roi_pct"])}%</td></tr>'
    for r in SP["by_month"])

HTML = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Playbook (backtested) — one rule, no fees</title>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@800&family=DM+Sans:wght@400;700&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.3/dist/chart.umd.min.js"></script>
<style>{CSS}
.banner b {{ color:var(--warn); }}
.rule {{ font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; font-size:1.05rem;
        background:var(--accbg); border-left:6px solid var(--acc); border-radius:0 14px 14px 0;
        padding:16px 20px; margin:14px 0; line-height:1.5; }}</style></head><body><div class="wrap">

<h1>Playbook (backtested)</h1>
<div class="tagline">One rule. Pre-match YES price, 5pp threshold, no fees.</div>
{NAV}

<div class="banner" style="background:var(--warnbg);border-color:#eddcc9;border-left-color:var(--warn)">
<b>THE ANSWER — IT MAKES MONEY, BUT ALMOST NONE OF IT IS THE MODEL.</b>
The rule returns <b>{sg(t2['roi_pct'])}%</b> on the T&minus;2h price and
<b>{sg(t1['roi_pct'])}%</b> on T&minus;1h, over {SP['n_matches']} matches, stable
across both halves of the window. But backing <em>every</em> market underdog with
BenPom switched off returns <b>{sg(cd['roi_pct'])}%</b>. The rule is worth about
<b>{sg(gap, 1)} points</b> on top of that. What is being harvested is the book's
favourite&ndash;longshot bias &mdash; a 65c favourite wins only 58.5% &mdash; not
a model insight.
</div>

<section>
<h2><span class="n">1</span>The rule</h2>
<div class="rule">Take the YES price an hour or two before the match starts.
Buy the side where BenPom differs from it by 5 percentage points or more.
One contract. Hold to settlement. No fees.</div>
<p>The 5pp threshold is <b>given, not fitted</b> &mdash; so there is no swept
maximum here, no train/test split to argue about, and the whole sample counts.
Prices come from the raw candle book rebuilt against the real scheduled start,
so nothing is sampled after a match began.</p>
<div class="cards">
  <div class="card"><div class="lbl">T&minus;1h price</div>
    <div class="big good">{sg(t1['roi_pct'])}%</div>
    <div class="sub">n={t1['n_bets']}, CI [{t1['ci95'][0]:.0f}, {t1['ci95'][1]:.0f}]</div></div>
  <div class="card"><div class="lbl">T&minus;2h price</div>
    <div class="big good">{sg(t2['roi_pct'])}%</div>
    <div class="sub">n={t2['n_bets']}, CI [{t2['ci95'][0]:.0f}, {t2['ci95'][1]:.0f}]</div></div>
  <div class="card"><div class="lbl">Hit rate</div><div class="big">{t2['hit_pct']}%</div>
    <div class="sub">at an average {t2['mean_price']}c</div></div>
  <div class="card"><div class="lbl">Model's share</div>
    <div class="big warn">{sg(gap, 1)} pts</div>
    <div class="sub">over backing every underdog</div></div>
</div>
</section>

<section>
<h2><span class="n">2</span>The rule vs the control</h2>
<div class="chartbox"><canvas id="c_bars"></canvas></div>
<p class="cap">No fees anywhere on this chart. The third bar uses no model at all
&mdash; it just buys whichever side the market has below 50c. It captures nearly
the whole return.</p>
{dl('v10_page_pb_bars.json')}
<div class="scroll"><table>
<thead><tr><th>What the rule bought</th><th>n</th><th>Hit</th><th>ROI</th></tr></thead><tbody>
<tr><td>underdog side</td><td class="mono">{od['n']}</td><td class="mono">{od['hit_pct']}%</td>
    <td class="mono good">{sg(od['roi_pct'])}%</td></tr>
<tr><td>favourite side</td><td class="mono">{of['n']}</td><td class="mono">{of['hit_pct']}%</td>
    <td class="mono bad">{sg(of['roi_pct'])}%</td></tr>
</tbody></table></div>
<p class="cap">{od['n']} of the {t2['n_bets']} bets are underdogs and they carry
all the profit; the {of['n']} favourite bets lose money. In practice the rule is
an underdog filter.</p>
</section>

<section>
<h2><span class="n">3</span>Is it stable?</h2>
<div class="cards">
  <div class="card"><div class="lbl">First half</div>
    <div class="big">{sg(h1['roi_pct'])}%</div><div class="sub">n={h1['n_bets']}</div></div>
  <div class="card"><div class="lbl">Second half</div>
    <div class="big">{sg(h2['roi_pct'])}%</div><div class="sub">n={h2['n_bets']}</div></div>
</div>
<p>Split the window down the middle and it returns {sg(h1['roi_pct'])}% then
{sg(h2['roi_pct'])}%. That is about as stable as a three-month sample can look.</p>
<div class="chartbox"><canvas id="c_mon"></canvas></div>
<div class="scroll"><table>
<thead><tr><th>Month</th><th>Bets</th><th>Hit</th><th>ROI</th></tr></thead><tbody>{monrows}</tbody></table></div>
<p class="cap">Month to month is far less comfortable &mdash; one month at
{min(r['roi_pct'] for r in SP['by_month']):.0f}% and one at
+{max(r['roi_pct'] for r in SP['by_month']):.0f}%. With a dozen to seventy bets a
month that spread is normal, and it is what a losing quarter would feel like.</p>
{dl('v10_simple.json')}
</section>

<section>
<h2><span class="n">4</span>What this does and does not establish</h2>
<p><b>Does:</b> over {SP['n_matches']} matches from {SP['window'][0]} to
{SP['window'][1]}, at pre-match prices taken against the real scheduled start,
the rule was profitable and stable across halves. The bias driving it is
measurable and in the expected direction &mdash; the market's favourites are
priced at 65c and win 58.5%.</p>
<p><b>Does not:</b></p>
<ul>
<li><b>Show a model edge.</b> {sg(gap, 1)} points over a no-model control, on a
sample whose interval is roughly &plusmn;25 points, is not evidence that BenPom
is beating the book. The honest description of this rule is "buy underdogs".</li>
<li><b>Include fees.</b> None are charged, per the spec. For reference only:
a taker paying Kalshi's
<span class="mono">ceil(0.07&middot;C&middot;P&middot;(1&minus;P))</span> would
give back roughly six points on a {t2['mean_price']}c ticket. Maker fees on this
series are genuinely zero.</li>
<li><b>Include adverse selection.</b> The live week measured
<b>{abs(LA['adverse_selection']['adverse_selection_cents_per_contract']):.1f}c per
contract</b> &mdash; you get filled preferentially when you are wrong. On a
{t2['mean_price']}c ticket that is most of the margin, and no backtest that
assumes you transact at the quoted price can see it. That week lost
<b>${abs(LA['realized_total']):,.0f}</b> over {LA['fills']} fills.</li>
<li><b>Cover a full season.</b> Three months, one title, top-tier VCT only.</li>
</ul>
<div class="callout warn"><b>Standing.</b> Reporting only. Market data is never a
fitting target or a selection signal, and nothing here changes the deployed model
or the trading configuration.</div>
</section>

<script>
const PAL = __PAL__, BARS = __BARS__, MON = __MON__;
Chart.defaults.font.family = "'DM Sans',system-ui,sans-serif";
Chart.defaults.color = '#6b6478';
Chart.defaults.animation = false;

new Chart(document.getElementById('c_bars'), {{ type:'bar',
 data:{{ labels: BARS.labels.map(l=>l.split('\\n')), datasets:[
  {{ data: BARS.roi, borderRadius:4, maxBarThickness:62,
     backgroundColor: BARS.roi.map((v,i)=> i<2 ? PAL.s1 : (v>=0?PAL.d5:PAL.bad)) }} ] }},
 options:{{ responsive:true, maintainAspectRatio:false,
  plugins:{{ legend:{{display:false}},
   tooltip:{{callbacks:{{label:(t)=>'ROI '+t.parsed.y.toFixed(2)+'%'}}}} }},
  scales:{{ x:{{grid:{{display:false}}, ticks:{{font:{{size:10}}}}}},
   y:{{grid:{{color:'#eceef2'}}, title:{{display:true, text:'ROI % (no fees)'}}}} }} }} }});

new Chart(document.getElementById('c_mon'), {{ type:'bar',
 data:{{ labels: MON.labels, datasets:[{{ data: MON.roi, borderRadius:4, maxBarThickness:54,
   backgroundColor: MON.roi.map(v=>v>=0?PAL.good:PAL.bad) }}] }},
 options:{{ responsive:true, maintainAspectRatio:false,
  plugins:{{ legend:{{display:false}},
   tooltip:{{callbacks:{{label:(t)=>t.parsed.y.toFixed(1)+'%  (n='+MON.n[t.dataIndex]+')'}}}} }},
  scales:{{ x:{{grid:{{display:false}}}},
   y:{{grid:{{color:'#eceef2'}}, title:{{display:true, text:'ROI %'}}}} }} }} }});
</script>
</div></body></html>"""

html = (HTML.replace("__PAL__", json.dumps(C))
            .replace("__BARS__", json.dumps(page_bars))
            .replace("__MON__", json.dumps(page_month)))
out = os.path.join(RD, "playbook_bt.html")
with open(out, "w") as f:
    f.write(html)
print(f"wrote {out}  ({len(html)} bytes)")
