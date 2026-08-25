"""Article: Championship DNA — Historical Trends To Note For Champions Shanghai

Scaffold only. Title + hero image; the body is intentionally EMPTY, waiting on
the author's copy. Nothing in here invents prose — no deck, no caption, no
section headings, no category label — so whatever lands in `.content` is the
author's words and only the author's words.

Layout mirrors AspasGreatestPrime / the Masters previews: centred 860px column,
label / h1 / byline / cover / content. Add the `.toc` nav and a `.label` when
the sections and category exist.
"""

import os
import json
from flask import Blueprint, render_template_string

article_championship_dna_bp = Blueprint("article_championship_dna", __name__)

_ROOT = os.path.dirname(os.path.abspath(__file__))
_LANDSCAPE = os.path.join(_ROOT, "data", "enriched", "side_landscape.json")

_ls_cache = (None, -1.0)


def _landscape():
    """Attack/defense win% for every international-attending team over the split
    before that international. Built by scrapers/BuildSideLandscape.py — see
    there for why it is precomputed rather than derived per request."""
    global _ls_cache
    try:
        stamp = os.path.getmtime(_LANDSCAPE)
    except OSError:
        return {"points": [], "internationals": []}
    if _ls_cache[0] is not None and _ls_cache[1] == stamp:
        return _ls_cache[0]
    try:
        with open(_LANDSCAPE) as f:
            data = json.load(f)
    except Exception:
        data = {"points": [], "internationals": []}
    _ls_cache = (data, stamp)
    return data


PAGE_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=820">
<title>Championship DNA: Historical Trends To Note For Champions Shanghai &mdash; Bobo's VCT Database</title>
<!-- Open Graph / Twitter link-preview cards. The description is the author's
     articles-index blurb, reused verbatim — same job, same words. -->
<meta property="og:type" content="article">
<meta property="og:site_name" content="Bobo gg">
<meta property="og:title" content="Championship DNA: Historical Trends To Note For Champions Shanghai">
<meta property="og:description" content="Understanding the indicators of a championship team - by the numbers, by the rosters, by the regions, and other miscellaneous trends.">
<meta property="og:url" content="https://bobo-gg.net/articles/championship-dna/">
<meta property="og:image" content="https://bobo-gg.net/championshipdna.jpg">
<meta property="og:image:secure_url" content="https://bobo-gg.net/championshipdna.jpg">
<meta property="og:image:type" content="image/jpeg">
<meta property="og:image:width" content="2048">
<meta property="og:image:height" content="1404">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="Championship DNA: Historical Trends To Note For Champions Shanghai">
<meta name="twitter:description" content="Understanding the indicators of a championship team - by the numbers, by the rosters, by the regions, and other miscellaneous trends.">
<meta name="twitter:image" content="https://bobo-gg.net/championshipdna.jpg">
<link rel="icon" type="image/svg+xml" href="/favicon.svg">
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@700;800&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
<link rel="stylesheet" href="/static/base.css">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"></script>
<style>
  html { -webkit-text-size-adjust:100%; text-size-adjust:100%; }
  .page { position:relative; z-index:1; flex:1; display:flex; flex-direction:column; align-items:center; padding:60px 32px 80px; }
  .article { max-width:860px; width:100%; }
  .label { font-family:'Plus Jakarta Sans',sans-serif; font-size:0.77rem; font-weight:800; letter-spacing:.14em; text-transform:uppercase; color:var(--soft); margin-bottom:16px; text-align:center; }
  /* This title is 66 characters — half again as long as the other articles'.
     At their 3.52rem it strands orphans, so the ceiling drops to 2.92rem
     and it sits on two even lines instead.
     text-wrap:balance rather than a hard <br>, so it stays even at any
     width instead of breaking in a fixed place that only suits one. */
  h1 { font-family:'Plus Jakarta Sans',sans-serif; font-size:clamp(1.55rem,3.1vw,2.25rem); font-weight:800; letter-spacing:-1px; line-height:1.14; margin-bottom:24px; text-align:center; }
  .nb { white-space:nowrap; }
  .deck { font-size:1.06rem; font-weight:300; color:var(--soft); line-height:1.6; text-align:center; max-width:640px; margin:-8px auto 22px; }
  .byline { font-size:.82rem; color:var(--soft); font-weight:300; margin-bottom:48px; padding-bottom:32px; border-bottom:1px solid #e8e0ec; text-align:center; }
  .cover { width:100%; border-radius:16px; overflow:hidden; margin-bottom:12px; }
  .cover img { width:100%; height:auto; display:block; }
  .cover-caption { font-size:.75rem; color:var(--soft); font-weight:300; font-style:italic; margin-bottom:48px; text-align:center; }
  .content p { font-size:1rem; font-weight:300; line-height:1.8; color:var(--ink); margin-bottom:24px; }
  .content h2 { font-family:'Plus Jakarta Sans',sans-serif; font-size:1.54rem; font-weight:800; letter-spacing:-0.5px; margin:48px 0 20px; }
  .content h2, .cover { scroll-margin-top:84px; }
  .fig { margin:34px 0 40px; }
  .fig-wrap { position:relative; width:100%; aspect-ratio:1/1; background:#fff;
              border:1px solid #ece6f2; border-radius:14px; box-shadow:0 4px 24px #0000000a;
              overflow:hidden; }
  .fig-filter { display:flex; flex-wrap:wrap; gap:6px; justify-content:center; margin-bottom:14px; }
  .lsb { font-family:'DM Sans',sans-serif; font-size:.7rem; font-weight:700; color:var(--soft);
         background:#fff; border:1px solid #e8e0ec; border-radius:99px; padding:5px 12px; cursor:pointer;
         transition:background .15s,border-color .15s,color .15s; }
  .lsb:hover { color:var(--ink); border-color:#c9b8d8; }
  .lsb.on { background:#7c4dd6; border-color:#7c4dd6; color:#fff; }
  .lsb-win { border-color:#e0c48a; color:#8a6a1a; }
  .lsb-win.on { background:#d8a93a; border-color:#d8a93a; color:#fff; }
  @media (max-width:820px) {
    .page { padding:40px 18px 60px; }
  }
</style>
</head>
<body>
<div class="page">
  <div class="article">
    <!-- Explicit break after the colon: the series name owns line 1, the
         subject owns line 2. The nowrap span keeps the event name whole if
         line 2 ever has to wrap on a narrow screen. Wording untouched. -->
    <h1>Championship DNA:<br>Historical Trends To Note For <span class="nb">Champions Shanghai</span></h1>
    <div class="byline">Bobo &mdash; August 2026</div>
    <div class="cover">
      <img src="/championshipdna.jpg" alt="VCT champions lifting trophies">
    </div>
    <p class="cover-caption">International winners from each year of franchised VCT<br>FNATIC at Tokyo in 2023, Sentinels at Madrid in 2024, Paper Rex in 2025, and Leviatán at London in 2026</p>
    <div class="content">
      <p>Champions, the biggest event in the year for VCT, is right around the corner. This means that the entirety of the fanbase will be making Pick&rsquo;Ems, discussing their predictions online, casting bets, and constructing fantasy teams. Furthermore, Champions involves 4 teams from each region, as opposed to the 3 (and sometimes 2, in the past) from each region at Masters events. With such a wide pool of teams alongside a large, captive, and opinionated audience, there is no better time to look back on the (almost) 4 years of franchised VCT history! How can we use history to sort through these teams and see who is expected to fail and who are true favorites?</p>

      <p>This tradition of historical and analytical trends exists heavily in other sports and the results are often fascinating (and accurate). I am excited to borrow these ideas, frameworks, and visualizations I&rsquo;ve read over the years and bring them into the world of VCT! In this article, you will likely see a few references to college basketball/baseball analytics, so bear with me if that&rsquo;s unfamiliar.</p>

      <h2 id="by-the-numbers">The Winners: By The Numbers</h2>

      <p>One of the simplest ways that a championship team is understood in any sport is by their offensive and defensive strength levels. Rely too heavily on one of these sides, and imbalance can often lead to failure. VCT is no different, except we&rsquo;re dealing with attack and defense rather than offense and defense. Here is a graph of every international-attending team, mapped by their attack win% and defense win% in the split prior (e.g. Leviatan at London uses their numbers from Stage 1 of 2026).</p>

      <figure class="fig">
        <div class="fig-filter" id="lsFilter"></div>
        <div class="fig-wrap"><canvas id="sideLandscape"></canvas></div>
      </figure>
    </div>
  </div>
</div>

<script>
const LS = __LANDSCAPE_JSON__;
const logoCache = {};
function logo(org) {
  if (!logoCache[org]) { const i = new Image(); i.src = '/logos/' + org + '.png'; logoCache[org] = i; }
  return logoCache[org];
}
const pct = v => (v * 100).toFixed(1) + '%';

let hovered = null, active = 'All', chart;

// White plate behind everything, so the figure reads as its own card instead of
// sitting on the page's gradient.
const plate = {
  id: 'plate',
  beforeDraw(c) {
    const {ctx} = c;
    ctx.save(); ctx.fillStyle = '#fff';
    ctx.fillRect(0, 0, c.width, c.height); ctx.restore();
  }
};


// Logos and their event labels are drawn here rather than via pointStyle. Two
// reasons: Chart.js swaps the point for its hover style on hover, which made the
// logo vanish under the cursor, and drawing by hand is what lets everything
// except the hovered team drop to greyscale.
const marks = {
  id: 'marks',
  afterDatasetsDraw(c) {
    const {ctx} = c;
    const meta = c.getDatasetMeta(0);
    // Hovered one last, so it draws over the neighbours it grows into.
    const order = meta.data.map((_, i) => i)
                           .sort((a, b) => (a === hovered) - (b === hovered));
    // Labels are skipped where they'd land on one already drawn. With all 122
    // points shown they otherwise pile into an unreadable smear through the
    // middle of the cloud; this keeps whichever ones can actually be read, and
    // the hovered label is always drawn regardless.
    const placed = [];
    const fits = (x, y, w) => {
      const r = {l: x - w / 2, r: x + w / 2, t: y, b: y + 11};
      for (const q of placed) {
        if (r.l < q.r && r.r > q.l && r.t < q.b && r.b > q.t) return false;
      }
      placed.push(r); return true;
    };
    order.forEach(i => {
      const pt = meta.data[i];
      const p = c.data.datasets[0].data[i].p;
      const img = logo(p.org);
      const on  = hovered === i;
      const dim = hovered !== null && !on;
      const S = (p.won ? 30 : 25) * (on ? 1.7 : 1);
      ctx.save();
      ctx.globalAlpha = dim ? 0.28 : 1;
      if (dim) ctx.filter = 'grayscale(1)';
      if (img.complete && img.naturalWidth) ctx.drawImage(img, pt.x - S / 2, pt.y - S / 2, S, S);
      ctx.filter = 'none';
      ctx.globalAlpha = 1;
      ctx.font = (on ? "800 12px " : "500 9px ") + "'DM Sans',sans-serif";
      ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      if (on) {
        // Plate under the enlarged label so it stays readable over other marks.
        const w = ctx.measureText(p.intl).width + 10;
        ctx.fillStyle = 'rgba(255,255,255,.92)';
        ctx.fillRect(pt.x - w / 2, pt.y + S / 2 + 1, w, 15);
      }
      const ly = pt.y + S / 2 + 3;
      if (on || fits(pt.x, ly, ctx.measureText(p.intl).width)) {
        ctx.fillStyle = dim ? 'rgba(150,142,158,.45)' : (on ? '#3d1a6e' : '#8a7f94');
        ctx.fillText(p.intl, pt.x, ly);
      }
      ctx.restore();
    });
  }
};

function visible() {
  return LS.points.filter(p =>
    active === 'All' ? true : (active === 'Winners' ? p.won : p.intl === active)
  ).map(p => ({x: p.dfn, y: p.atk, p}));
}

function draw() {
  const data = visible();
  if (chart) { chart.data.datasets[0].data = data; hovered = null; chart.update(); return; }
  chart = new Chart(document.getElementById('sideLandscape'), {
    type: 'scatter',
    data: {datasets: [{data, pointRadius: 0, pointHoverRadius: 0, hitRadius: 14}]},
    options: {
      responsive: true, maintainAspectRatio: false,
      layout: {padding: {top: 14, right: 16, bottom: 4, left: 4}},
      // nearest + intersect:false = exactly one point, the closest to the
      // cursor. The default mode returns everything within hitRadius, which in
      // a cluster this dense meant five teams in one tooltip.
      interaction: {mode: 'nearest', intersect: false, axis: 'xy'},
      onHover(e, els) {
        const i = els.length ? els[0].index : null;
        if (i !== hovered) { hovered = i; chart.draw(); }
      },
      scales: {
        x: {title: {display: true, text: 'Defense win% in the split before the event',
                    font: {family: "'DM Sans',sans-serif", size: 11, weight: 600}, color: '#7a6e7e'},
            ticks: {callback: v => pct(v), font: {size: 10}, color: '#9a8fa4'},
            grid: {color: 'rgba(0,0,0,.05)'}},
        y: {title: {display: true, text: 'Attack win% in the split before the event',
                    font: {family: "'DM Sans',sans-serif", size: 11, weight: 600}, color: '#7a6e7e'},
            ticks: {callback: v => pct(v), font: {size: 10}, color: '#9a8fa4'},
            grid: {color: 'rgba(0,0,0,.05)'}}
      },
      plugins: {
        legend: {display: false},
        tooltip: {
          displayColors: false, backgroundColor: 'rgba(22,18,29,.94)', padding: 10,
          callbacks: {
            title: it => it[0].raw.p.org + ' — ' + it[0].raw.p.intl + (it[0].raw.p.won ? '  (won it)' : ''),
            label: it => {
              const p = it.raw.p;
              return ['from ' + p.prior,
                      'attack  ' + pct(p.atk) + '  (' + p.atk_w + '/' + p.atk_n + ')',
                      'defense ' + pct(p.dfn) + '  (' + p.def_w + '/' + p.def_n + ')'];
            }
          }
        }
      }
    },
    plugins: [plate, marks]
  });
}

(function () {
  const box = document.getElementById('lsFilter');
  const opts = [['All', 'All'], ['Winners', 'Winners only']].concat(
    LS.internationals.map(n => [n, n]));
  opts.forEach(([val, text]) => {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'lsb' + (val === 'All' ? ' on' : '') + (val === 'Winners' ? ' lsb-win' : '');
    b.textContent = text;
    b.onclick = () => {
      active = val;
      [].forEach.call(box.children, c => c.classList.toggle('on', c === b));
      draw();
    };
    box.appendChild(b);
  });
  draw();
})();
</script>
</body>
</html>
"""


@article_championship_dna_bp.route("/")
def article_championship_dna():
    # Straight substitution rather than a Jinja variable: the chart script is
    # full of JS object literals, and render_template_string would try to read
    # some of them as Jinja delimiters.
    return PAGE_HTML.replace("__LANDSCAPE_JSON__", json.dumps(_landscape()))
