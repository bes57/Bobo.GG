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
  /* Breaks out of the 860px text column. Ten event names cannot fit one line
     at any readable size inside 860px, and the scatter is dense enough that
     the extra width helps it too. Centred on the viewport, capped so it never
     runs to the screen edge. */
  .fig { margin:34px 0 40px; width:min(1120px, calc(100vw - 56px));
         margin-left:50%; transform:translateX(-50%); }
  .fig-wrap { position:relative; width:100%; aspect-ratio:1.25/1; background:#fff;
              border:1px solid #ece6f2; box-shadow:0 4px 24px #0000000a;
              overflow:hidden; }
  /* Scoped under .content: `.content p` is class+element, which outranks a bare
     .fig-note class, so an unscoped rule here silently lost every property to
     the body-paragraph style. */
  /* Centred in the bullet, and pulled back over the list indent so it lines up
     with the body column rather than sitting off to the right. */
  .inline-fig { margin:18px 0 18px -22px; }
  .content .fig-credit { font-size:.68rem; font-weight:300; color:var(--soft);
                         text-align:center; margin-top:8px; word-break:break-all; }
  .content .fig-credit a { color:var(--soft); text-decoration:underline; }
  .content .fig-credit a:hover { color:#7c4dd6; }
  .inline-fig.wide img { display:block; width:100%; max-width:820px; height:auto; margin:0 auto;
                        border:1px solid #ece6f2; }
  .inline-fig-wrap { position:relative; width:100%; max-width:760px; aspect-ratio:1.15/1;
                     margin:0 auto; background:#fff; border:1px solid #ece6f2; overflow:hidden; }
  .content ul.notes { margin:0 0 24px; padding-left:22px; }
  .content ul.notes li { font-size:1rem; font-weight:300; line-height:1.8; color:var(--ink);
                         margin-bottom:12px; }
  .content .pin { color:#7c4dd6; font-weight:500; text-decoration:underline;
                  text-decoration-thickness:1px; text-underline-offset:2px; cursor:pointer; }
  .content .pin:hover { color:#5b21b6; }
  .content .fig-note { font-size:.68rem; font-weight:300; color:var(--soft);
                       text-align:center; margin:0 0 14px; line-height:1.5; }
  .fig-filter { margin-bottom:14px; }
  .ls-win { display:flex; justify-content:center; margin-bottom:8px; }
  /* One line, always. The ten event names are far wider than the 860px column,
     so the row scrolls sideways rather than wrapping — same treatment as the
     site nav. Scrollbar hidden; it's still swipeable/shift-scrollable. */
  .ls-events { display:flex; flex-wrap:nowrap; gap:5px; justify-content:safe center;
               overflow-x:auto; scrollbar-width:none; padding-bottom:2px;
               --lsb-fs:.68rem; --lsb-px:10px; }
  .ls-events .lsb { font-size:var(--lsb-fs); padding:4px var(--lsb-px); }
  .ls-events::-webkit-scrollbar { display:none; }
  .ls-events .lsb { flex:0 0 auto; }
  .lsb { font-family:'DM Sans',sans-serif; font-size:.62rem; font-weight:700; color:var(--soft);
         background:#fff; border:1px solid #e8e0ec; border-radius:99px; padding:4px 9px; cursor:pointer;
         white-space:nowrap; transition:background .15s,border-color .15s,color .15s; }
  .lsb:hover { color:var(--ink); border-color:#c9b8d8; }
  .lsb.on { background:#7c4dd6; border-color:#7c4dd6; color:#fff; }
  .lsb-win { border-color:#e0c48a; color:#8a6a1a; font-size:.76rem; padding:7px 16px; }
  .lsb-state { display:inline-block; margin-left:7px; padding:1px 7px; border-radius:99px;
               font-size:.62rem; font-weight:800; letter-spacing:.05em;
               background:rgba(138,106,26,.14); color:#8a6a1a; }
  .lsb-win.on .lsb-state { background:rgba(255,255,255,.28); color:#fff; }
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
        <p class="fig-note"><em>Note: Champions 2023 was not included, since there was no domestic split prior to the tournament</em></p>
        <div class="fig-filter">
          <div class="ls-win" id="lsWin"></div>
          <div class="ls-events" id="lsEvents"></div>
        </div>
        <div class="fig-wrap"><canvas id="sideLandscape"></canvas></div>
      </figure>

      <p>This is an awesome visualization that's fun to play around with! Some notes:</p>

      <ul class="notes">
        <li>We can see teams that were expected to do better than they did: for example, <a class="pin" data-org="LOUD" data-intl="Masters Tokyo 2023">LOUD at Tokyo</a>. This is a favorite example of mine, with a great narrative. 2023 LOUD was an amazing team with intense success before Masters Tokyo (2nd at LOCK//IN and then won Americas Stage 1) and after Masters Tokyo (3rd at Champions LA). They were a consensus top-2 favorite to win the event (Platchat put them above FNATIC, in fact, as favorites for the event.) Their flop at Tokyo was shocking and historic - what happened? As I recall, Masters Tokyo was the start of a rift between Less/Saadhak and Aspas, a reminder that this game cannot be just broken down into numbers. Also, it’s a reminder that Valorant is random.</li>
        <li>We can also see which teams overshot their previous domestic performance! <a class="pin" data-org="T1" data-intl="Masters Bangkok 2025">T1 at Bangkok</a> is the most obvious one. I mean seriously, how did they win this tournament:
          <figure class="inline-fig">
            <div class="inline-fig-wrap"><canvas id="bangkokInset"></canvas></div>
          </figure>
          <a class="pin" data-org="MIBR" data-intl="Champions 2025">MIBR at Champions Paris</a> and <a class="pin" data-org="WOL" data-intl="Masters Toronto 2025">Wolves at Masters Toronto</a> are also worth mentioning, though!</li>
        <li>We can see that Chinese teams get consistently overrated by this visualization, due to the less competitive state of domestic CN Valorant (e.g. <a class="pin" data-org="FPX" data-intl="Masters Shanghai 2024">FPX at Shanghai</a> and <a class="pin" data-org="XLG" data-intl="Masters Santiago 2026">XLG at Santiago</a> are placed impressively on this graph - they also went 1-2 and 0-2 in their respective events)</li>
      </ul>

      <p>Lastly, this visualization was inspired by EvanMiya&rsquo;s March Madness Efficiency Landscape graph that he uses to put teams in tiers of favoritism to win the NCAA Tournament.</p>

      <figure class="inline-fig wide">
        <img src="/evanmiya-landscape.jpg" alt="EvanMiya&rsquo;s March Madness Predicted Efficiency Landscape">
        <figcaption class="fig-credit"><a href="https://substack.com/home/post/p-191132130" target="_blank" rel="noopener">https://substack.com/home/post/p-191132130</a></figcaption>
      </figure>

      <p>I highly recommend checking out EvanMiya&rsquo;s work for anyone interested in basketball.</p>

      <p>In accordance with him, let&rsquo;s make our own bands of favoritism to win VCT tournaments based on this graph.</p>

      <figure class="fig">
        <div class="fig-filter">
          <div class="ls-win" id="tierWin"></div>
          <div class="ls-events" id="tierEvents"></div>
        </div>
        <div class="fig-wrap"><canvas id="tierLandscape"></canvas></div>
      </figure>
    </div>
  </div>
</div>

<script>
const LS = __LANDSCAPE_JSON__;
const logoCache = {}, greyCache = {}, wCache = {};
const charts = [];
let redrawQueued = false;
function queueRedraw() {
  if (redrawQueued) return;
  redrawQueued = true;
  requestAnimationFrame(() => {
    redrawQueued = false;
    charts.forEach(c => { try { c.draw(); } catch (e) {} });
  });
}
function logo(org) {
  if (!logoCache[org]) {
    const i = new Image();
    i.onload = queueRedraw;
    i.src = '/logos/' + org + '.png';
    logoCache[org] = i;
  }
  return logoCache[org];
}
// Greyscale baked once per logo. Doing it with ctx.filter per image per frame
// is a full filter pass 100+ times a frame, which is what made hovering crawl.
function grey(org) {
  const img = logo(org);
  if (!img.complete || !img.naturalWidth) return null;
  if (!greyCache[org]) {
    const c = document.createElement('canvas');
    c.width = img.naturalWidth; c.height = img.naturalHeight;
    const g = c.getContext('2d');
    g.filter = 'grayscale(1)';
    g.drawImage(img, 0, 0);
    greyCache[org] = c;
  }
  return greyCache[org];
}
function textW(ctx, t, font) {
  const k = font + '|' + t;
  if (wCache[k] === undefined) { ctx.font = font; wCache[k] = ctx.measureText(t).width; }
  return wCache[k];
}
const pct = v => (v * 100).toFixed(1) + '%';
const ringFor = intl => intl.indexOf('Champions') === 0 ? '#d8a93a' : '#7c4dd6';

Chart.Tooltip.positioners.aboveMark = function (items) {
  if (!items.length) return false;
  const e = items[0].element;
  return {x: e.x, y: e.y - 30};
};

const plate = {
  id: 'plate',
  beforeDraw(c) {
    const {ctx} = c;
    ctx.save(); ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, c.width, c.height); ctx.restore();
  }
};

const fifty = {
  id: 'fifty',
  beforeDatasetsDraw(c) {
    const {ctx, chartArea: a, scales: {x, y}} = c;
    ctx.save();
    ctx.strokeStyle = 'rgba(61,26,110,.38)'; ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(a.left, y.getPixelForValue(0.5)); ctx.lineTo(a.right, y.getPixelForValue(0.5));
    ctx.moveTo(x.getPixelForValue(0.5), a.top);  ctx.lineTo(x.getPixelForValue(0.5), a.bottom);
    ctx.stroke();
    ctx.restore();
  }
};

// Tiers of favouritism, EvanMiya-style. A band is a range of attack% + defense%,
// so every boundary is the same diagonal slope — a team can reach a tier by
// being strong on either side or balanced across both. Thresholds are set from
// the teams the author named: 1.168 clears FPX at Shanghai (1.160) while taking
// in G2 (1.176) and Vitality (1.177) at Bangkok; 1.08 sits just under NRG at
// Champions 2025 (1.086) and PRX at Toronto (1.084); 1.015 sits just above T1
// at Bangkok (1.008), who names the floor tier.
const BANDS = [
  {lo: 1.168, hi: 2.00,  label: 'Trophy Favorites',  fill: 'rgba(216,169,58,.20)', ink: '#8a6a1a'},
  {lo: 1.08,  hi: 1.168, label: 'Trophy Contenders', fill: 'rgba(124,77,214,.16)', ink: '#5b21b6'},
  {lo: 1.015, hi: 1.08,  label: 'Trophy Believers',  fill: 'rgba(37,99,235,.14)',  ink: '#1d4ed8'},
  {lo: 0.00,  hi: 1.015, label: 'T1 Tier',           fill: 'rgba(220,38,38,.12)',  ink: '#b91c1c'}
];

const bands = {
  id: 'bands',
  beforeDatasetsDraw(c) {
    if (!c.$bands) return;
    const {ctx, chartArea: a, scales: {x, y}} = c;
    const px = (vx, vy) => [x.getPixelForValue(vx), y.getPixelForValue(vy)];
    ctx.save();
    ctx.beginPath(); ctx.rect(a.left, a.top, a.right - a.left, a.bottom - a.top); ctx.clip();
    // Each band is the strip between two constant-sum diagonals. Drawn as a
    // parallelogram running well past the axes and clipped to the plot.
    c.$bands.forEach(b => {
      const p1 = px(0.20, b.hi - 0.20), p2 = px(0.90, b.hi - 0.90);
      const p3 = px(0.90, b.lo - 0.90), p4 = px(0.20, b.lo - 0.20);
      ctx.beginPath();
      ctx.moveTo(p1[0], p1[1]); ctx.lineTo(p2[0], p2[1]);
      ctx.lineTo(p3[0], p3[1]); ctx.lineTo(p4[0], p4[1]);
      ctx.closePath();
      ctx.fillStyle = b.fill; ctx.fill();
      if (b.lo > 0.01) {
        const l1 = px(0.20, b.lo - 0.20), l2 = px(0.90, b.lo - 0.90);
        ctx.beginPath(); ctx.moveTo(l1[0], l1[1]); ctx.lineTo(l2[0], l2[1]);
        ctx.setLineDash([6, 5]); ctx.lineWidth = 1.25;
        ctx.strokeStyle = 'rgba(61,26,110,.35)'; ctx.stroke(); ctx.setLineDash([]);
      }
    });
    ctx.restore();

    // Titles, placed at the middle of each band's visible run so they sit in
    // open space rather than on the diagonal's ends.
    ctx.save();
    c.$bands.forEach(b => {
      const m = (b.lo + Math.min(b.hi, 1.40)) / 2;
      let x0 = Math.max(0.40, m - 0.70), x1 = Math.min(0.70, m - 0.40);
      if (x1 <= x0) return;
      // Near the LEFT end of the strip rather than its middle: a diagonal band's
      // centre runs through the thick of the cloud, where the title lands on
      // top of teams. The upper-left of each band is open space.
      const cx = x0 + 0.12 * (x1 - x0), cy = m - cx;
      if (cy < 0.40 || cy > 0.70) return;
      const [sx, sy] = px(cx, cy);
      const font = "800 14px 'DM Sans',sans-serif";
      const w = textW(ctx, b.label, font) + 22;
      ctx.font = font; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillStyle = 'rgba(255,255,255,.88)';
      ctx.strokeStyle = 'rgba(61,26,110,.20)'; ctx.lineWidth = 1;
      const h = 28, rx = sx - w / 2, ry = sy - h / 2;
      ctx.beginPath();
      if (ctx.roundRect) ctx.roundRect(rx, ry, w, h, 14);
      else ctx.rect(rx, ry, w, h);
      ctx.fill(); ctx.stroke();
      ctx.fillStyle = b.ink;
      ctx.fillText(b.label, sx, sy + 0.5);
    });
    ctx.restore();
  }
};

const marks = {
  id: 'marks',
  afterDatasetsDraw(c) {
    const {ctx} = c;
    const st = c.$state ? c.$state() : {hovered: null, pinned: null, winnersOn: false};
    const focusIdx = st.hovered !== null ? st.hovered : st.pinned;
    const meta = c.getDatasetMeta(0);
    const rank = i => {
      if (i === focusIdx) return 2;
      const p = c.data.datasets[0].data[i].p;
      const lit = focusIdx !== null ? false : (!st.winnersOn || p.won);
      return lit ? 1 : 0;
    };
    const order = meta.data.map((_, i) => i).sort((a, b) => rank(a) - rank(b));
    order.forEach(i => {
      const pt = meta.data[i];
      const p = c.data.datasets[0].data[i].p;
      const img = logo(p.org);
      const on  = focusIdx === i;
      const dim = focusIdx !== null ? !on : (st.winnersOn && !p.won);
      const S = (p.won ? 30 : 25) * (on ? 1.7 : 1);
      const src = dim ? grey(p.org) : img;
      ctx.save();
      ctx.globalAlpha = dim ? 0.45 : 1;
      if (src && (dim || (img.complete && img.naturalWidth))) {
        ctx.drawImage(src, pt.x - S / 2, pt.y - S / 2, S, S);
      }
      ctx.globalAlpha = 1;
      const font = (on ? "800 12px " : "700 9px ") + "'DM Sans',sans-serif";
      const w = textW(ctx, p.intl, font);
      ctx.font = font;
      ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      const ly = pt.y + S / 2 + 3;
      ctx.lineWidth = on ? 3.5 : 2;
      ctx.lineJoin = 'round';
      ctx.strokeStyle = dim ? 'rgba(190,184,196,.5)' : ringFor(p.intl);
      ctx.strokeText(p.intl, pt.x, ly);
      ctx.fillStyle = dim ? 'rgba(150,142,158,.55)' : '#ffffff';
      ctx.fillText(p.intl, pt.x, ly);
      ctx.restore();
    });
  }
};

const AXIS = title => ({
  min: 0.40, max: 0.70,
  title: {display: true, text: title,
          font: {family: "'DM Sans',sans-serif", size: 11, weight: 600}, color: '#7a6e7e'},
  ticks: {callback: v => pct(v), stepSize: 0.05, font: {size: 10}, color: '#9a8fa4'},
  grid: {color: 'rgba(0,0,0,.05)'}
});

// One interactive landscape. Built twice — plain, then with tier bands — so its
// state lives in a closure rather than on the page, or the two would share a
// hover.
function buildLandscape({canvas, winBox, eventBox, bandDefs, winnersDefault = true}) {
  let hovered = null, pinned = null, active = 'All', winnersOn = winnersDefault, chart;
  const HOVER_PX = 20;

  const visible = () => LS.points.filter(p => active === 'All' || p.intl === active)
                                 .map(p => ({x: p.dfn, y: p.atk, p}));
  const hitRadii = data => data.map(d => (winnersOn && !d.p.won) ? 0 : 16);

  function draw() {
    const data = visible();
    if (chart) {
      chart.data.datasets[0].data = data;
      chart.data.datasets[0].hitRadius = hitRadii(data);
      hovered = null; chart.update(); return;
    }
    chart = new Chart(document.getElementById(canvas), {
      type: 'scatter',
      data: {datasets: [{data, pointRadius: 0, pointHoverRadius: 0, hitRadius: hitRadii(data)}]},
      options: {
        responsive: true, maintainAspectRatio: false, animation: false,
        layout: {padding: {top: 14, right: 16, bottom: 4, left: 4}},
        interaction: {mode: 'nearest', intersect: true, axis: 'xy'},
        events: ['mousemove', 'mouseout', 'click', 'touchstart', 'touchmove'],
        onHover(e, els, c) {
          if (pinned !== null) { pinned = null; c.draw(); }
          let i = null;
          if (els.length) {
            const pt = c.getDatasetMeta(0).data[els[0].index];
            if (Math.hypot(e.x - pt.x, e.y - pt.y) <= HOVER_PX) i = els[0].index;
          }
          if (i !== hovered) { hovered = i; c.draw(); }
        },
        scales: {x: AXIS('Defense win%'), y: AXIS('Attack win%')},
        plugins: {
          legend: {display: false},
          tooltip: {
            displayColors: false, backgroundColor: 'rgba(22,18,29,.94)', padding: 10,
            position: 'aboveMark', yAlign: 'bottom', xAlign: 'center',
            animation: {duration: 140},
            animations: {numbers: {duration: 0}, opacity: {duration: 140, easing: 'linear'}},
            caretSize: 5,
            callbacks: {
              title: it => it[0].raw.p.org + ' — ' + it[0].raw.p.intl + (it[0].raw.p.won ? '  (won it)' : ''),
              label: it => {
                const p = it.raw.p;
                return ['from ' + p.prior,
                        'Attack  ' + pct(p.atk) + '  (' + p.atk_w + '/' + p.atk_n + ')',
                        'Defense ' + pct(p.dfn) + '  (' + p.def_w + '/' + p.def_n + ')'];
              }
            }
          }
        }
      },
      plugins: [plate, bands, fifty, marks]
    });
    chart.$state = () => ({hovered, pinned, winnersOn});
    if (bandDefs) chart.$bands = bandDefs;
    charts.push(chart);
    chart.canvas.addEventListener('mouseleave', () => {
      if (hovered !== null || pinned !== null) { hovered = null; pinned = null; chart.draw(); }
    });
  }

  const box = document.getElementById(eventBox);
  const evBtns = [];
  ['All'].concat(LS.internationals).forEach(name => {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'lsb' + (name === 'All' ? ' on' : '');
    b.textContent = name;
    b.onclick = () => {
      active = name; pinned = null;
      if (chart) chart.tooltip.setActiveElements([], {});
      evBtns.forEach(c => c.classList.toggle('on', c === b));
      draw();
    };
    evBtns.push(b);
    box.appendChild(b);
  });
  const w = document.createElement('button');
  w.type = 'button';
  const setLabel = () => {
    w.innerHTML = 'Highlight winners <span class="lsb-state">' + (winnersOn ? 'ON' : 'OFF') + '</span>';
  };
  w.className = 'lsb lsb-win' + (winnersOn ? ' on' : '');
  setLabel();
  w.onclick = () => {
    winnersOn = !winnersOn; pinned = null;
    if (chart) chart.tooltip.setActiveElements([], {});
    w.classList.toggle('on', winnersOn);
    setLabel();
    draw();
  };
  document.getElementById(winBox).appendChild(w);

  // Fit the row to one line by measurement — ten event names are wider than the
  // column at any comfortable size, and a tuned constant breaks when an event
  // is added.
  function fitRow() {
    const steps = [[.68, 10], [.64, 9], [.60, 8], [.56, 7], [.52, 6], [.48, 5], [.44, 4], [.40, 4]];
    for (const [fs, px] of steps) {
      box.style.setProperty('--lsb-fs', fs + 'rem');
      box.style.setProperty('--lsb-px', px + 'px');
      if (box.scrollWidth <= box.clientWidth) return;
    }
  }
  fitRow();
  addEventListener('resize', fitRow);
  draw();

  return {
    pin(org, intl) {
      active = 'All';
      evBtns.forEach(c => c.classList.toggle('on', c.textContent === 'All'));
      if (winnersOn) { winnersOn = false; w.classList.remove('on'); setLabel(); }
      draw();
      const i = chart.data.datasets[0].data.findIndex(d => d.p.org === org && d.p.intl === intl);
      pinned = i >= 0 ? i : null;
      chart.draw();
      if (pinned !== null) {
        const el = chart.getDatasetMeta(0).data[pinned];
        chart.tooltip.setActiveElements([{datasetIndex: 0, index: pinned}], {x: el.x, y: el.y});
        chart.update();
      }
    }
  };
}

const mainLandscape = buildLandscape({canvas: 'sideLandscape', winBox: 'lsWin', eventBox: 'lsEvents'});
buildLandscape({canvas: 'tierLandscape', winBox: 'tierWin', eventBox: 'tierEvents',
                bandDefs: BANDS, winnersDefault: false});

// Static, non-interactive: the cluster the T1 sentence points at.
(function () {
  const el = document.getElementById('bangkokInset');
  if (!el) return;
  const pts = LS.points.filter(p => p.intl === 'Masters Bangkok 2025')
                       .map(p => ({x: p.dfn, y: p.atk, p}));
  const c = new Chart(el, {
    type: 'scatter',
    data: {datasets: [{data: pts, pointRadius: 0, pointHoverRadius: 0, hitRadius: 0}]},
    options: {
      responsive: true, maintainAspectRatio: false, animation: false,
      layout: {padding: {top: 12, right: 14, bottom: 2, left: 2}},
      events: [],
      scales: {x: AXIS('Defense win%'), y: AXIS('Attack win%')},
      plugins: {legend: {display: false}, tooltip: {enabled: false}}
    },
    plugins: [plate, fifty, marks]
  });
  c.$state = () => ({hovered: null, pinned: null, winnersOn: false});
  charts.push(c);
})();

// In-text mentions drive the first chart.
document.querySelectorAll('.pin').forEach(a => {
  a.addEventListener('click', ev => {
    ev.preventDefault();
    mainLandscape.pin(a.dataset.org, a.dataset.intl);
    document.querySelector('.fig').scrollIntoView({behavior: 'smooth', block: 'center'});
  });
});
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
