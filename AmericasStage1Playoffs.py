"""Article: Americas Stage 1 Playoffs Preview.

Mirrors the layout of IdentifyingOverUnderPerformers (label / h1 / dek /
byline / hero image / body). Body is a placeholder for now — fill in prose
and any charts the same way the over-underperformers piece does.
"""

import os
from flask import Blueprint, render_template_string

article_americas_stage1_bp = Blueprint("article_americas_stage1", __name__)

PAGE_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Americas Stage 1 Playoffs Preview &mdash; Bobo's VCT Database</title>
<link rel="icon" type="image/svg+xml" href="/favicon.svg">
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
<style>
  :root {
    --rose:#f4b8c1; --peach:#f9cba7; --mint:#b8e8d4;
    --sky:#b8d8f4; --lavender:#d4b8f4; --lemon:#f4edb8;
    --cream:#fdf6f0; --ink:#2a1f2d; --soft:#7a6e7e;
  }
  * { box-sizing:border-box; margin:0; padding:0; }
  body { background:var(--cream); font-family:'DM Sans',sans-serif; color:var(--ink); min-height:100vh; display:flex; flex-direction:column; }
  body::before {
    content:''; position:fixed; inset:0; pointer-events:none; z-index:0;
    background:
      radial-gradient(ellipse 60% 50% at 10% 10%,#f4b8c155 0%,transparent 70%),
      radial-gradient(ellipse 50% 60% at 90% 20%,#b8d8f455 0%,transparent 70%),
      radial-gradient(ellipse 55% 45% at 15% 85%,#b8e8d455 0%,transparent 70%),
      radial-gradient(ellipse 60% 50% at 85% 80%,#d4b8f455 0%,transparent 70%);
  }
  body::after {
    content:''; position:fixed; inset:-50%; pointer-events:none; z-index:0;
    background:
      radial-gradient(ellipse 60% 50% at 60% 55%,#c4a0f099 0%,transparent 55%),
      radial-gradient(ellipse 50% 60% at 38% 42%,#d4a97477 0%,transparent 55%);
    animation:purpleFloat 12s ease-in-out infinite alternate;
  }
  @keyframes purpleFloat {
    0%   { transform:translate(0,0) scale(1); }
    33%  { transform:translate(10%,-9%) scale(1.14); }
    66%  { transform:translate(-9%,12%) scale(0.9); }
    100% { transform:translate(7%,5%) scale(1.1); }
  }
  .top-nav { padding:32px 32px 0; position:relative; z-index:1; }
  .home-logo { height:80px; width:auto; display:block; opacity:.85; transition:opacity .2s; }
  .home-logo:hover { opacity:1; }
  .toc { position:fixed; top:32px; right:32px; background:white; border-radius:16px; padding:20px 24px; box-shadow:0 4px 24px #0000000f; display:flex; flex-direction:column; gap:6px; z-index:100; max-width:240px; }
  .toc-title { font-family:'Syne',sans-serif; font-size:.7rem; font-weight:800; letter-spacing:.1em; text-transform:uppercase; color:var(--soft); margin-bottom:4px; }
  .toc a { font-size:.78rem; color:var(--soft); text-decoration:none; font-weight:400; transition:color .15s; line-height:1.4; }
  .toc a:hover { color:var(--ink); }
  .toc a.active { color:var(--ink); font-weight:500; }
  .toc a.toc-sub { padding-left:16px; font-size:.74rem; border-left:2px solid #ede5f3; margin-left:4px; }
  .toc a.toc-sub.active { border-left-color:#7c3aed; color:var(--ink); font-weight:600; }
  @media(max-width:1180px) { .toc { display:none; } }
  .page { position:relative; z-index:1; flex:1; display:flex; flex-direction:column; align-items:center; padding:60px 32px 80px; }
  .article { max-width:860px; width:100%; }
  .label { font-family:'Syne',sans-serif; font-size:.7rem; font-weight:800; letter-spacing:.14em; text-transform:uppercase; color:var(--soft); margin-bottom:16px; }
  h1 { font-family:'Syne',sans-serif; font-size:clamp(2rem,5vw,3.2rem); font-weight:800; letter-spacing:-1px; line-height:1.1; margin-bottom:16px; }
  .dek { font-size:1.05rem; font-weight:300; line-height:1.55; color:var(--ink); margin-bottom:24px; opacity:.85; }
  .byline { font-size:.82rem; color:var(--soft); font-weight:300; margin-bottom:48px; padding-bottom:32px; border-bottom:1px solid #e8e0ec; }
  .cover { width:100%; border-radius:16px; overflow:hidden; margin-bottom:12px; }
  .cover img { width:100%; height:auto; display:block; }
  .cover-caption { font-size:.75rem; color:var(--soft); font-weight:300; font-style:italic; margin-bottom:48px; text-align:center; }
  .content p { font-size:1rem; font-weight:300; line-height:1.8; color:var(--ink); margin-bottom:24px; }
  .content h2 { font-family:'Syne',sans-serif; font-size:1.4rem; font-weight:800; letter-spacing:-0.5px; margin:48px 0 20px; }
  .content h2.centered { text-align:center; }
  /* Bubble-style section header — white pill with pink→purple gradient
     text. Two elements: .section-bubble is the white pill; .section-bubble
     -text receives the gradient via background-clip:text. */
  .section-bubble { display:inline-block; background:white; padding:16px 44px; border-radius:999px; box-shadow:0 6px 32px #0000001a; }
  .section-bubble-text { font-family:'Syne',sans-serif; font-size:1.9rem; font-weight:800; letter-spacing:.01em;
    background-image:linear-gradient(95deg,#f472b6 0%,#a855f7 55%,#7c3aed 100%);
    -webkit-background-clip:text; background-clip:text;
    -webkit-text-fill-color:transparent; color:transparent;
  }
  .section-bubble-wrap { text-align:center; margin:96px 0 32px; }
  .section-bubble-wrap.section-bubble-tight { margin-top:40px; }
  /* Per-team subsection header: logo + name on one line. */
  .team-heading { display:flex; align-items:center; gap:14px; margin:48px 0 20px; }
  .team-heading img { width:42px; height:42px; object-fit:contain; flex-shrink:0; }
  .team-heading h2 { margin:0; }
  .inline-figure { margin:24px 0 32px; text-align:center; }
  .inline-figure img { max-width:100%; border-radius:14px; box-shadow:0 6px 24px #0000000f; display:inline-block; }
  .inline-figure-cap { font-size:.75rem; color:var(--soft); font-weight:300; font-style:italic; margin-top:8px; }
  .expand-card { background:white; border-radius:14px; padding:0; box-shadow:0 4px 24px #0000000a; margin:8px 0 32px; overflow:hidden; }
  .expand-card summary { list-style:none; cursor:pointer; padding:14px 22px; display:flex; align-items:center; justify-content:space-between; gap:16px; user-select:none; }
  .expand-card summary::-webkit-details-marker { display:none; }
  .expand-card summary::after { content:'▾'; font-size:.85rem; color:var(--soft); transition:transform .2s; }
  .expand-card[open] summary::after { transform:rotate(180deg); }
  .expand-card-label { font-family:'Syne',sans-serif; font-size:.7rem; font-weight:800; letter-spacing:.1em; text-transform:uppercase; color:var(--soft); }
  .expand-card-headline { font-family:'Syne',sans-serif; font-size:1.1rem; font-weight:800; color:var(--ink); font-variant-numeric:tabular-nums; }
  .expand-card-body { padding:0 22px 16px; border-top:1px solid #f5eff8; }
  .expand-card-body table { width:100%; border-collapse:collapse; font-size:.9rem; margin-top:8px; }
  .expand-card-body td { padding:8px 4px; border-bottom:1px solid #f5eff8; }
  .expand-card-body tr:last-child td { border-bottom:none; }
  .expand-card-body .L { color:#a33247; font-weight:600; }
  .expand-card-body .W { color:#1a6a4a; font-weight:600; }
  /* Bulleted list used in the per-team thought sections. Em-dash bullet for
     editorial feel — literal em-dash char (not the \\2014 escape, which gets
     eaten by Python's octal-escape parsing in this triple-quoted string). */
  .content ul { list-style:none; margin:-8px 0 24px; display:flex; flex-direction:column; gap:8px; }
  .content ul li { font-size:1rem; font-weight:300; line-height:1.8; padding-left:20px; position:relative; color:var(--ink); }
  .content ul li::before { content:'—'; position:absolute; left:0; color:var(--soft); }
  /* Per-team stat card shown right under each team's heading. BenPom + Stage
     1 Pythagorean on top, match-by-match Stage 1 results below. */
  .team-stat-card { background:white; border-radius:14px; padding:18px 22px; box-shadow:0 4px 24px #0000000a; margin:0 0 28px; }
  .team-stat-row { display:flex; gap:56px; margin-bottom:16px; flex-wrap:wrap; justify-content:center; }
  .team-stat-block { display:flex; flex-direction:column; gap:3px; align-items:center; text-align:center; }
  .team-stat-label { font-family:'Syne',sans-serif; font-size:.62rem; font-weight:800; letter-spacing:.1em; text-transform:uppercase; color:var(--soft); }
  .team-stat-value { font-family:'Syne',sans-serif; font-size:1.45rem; font-weight:800; font-variant-numeric:tabular-nums; }
  .team-stat-value.pos { color:#1a6a4a; }
  .team-stat-value.neg { color:#a33247; }
  .team-stat-matches { display:flex; flex-wrap:wrap; gap:10px; align-items:flex-end; justify-content:center; }
  .team-stat-matches-label { font-family:'Syne',sans-serif; font-size:1rem; font-weight:800; letter-spacing:.08em; text-transform:uppercase; color:var(--ink); margin:0 0 10px; display:block; text-align:center; }
  .match-col { display:flex; flex-direction:column; align-items:stretch; gap:3px; }
  .match-week { font-family:'Syne',sans-serif; font-size:.55rem; font-weight:800; letter-spacing:.08em; text-transform:uppercase; color:var(--soft); text-align:center; }
  .match-chip { display:inline-flex; align-items:center; gap:6px; padding:5px 10px; border-radius:7px; font-size:.78rem; font-weight:500; font-variant-numeric:tabular-nums; }
  .match-chip .res { font-weight:800; font-family:'Syne',sans-serif; }
  .match-chip.win { background:rgba(34,197,94,.13); color:#176a47; }
  .match-chip.loss { background:rgba(220,38,38,.11); color:#a33247; }
  .content em { font-style:italic; }
  .content a { color:var(--ink); font-weight:400; }
  .content a:hover { opacity:.7; }
  .data-table-wrap { background:white; border-radius:16px; padding:20px 24px; box-shadow:0 4px 24px #0000000a; margin:24px 0 32px; }
  .data-table-label { font-family:'Syne',sans-serif; font-size:.7rem; font-weight:800; letter-spacing:.1em; text-transform:uppercase; color:var(--soft); margin-bottom:12px; }
  .data-table { width:100%; border-collapse:collapse; font-size:.9rem; font-weight:400; }
  .data-table th { font-family:'Syne',sans-serif; font-size:.68rem; font-weight:800; text-transform:uppercase; letter-spacing:.08em; color:var(--soft); padding:8px 12px; text-align:left; border-bottom:2px solid #f0eaf4; }
  .data-table td { padding:10px 12px; border-bottom:1px solid #f5eff8; }
  .data-table tr:last-child td { border-bottom:none; }
  .data-table .num { font-variant-numeric:tabular-nums; }
  .data-table .rank { color:var(--soft); font-weight:500; width:36px; }
  .data-table .team { font-weight:600; }
  .data-table .team-cell { display:flex; align-items:center; gap:10px; }
  .data-table .team-logo { width:22px; height:22px; object-fit:contain; flex-shrink:0; }
  /* VLR-style pick'em bracket: each match is a 2-row card (one team per row),
     winners highlighted with a green bar + bold name, losers muted. Rounds
     flow left to right; upper bracket sits above lower, GF centered below. */
  /* Break out of the article's 860px max-width so the whole bracket fits in
     one view. Pinned to the viewport center; never wider than 1200px. */
  .bracket-wrap { background:white; border-radius:16px; padding:24px 28px; box-shadow:0 4px 24px #0000000a; margin:24px 0 32px; overflow-x:auto; position:relative; left:50%; transform:translateX(-50%); width:min(96vw,1200px); max-width:none; }
  .bracket-label { font-family:'Syne',sans-serif; font-size:.7rem; font-weight:800; letter-spacing:.1em; text-transform:uppercase; color:var(--soft); margin-bottom:18px; text-align:center; }
  .bracket-section { margin-bottom:32px; min-width:fit-content; }
  .bracket-section-title { font-family:'Syne',sans-serif; font-size:.8rem; font-weight:800; letter-spacing:.05em; color:var(--ink); margin-bottom:10px; padding-bottom:6px; border-bottom:1px solid #f0eaf4; }
  .bracket-rounds { display:flex; gap:28px; align-items:stretch; }
  .bracket-round { flex:1; min-width:160px; display:flex; flex-direction:column; justify-content:space-around; gap:18px; }
  .bracket-round-title { font-family:'Syne',sans-serif; font-size:.6rem; font-weight:800; letter-spacing:.1em; text-transform:uppercase; color:var(--soft); text-align:center; margin-bottom:4px; }
  .bracket-cell { background:#faf6fd; border-radius:10px; padding:0; overflow:hidden; font-size:.82rem; box-shadow:0 1px 4px #00000008; }
  .bracket-cell .slot { display:flex; align-items:center; gap:8px; padding:8px 12px; }
  .bracket-cell .slot + .slot { border-top:1px solid #ece5f3; }
  .bracket-cell .slot img { width:18px; height:18px; object-fit:contain; flex-shrink:0; }
  .bracket-cell .slot .name { flex:1; font-weight:500; }
  .bracket-cell .slot .score { font-family:'Syne',sans-serif; font-weight:800; font-variant-numeric:tabular-nums; font-size:.78rem; color:var(--soft); }
  .bracket-cell .slot.winner { background:rgba(34,197,94,.13); }
  .bracket-cell .slot.winner .name { color:#176a47; font-weight:700; }
  .bracket-cell .slot.winner .score { color:#176a47; }
  .bracket-cell .slot.loser .name { color:#a89db4; }
  .bracket-cell .slot.loser img { opacity:.6; }
  .bracket-cell .slot.tbd .name { color:#c2b8ce; font-style:italic; }
  .bracket-cell-pct { font-family:'Syne',sans-serif; font-size:.58rem; font-weight:800; letter-spacing:.08em; text-transform:uppercase; color:var(--soft); padding:4px 12px; background:#f3eef8; text-align:right; }
  /* Bracket body splits into two columns: main (upper + lower stacked) on
     the left, the Grand Final pinned to the right and vertically centered. */
  .bracket-body { display:flex; gap:32px; align-items:stretch; min-width:fit-content; }
  .bracket-main { flex:1; min-width:0; }
  .bracket-gf-col { display:flex; flex-direction:column; justify-content:center; align-items:center; padding-left:8px; border-left:1px dashed #ece5f3; }
  .bracket-gf { display:flex; flex-direction:column; align-items:center; }
  .bracket-gf .bracket-round-title { font-size:.78rem; color:#5a2a7a; margin-bottom:8px; letter-spacing:.12em; }
  .bracket-gf .bracket-cell { min-width:220px; background:linear-gradient(180deg,#faf2f8,#f4eaf8); box-shadow:0 4px 18px #5a2a7a15; }
  .bracket-gf .bracket-cell .slot.winner { background:linear-gradient(90deg,rgba(124,58,237,.16),rgba(34,197,94,.18)); }
  .bracket-final-line { margin-top:18px; font-family:'Syne',sans-serif; font-size:.72rem; font-weight:800; letter-spacing:.1em; text-transform:uppercase; color:var(--soft); text-align:center; max-width:240px; }
  .bracket-final-line .pod { color:#5a2a7a; display:block; margin-top:4px; }
  .bracket-snapshot { font-size:.7rem; color:var(--soft); font-weight:300; font-style:italic; text-align:center; margin-top:14px; }
  .stat-callout { background:white; border-radius:14px; padding:18px 24px; box-shadow:0 4px 24px #0000000a; margin:8px 0 32px; display:flex; align-items:center; justify-content:center; gap:16px; }
  .stat-callout-label { font-family:'Syne',sans-serif; font-size:.68rem; font-weight:800; letter-spacing:.1em; text-transform:uppercase; color:var(--soft); }
  .stat-callout-team { display:flex; align-items:center; gap:8px; font-weight:600; }
  .stat-callout-team img { width:24px; height:24px; object-fit:contain; }
  .stat-callout-rating { font-family:'Syne',sans-serif; font-size:1.6rem; font-weight:800; color:#1a6a4a; font-variant-numeric:tabular-nums; }
  footer { position:relative; z-index:1; text-align:center; padding:24px; color:var(--soft); font-size:.75rem; font-weight:300; }
  @keyframes fadeUp{from{opacity:0;transform:translateY(24px)}to{opacity:1;transform:translateY(0)}}
  .page { animation:fadeUp .6s ease both; }
</style>
</head>
<body>
<nav class="toc">
  <div class="toc-title">Sections</div>
  <a href="#intro">Introduction</a>
  <a href="#teams-to-watch">Teams To Watch</a>
  <a class="toc-sub" href="#loud">LOUD</a>
  <a class="toc-sub" href="#100t">100 Thieves</a>
  <a href="#shorter-team-thoughts">Shorter Team Thoughts</a>
  <a class="toc-sub" href="#g2">G2</a>
  <a class="toc-sub" href="#furia">FURIA</a>
  <a href="#benpoms-predictions">BenPom&rsquo;s Predictions</a>
  <a href="#my-predictions">My Predictions</a>
</nav>
<div class="top-nav">
  <a href="/"><img src="/logo.svg" alt="Home" class="home-logo"></a>
</div>
<div class="page">
  <div class="article">
    <div class="label">Research / Opinion</div>
    <h1>Americas Stage 1 Playoffs Preview</h1>
    <div class="byline">Bobo &mdash; May 2026</div>
    <div class="cover">
      <img src="/loudlev26.jpg" alt="LOUD vs Leviat&aacute;n at VCT Americas Stage 1 2026">
    </div>
    <p class="cover-caption">Lukxo and Sato hug after LOUD beat the heavily-favored Leviat&aacute;n 2&ndash;1 &mdash; a key moment in LOUD&rsquo;s resurgence all the way into Playoffs.</p>
    <div class="content">
      <p id="intro">Looking ahead into Americas Stage 1 Playoffs, BenPom says that G2 are the no.&nbsp;1 team to beat, which makes sense as the perennial powerhouse of Americas. Yawn. After that it&rsquo;s&hellip; 100 Thieves? And then Leviat&aacute;n? Also, LOUD are in the Playoffs? Suddenly, things look a lot more interesting! This split, we have one of the most interesting fields in domestic history, with narratives galore. Let&rsquo;s discuss and make some predictions!</p>

      <div class="section-bubble-wrap"><span class="section-bubble" id="teams-to-watch"><span class="section-bubble-text">Teams To Watch</span></span></div>

      <div class="team-heading">
        <img src="/logos/LOUD.png" alt="LOUD" onerror="this.style.display='none'">
        <h2 id="loud">LOUD</h2>
      </div>

      <div class="team-stat-card">
        <div class="team-stat-row">
          <div class="team-stat-block">
            <div class="team-stat-label">BenPom</div>
            <div class="team-stat-value neg">&minus;1.02</div>
          </div>
          <div class="team-stat-block">
            <div class="team-stat-label">Stage&nbsp;1 Pythagorean</div>
            <div class="team-stat-value">41.4%</div>
          </div>
        </div>
        <div class="team-stat-matches">
          <div class="match-col"><div class="match-week">Week 1</div><span class="match-chip win"><span class="res">W</span>ENVY 2&ndash;1</span></div>
          <div class="match-col"><div class="match-week">Week 2</div><span class="match-chip loss"><span class="res">L</span>MIBR 0&ndash;2</span></div>
          <div class="match-col"><div class="match-week">Week 3</div><span class="match-chip loss"><span class="res">L</span>G2 0&ndash;2</span></div>
          <div class="match-col"><div class="match-week">Week 4</div><span class="match-chip win"><span class="res">W</span>C9 2&ndash;1</span></div>
          <div class="match-col"><div class="match-week">Week 5</div><span class="match-chip win"><span class="res">W</span>LEV 2&ndash;1</span></div>
        </div>
      </div>

      <p>As previously alluded to, LOUD&rsquo;s appearance in Stage&nbsp;1 Playoffs is shocking. A few years back, LOUD were a staple of VCT &mdash; 2022 Champions, 2023 Americas Champions, 2nd at LOCK//IN, and 3rd at 2023 Champions. Even in 2024, LOUD were at least 2nd at Kickoff. Since then, it&rsquo;s been nothing but disappointment for the Brazilian fans. In 2025, not only did they fail to make a single international event, they didn&rsquo;t even make one of the two Americas Playoffs (8 out of 12 teams in the Americas do each split). In 2026, any hope you might&rsquo;ve had for LOUD seemed for naught, as they not only had the worst Pythagorean rating at Americas Kickoff, but they had the worst Pythagorean rating of any team across any region during 2026 Kickoff.</p>

      <div class="data-table-wrap">
        <div class="data-table-label">Bottom 5 Pythagorean Ratings &mdash; VCT Kickoff 2026 (all regions)</div>
        <table class="data-table">
          <thead>
            <tr>
              <th>#</th>
              <th>Team</th>
              <th>Region</th>
              <th class="num">Pyth&nbsp;Win%</th>
              <th class="num">Map W&ndash;L</th>
              <th class="num">Rounds W&ndash;L</th>
            </tr>
          </thead>
          <tbody>
            <tr><td class="rank">1</td><td class="team"><div class="team-cell"><img class="team-logo" src="/logos/TS.png"   alt="TS"   onerror="this.style.display='none'">TS</div></td>     <td>Pacific</td>  <td class="num">17.5%</td><td class="num">1&ndash;6</td><td class="num">51&ndash;82</td></tr>
            <tr><td class="rank">2</td><td class="team"><div class="team-cell"><img class="team-logo" src="/logos/LOUD.png" alt="LOUD" onerror="this.style.display='none'">LOUD</div></td>   <td>Americas</td> <td class="num">17.5%</td><td class="num">1&ndash;6</td><td class="num">56&ndash;90</td></tr>
            <tr><td class="rank">3</td><td class="team"><div class="team-cell"><img class="team-logo" src="/logos/KR%C3%9C.png" alt="KR&Uuml;" onerror="this.style.display='none'">KR&Uuml;</div></td><td>Americas</td> <td class="num">23.0%</td><td class="num">2&ndash;6</td><td class="num">67&ndash;97</td></tr>
            <tr><td class="rank">4</td><td class="team"><div class="team-cell"><img class="team-logo" src="/logos/PCF.png"  alt="PCF"  onerror="this.style.display='none'">PCF</div></td>    <td>EMEA</td>     <td class="num">27.4%</td><td class="num">2&ndash;6</td><td class="num">72&ndash;97</td></tr>
            <tr><td class="rank">5</td><td class="team"><div class="team-cell"><img class="team-logo" src="/logos/ULF.png"  alt="ULF"  onerror="this.style.display='none'">ULF</div></td>    <td>EMEA</td>     <td class="num">28.1%</td><td class="num">3&ndash;6</td><td class="num">75&ndash;100</td></tr>
          </tbody>
        </table>
      </div>

      <p>Somehow, in Stage&nbsp;1, they&rsquo;ve begun to turn things around. Last week, they managed to get to 2&ndash;2 (with two reasonable losses to G2 and MIBR and two reasonable wins against Envy and C9). At that point, it seemed like they were no longer bottom-of-the-barrel in VCT Americas. However, with their most recent win against Leviat&aacute;n (who BenPom had as a top-10 team in the world before that match), LOUD is making a push to be considered an international-caliber team.</p>

      <p>Before I go on to question LOUD&rsquo;s legitimacy, it&rsquo;s important to note how great this is as a fan. LOUD has one of the largest fanbases in VCT (perhaps the largest, with 11 million followers on Instagram), a hallowed legacy, and deserving players (this is not just referring to Luk&nbsp;xo; Darker and Erde have proved their worth as rookies). As a neutral party, I give them my full support and would love to see this team succeed.</p>

      <p>However, it&rsquo;s one thing for me to give them my support and another to give them my belief. LOUD were one round away from losing 0&ndash;2 to Leviat&aacute;n <em>twice</em> (rounds 23 and 24 of Breeze). After losing on Bind 13&ndash;3, losing on Split 13&ndash;10 (or 13&ndash;11) would have made this a convincing match loss for LOUD. Let&rsquo;s also remember that this Leviat&aacute;n team has been faltering recently, losing to MIBR the match before this one. On the other hand, they played G2 close (9&ndash;13 and 10&ndash;13), as they actually <em>gained</em> BenPom rating for that loss. Secondly, their roster is new (and young) as Erde has only played the two most recent after just turning 18, notching 1.31 and 1.10 ratings. They should only get better with time. Thirdly, Bind (a map they went 0&ndash;2 on in Split&nbsp;1) is leaving the map pool.</p>

      <p>BenPom has them as the worst team in Americas Playoffs by a healthy margin, at a &minus;1.02 rating. However, I find this to be a bit harsh. These past matches with different roster iterations carry a lot of weight in their rating and perhaps undersell the difference Erde could make (and has made) in their performances.</p>

      <p>If their previous two matches (the only ones with their full roster) were the only ones in the dataset, their BenPom rating would be:</p>

      <div class="stat-callout">
        <div class="stat-callout-team">
          <img src="/logos/LOUD.png" alt="LOUD" onerror="this.style.display='none'">LOUD
        </div>
        <div class="stat-callout-label">BenPom Rating (full-roster only)</div>
        <div class="stat-callout-rating">+0.42</div>
      </div>

      <p>This is (of course) not a statistically sound method, but it&rsquo;s food for thought. This rating would put them above MIBR and just below Furia.</p>

      <p>With the new map pool, more practice time together, and their momentum, this is a team that could easily punch above its weight. Are they the worst team in Americas Playoffs? Probably. Could they beat a top team that&rsquo;s been stumbling recently? 100%. I&rsquo;d like their odds against teams like Kr&uuml;, MIBR, or Furia. With an exciting young roster that&rsquo;s reestablishing LOUD&rsquo;s standards of winning, I will be rooting for LOUD, and you should be too.</p>

      <div class="team-heading">
        <img src="/logos/100T.png" alt="100 Thieves" onerror="this.style.display='none'">
        <h2 id="100t">100 Thieves</h2>
      </div>

      <div class="team-stat-card">
        <div class="team-stat-row">
          <div class="team-stat-block">
            <div class="team-stat-label">BenPom</div>
            <div class="team-stat-value pos">+1.84</div>
          </div>
          <div class="team-stat-block">
            <div class="team-stat-label">Stage&nbsp;1 Pythagorean</div>
            <div class="team-stat-value">64.0%</div>
          </div>
        </div>
        <div class="team-stat-matches">
          <div class="match-col"><div class="match-week">Week 1</div><span class="match-chip win"><span class="res">W</span>EG 2&ndash;0</span></div>
          <div class="match-col"><div class="match-week">Week 2</div><span class="match-chip loss"><span class="res">L</span>SEN 0&ndash;2</span></div>
          <div class="match-col"><div class="match-week">Week 3</div><span class="match-chip loss"><span class="res">L</span>NRG 1&ndash;2</span></div>
          <div class="match-col"><div class="match-week">Week 4</div><span class="match-chip win"><span class="res">W</span>KR&Uuml; 2&ndash;1</span></div>
          <div class="match-col"><div class="match-week">Week 5</div><span class="match-chip win"><span class="res">W</span>FUR 2&ndash;1</span></div>
        </div>
      </div>

      <p>What I find most surprising in BenPom&rsquo;s Americas rankings ahead of Playoffs is 100 Thieves being in second. An important caveat to this statement is the fact that the difference between 1st (G2) and 2nd (100 Thieves) &mdash; a 1.11 differential &mdash; is larger than that between 2nd (100 Thieves) and 5th (Kr&uuml;) &mdash; a 0.63 differential.</p>

      <div class="data-table-wrap">
        <div class="data-table-label">Americas BenPom Ranking &mdash; Ahead of Stage&nbsp;1 Playoffs</div>
        <table class="data-table">
          <thead>
            <tr><th>#</th><th>Team</th><th class="num">BenPom</th></tr>
          </thead>
          <tbody>
            <tr><td class="rank">1</td>  <td class="team"><div class="team-cell"><img class="team-logo" src="/logos/G2.png"   alt="G2"   onerror="this.style.display='none'">G2</div></td>     <td class="num">+2.95</td></tr>
            <tr><td class="rank">2</td>  <td class="team"><div class="team-cell"><img class="team-logo" src="/logos/100T.png" alt="100T" onerror="this.style.display='none'">100T</div></td>   <td class="num">+1.84</td></tr>
            <tr><td class="rank">3</td>  <td class="team"><div class="team-cell"><img class="team-logo" src="/logos/LEV.png"  alt="LEV"  onerror="this.style.display='none'">LEV</div></td>    <td class="num">+1.49</td></tr>
            <tr><td class="rank">4</td>  <td class="team"><div class="team-cell"><img class="team-logo" src="/logos/NRG.png"  alt="NRG"  onerror="this.style.display='none'">NRG</div></td>    <td class="num">+1.49</td></tr>
            <tr><td class="rank">5</td>  <td class="team"><div class="team-cell"><img class="team-logo" src="/logos/KR%C3%9C.png" alt="KR&Uuml;" onerror="this.style.display='none'">KR&Uuml;</div></td><td class="num">+1.21</td></tr>
            <tr><td class="rank">6</td>  <td class="team"><div class="team-cell"><img class="team-logo" src="/logos/FUR.png"  alt="FUR"  onerror="this.style.display='none'">FUR</div></td>    <td class="num">+0.57</td></tr>
            <tr><td class="rank">7</td>  <td class="team"><div class="team-cell"><img class="team-logo" src="/logos/SEN.png"  alt="SEN"  onerror="this.style.display='none'">SEN</div></td>    <td class="num">&minus;0.12</td></tr>
            <tr><td class="rank">8</td>  <td class="team"><div class="team-cell"><img class="team-logo" src="/logos/ENVY.png" alt="ENVY" onerror="this.style.display='none'">ENVY</div></td>   <td class="num">&minus;0.34</td></tr>
            <tr><td class="rank">9</td>  <td class="team"><div class="team-cell"><img class="team-logo" src="/logos/MIBR.png" alt="MIBR" onerror="this.style.display='none'">MIBR</div></td>   <td class="num">&minus;0.63</td></tr>
            <tr><td class="rank">10</td> <td class="team"><div class="team-cell"><img class="team-logo" src="/logos/LOUD.png" alt="LOUD" onerror="this.style.display='none'">LOUD</div></td>   <td class="num">&minus;1.02</td></tr>
            <tr><td class="rank">11</td> <td class="team"><div class="team-cell"><img class="team-logo" src="/logos/C9.png"   alt="C9"   onerror="this.style.display='none'">C9</div></td>     <td class="num">&minus;2.14</td></tr>
            <tr><td class="rank">12</td> <td class="team"><div class="team-cell"><img class="team-logo" src="/logos/EG.png"   alt="EG"   onerror="this.style.display='none'">EG</div></td>     <td class="num">&minus;2.24</td></tr>
          </tbody>
        </table>
      </div>

      <p>Last I remember, 100 Thieves were likely to miss Playoffs after only beating EG and then falling to Sentinels and NRG; neither of these teams looked that convincing, and thus 100 Thieves looked doubly unconvincing in losing to them. Their two wins afterwards must&rsquo;ve done a <em>lot</em> of work.</p>

      <p>First, they beat Kr&uuml; 2&ndash;1, who were the highest rated team in VCT at the time:</p>

      <div class="inline-figure">
        <img src="/krustage1.png" alt="Kr&uuml; as the highest-rated team in VCT during Stage 1">
      </div>

      <p>Then, they beat Furia 2&ndash;1, another great team. The key behind this rating, though, is their round differentials. When they beat Kr&uuml;, the only map they lost was decided by 3 rounds while their wins were by 9 and 7. When they beat Furia, the only map they lost was 4 while their wins were by 5 and 7. These round differentials mean something, and BenPom lets us know that.</p>

      <p>Similarly to LOUD, they&rsquo;re another team with deserving players who seem to be catching momentum at the right time, against the right opponents (re: Kr&uuml;). Do I buy it? Honestly, not really. A highly variable 100 Thieves team seems like something to be wary of, especially when they&rsquo;re losing their best map (that 9-round win against Kr&uuml; was on Bind). Additionally, look at their record in Split&nbsp;1 when the game is decided by 3 or fewer rounds:</p>

      <details class="expand-card">
        <summary>
          <span>
            <span class="expand-card-label">100T in close maps (&le;3 rounds) &mdash; Stage 1</span><br>
            <span class="expand-card-headline">0&ndash;4 record</span>
          </span>
        </summary>
        <div class="expand-card-body">
          <table>
            <tr><td>vs SEN</td><td>Haven</td><td class="L">L 10&ndash;13</td><td>(&Delta; &minus;3)</td><td>Apr 19</td></tr>
            <tr><td>vs SEN</td><td>Split</td><td class="L">L 17&ndash;19</td><td>(&Delta; &minus;2)</td><td>Apr 19</td></tr>
            <tr><td>vs NRG</td><td>Lotus</td><td class="L">L 10&ndash;13</td><td>(&Delta; &minus;3)</td><td>Apr 25</td></tr>
            <tr><td>vs Kr&uuml;</td><td>Haven</td><td class="L">L 10&ndash;13</td><td>(&Delta; &minus;3)</td><td>May 2</td></tr>
          </table>
        </div>
      </details>

      <p>That&rsquo;s not a good sign where they&rsquo;re gonna have to play close games against good teams.</p>

      <p>In a bracket where there are slim strength margins between 2nd and 5th, every small detraction can have big implications &mdash; especially when only the top 3 teams qualify. For Cryo&rsquo;s sake, maybe this 100 Thieves team can live up to their rating and qualify for their first international since Masters Shanghai in 2024.</p>

      <div class="section-bubble-wrap"><span class="section-bubble" id="shorter-team-thoughts"><span class="section-bubble-text">Shorter Team Thoughts</span></span></div>

      <div class="team-heading">
        <img src="/logos/G2.png" alt="G2" onerror="this.style.display='none'">
        <h2 id="g2">G2</h2>
      </div>

      <div class="team-stat-card">
        <div class="team-stat-row">
          <div class="team-stat-block">
            <div class="team-stat-label">BenPom</div>
            <div class="team-stat-value pos">+2.95</div>
          </div>
          <div class="team-stat-block">
            <div class="team-stat-label">Stage&nbsp;1 Pythagorean</div>
            <div class="team-stat-value">73.0%</div>
          </div>
        </div>
        <div class="team-stat-matches">
          <div class="match-col"><div class="match-week">Week 1</div><span class="match-chip loss"><span class="res">L</span>MIBR 0&ndash;2</span></div>
          <div class="match-col"><div class="match-week">Week 2</div><span class="match-chip loss"><span class="res">L</span>LEV 1&ndash;2</span></div>
          <div class="match-col"><div class="match-week">Week 3</div><span class="match-chip win"><span class="res">W</span>LOUD 2&ndash;0</span></div>
          <div class="match-col"><div class="match-week">Week 4</div><span class="match-chip win"><span class="res">W</span>ENVY 2&ndash;0</span></div>
          <div class="match-col"><div class="match-week">Week 5</div><span class="match-chip win"><span class="res">W</span>C9 2&ndash;0</span></div>
        </div>
      </div>

      <p>While they&rsquo;re 1st in BenPom by a healthy margin, I&rsquo;m not completely convinced. The three teams they&rsquo;ve beaten this split sit at the bottom of the BenPom rankings for Americas. Meanwhile, the two other teams they played, they lost to. They&rsquo;re (justly) given a cushion for:</p>

      <ul>
        <li>Performing well at Santiago</li>
        <li>Losing by narrow margins (e.g. they won more rounds than they lost in losing to Leviat&aacute;n)</li>
        <li>Winning by healthy margins</li>
      </ul>

      <p>Still, they&rsquo;re not as indomitable as their ranking might suggest.</p>

      <div class="team-heading">
        <img src="/logos/FUR.png" alt="FURIA" onerror="this.style.display='none'">
        <h2 id="furia">FURIA</h2>
      </div>

      <div class="team-stat-card">
        <div class="team-stat-row">
          <div class="team-stat-block">
            <div class="team-stat-label">BenPom</div>
            <div class="team-stat-value pos">+0.57</div>
          </div>
          <div class="team-stat-block">
            <div class="team-stat-label">Stage&nbsp;1 Pythagorean</div>
            <div class="team-stat-value">55.2%</div>
          </div>
        </div>
        <div class="team-stat-matches">
          <div class="match-col"><div class="match-week">Week 1</div><span class="match-chip win"><span class="res">W</span>NRG 2&ndash;0</span></div>
          <div class="match-col"><div class="match-week">Week 2</div><span class="match-chip win"><span class="res">W</span>EG 2&ndash;1</span></div>
          <div class="match-col"><div class="match-week">Week 3</div><span class="match-chip loss"><span class="res">L</span>KR&Uuml; 0&ndash;2</span></div>
          <div class="match-col"><div class="match-week">Week 4</div><span class="match-chip win"><span class="res">W</span>SEN 2&ndash;1</span></div>
          <div class="match-col"><div class="match-week">Week 5</div><span class="match-chip loss"><span class="res">L</span>100T 1&ndash;2</span></div>
        </div>
      </div>

      <p>To me, Furia are the biggest question mark. Not only is it hard to figure out how well they will do, but even to figure out how well they <em>should</em> do. On one hand, their rating is much lower than other Americas teams. On the other hand, this is the same team that went flawless in the last Americas bracket we watched (Kickoff). On one hand, they&rsquo;ve lost 2 of their past 3 matches (handily). On the other hand, their only winless map (Bind) is getting removed from the map pool.</p>

      <p>My thoughts are this: they will not put up poor performances as a proven playoff team. However, in order to win (let alone qualify) they&rsquo;ll need some of these other &ldquo;question mark&rdquo; teams to regress. If teams like Kr&uuml; and Leviat&aacute;n are able to put aside their recent struggles, I&rsquo;d be pessimistic about Furia.</p>

      <div class="section-bubble-wrap section-bubble-tight"><span class="section-bubble" id="benpoms-predictions"><span class="section-bubble-text">BenPom&rsquo;s Predictions</span></span></div>

      <p>Enough narrative and &ldquo;my thoughts.&rdquo; Instead, let&rsquo;s see what BenPom thinks:</p>

      <div class="bracket-wrap">
        <div class="bracket-label">BenPom Prediction</div>
       <div class="bracket-body">
        <div class="bracket-main">
        <div class="bracket-section">
          <div class="bracket-section-title">Upper Bracket</div>
          <div class="bracket-rounds">
            <div class="bracket-round">
              <div class="bracket-round-title">Upper Round 1</div>
              <div class="bracket-cell">
                <div class="slot loser"><img src="/logos/FUR.png" onerror="this.style.display='none'"><span class="name">FURIA</span><span class="score">0</span></div>
                <div class="slot winner"><img src="/logos/LEV.png" onerror="this.style.display='none'"><span class="name">Leviat&aacute;n</span><span class="score">2</span></div>
                <div class="bracket-cell-pct">55%</div>
              </div>
              <div class="bracket-cell">
                <div class="slot winner"><img src="/logos/G2.png" onerror="this.style.display='none'"><span class="name">G2</span><span class="score">2</span></div>
                <div class="slot loser"><img src="/logos/100T.png" onerror="this.style.display='none'"><span class="name">100 Thieves</span><span class="score">0</span></div>
                <div class="bracket-cell-pct">59%</div>
              </div>
            </div>
            <div class="bracket-round">
              <div class="bracket-round-title">Upper Semifinals</div>
              <div class="bracket-cell">
                <div class="slot loser"><img src="/logos/MIBR.png" onerror="this.style.display='none'"><span class="name">MIBR</span><span class="score">0</span></div>
                <div class="slot winner"><img src="/logos/LEV.png" onerror="this.style.display='none'"><span class="name">Leviat&aacute;n</span><span class="score">2</span></div>
                <div class="bracket-cell-pct">60%</div>
              </div>
              <div class="bracket-cell">
                <div class="slot loser"><img src="/logos/KR%C3%9C.png" onerror="this.style.display='none'"><span class="name">KR&Uuml;</span><span class="score">0</span></div>
                <div class="slot winner"><img src="/logos/G2.png" onerror="this.style.display='none'"><span class="name">G2</span><span class="score">2</span></div>
                <div class="bracket-cell-pct">62%</div>
              </div>
            </div>
            <div class="bracket-round">
              <div class="bracket-round-title">Upper Final</div>
              <div class="bracket-cell">
                <div class="slot loser"><img src="/logos/LEV.png" onerror="this.style.display='none'"><span class="name">Leviat&aacute;n</span><span class="score">0</span></div>
                <div class="slot winner"><img src="/logos/G2.png" onerror="this.style.display='none'"><span class="name">G2</span><span class="score">2</span></div>
                <div class="bracket-cell-pct">62%</div>
              </div>
            </div>
          </div>
        </div>

        <div class="bracket-section">
          <div class="bracket-section-title">Lower Bracket</div>
          <div class="bracket-rounds">
            <div class="bracket-round">
              <div class="bracket-round-title">Lower Round 1</div>
              <div class="bracket-cell">
                <div class="slot loser"><img src="/logos/FUR.png" onerror="this.style.display='none'"><span class="name">FURIA</span><span class="score">1</span></div>
                <div class="slot winner"><img src="/logos/NRG.png" onerror="this.style.display='none'"><span class="name">NRG</span><span class="score">2</span></div>
                <div class="bracket-cell-pct">58%</div>
              </div>
              <div class="bracket-cell">
                <div class="slot winner"><img src="/logos/100T.png" onerror="this.style.display='none'"><span class="name">100 Thieves</span><span class="score">2</span></div>
                <div class="slot loser"><img src="/logos/LOUD.png" onerror="this.style.display='none'"><span class="name">LOUD</span><span class="score">0</span></div>
                <div class="bracket-cell-pct">66%</div>
              </div>
            </div>
            <div class="bracket-round">
              <div class="bracket-round-title">Lower Round 2</div>
              <div class="bracket-cell">
                <div class="slot winner"><img src="/logos/NRG.png" onerror="this.style.display='none'"><span class="name">NRG</span><span class="score">2</span></div>
                <div class="slot loser"><img src="/logos/KR%C3%9C.png" onerror="this.style.display='none'"><span class="name">KR&Uuml;</span><span class="score">1</span></div>
                <div class="bracket-cell-pct">54%</div>
              </div>
              <div class="bracket-cell">
                <div class="slot winner"><img src="/logos/100T.png" onerror="this.style.display='none'"><span class="name">100 Thieves</span><span class="score">2</span></div>
                <div class="slot loser"><img src="/logos/MIBR.png" onerror="this.style.display='none'"><span class="name">MIBR</span><span class="score">1</span></div>
                <div class="bracket-cell-pct">61%</div>
              </div>
            </div>
            <div class="bracket-round">
              <div class="bracket-round-title">Lower Round 3</div>
              <div class="bracket-cell">
                <div class="slot winner"><img src="/logos/NRG.png" onerror="this.style.display='none'"><span class="name">NRG</span><span class="score">2</span></div>
                <div class="slot loser"><img src="/logos/100T.png" onerror="this.style.display='none'"><span class="name">100 Thieves</span><span class="score">1</span></div>
                <div class="bracket-cell-pct">51%</div>
              </div>
            </div>
            <div class="bracket-round">
              <div class="bracket-round-title">Lower Final</div>
              <div class="bracket-cell">
                <div class="slot winner"><img src="/logos/NRG.png" onerror="this.style.display='none'"><span class="name">NRG</span><span class="score">2</span></div>
                <div class="slot loser"><img src="/logos/LEV.png" onerror="this.style.display='none'"><span class="name">Leviat&aacute;n</span><span class="score">1</span></div>
                <div class="bracket-cell-pct">53%</div>
              </div>
            </div>
          </div>
        </div>
        </div><!-- /bracket-main -->

        <div class="bracket-gf-col">
         <div class="bracket-gf">
          <div class="bracket-round-title">Grand Final &middot; May 24</div>
          <div class="bracket-cell">
            <div class="slot winner"><img src="/logos/G2.png" onerror="this.style.display='none'"><span class="name">G2</span><span class="score">3</span></div>
            <div class="slot loser"><img src="/logos/NRG.png" onerror="this.style.display='none'"><span class="name">NRG</span><span class="score">1</span></div>
            <div class="bracket-cell-pct">59%</div>
          </div>
          <div class="bracket-final-line">
            Top 3 to Masters London: <span class="pod">G2 &middot; NRG &middot; Leviat&aacute;n</span>
          </div>
         </div>
        </div><!-- /bracket-gf-col -->
       </div><!-- /bracket-body -->
        <div class="bracket-snapshot">Static snapshot of BenPom &amp; per-map ratings as of May&nbsp;12, 2026 (before Playoffs).</div>
      </div>

      <p>After 20,000 simulated brackets, here&rsquo;s how often each team wins it all:</p>

      <div class="data-table-wrap">
        <div class="data-table-label">Odds of winning Americas Stage&nbsp;1 Playoffs</div>
        <table class="data-table">
          <thead><tr><th>#</th><th>Team</th><th class="num">Win%</th></tr></thead>
          <tbody>
            <tr><td class="rank">1</td><td class="team"><div class="team-cell"><img class="team-logo" src="/logos/G2.png"   alt="G2"   onerror="this.style.display='none'">G2</div></td>     <td class="num">29.4%</td></tr>
            <tr><td class="rank">2</td><td class="team"><div class="team-cell"><img class="team-logo" src="/logos/KR%C3%9C.png" alt="KR&Uuml;" onerror="this.style.display='none'">KR&Uuml;</div></td><td class="num">18.9%</td></tr>
            <tr><td class="rank">3</td><td class="team"><div class="team-cell"><img class="team-logo" src="/logos/100T.png" alt="100T" onerror="this.style.display='none'">100T</div></td>   <td class="num">14.1%</td></tr>
            <tr><td class="rank">4</td><td class="team"><div class="team-cell"><img class="team-logo" src="/logos/LEV.png"  alt="LEV"  onerror="this.style.display='none'">LEV</div></td>    <td class="num">13.2%</td></tr>
            <tr><td class="rank">5</td><td class="team"><div class="team-cell"><img class="team-logo" src="/logos/MIBR.png" alt="MIBR" onerror="this.style.display='none'">MIBR</div></td>   <td class="num">10.9%</td></tr>
            <tr><td class="rank">6</td><td class="team"><div class="team-cell"><img class="team-logo" src="/logos/FUR.png"  alt="FUR"  onerror="this.style.display='none'">FUR</div></td>    <td class="num">8.8%</td></tr>
            <tr><td class="rank">7</td><td class="team"><div class="team-cell"><img class="team-logo" src="/logos/NRG.png"  alt="NRG"  onerror="this.style.display='none'">NRG</div></td>    <td class="num">4.1%</td></tr>
            <tr><td class="rank">8</td><td class="team"><div class="team-cell"><img class="team-logo" src="/logos/LOUD.png" alt="LOUD" onerror="this.style.display='none'">LOUD</div></td>   <td class="num">0.6%</td></tr>
          </tbody>
        </table>
      </div>

      <p>And here&rsquo;s how often each team qualifies to London:</p>

      <div class="data-table-wrap">
        <div class="data-table-label">Odds of qualifying for Masters London (top 3 finish)</div>
        <table class="data-table">
          <thead><tr><th>#</th><th>Team</th><th class="num">Top-3 %</th></tr></thead>
          <tbody>
            <tr><td class="rank">1</td><td class="team"><div class="team-cell"><img class="team-logo" src="/logos/G2.png"   alt="G2"   onerror="this.style.display='none'">G2</div></td>     <td class="num">58.1%</td></tr>
            <tr><td class="rank">2</td><td class="team"><div class="team-cell"><img class="team-logo" src="/logos/KR%C3%9C.png" alt="KR&Uuml;" onerror="this.style.display='none'">KR&Uuml;</div></td><td class="num">56.4%</td></tr>
            <tr><td class="rank">3</td><td class="team"><div class="team-cell"><img class="team-logo" src="/logos/MIBR.png" alt="MIBR" onerror="this.style.display='none'">MIBR</div></td>   <td class="num">50.8%</td></tr>
            <tr><td class="rank">4</td><td class="team"><div class="team-cell"><img class="team-logo" src="/logos/LEV.png"  alt="LEV"  onerror="this.style.display='none'">LEV</div></td>    <td class="num">42.7%</td></tr>
            <tr><td class="rank">5</td><td class="team"><div class="team-cell"><img class="team-logo" src="/logos/100T.png" alt="100T" onerror="this.style.display='none'">100T</div></td>   <td class="num">38.2%</td></tr>
            <tr><td class="rank">6</td><td class="team"><div class="team-cell"><img class="team-logo" src="/logos/FUR.png"  alt="FUR"  onerror="this.style.display='none'">FUR</div></td>    <td class="num">34.5%</td></tr>
            <tr><td class="rank">7</td><td class="team"><div class="team-cell"><img class="team-logo" src="/logos/NRG.png"  alt="NRG"  onerror="this.style.display='none'">NRG</div></td>    <td class="num">14.9%</td></tr>
            <tr><td class="rank">8</td><td class="team"><div class="team-cell"><img class="team-logo" src="/logos/LOUD.png" alt="LOUD" onerror="this.style.display='none'">LOUD</div></td>   <td class="num">4.3%</td></tr>
          </tbody>
        </table>
      </div>

      <p>The difference between BenPom predicting NRG to make the Grand Finals and also having the second-worst odds of winning Playoffs highlights the disadvantage of coming in as a 4-seed. BenPom likes NRG in each of its matchups up until G2; the problem is just winning that amount of (borderline) coinflips. On the opposite end of the spectrum, Kr&uuml; is predicted to bomb out of the tournament 0&ndash;2, but has the second-best odds of qualifying, highlighting the advantage of being a 1-seed.</p>

      <p>The only surprising thing here is the amount of faith BenPom has in MIBR. Honestly, I don&rsquo;t think it&rsquo;s wrong, I just thought the margin-heavy algorithm would be too dissuaded by MIBR losing to ENVY 1&ndash;13, 3&ndash;13.</p>

      <div class="section-bubble-wrap"><span class="section-bubble" id="my-predictions"><span class="section-bubble-text">My Predictions</span></span></div>

      <div class="bracket-wrap">
        <div class="bracket-label">My Pick&rsquo;em</div>
       <div class="bracket-body">
        <div class="bracket-main">
        <div class="bracket-section">
          <div class="bracket-section-title">Upper Bracket</div>
          <div class="bracket-rounds">
            <div class="bracket-round">
              <div class="bracket-round-title">Upper Round 1</div>
              <div class="bracket-cell">
                <div class="slot loser"><img src="/logos/FUR.png" onerror="this.style.display='none'"><span class="name">FURIA</span><span class="score">0</span></div>
                <div class="slot winner"><img src="/logos/LEV.png" onerror="this.style.display='none'"><span class="name">Leviat&aacute;n</span><span class="score">2</span></div>
              </div>
              <div class="bracket-cell">
                <div class="slot winner"><img src="/logos/G2.png" onerror="this.style.display='none'"><span class="name">G2</span><span class="score">2</span></div>
                <div class="slot loser"><img src="/logos/100T.png" onerror="this.style.display='none'"><span class="name">100 Thieves</span><span class="score">0</span></div>
              </div>
            </div>
            <div class="bracket-round">
              <div class="bracket-round-title">Upper Semifinals</div>
              <div class="bracket-cell">
                <div class="slot loser"><img src="/logos/MIBR.png" onerror="this.style.display='none'"><span class="name">MIBR</span><span class="score">0</span></div>
                <div class="slot winner"><img src="/logos/LEV.png" onerror="this.style.display='none'"><span class="name">Leviat&aacute;n</span><span class="score">2</span></div>
              </div>
              <div class="bracket-cell">
                <div class="slot winner"><img src="/logos/KR%C3%9C.png" onerror="this.style.display='none'"><span class="name">KR&Uuml;</span><span class="score">2</span></div>
                <div class="slot loser"><img src="/logos/G2.png" onerror="this.style.display='none'"><span class="name">G2</span><span class="score">1</span></div>
              </div>
            </div>
            <div class="bracket-round">
              <div class="bracket-round-title">Upper Final</div>
              <div class="bracket-cell">
                <div class="slot loser"><img src="/logos/LEV.png" onerror="this.style.display='none'"><span class="name">Leviat&aacute;n</span><span class="score">1</span></div>
                <div class="slot winner"><img src="/logos/KR%C3%9C.png" onerror="this.style.display='none'"><span class="name">KR&Uuml;</span><span class="score">2</span></div>
              </div>
            </div>
          </div>
        </div>

        <div class="bracket-section">
          <div class="bracket-section-title">Lower Bracket</div>
          <div class="bracket-rounds">
            <div class="bracket-round">
              <div class="bracket-round-title">Lower Round 1</div>
              <div class="bracket-cell">
                <div class="slot loser"><img src="/logos/FUR.png" onerror="this.style.display='none'"><span class="name">FURIA</span><span class="score">0</span></div>
                <div class="slot winner"><img src="/logos/NRG.png" onerror="this.style.display='none'"><span class="name">NRG</span><span class="score">2</span></div>
              </div>
              <div class="bracket-cell">
                <div class="slot loser"><img src="/logos/100T.png" onerror="this.style.display='none'"><span class="name">100 Thieves</span><span class="score">1</span></div>
                <div class="slot winner"><img src="/logos/LOUD.png" onerror="this.style.display='none'"><span class="name">LOUD</span><span class="score">2</span></div>
              </div>
            </div>
            <div class="bracket-round">
              <div class="bracket-round-title">Lower Round 2</div>
              <div class="bracket-cell">
                <div class="slot winner"><img src="/logos/G2.png" onerror="this.style.display='none'"><span class="name">G2</span><span class="score">2</span></div>
                <div class="slot loser"><img src="/logos/NRG.png" onerror="this.style.display='none'"><span class="name">NRG</span><span class="score">1</span></div>
              </div>
              <div class="bracket-cell">
                <div class="slot winner"><img src="/logos/MIBR.png" onerror="this.style.display='none'"><span class="name">MIBR</span><span class="score">2</span></div>
                <div class="slot loser"><img src="/logos/LOUD.png" onerror="this.style.display='none'"><span class="name">LOUD</span><span class="score">0</span></div>
              </div>
            </div>
            <div class="bracket-round">
              <div class="bracket-round-title">Lower Round 3</div>
              <div class="bracket-cell">
                <div class="slot winner"><img src="/logos/G2.png" onerror="this.style.display='none'"><span class="name">G2</span><span class="score">2</span></div>
                <div class="slot loser"><img src="/logos/MIBR.png" onerror="this.style.display='none'"><span class="name">MIBR</span><span class="score">1</span></div>
              </div>
            </div>
            <div class="bracket-round">
              <div class="bracket-round-title">Lower Final</div>
              <div class="bracket-cell">
                <div class="slot loser"><img src="/logos/LEV.png" onerror="this.style.display='none'"><span class="name">Leviat&aacute;n</span><span class="score">0</span></div>
                <div class="slot winner"><img src="/logos/G2.png" onerror="this.style.display='none'"><span class="name">G2</span><span class="score">2</span></div>
              </div>
            </div>
          </div>
        </div>
        </div><!-- /bracket-main -->

        <div class="bracket-gf-col">
         <div class="bracket-gf">
          <div class="bracket-round-title">Grand Final &middot; May 24</div>
          <div class="bracket-cell">
            <div class="slot winner"><img src="/logos/KR%C3%9C.png" onerror="this.style.display='none'"><span class="name">KR&Uuml;</span><span class="score">3</span></div>
            <div class="slot loser"><img src="/logos/G2.png" onerror="this.style.display='none'"><span class="name">G2</span><span class="score">2</span></div>
          </div>
          <div class="bracket-final-line">
            Top 3 to Masters London: <span class="pod">KR&Uuml; &middot; G2 &middot; Leviat&aacute;n</span>
          </div>
         </div>
        </div><!-- /bracket-gf-col -->
       </div><!-- /bracket-body -->
      </div>
    </div>
  </div>
</div>
<footer>Data sourced from VLR.gg</footer>
<script>
(function() {
  var tocLinks = document.querySelectorAll('.toc a');
  var ids = Array.from(tocLinks).map(function(a) { return a.getAttribute('href').slice(1); });
  function onScroll() {
    var y = window.scrollY + 120;
    var active = ids[0];
    ids.forEach(function(id) {
      var el = document.getElementById(id);
      if (el && el.offsetTop <= y) active = id;
    });
    tocLinks.forEach(function(a) {
      a.classList.toggle('active', a.getAttribute('href') === '#' + active);
    });
  }
  window.addEventListener('scroll', onScroll, { passive: true });
  onScroll();
})();
</script>
</body>
</html>
"""


@article_americas_stage1_bp.route("/")
def index():
    return render_template_string(PAGE_HTML)
