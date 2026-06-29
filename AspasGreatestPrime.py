"""
Article: "The Greatest Prime in VCT History Isn't a Debate"

A data-driven argument that aspas at Champions 2025 (Paris) is, by analytics,
the greatest individual prime VCT has ever seen.

All leaderboards/numbers are computed offline by scrapers/BuildAspasPrimeData.py
into data/article_aspas_prime.json and rendered here.  Methodology:
international events only, minimum 150 rounds played ("candidates").  Masters
London is excluded and 2024 Masters Shanghai has no VLR ratings published, so it
contributes no rated candidates.

Blueprint mounted at /articles/greatest-prime/ by BobosHome.py.
"""

import os
import json

from flask import Blueprint, render_template_string

article_aspas_prime_bp = Blueprint("article_aspas_prime", __name__)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DATA_PATH = os.path.join(DATA_DIR, "article_aspas_prime.json")


def _load():
    with open(DATA_PATH) as f:
        return json.load(f)


def _img(hs):
    return ('<img src="' + hs + '" alt="" loading="lazy">') if hs else '<span class="noimg"></span>'


def _leaderboard_table(rows, stat_label):
    """Build a ranked leaderboard table; the aspas-at-Champions-2025 row is lit."""
    out = [
        '<div class="lb-wrap"><table class="lb-table"><thead><tr>',
        '<th class="c-rank">#</th><th class="c-player">Player</th>',
        '<th class="c-team">Team</th><th class="c-event">Event</th>',
        '<th class="c-stat">' + stat_label + '</th></tr></thead><tbody>',
    ]
    for r in rows:
        cls = ' class="is-aspas"' if r.get("is_aspas") else ""
        out.append(
            "<tr" + cls + ">"
            + '<td class="c-rank">' + str(r["rank"]) + "</td>"
            + '<td class="c-player"><span class="pcell"><span class="phead">' + _img(r["headshot"]) + "</span>"
            + '<span class="pname">' + r["player"] + "</span></span></td>"
            + '<td class="c-team"><span class="tcell">'
            + (('<img class="tlogo" src="' + r["logo"] + '" alt="">') if r.get("logo") else "")
            + "<span>" + r["org"] + "</span></span></td>"
            + '<td class="c-event">' + r["evlabel"] + "</td>"
            + '<td class="c-stat">' + r["disp"] + "</td>"
            + "</tr>"
        )
    out.append("</tbody></table></div>")
    return "".join(out)


def _role_table(rows):
    out = [
        '<div class="lb-wrap"><table class="lb-table role"><thead><tr>',
        '<th class="c-team">Role</th><th class="c-stat">Avg. KAST%</th>',
        "</tr></thead><tbody>",
    ]
    for r in rows:
        cls = ' class="is-aspas"' if r["role"] == "Duelist" else ""
        out.append(
            "<tr" + cls + ">"
            + '<td class="c-team">' + r["role"] + "</td>"
            + '<td class="c-stat">' + ("%.1f%%" % r["kast"]) + "</td></tr>"
        )
    out.append("</tbody></table></div>")
    return "".join(out)


def _fiwr_fi_table(rows):
    """FIWR top-10 with a total-first-interactions column."""
    out = [
        '<div class="lb-wrap"><table class="lb-table"><thead><tr>',
        '<th class="c-rank">#</th><th class="c-player">Player</th>',
        '<th class="c-team">Team</th><th class="c-event">Event</th>',
        '<th class="c-fi">First Ints</th><th class="c-stat">FIWR</th></tr></thead><tbody>',
    ]
    for r in rows:
        cls = ' class="is-aspas"' if r.get("is_aspas") else ""
        out.append(
            "<tr" + cls + ">"
            + '<td class="c-rank">' + str(r["rank"]) + "</td>"
            + '<td class="c-player"><span class="pcell"><span class="phead">' + _img(r["headshot"]) + "</span>"
            + '<span class="pname">' + r["player"] + "</span></span></td>"
            + '<td class="c-team"><span class="tcell">'
            + (('<img class="tlogo" src="' + r["logo"] + '" alt="">') if r.get("logo") else "")
            + "<span>" + r["org"] + "</span></span></td>"
            + '<td class="c-event">' + r["evlabel"] + "</td>"
            + '<td class="c-fi">' + str(r["fi"]) + "</td>"
            + '<td class="c-stat">' + r["disp"] + "</td>"
            + "</tr>"
        )
    out.append("</tbody></table></div>")
    return "".join(out)


def _role_stat_table(rows, stat_label):
    out = [
        '<div class="lb-wrap"><table class="lb-table role"><thead><tr>',
        '<th class="c-team">Role</th><th class="c-stat">' + stat_label + "</th>",
        "</tr></thead><tbody>",
    ]
    for r in rows:
        cls = ' class="is-aspas"' if r["role"] == "Duelist" else ""
        out.append(
            "<tr" + cls + '><td class="c-team">' + r["role"] + "</td>"
            + '<td class="c-stat">' + r["disp"] + "</td></tr>"
        )
    out.append("</tbody></table></div>")
    return "".join(out)


@article_aspas_prime_bp.route("/")
def index():
    d = _load()
    return render_template_string(
        PAGE_HTML,
        data_json=json.dumps(d),
        rating_table=_leaderboard_table(d["rating_top10"], "Rating"),
        kd_table=_leaderboard_table(d["kd_top10"], "K:D"),
        kpr_table=_leaderboard_table(d["kpr_top10"], "KPR"),
        kast_table=_leaderboard_table(d["kast_top15"], "KAST%"),
        role_table=_role_table(d["role_kast"]),
        duelist_table=_leaderboard_table(d["duelist_kast_top5"], "KAST%"),
        fiwr_table=_leaderboard_table(d["baiting"]["fiwr_top10"], "FIWR"),
        fiwr_fi_table=_fiwr_fi_table(d["baiting"]["fiwr_top10"]),
        role_fiwr_table=_role_stat_table(d["baiting"]["role_fiwr"], "Avg. FIWR"),
        role_fipr_table=_role_stat_table(d["baiting"]["role_fipr"], "Avg. FIPR"),
        duelist_fipr_pctile=d["baiting"]["duelist_fipr_pctile"],
        counts=d["counts"],
        pct=d["avg_1p3_pct"],
        duelist_n=d["duelist_n"],
        aspas=d["aspas"],
    )


PAGE_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=820">
<title>The Greatest Prime in VCT History Isn't a Debate — Bobo's VCT Database</title>
<link rel="icon" type="image/svg+xml" href="/favicon.svg">
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@700;800&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
<link rel="stylesheet" href="/static/base.css">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"></script>
<style>
  html { -webkit-text-size-adjust:100%; text-size-adjust:100%; }
  .page { position:relative; z-index:1; flex:1; display:flex; flex-direction:column; align-items:center; padding:60px 32px 80px; }
  .top-nav { padding:32px 32px 0; position:relative; z-index:1; }
  .home-logo { height:80px; width:auto; display:block; opacity:.85; transition:opacity .2s; }
  .home-logo:hover { opacity:1; }
  .toc { position:fixed; top:32px; right:32px; background:white; border-radius:16px; padding:20px 24px; box-shadow:0 4px 24px #0000000f; display:flex; flex-direction:column; gap:6px; z-index:100; max-width:230px; }
  .toc-title { font-family:'Plus Jakarta Sans',sans-serif; font-size:0.77rem; font-weight:800; letter-spacing:.1em; text-transform:uppercase; color:var(--soft); margin-bottom:4px; }
  .toc a { font-size:.78rem; color:var(--soft); text-decoration:none; font-weight:400; transition:color .15s; line-height:1.4; }
  .toc a:hover { color:var(--ink); }
  .toc a.active { color:var(--ink); font-weight:500; }
  .toc a.sub { padding-left:14px; font-size:.74rem; }
  /* When the Alpha nav bar is injected (fixed at top), drop the Sections box
     below it so they don't overlap. Classic mode (no bar) keeps top:32px. */
  .alpha-navbar ~ .toc { top:72px; }
  @media(max-width:1000px) { .toc { display:none; } }
  .article { max-width:860px; width:100%; }
  .label { font-family:'Plus Jakarta Sans',sans-serif; font-size:0.77rem; font-weight:800; letter-spacing:.14em; text-transform:uppercase; color:var(--soft); margin-bottom:16px; text-align:center; }
  h1 { font-family:'Plus Jakarta Sans',sans-serif; font-size:clamp(2.2rem,5vw,3.52rem); font-weight:800; letter-spacing:-1px; line-height:1.1; margin-bottom:24px; text-align:center; }
  .deck { font-size:1.06rem; font-weight:300; color:var(--soft); line-height:1.6; text-align:center; max-width:640px; margin:-8px auto 22px; }
  .byline { font-size:.82rem; color:var(--soft); font-weight:300; margin-bottom:48px; padding-bottom:32px; border-bottom:1px solid #e8e0ec; text-align:center; }
  .cover { width:100%; border-radius:16px; overflow:hidden; margin-bottom:12px; }
  .cover img { width:100%; height:auto; display:block; }
  .cover-caption { font-size:.75rem; color:var(--soft); font-weight:300; font-style:italic; margin-bottom:48px; text-align:center; }
  .content p { font-size:1rem; font-weight:300; line-height:1.8; color:var(--ink); margin-bottom:24px; }
  .content h2 { font-family:'Plus Jakarta Sans',sans-serif; font-size:1.54rem; font-weight:800; letter-spacing:-0.5px; margin:48px 0 20px; }
  .content p.gs { font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; font-size:1.24rem; letter-spacing:-0.3px; color:var(--ink); line-height:1.35; margin:44px 0 4px; }
  .note { font-weight:400; font-size:.8em; color:var(--soft); letter-spacing:0; margin-left:.35em; white-space:nowrap; }
  .blank { display:inline-block; width:2.4em; border-bottom:1.5px solid currentColor; }
  .qb { white-space:nowrap; }
  /* land anchored titles below the fixed top nav bar instead of under it */
  .content h2, .content p.gs, .cover { scroll-margin-top:84px; }
  .qbubble { background:#faf6ff; border-left:4px solid #b09ad4; border-radius:0 14px 14px 0; padding:18px 24px; margin:10px 0 28px; font-size:1.08rem; font-weight:400; font-style:italic; line-height:1.6; color:var(--ink); position:relative; }
  .qbubble .qsrc { display:block; margin-top:10px; font-size:.8rem; font-style:normal; font-weight:500; color:var(--soft); }
  .syn-card { background:white; border-radius:18px; box-shadow:0 4px 24px #0000000a; padding:24px 18px 26px; margin:22px 0 34px; display:flex; flex-wrap:wrap; justify-content:center; gap:14px 10px; }
  .syn-title { flex:1 1 100%; text-align:center; font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; font-size:1.08rem; letter-spacing:-0.2px; color:var(--ink); margin-bottom:6px; }
  .syn-stat { flex:1 1 124px; min-width:110px; text-align:center; padding:8px 6px; }
  .syn-rank { font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; font-size:2.6rem; line-height:1; color:#e0992a; letter-spacing:-1.5px; }
  .syn-rank.second { color:#9a7ab4; }
  .syn-label { font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; font-size:.82rem; letter-spacing:.03em; text-transform:uppercase; color:var(--ink); margin-top:9px; }
  .syn-label span { display:block; font-size:.64rem; font-weight:500; color:var(--soft); letter-spacing:.08em; margin-top:3px; }
  .syn-of { font-size:.68rem; font-weight:500; color:var(--soft); letter-spacing:.02em; margin-top:6px; }
  .content p.syn-cap { font-size:.62rem; font-style:italic; font-weight:300; color:var(--soft); text-align:center; line-height:1.5; white-space:nowrap; margin:-24px auto 32px; }
  .content a { color:var(--ink); font-weight:400; }
  .content a:hover { opacity:.7; }
  .content strong { font-weight:500; }
  .content em { font-style:italic; }
  .content ul { list-style:none; margin:-8px 0 24px; display:flex; flex-direction:column; gap:8px; }
  .content ul li { font-size:1rem; font-weight:300; line-height:1.8; padding-left:20px; position:relative; }
  .content ul li::before { content:'—'; position:absolute; left:0; color:var(--soft); }
  .content ol.method { counter-reset:m; list-style:none; margin:6px 0 24px; padding:0; }
  .content ol.method > li { counter-increment:m; padding-left:40px; position:relative; margin-bottom:24px; }
  .content ol.method > li:last-child { margin-bottom:0; }
  .content ol.method > li::before { content:counter(m); position:absolute; left:0; top:1px; width:27px; height:27px; border-radius:50%; background:#ece4f4; color:#16121d; font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; font-size:.92rem; display:flex; align-items:center; justify-content:center; }
  .content ol.method > li > p { margin-bottom:14px; }
  .content ol.method > li > p:last-child { margin-bottom:0; }
  .content ol.method > li > ul { margin:14px 0; }
  .pull { font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; font-size:1.18rem; line-height:1.45; letter-spacing:-0.4px; color:var(--ink); text-align:center; margin:8px auto 4px; }
  .kicker { font-family:'Plus Jakarta Sans',sans-serif; font-size:0.77rem; font-weight:800; letter-spacing:.12em; text-transform:uppercase; color:#9a7ab4; margin:40px 0 4px; }
  /* ── leaderboard tables ── */
  .lb-wrap { background:white; border-radius:18px; padding:10px 14px; box-shadow:0 4px 24px #0000000a; margin:20px 0 32px; overflow-x:auto; }
  .lb-table { width:100%; border-collapse:collapse; font-size:.9rem; font-weight:300; }
  .lb-table th { font-family:'Plus Jakarta Sans',sans-serif; font-size:0.7rem; font-weight:800; text-transform:uppercase; letter-spacing:.07em; color:var(--soft); padding:12px 12px; text-align:left; border-bottom:2px solid #f0eaf4; }
  .lb-table td { padding:9px 12px; border-bottom:1px solid #f4eff8; color:var(--ink); vertical-align:middle; }
  .lb-table tr:last-child td { border-bottom:none; }
  .lb-table .c-rank { width:34px; color:var(--soft); font-weight:500; font-family:'Plus Jakarta Sans',sans-serif; }
  .lb-table .c-stat { text-align:right; font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; font-size:.98rem; white-space:nowrap; }
  .lb-table th.c-stat { text-align:right; }
  .lb-table .pcell { display:flex; align-items:center; gap:10px; }
  .lb-table .phead img, .lb-table .phead .noimg { width:30px; height:30px; border-radius:50%; object-fit:cover; display:block; background:#ece4f4; }
  .lb-table .pname { font-weight:400; }
  .lb-table .c-team { color:var(--soft); font-weight:400; white-space:nowrap; }
  .lb-table .tcell { display:flex; align-items:center; gap:7px; }
  .lb-table .tlogo { width:20px; height:20px; object-fit:contain; display:block; flex:none; }
  .lb-table .c-fi { text-align:right; color:var(--soft); font-weight:400; white-space:nowrap; }
  .lb-table th.c-fi { text-align:right; }
  .lb-table .c-event { color:var(--soft); font-weight:300; white-space:nowrap; }
  .lb-table tr.is-aspas td { background:linear-gradient(90deg,#fff5e6,#fffaf2); }
  .lb-table tr.is-aspas .pname { font-weight:700; }
  .lb-table tr.is-aspas .c-rank { color:#e0992a; font-weight:800; }
  .lb-table tr.is-aspas .c-stat { color:#d98410; }
  .lb-table.role tr.is-aspas td { background:#faf6ff; }
  .lb-table.role tr.is-aspas .c-team { color:var(--ink); font-weight:600; }
  /* ── strip / dot chart ── */
  .chart-wrap { background:white; border-radius:20px; padding:24px 26px 18px; box-shadow:0 4px 24px #0000000a; margin:28px 0 32px; }
  .chart-title { font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; font-size:1.05rem; letter-spacing:-0.2px; color:var(--ink); text-align:center; margin-bottom:14px; }
  .chart-label { font-family:'Plus Jakarta Sans',sans-serif; font-size:0.77rem; font-weight:800; letter-spacing:.1em; text-transform:uppercase; color:var(--soft); margin-bottom:6px; }
  .chart-sub { font-size:.78rem; color:var(--soft); font-weight:300; margin-bottom:14px; }
  .strip-box { position:relative; height:220px; }
  .scatter-box { position:relative; height:340px; }
  .aspas-card { position:absolute; transform:translate(-50%,-100%); margin-top:-14px; background:white; border-radius:14px; padding:10px 12px 11px; box-shadow:0 8px 30px #00000022; text-align:center; min-width:120px; z-index:5; pointer-events:none; border:1.5px solid #f0d9a8; }
  .aspas-card::after { content:''; position:absolute; left:calc(50% + var(--arrow-dx, 0px)); bottom:-7px; transform:translateX(-50%) rotate(45deg); width:12px; height:12px; background:white; border-right:1.5px solid #f0d9a8; border-bottom:1.5px solid #f0d9a8; }
  .aspas-card.side { transform:translate(-100%,-50%); margin-top:0; margin-left:-14px; }
  .aspas-card.side::after { left:auto; right:-7px; bottom:auto; top:50%; transform:translateY(-50%) rotate(45deg); border:none; border-top:1.5px solid #f0d9a8; border-right:1.5px solid #f0d9a8; }
  .aspas-card img { width:46px; height:46px; border-radius:50%; object-fit:cover; display:block; margin:0 auto 6px; border:2px solid #efc874; }
  .aspas-card .ac-name { font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; font-size:.92rem; color:var(--ink); }
  .aspas-card .ac-stat { font-size:.72rem; color:#d98410; font-weight:500; margin-top:1px; }
  footer { position:relative; z-index:1; text-align:center; padding:24px; color:var(--soft); font-size:.75rem; font-weight:300; }
  @keyframes fadeUp{from{opacity:0;transform:translateY(24px)}to{opacity:1;transform:translateY(0)}}
  .page { animation:fadeUp .6s ease both; }
  /* Larger body text on the fixed-width mobile viewport (fires at width=820). */
  @media (max-width:1000px){
    .content p, .content ul li { font-size:1.3rem; }
    .content p.gs { font-size:1.55rem; }
    .deck { font-size:1.22rem; }
    .chart-title { font-size:1.3rem; }
  }
  @media (max-width:600px){
    .page{padding:24px 14px 48px}
    .content p, .content ul li { font-size:.94rem; }
    .pull { font-size:1.12rem; }
    .chart-wrap{padding:18px 14px 14px}
    .strip-box{height:220px}
    .lb-table{font-size:.8rem;min-width:520px}
    .lb-table th,.lb-table td{padding:8px 8px}
    .lb-table .phead img,.lb-table .phead .noimg{width:26px;height:26px}
  }
</style>
</head>
<body>
<nav class="toc">
  <div class="toc-title">Sections</div>
  <a href="#intro">Introduction</a>
  <a href="#methodology">Quick Notes on Methodology</a>
  <a href="#numbers">The Numbers</a>
  <a href="#stat-rating" class="sub">VLR Rating</a>
  <a href="#stat-kd" class="sub">K/D</a>
  <a href="#stat-kpr" class="sub">KPR</a>
  <a href="#stat-kast" class="sub">KAST%</a>
  <a href="#baiting">What About Baiting?</a>
  <a href="#macro">The Macro Argument behind the Micro</a>
  <a href="#synopsis">Synopsis</a>
</nav>
<div class="top-nav">
  <a href="/"><img src="/logo.svg" alt="Home" class="home-logo"></a>
</div>
<div class="page">
  <div class="article">
    <div class="label">Research / Opinion</div>
    <h1>The Greatest Prime in<br>VCT History Isn&rsquo;t a Debate</h1>
    <div class="byline">Bobo &mdash; June 2026</div>
    <div class="cover" id="intro">
      <img src="/aspas25corrode.jpg" alt="Aspas at Champions 2025">
    </div>
    <p class="cover-caption">Aspas pictured after winning MIBR&rsquo;s opening match at Champs 2025 and going 35/17 - his first step in a ridiculous stretch of performances over the event</p>
    <div class="content">

      <p>Despite this article&rsquo;s title, I don&rsquo;t want to seem like I&rsquo;m dismissing the legacies that exist within this conversation. Leo showcased some of the most perfect macro/micro at LOCK//IN and Masters Tokyo. Demon1 had a mechanical peak (and storyline) at Champions LA that has been rivaled by few. Trent&rsquo;s mastery of Tejo at Master Bangkok is, in my opinion, an unmatched level of dominance over one individual agent. Even just recently, Marteen set the record for VLR rating at an event with his 1.41 average at Masters Santiago.</p>

      <p>This website, though, focuses on analytics. If you add the word &ldquo;by analytics&rdquo; to the title, then it&rsquo;s clearer: The greatest prime in VCT History isn&rsquo;t a debate <em>by analytics</em>. Aspas at Champions Paris was, by analytics, the greatest prime in VCT history. The more you break down the numbers, the more ludicrous his performance was.</p>

      <p>In this article I&rsquo;ll do exactly that. This will be a shorter read, as I&rsquo;ll just go over some numbers and their context.</p>

      <h2 id="methodology">Quick Notes on Methodology</h2>

      <ol class="method">
        <li>
          <p>While other variations are valid, I only considered international performances. Players and rosters fluctuate too much over an entire year. Domestic splits involve lesser and uneven competition, further clouded by discussions of regional power levels. Internationals are consistent, with high-level teams from all regions and high-stakes matches.</p>
          <p>I consider internationals to be a true reflection of player form against the best teams in the world and, importantly, comparable from one to another.</p>
          <p>I&rsquo;m looking at you, Marteen at Masters Santiago.</p>
        </li>
        <li>
          <p>I use a filter of minimum rounds: 150+ when looking at contenders/statistics. This omits players whose statistics are anomalies based on variance + low sample size. It also eliminates players whose teams did extremely poorly, which is fine, as no player whose team couldn&rsquo;t win a match at an international is a true candidate for having &ldquo;the greatest prime in VCT history&rdquo;. Even players whose teams were mediocre, like Marteen at Santiago (where Gentle Mates went 2-2, not beating a single team who made Playoffs), are still in contention with this filter. Here are some examples of players who were omitted based on this filter:</p>
          <ul>
            <li>qRaxs at LOCK//IN has the 3rd-highest FIWR (first interaction win-rate) in VCT history at 83.3%. He only took 6 first interactions the entire event (winning 5 and losing 1).</li>
            <li>Life at Masters London has the highest Headshot % in VCT history at 45%, but his team went out 0-2.</li>
          </ul>
          <p>I could go on, but you get the point. No meaningful contributions to leaderboards or contenders for this debate are omitted.</p>
          <p>The number of entries goes from {{ counts.total }} to {{ counts.qualifying }}.</p>
        </li>
        <li>
          <p>Not all stats share the same pool. Masters Shanghai published no VLR ratings or KAST%, so those two draw from {{ counts.qualifying_rated }} entries, while K/D, KPR, and the first-interaction stats use the full {{ counts.qualifying }}.</p>
        </li>
      </ol>

      <h2 id="numbers">The Numbers</h2>

      <p class="gs" id="stat-rating">Aspas (Champions 2025) has the second-highest VLR rating in VCT tournament history <span class="note">(min 150+ rounds)</span></p>

      {{ rating_table | safe }}

      <p>This is a simple statistics to gloss over, but it&rsquo;s probably the most important. Out of {{ counts.qualifying_rated }} candidates, he is one of only 5 players (including Leo twice) to have a 1.3+ rating over an event, which is ridiculously impressive.</p>

      <p>What&rsquo;s more impressive is that, of these 6 entries with a 1.3+ rating, half of them ended up winning their tournament (Leo twice and Alfajer). Of the 3 non-winners, 2 of them played only 4 matches at the international (Shao at LOCK//IN and Marteen at Masters Santiago). Aspas is the only player to notch a 1.3+ rating while:</p>

      <ul>
        <li>Having a sample size greater than 4 matches</li>
        <li>Not winning the international</li>
      </ul>

      <p class="gs" id="stat-kd">Aspas (Champions 2025) has the highest K/D in VCT tournament history <span class="note">(min 150+ rounds)</span></p>

      {{ kd_table | safe }}

      <p>Here it is in context:</p>

      <div class="chart-wrap">
        <div class="chart-title">International K/D&rsquo;s <span class="note">(min 150+ rounds)</span></div>
        <div class="strip-box"><canvas id="kdStrip"></canvas><div class="aspas-card" id="kdCard" style="display:none"></div></div>
      </div>

      <p>Not only does he exist within that top cluster, he is above them all.</p>

      <p class="gs" id="stat-kpr">Aspas (Champions 2025) has the highest KPR in VCT tournament history <span class="note">(min 150+ rounds)</span></p>

      {{ kpr_table | safe }}

      <p>He was 0.03 away from averaging a kill a round, the closest anyone has ever come given these conditions.</p>

      <p class="gs" id="stat-kast">Aspas (Champions 2025) has the 11th-highest KAST% in VCT tournament history <span class="note">(min 150+ rounds)</span></p>

      {{ kast_table | safe }}

      <p>While 11th is not as groundbreaking as previous statistics, consider how biased KAST% is towards non-Duelists:</p>

      <div class="chart-title" style="margin-bottom:0">Average KAST% by Role <span class="note">(min 150+ rounds)</span></div>
      {{ role_table | safe }}

      <p>That makes his 11th-place more impressive. In fact, there&rsquo;s not another duelist in that top-15 list. What if we just looked at duelists?</p>

      {{ duelist_table | safe }}

      <div class="chart-wrap">
        <div class="chart-title">International Duelist&rsquo;s KAST% <span class="note">(min 150+ rounds)</span></div>
        <div class="strip-box"><canvas id="duelStrip"></canvas><div class="aspas-card" id="duelCard" style="display:none"></div></div>
      </div>

      <p>Aaaaand we&rsquo;re back to breaking records. At Champions 2025, Aspas notched the highest KAST% ever by a duelist at an international (min 150+ rounds) by a good margin.</p>

      <h2 id="baiting">What About Baiting?</h2>

      <p>In the words of Bren from Platchat:</p>

      <blockquote class="qbubble">[Aspas] is a great player, but I think he&rsquo;s found out the formula for the VLR rating, knows that if you die less, you get fucking rated higher</blockquote>

      <p>How true was that at Champions 2025? Unsurprisingly, he certainly died less frequently than most:</p>

      <div class="chart-wrap">
        <div class="chart-title">International DPR <span class="note">(min 150+ rounds)</span></div>
        <div class="strip-box"><canvas id="dprStrip"></canvas><div class="aspas-card" id="dprCard" style="display:none"></div></div>
      </div>

      <p>Out of the 437 international performances with 150+ rounds, Aspas ranks 428th for DPR (Deaths per Round). In essence, dying less is considered a <em>good</em> thing, but if you&rsquo;re dying <em>too</em> infrequently it can be a symptom of playing for your life instead of the team. Especially if you&rsquo;re a duelist player. So, was he truly selfish? Is Bren right?</p>

      <p>Let&rsquo;s look at first interactions:</p>

      <div class="chart-wrap">
        <div class="chart-title">International FIPR <span class="note">(min 150+ rounds)</span></div>
        <div class="strip-box"><canvas id="fiprStrip"></canvas><div class="aspas-card" id="fiprCard" style="display:none"></div></div>
      </div>

      <p>Aspas&rsquo; 0.27 FIPR (First Interactions per Round) puts him in the 79th percentile given the filter. However, it&rsquo;s unfair to forget that he&rsquo;s a duelist. If I used his role to contextualize KAST% in a positive light, it&rsquo;s only fair to factor in his role as a duelist into FIPR (even if it&rsquo;s in a negative light):</p>

      <div class="chart-wrap">
        <div class="chart-title">International Duelist&rsquo;s FIPR <span class="note">(min 150+ rounds)</span></div>
        <div class="strip-box"><canvas id="fiprDuelStrip"></canvas><div class="aspas-card" id="fiprDuelCard" style="display:none"></div></div>
      </div>

      <p>Now, he&rsquo;s in the {{ duelist_fipr_pctile }}rd percentile for duelists. It&rsquo;s lower than average, but not crazy low.</p>

      <p>However, this isn&rsquo;t the full story. How often was he winning those first duels?</p>

      {{ fiwr_table | safe }}

      <p>Holy shit. Not only is his 74.44% FIWR (First Interaction Win Rate) the 4th-best in international VCT history (min 150+ rounds), he&rsquo;s doing so as a <em>duelist</em>. Look again at the top-10 list; not a single other member of the top 10 plays duelist. In fact, the second-highest FIWR by a duelist at an international (150+ rounds) is garnetS at LOCK//IN, with a FIWR of 63.83% at 23rd. This is followed by Aspas at Champions LA, with a FIWR of 63.16% at 29th. Aspas&rsquo; FIWR at Champions Paris is, when compared to other duelists, &gt;10% higher than the next highest FIWR.</p>

      <p>Here&rsquo;s how abnormal Aspas&rsquo; FIWR at Champions Paris was when compared to other duelists:</p>

      <div class="chart-wrap">
        <div class="chart-title">International Duelist&rsquo;s FIWR <span class="note">(min 150+ rounds)</span></div>
        <div class="strip-box"><canvas id="fiwrDuelStrip"></canvas><div class="aspas-card" id="fiwrDuelCard" style="display:none"></div></div>
      </div>

      <p>I cannot understate how impressive that is.</p>

      <p>Duelists are expected to constantly take first contact, while other roles typically hold peaks, lurk, or play off of/with their duelist&rsquo;s entries. While this may sound like pure narrative, look at the numbers:</p>

      <div class="chart-title" style="margin-bottom:0">Average FIPR by Role <span class="note">(min 150+ rounds)</span></div>
      {{ role_fipr_table | safe }}

      <p>The impact that this has on FIWR is, with higher first interaction frequency, it&rsquo;s difficult to have extremely high (or low) win-rate numbers. Let&rsquo;s look again at the top-10 list for FIWR at international events, but this time include total first interactions.</p>

      {{ fiwr_fi_table | safe }}

      <p>Again, it&rsquo;s just incredible what he was doing. No one above Aspas in FIWR has 1/4th of the first interactions he had at Champions Paris.</p>

      <p>As a final illustration of this point, here&rsquo;s a scatterplot of the relationship between number of first interactions and first interaction win rate at internationals (150+ rounds played) <em>without Aspas at Champions Paris</em>.</p>

      <div class="chart-wrap">
        <div class="chart-title">International FIWR vs. First Interactions <span class="note">(min 150+ rounds)</span></div>
        <div class="scatter-box"><canvas id="fiwrScatterNo"></canvas></div>
      </div>

      <p>Now, let&rsquo;s include him:</p>

      <div class="chart-wrap">
        <div class="chart-title">International FIWR vs. First Interactions <span class="note">(min 150+ rounds)</span></div>
        <div class="scatter-box"><canvas id="fiwrScatterYes"></canvas><div class="aspas-card" id="fiwrScatterCard" style="display:none"></div></div>
      </div>

      <p>He&rsquo;s hard to miss.</p>

      <p><strong>Aspas&rsquo; consistency with which he won first interactions at Champions Paris should not be possible. Not for a duelist with 90 first interactions.</strong></p>

      <p>Aspas did play safer than other duelists, that&rsquo;s true. He would also save frequently.</p>

      <p>However, his borderline impossible level of FIWR proves that he wasn&rsquo;t just staying alive because he avoided first interactions, he was staying alive because he kept <em>winning</em> them. Considering the high volume with which he took them, it&rsquo;s fair to say that Aspas at Champions Paris was the deadliest player in VCT history to face in a first interaction.</p>

      <p>Furthermore, to claim that Aspas was just avoiding interactions/baiting at Champions Paris when he recorded the highest KPR in international VCT history (150+ rounds) is inane.</p>

      <h2 id="macro">The Macro Argument behind the Micro</h2>

      <p>One other point worth noting is this:</p>

      <p>During MIBR&rsquo;s Champions Paris run, they played each of the top-3 teams - NRG, FNATIC, and DRX. Those were the only teams they lost to, each loss being 1-2. Every other game was a 2-0, including against Team Heretics who finished 5th-6th alongside MIBR.</p>

      <p>Aspas was dropping these numbers against the best in the world. There&rsquo;s no caveat of &ldquo;opponent quality&rdquo; (I&rsquo;m looking at you Marteen at Masters Santiago).</p>

      <h2 id="synopsis">Synopsis</h2>

      <p>Summarizing these findings, let&rsquo;s look back at how Aspas compares with other international performances (150+ rounds):</p>

      <div class="syn-card">
        <div class="syn-title">Aspas at Champions Paris &mdash; International Historical Rankings</div>
        <div class="syn-stat"><div class="syn-rank second">2nd</div><div class="syn-label">VLR Rating</div><div class="syn-of">of 397</div></div>
        <div class="syn-stat"><div class="syn-rank">1st</div><div class="syn-label">K/D</div><div class="syn-of">of 437</div></div>
        <div class="syn-stat"><div class="syn-rank">1st</div><div class="syn-label">KPR</div><div class="syn-of">of 437</div></div>
        <div class="syn-stat"><div class="syn-rank">1st</div><div class="syn-label">KAST%<span>by a Duelist</span></div><div class="syn-of">of 81</div></div>
        <div class="syn-stat"><div class="syn-rank">1st</div><div class="syn-label">FIWR<span>by a Duelist</span></div><div class="syn-of">of 91</div></div>
      </div>

      <p class="syn-cap">Pools differ slightly: Masters Shanghai (no published ratings or KAST%) is excluded from VLR Rating and KAST% but counts toward K/D, KPR, and FIWR.</p>

      <p>And my three favorite visualizations:</p>

      <div class="chart-wrap">
        <div class="chart-title">International Duelist&rsquo;s KAST% <span class="note">(min 150+ rounds)</span></div>
        <div class="strip-box"><canvas id="synKast"></canvas><div class="aspas-card" id="synKastCard" style="display:none"></div></div>
      </div>

      <div class="chart-wrap">
        <div class="chart-title">International Duelist&rsquo;s FIWR <span class="note">(min 150+ rounds)</span></div>
        <div class="strip-box"><canvas id="synFiwr"></canvas><div class="aspas-card" id="synFiwrCard" style="display:none"></div></div>
      </div>

      <div class="chart-wrap">
        <div class="chart-title">International FIWR vs. First Interactions <span class="note">(min 150+ rounds)</span></div>
        <div class="scatter-box"><canvas id="synScatter"></canvas><div class="aspas-card" id="synScatterCard" style="display:none"></div></div>
      </div>

      <p>Aspas at Champions Paris wasn&rsquo;t just the best at what he does, he was the best we&rsquo;ve ever seen by a large, large margin.</p>

      <p>Analytically, the greatest prime in VCT history is not a debate.</p>

    </div>
  </div>
</div>
<footer>
  Data sourced from VLR.gg
  <div style="margin-top:8px;">Like my work? Tips are appreciated! <a href="https://ko-fi.com/bobovct" target="_blank" rel="noopener" style="color:inherit;text-decoration:underline;">ko-fi.com/bobovct</a></div>
</footer>

<script>
var DATA = {{ data_json | safe }};

function buildStrip(canvasId, cardId, points, axisLabel, fmt, axisMin, axisMax) {
  var aspas = null, field = [];
  points.forEach(function(p) {
    if (p.is_aspas) aspas = { x: p.value, y: 0, raw: p }; else field.push(p);
  });
  // Stack dots that share the same value into a centered column (swarm), so the
  // dense middle reads as distinct dots instead of one overlapping blob.
  var groups = {};
  field.forEach(function(p) { var k = p.value.toFixed(3); (groups[k] = groups[k] || []).push(p); });
  var maxStack = 1;
  Object.keys(groups).forEach(function(k) { if (groups[k].length > maxStack) maxStack = groups[k].length; });
  var step = Math.min(0.16, 1.55 / maxStack);
  var others = [];
  Object.keys(groups).forEach(function(k) {
    var arr = groups[k];
    for (var i = 0; i < arr.length; i++) {
      others.push({ x: arr[i].value, y: (i - (arr.length - 1) / 2) * step, raw: arr[i] });
    }
  });
  var ctx = document.getElementById(canvasId).getContext('2d');
  var chart = new Chart(ctx, {
    type: 'scatter',
    data: { datasets: [
      { label: 'field', data: others, backgroundColor: 'rgba(149,118,184,0.55)', borderColor: '#fff', borderWidth: 0.5, pointRadius: 3.5, pointHoverRadius: 5 },
      { label: 'aspas', data: aspas ? [aspas] : [], backgroundColor: '#e8a33d', borderColor: '#fff', borderWidth: 2, pointRadius: 8, pointHoverRadius: 9 }
    ] },
    options: {
      responsive: true, maintainAspectRatio: false,
      interaction: { mode: 'nearest', intersect: true },
      layout: { padding: { top: 105, left: 4, right: 12 } },
      scales: {
        x: { min: axisMin, max: axisMax, title: { display: true, text: axisLabel, color: '#9b8fae', font: { family: 'Plus Jakarta Sans', weight: '800', size: 11 } }, grid: { color: '#f1ecf6' }, ticks: { color: '#9b8fae', font: { size: 11 } } },
        y: { min: -1, max: 1, display: false, grid: { display: false } }
      },
      plugins: {
        legend: { display: false },
        tooltip: { mode: 'nearest', intersect: true, displayColors: false, callbacks: { label: function(c) { var r = c.raw.raw; return r.player + ' (' + r.org + ', ' + r.evlabel + ') — ' + fmt(r.value); } } }
      }
    }
  });
  var card = document.getElementById(cardId);
  if (aspas) {
    var a = aspas.raw;
    var nm = a.player.charAt(0).toUpperCase() + a.player.slice(1);
    card.innerHTML = (a.headshot ? '<img src="' + a.headshot + '" alt="">' : '') +
      '<div class="ac-name">' + nm + '</div>' +
      '<div class="ac-stat">' + fmt(a.value) + '</div>';
    var place = function() {
      var meta = chart.getDatasetMeta(1);
      if (!meta.data || !meta.data[0]) return;
      var el = meta.data[0];
      card.style.top = el.y + 'px';
      card.style.display = 'block';
      var box = card.parentElement, half = card.offsetWidth / 2, pad = 6;
      var center = Math.max(half + pad, Math.min(el.x, box.clientWidth - half - pad));
      card.style.left = center + 'px';
      card.style.setProperty('--arrow-dx', (el.x - center) + 'px');
    };
    place();
    chart.options.animation = { onComplete: place };
    chart.update();
    window.addEventListener('resize', function() { setTimeout(place, 60); });
  }
  return chart;
}

function buildScatter(canvasId, cardId, points, withAspas) {
  var field = [], aspas = null;
  points.forEach(function(p) {
    var pt = { x: p.x, y: p.y, raw: p };
    if (p.is_aspas) { if (withAspas) aspas = pt; } else field.push(pt);
  });
  var datasets = [{ label: 'field', data: field, backgroundColor: 'rgba(149,118,184,0.4)', borderColor: '#fff', borderWidth: 0.5, pointRadius: 4, pointHoverRadius: 6 }];
  if (aspas) datasets.push({ label: 'aspas', data: [aspas], backgroundColor: '#e8a33d', borderColor: '#fff', borderWidth: 2, pointRadius: 9, pointHoverRadius: 11 });
  var chart = new Chart(document.getElementById(canvasId).getContext('2d'), {
    type: 'scatter',
    data: { datasets: datasets },
    options: {
      responsive: true, maintainAspectRatio: false,
      interaction: { mode: 'nearest', intersect: true },
      layout: { padding: { top: 14, left: 4, right: 16, bottom: 2 } },
      scales: {
        x: { title: { display: true, text: 'First Interactions', color: '#9b8fae', font: { family: 'Plus Jakarta Sans', weight: '800', size: 11 } }, grid: { color: '#f1ecf6' }, ticks: { color: '#9b8fae', font: { size: 11 } } },
        y: { title: { display: true, text: 'FIWR %', color: '#9b8fae', font: { family: 'Plus Jakarta Sans', weight: '800', size: 11 } }, grid: { color: '#f1ecf6' }, ticks: { color: '#9b8fae', font: { size: 11 } } }
      },
      plugins: {
        legend: { display: false },
        tooltip: { mode: 'nearest', intersect: true, displayColors: false, callbacks: { label: function(c) { var r = c.raw.raw; return r.player + ' (' + r.org + ', ' + r.evlabel + ') — ' + r.y + '% on ' + r.x + ' FIs'; } } }
      }
    }
  });
  var card = cardId ? document.getElementById(cardId) : null;
  if (aspas && card) {
    var a = aspas.raw;
    card.innerHTML = (a.headshot ? '<img src="' + a.headshot + '" alt="">' : '') + '<div class="ac-name">Aspas</div><div class="ac-stat">' + a.y + '% on ' + a.x + ' FIs</div>';
    card.classList.add('side');
    var place = function() { var meta = chart.getDatasetMeta(1); if (!meta.data || !meta.data[0]) return; var el = meta.data[0]; card.style.left = el.x + 'px'; card.style.top = el.y + 'px'; card.style.display = 'block'; };
    place(); chart.options.animation = { onComplete: place }; chart.update();
    window.addEventListener('resize', function() { setTimeout(place, 60); });
  }
  return chart;
}

var kdMin = Math.floor((Math.min.apply(null, DATA.kd_strip.map(function(p){return p.value;})) - 0.05) * 10) / 10;
buildStrip('kdStrip', 'kdCard', DATA.kd_strip, 'K : D', function(v){ return v.toFixed(2) + ' K:D'; }, kdMin, 1.75);

var dMin = 60;
var dMax = 80;
buildStrip('duelStrip', 'duelCard', DATA.duelist_strip, 'KAST %', function(v){ return v.toFixed(1) + '% KAST'; }, dMin, dMax);

var bait = DATA.baiting;
function rng(arr, pad) { var vs = arr.map(function(p){ return p.value; }); return [Math.min.apply(null, vs) - pad, Math.max.apply(null, vs) + pad]; }
var dprR = rng(bait.dpr_strip, 0.03);
buildStrip('dprStrip', 'dprCard', bait.dpr_strip, 'DPR', function(v){ return v.toFixed(2) + ' DPR'; }, dprR[0], dprR[1]);
var fiR = rng(bait.fipr_strip, 0.02);
buildStrip('fiprStrip', 'fiprCard', bait.fipr_strip, 'FIPR', function(v){ return v.toFixed(2) + ' FIPR'; }, fiR[0], fiR[1]);
var fdR = rng(bait.fipr_duelist_strip, 0.02);
buildStrip('fiprDuelStrip', 'fiprDuelCard', bait.fipr_duelist_strip, 'FIPR', function(v){ return v.toFixed(2) + ' FIPR'; }, fdR[0], fdR[1]);
var fwR = [35, 75];
buildStrip('fiwrDuelStrip', 'fiwrDuelCard', bait.fiwr_duelist_strip, 'FIWR %', function(v){ return v.toFixed(2) + '% FIWR'; }, fwR[0], fwR[1]);
buildScatter('fiwrScatterNo', null, bait.fiwr_scatter, false);
buildScatter('fiwrScatterYes', 'fiwrScatterCard', bait.fiwr_scatter, true);

// Synopsis — re-render the three favorite visualizations
buildStrip('synKast', 'synKastCard', DATA.duelist_strip, 'KAST %', function(v){ return v.toFixed(1) + '% KAST'; }, dMin, dMax);
buildStrip('synFiwr', 'synFiwrCard', bait.fiwr_duelist_strip, 'FIWR %', function(v){ return v.toFixed(2) + '% FIWR'; }, fwR[0], fwR[1]);
buildScatter('synScatter', 'synScatterCard', bait.fiwr_scatter, true);

(function() {
  var tocLinks = document.querySelectorAll('.toc a');
  var ids = Array.from(tocLinks).map(function(a) { return a.getAttribute('href').slice(1); });
  function onScroll() {
    var scrollY = window.scrollY + 120;
    var active = ids[0];
    ids.forEach(function(id) {
      var el = document.getElementById(id);
      if (el && el.offsetTop <= scrollY) active = id;
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
