"""Article: Masters London Preview.

Mirrors the layout of AmericasStage1Playoffs (label / h1 / dek / byline /
hero image / body). Body is a placeholder for now — fill in prose and any
charts the same way the other articles do.
"""

from flask import Blueprint, render_template_string

article_masters_london_bp = Blueprint("article_masters_london", __name__)

PAGE_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=820">
<title>Masters London Preview &mdash; Bobo's VCT Database</title>
<link rel="icon" type="image/svg+xml" href="/favicon.svg">
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@700;800&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
<link rel="stylesheet" href="/static/base.css">
<style>
  html { -webkit-text-size-adjust:100%; text-size-adjust:100%; }
  .top-nav { padding:32px 32px 0; position:relative; z-index:1; }
  .home-logo { height:80px; width:auto; display:block; opacity:.85; transition:opacity .2s; }
  .home-logo:hover { opacity:1; }
  .toc { position:fixed; top:32px; right:32px; background:white; border-radius:16px; padding:20px 24px; box-shadow:0 4px 24px #0000000f; display:flex; flex-direction:column; gap:6px; z-index:100; max-width:240px; }
  .toc-title { font-family:'Plus Jakarta Sans',sans-serif; font-size:0.77rem; font-weight:800; letter-spacing:.1em; text-transform:uppercase; color:var(--soft); margin-bottom:4px; }
  .toc a { font-size:.78rem; color:var(--soft); text-decoration:none; font-weight:400; transition:color .15s; line-height:1.4; }
  .toc a:hover { color:var(--ink); }
  .toc a.active { color:var(--ink); font-weight:500; }
  .toc a.toc-sub { padding-left:16px; font-size:.74rem; border-left:2px solid #ede5f3; margin-left:4px; }
  .toc a.toc-sub.active { border-left-color:#7c3aed; color:var(--ink); font-weight:600; }
  .alpha-navbar ~ .toc { top:72px; }
  @media(max-width:1180px) { .toc { display:none; } }
  .page { position:relative; z-index:1; flex:1; display:flex; flex-direction:column; align-items:center; padding:60px 32px 80px; }
  .article { max-width:860px; width:100%; }
  .label { font-family:'Plus Jakarta Sans',sans-serif; font-size:0.77rem; font-weight:800; letter-spacing:.14em; text-transform:uppercase; color:var(--soft); margin-bottom:16px; text-align:center; }
  h1 { font-family:'Plus Jakarta Sans',sans-serif; font-size:clamp(2.2rem,5vw,3.52rem); font-weight:800; letter-spacing:-1px; line-height:1.1; margin-bottom:16px; text-align:center; }
  .dek { font-size:1.05rem; font-weight:300; line-height:1.55; color:var(--ink); margin-bottom:24px; opacity:.85; }
  .byline { font-size:.82rem; color:var(--soft); font-weight:300; margin-bottom:48px; padding-bottom:32px; border-bottom:1px solid #e8e0ec; text-align:center; }
  .cover { width:100%; border-radius:16px; overflow:hidden; margin-bottom:12px; }
  .cover img { width:100%; height:auto; display:block; }
  .cover-caption { font-size:.75rem; color:var(--soft); font-weight:300; font-style:italic; margin-bottom:48px; text-align:center; }
  .content p { font-size:1rem; font-weight:300; line-height:1.8; color:var(--ink); margin-bottom:24px; }
  .section-bubble { display:inline-block; background:white; padding:16px 44px; border-radius:999px; box-shadow:0 6px 32px #0000001a; }
  .section-bubble-text { font-family:'Plus Jakarta Sans',sans-serif; font-size:2.09rem; font-weight:800; letter-spacing:.01em;
    background-image:linear-gradient(95deg,#f472b6 0%,#a855f7 55%,#7c3aed 100%);
    -webkit-background-clip:text; background-clip:text;
    -webkit-text-fill-color:transparent; color:transparent;
  }
  .section-bubble-wrap { text-align:center; margin:96px 0 32px; }
  .section-bubble-wrap.section-bubble-tight { margin-top:40px; }
  .section-note { font-size:.85rem; color:var(--soft); font-weight:400; font-style:italic; line-height:1.55; text-align:center; max-width:680px; margin:-12px auto 28px; padding:0 24px; }

  /* Two-team comparison panel (Gen.G vs PRX) */
  .comparison-chart { background:white; border-radius:16px; padding:24px 28px; box-shadow:0 4px 24px #0000000a; margin:24px 0 32px; }
  .comparison-header { display:grid; grid-template-columns:1fr 150px 1fr; gap:16px; padding-bottom:14px; margin-bottom:8px; border-bottom:2px solid #f0eaf4; align-items:end; }
  .comparison-team { display:flex; flex-direction:column; align-items:center; gap:6px; }
  .comparison-team img { width:38px; height:38px; object-fit:contain; }
  .comparison-team-name { font-family:'Plus Jakarta Sans',sans-serif; font-size:1.155rem; font-weight:800; }
  .comparison-team-sub { font-size:.7rem; color:var(--soft); font-weight:300; text-align:center; line-height:1.3; }
  .comparison-row { display:grid; grid-template-columns:1fr 150px 1fr; gap:16px; padding:14px 0; align-items:center; border-bottom:1px solid #f5eff8; }
  .comparison-row:last-child { border-bottom:none; }
  .comparison-value { text-align:center; font-weight:500; line-height:1.4; font-size:.95rem; }
  .comparison-value.num { font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; font-size:1.43rem; font-variant-numeric:tabular-nums; }
  .comparison-label { font-family:'Plus Jakarta Sans',sans-serif; font-size:0.682rem; font-weight:800; letter-spacing:.1em; text-transform:uppercase; color:var(--soft); text-align:center; }

  /* Tier list overview (click-to-jump chips) */
  .tier-overview { background:white; border-radius:16px; padding:20px 24px; box-shadow:0 4px 24px #0000000a; margin:24px 0 18px; }
  .tier-overview-label { font-family:'Plus Jakarta Sans',sans-serif; font-size:0.77rem; font-weight:800; letter-spacing:.1em; text-transform:uppercase; color:var(--soft); margin-bottom:14px; text-align:center; }
  .tier-row { display:grid; grid-template-columns:54px 1fr; gap:14px; align-items:center; padding:10px 0; border-bottom:1px solid #f5eff8; }
  .tier-row:last-child { border-bottom:none; }
  .tier-badge { font-family:'Plus Jakarta Sans',sans-serif; font-size:1.54rem; font-weight:800; height:46px; border-radius:10px; display:flex; align-items:center; justify-content:center; color:white; }
  .tier-badge.s { background:#e74c3c; }
  .tier-badge.a { background:#e67e22; }
  .tier-badge.b { background:#d4a017; }
  .tier-badge.c { background:#27ae60; }
  .tier-badge.d { background:#4a7fbf; }
  .tier-teams { display:flex; flex-wrap:wrap; gap:8px; }
  .tier-chip { display:inline-flex; align-items:center; justify-content:center; width:74px; height:74px; padding:8px; border-radius:14px; background:#faf6fd; text-decoration:none; transition:background .15s, transform .15s; }
  .tier-chip:hover { background:#efe2f7; transform:translateY(-2px) scale(1.05); }
  .tier-chip img { width:100%; height:100%; object-fit:contain; }

  /* Tier-section banner header */
  .tier-section-header { font-family:'Plus Jakarta Sans',sans-serif; font-size:1.54rem; font-weight:800; margin:32px 0 16px; padding:10px 28px; border-radius:12px; color:white; display:inline-block; }
  .tier-section-header.s { background:#e74c3c; }
  .tier-section-header.a { background:#e67e22; }
  .tier-section-header.b { background:#d4a017; }
  .tier-section-header.c { background:#27ae60; }
  .tier-section-header.d { background:#4a7fbf; }

  /* Per-team heading + stat card + why-blocks */
  .team-heading { display:flex; align-items:center; gap:14px; margin:48px 0 20px; flex-wrap:wrap; }
  .team-heading img { width:42px; height:42px; object-fit:contain; flex-shrink:0; }
  .team-heading h2 { margin:0; font-family:'Plus Jakarta Sans',sans-serif; font-size:1.54rem; font-weight:800; letter-spacing:-.5px; }
  .tier-pill { font-family:'Plus Jakarta Sans',sans-serif; font-size:0.748rem; font-weight:800; letter-spacing:.08em; padding:4px 10px; border-radius:6px; color:white; }
  .tier-pill.s { background:#e74c3c; }
  .tier-pill.a { background:#e67e22; }
  .tier-pill.b { background:#d4a017; }
  .tier-pill.c { background:#27ae60; }
  .tier-pill.d { background:#4a7fbf; }
  .team-stat-card { background:white; border-radius:14px; padding:18px 22px; box-shadow:0 4px 24px #0000000a; margin:0 0 24px; }
  .team-stat-row { display:flex; gap:28px; flex-wrap:wrap; justify-content:center; }
  .team-stat-block { display:flex; flex-direction:column; gap:5px; align-items:center; text-align:center; min-width:170px; flex:1; }
  .team-stat-label { font-family:'Plus Jakarta Sans',sans-serif; font-size:0.792rem; font-weight:800; letter-spacing:.1em; text-transform:uppercase; color:var(--soft); }
  .team-stat-value { font-family:'Plus Jakarta Sans',sans-serif; font-size:1.1rem; font-weight:700; }
  .team-stat-value.big { font-size:1.45rem; font-variant-numeric:tabular-nums; }
  .team-stat-value.muted { color:var(--soft); font-style:italic; font-weight:400; }

  .roster-row { margin-top:18px; padding-top:16px; border-top:1px solid #f5eff8; }
  .roster-row-label { font-family:'Plus Jakarta Sans',sans-serif; font-size:0.792rem; font-weight:800; letter-spacing:.1em; text-transform:uppercase; color:var(--soft); margin-bottom:12px; text-align:center; }
  .roster-headshots { display:flex; gap:18px; justify-content:center; flex-wrap:wrap; }
  .roster-player { display:flex; flex-direction:column; align-items:center; gap:7px; width:80px; }
  .roster-headshot { width:66px; height:66px; border-radius:50%; object-fit:cover; object-position:top center; background:#f0ecf4; flex-shrink:0; }
  .roster-player-name { font-size:.78rem; font-weight:600; color:var(--ink); text-align:center; line-height:1.2; word-break:break-word; }
  .team-stat-value.pos { color:#1a6a4a; }
  .team-stat-value.neg { color:#a33247; }

  .why-block { margin:32px 0 18px; }
  .why-label { font-family:'Plus Jakarta Sans',sans-serif; font-size:0.77rem; font-weight:800; letter-spacing:.1em; text-transform:uppercase; margin-bottom:6px; display:block; }
  .why-label.win { color:#1a6a4a; }
  .why-label.lose { color:#a33247; }
  .why-label.watch { color:#7c3aed; }
  .why-block .placeholder { font-style:italic; color:var(--soft); font-weight:300; margin:0; }
  .content strong, .why-block strong { font-weight:800; color:var(--ink); }
  .why-block p { font-size:1rem; font-weight:300; line-height:1.8; color:var(--ink); margin:0 0 12px; }
  .why-block p:last-child { margin-bottom:0; }
  .content ul { list-style:none; margin:-4px 0 14px; display:flex; flex-direction:column; gap:6px; padding-left:0; }
  .content ul li { font-size:1rem; font-weight:300; line-height:1.7; padding-left:20px; position:relative; color:var(--ink); }
  .content ul li::before { content:'—'; position:absolute; left:0; color:var(--soft); }
  .content ol.numbered { list-style:none; margin:-4px 0 14px; padding-left:0; display:flex; flex-direction:column; gap:8px; counter-reset:nl; }
  .content ol.numbered li { font-size:1rem; font-weight:300; line-height:1.7; padding-left:30px; position:relative; color:var(--ink); counter-increment:nl; }
  .content ol.numbered li::before { content:counter(nl) '.'; position:absolute; left:0; color:#ED1C7C; font-weight:800; font-family:'Plus Jakarta Sans',sans-serif; }
  .team-lede { font-size:1rem; font-weight:300; line-height:1.8; color:var(--ink); margin:4px 0 24px; }
  .player-delta-chart { display:flex; flex-direction:column; gap:14px; margin:16px 0 4px; }
  .player-delta-event { background:white; border-radius:14px; padding:14px 18px; box-shadow:0 4px 24px #0000000a; }
  .player-delta-title { font-family:'Plus Jakarta Sans',sans-serif; font-size:0.858rem; font-weight:800; letter-spacing:.04em; color:var(--ink); padding-bottom:9px; border-bottom:1px solid #f0eaf4; margin-bottom:6px; }
  .player-delta-table { width:100%; border-collapse:collapse; font-size:.88rem; font-variant-numeric:tabular-nums; }
  .player-delta-table th { font-family:'Plus Jakarta Sans',sans-serif; font-size:0.66rem; font-weight:800; text-transform:uppercase; letter-spacing:.08em; color:var(--soft); padding:6px 4px; text-align:right; }
  .player-delta-table th:first-child { text-align:left; }
  .player-delta-table td { padding:7px 4px; text-align:right; border-bottom:1px solid #f5eff8; }
  .player-delta-table td:first-child { text-align:left; font-weight:600; }
  .player-delta-table tr:last-child td { border-bottom:none; }
  .player-delta-table .pos { color:#1a6a4a; font-weight:700; }
  .player-delta-table .neg { color:#a33247; font-weight:700; }
  .player-delta-table .flat { color:var(--soft); font-weight:600; }
  .player-delta-title { display:flex; align-items:center; gap:10px; flex-wrap:wrap; }
  .delta-event-pill { font-family:'Plus Jakarta Sans',sans-serif; font-size:0.638rem; font-weight:800; letter-spacing:.1em; text-transform:uppercase; padding:3px 9px; border-radius:6px; }
  .delta-event-pill.won  { background:rgba(34,197,94,.18); color:#176a47; }
  .delta-event-pill.lost { background:rgba(220,38,38,.13); color:#a33247; }
  .data-table-wrap { background:white; border-radius:14px; padding:16px 20px; box-shadow:0 4px 24px #0000000a; margin:14px 0 4px; overflow-x:auto; -webkit-overflow-scrolling:touch; }
  .data-table-label { font-family:'Plus Jakarta Sans',sans-serif; font-size:0.748rem; font-weight:800; letter-spacing:.08em; text-transform:uppercase; color:var(--soft); margin-bottom:10px; }
  .data-table { width:100%; border-collapse:collapse; font-size:.88rem; font-weight:400; }
  .data-table th { font-family:'Plus Jakarta Sans',sans-serif; font-size:0.682rem; font-weight:800; text-transform:uppercase; letter-spacing:.08em; color:var(--soft); padding:7px 10px; text-align:left; border-bottom:2px solid #f0eaf4; }
  .data-table td { padding:8px 10px; border-bottom:1px solid #f5eff8; }
  .data-table tr:last-child td { border-bottom:none; }
  .data-table .num { font-variant-numeric:tabular-nums; text-align:right; }
  .data-table th.num { text-align:right; }
  .data-table .rank { color:var(--soft); font-weight:500; width:36px; }
  .data-table .team { font-weight:600; }
  .data-table tr.highlight td { background:rgba(168,85,247,.07); }
  .data-table tr.highlight .team { color:#7c3aed; }
  .data-table .team-cell { display:flex; align-items:center; gap:10px; }
  .data-table .team-logo { width:22px; height:22px; object-fit:contain; flex-shrink:0; }

  /* Player-rating heatmap (7 matches × 6 players) */
  .rating-heatmap-wrap { background:white; border-radius:14px; padding:18px 20px 16px; box-shadow:0 4px 24px #0000000a; margin:14px 0 6px; overflow-x:auto; }
  .rating-heatmap-title { font-family:'Plus Jakarta Sans',sans-serif; font-size:0.858rem; font-weight:800; letter-spacing:.04em; color:var(--ink); padding-bottom:10px; border-bottom:1px solid #f0eaf4; margin-bottom:12px; }
  .rating-heatmap { width:100%; min-width:680px; border-collapse:separate; border-spacing:4px; font-variant-numeric:tabular-nums; }
  .rating-heatmap th { font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; font-size:0.682rem; letter-spacing:.04em; color:var(--ink); padding:6px 4px; text-align:center; vertical-align:bottom; line-height:1.25; }
  .rating-heatmap th.player-col { text-align:left; font-size:.7rem; min-width:90px; padding-left:8px; color:var(--soft); text-transform:uppercase; letter-spacing:.08em; }
  .rating-heatmap th .ev-sub { display:block; font-size:.52rem; color:var(--soft); font-weight:700; margin-top:3px; text-transform:uppercase; letter-spacing:.06em; }
  .rating-heatmap th .wl { display:inline-block; padding:2px 7px; border-radius:5px; font-size:.5rem; margin-top:4px; letter-spacing:.06em; }
  .rating-heatmap th .wl.w { background:rgba(34,197,94,.2); color:#176a47; }
  .rating-heatmap th .wl.l { background:rgba(220,38,38,.14); color:#a33247; }
  .rating-heatmap td { padding:9px 4px; text-align:center; border-radius:6px; font-weight:700; font-size:.85rem; }
  .rating-heatmap td.name { text-align:left; font-family:'Plus Jakarta Sans',sans-serif; font-size:0.935rem; font-weight:800; padding-left:8px; color:var(--ink); background:transparent; border-radius:0; }
  .rating-heatmap tr.smth-row td.name { color:#ED1C7C; }
  .rating-heatmap tr.smth-row { box-shadow:inset 3px 0 0 #ED1C7C; }
  .rating-heatmap td.r1 { background:#fbe1e4; color:#9b2138; }
  .rating-heatmap td.r2 { background:#fceadf; color:#955a26; }
  .rating-heatmap td.r3 { background:#fbf6e0; color:#7a6814; }
  .rating-heatmap td.r4 { background:#e8f3df; color:#3d6b28; }
  .rating-heatmap td.r5 { background:#d2e9d5; color:#176a47; }
  .rating-heatmap td.dnp { background:#f7f3fa; color:#bbb1c7; font-style:italic; font-weight:500; font-size:.78rem; }
  .rating-heatmap td.lowest { outline:2px solid #ED1C7C; outline-offset:-2px; }
  .rating-heatmap-legend { font-size:.7rem; color:var(--soft); font-weight:400; margin-top:10px; text-align:center; font-style:italic; }
  .rating-heatmap-legend .swatch { display:inline-block; width:11px; height:11px; border-radius:3px; vertical-align:-1px; margin:0 4px 0 10px; }
  .rating-heatmap-legend .swatch.low { background:#fbe1e4; }
  .rating-heatmap-legend .swatch.high { background:#d2e9d5; }
  .rating-heatmap-legend .ring { display:inline-block; width:11px; height:11px; border-radius:3px; vertical-align:-1px; margin:0 4px 0 10px; border:2px solid #ED1C7C; }

  /* Top 15 players ranked list — vertical cards with descriptions */
  .player-rank-list { display:flex; flex-direction:column; gap:14px; margin:14px 0 24px; padding:0; list-style:none; }
  .player-rank-card { display:flex; align-items:center; gap:18px; background:white; border-radius:14px; padding:16px 20px; box-shadow:0 4px 24px #0000000a; }
  .player-rank-header { display:flex; flex-direction:column; align-items:center; gap:8px; width:96px; flex-shrink:0; padding-right:14px; border-right:1px solid #f0eaf4; }
  .player-rank-num { font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; font-size:1.705rem; color:var(--soft); line-height:1; font-variant-numeric:tabular-nums; }
  .player-rank-img { width:64px; height:64px; border-radius:50%; object-fit:cover; object-position:top center; background:#f0ecf4; flex-shrink:0; }
  .player-rank-name { font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; font-size:0.99rem; line-height:1.15; text-align:center; white-space:nowrap; }
  .player-rank-team { display:flex; align-items:center; gap:5px; font-family:'Plus Jakarta Sans',sans-serif; font-size:0.66rem; font-weight:800; letter-spacing:.08em; text-transform:uppercase; color:var(--soft); }
  .player-rank-team img { width:14px; height:14px; object-fit:contain; }
  .content p.player-rank-desc { font-size:.95rem; font-weight:300; line-height:1.65; color:var(--ink); margin:0; flex:1; align-self:center; }
  @media(max-width:560px) { .player-rank-card { flex-direction:column; align-items:stretch; } .player-rank-header { flex-direction:row; width:auto; padding-right:0; border-right:none; border-bottom:1px solid #f0eaf4; padding-bottom:12px; gap:14px; justify-content:flex-start; align-self:auto; } }

  /* Masters London FI ranking list */
  .fi-rank-wrap { background:white; border-radius:14px; padding:18px 22px 16px; box-shadow:0 4px 24px #0000000a; margin:14px 0 4px; }
  .fi-rank-title { font-family:'Plus Jakarta Sans',sans-serif; font-size:0.858rem; font-weight:800; letter-spacing:.04em; color:var(--ink); padding-bottom:9px; border-bottom:1px solid #f0eaf4; margin-bottom:12px; }
  .fi-rank-list { columns:3; column-gap:28px; padding:0; list-style:none; margin:0; }
  .fi-rank-list li { display:flex; align-items:center; gap:8px; padding:6px 4px; font-size:.8rem; break-inside:avoid; border-bottom:1px solid #f5eff8; }
  .fi-rank-list li .rank { font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; font-size:0.748rem; color:var(--soft); width:22px; flex-shrink:0; text-align:right; }
  .fi-rank-list li img { width:18px; height:18px; object-fit:contain; flex-shrink:0; }
  .fi-rank-list li .player { flex:1; font-weight:600; line-height:1.2; }
  .fi-rank-list li .org-tag { font-family:'Plus Jakarta Sans',sans-serif; font-size:0.605rem; font-weight:800; letter-spacing:.06em; color:var(--soft); margin-left:4px; vertical-align:1px; }
  .fi-rank-list li .pct { font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; font-size:0.935rem; font-variant-numeric:tabular-nums; }
  .fi-rank-list li.smth-highlight { background:rgba(237,28,124,.1); border-radius:6px; padding-left:6px; }
  .fi-rank-list li.smth-highlight .player { color:#ED1C7C; }
  .fi-rank-list li.smth-highlight .pct { color:#ED1C7C; }
  @media(max-width:720px) { .fi-rank-list { columns:2; } }
  @media(max-width:480px) { .fi-rank-list { columns:1; } }

  /* Inline tweet embeds */
  .tweet-embeds { display:flex; flex-direction:column; gap:14px; align-items:center; margin:14px 0 24px; }
  .tweet-embeds blockquote.twitter-tweet { max-width:550px; width:100%; margin:0 !important; }

  /* YouTube embed (16:9) */
  .yt-embed { position:relative; width:100%; padding-top:56.25%; border-radius:12px; overflow:hidden; margin:14px 0 4px; box-shadow:0 4px 24px #0000000a; background:#000; }
  .yt-embed iframe { position:absolute; inset:0; width:100%; height:100%; border:0; }

  /* Primmie FI-share split chart */
  .primmie-chart { background:white; border-radius:14px; padding:18px 22px; box-shadow:0 4px 24px #0000000a; margin:14px 0 24px; }
  .primmie-title { font-family:'Plus Jakarta Sans',sans-serif; font-size:0.858rem; font-weight:800; letter-spacing:.04em; color:var(--ink); padding-bottom:10px; border-bottom:1px solid #f0eaf4; margin-bottom:14px; }
  .primmie-grid { display:grid; grid-template-columns:1fr 40px 1fr; align-items:center; gap:14px; }
  .primmie-col { display:flex; flex-direction:column; align-items:center; gap:6px; padding:14px 12px; border-radius:12px; }
  .primmie-col.high { background:rgba(34,197,94,.10); }
  .primmie-col.low  { background:rgba(220,38,38,.08); }
  .primmie-pct { font-family:'Plus Jakarta Sans',sans-serif; font-size:2.64rem; font-weight:800; font-variant-numeric:tabular-nums; line-height:1; }
  .primmie-col.high .primmie-pct { color:#176a47; }
  .primmie-col.low  .primmie-pct { color:#a33247; }
  .primmie-label { font-size:.78rem; color:var(--ink); font-weight:500; text-align:center; line-height:1.3; }
  .primmie-record { font-family:'Plus Jakarta Sans',sans-serif; font-size:0.858rem; font-weight:800; letter-spacing:.05em; color:var(--soft); }
  .primmie-divider { font-family:'Plus Jakarta Sans',sans-serif; font-size:0.88rem; font-weight:800; letter-spacing:.1em; color:var(--soft); text-align:center; }
  @media(max-width:560px) { .primmie-grid { grid-template-columns:1fr; } .primmie-divider { padding:6px 0; } }

  /* Featured player stat-line */
  .player-statline { background:white; border-radius:14px; padding:16px 20px; box-shadow:0 4px 24px #0000000a; margin:14px 0 24px; }
  .player-statline-head { display:flex; align-items:center; gap:14px; margin-bottom:14px; padding-bottom:10px; border-bottom:1px solid #f0eaf4; }
  .player-statline-head img.headshot { width:44px; height:44px; border-radius:50%; object-fit:cover; object-position:top center; background:#f0ecf4; flex-shrink:0; }
  .player-statline-name { font-family:'Plus Jakarta Sans',sans-serif; font-size:1.1rem; font-weight:800; color:var(--ink); line-height:1.2; }
  .player-statline-sub { font-size:.72rem; color:var(--soft); font-weight:400; margin-top:2px; }
  .player-statline-stats { display:grid; grid-template-columns:repeat(6, 1fr); gap:10px; }
  .player-statline-stat { display:flex; flex-direction:column; gap:2px; align-items:center; }
  .player-statline-stat-label { font-family:'Plus Jakarta Sans',sans-serif; font-size:0.605rem; font-weight:800; letter-spacing:.08em; text-transform:uppercase; color:var(--soft); }
  .player-statline-stat-value { font-family:'Plus Jakarta Sans',sans-serif; font-size:1.155rem; font-weight:800; font-variant-numeric:tabular-nums; color:var(--ink); }
  @media(max-width:560px) { .player-statline-stats { grid-template-columns:repeat(3, 1fr); row-gap:14px; } }

  /* FB+FD comparison panel */
  .fbfd-wrap { background:white; border-radius:14px; padding:18px 22px; box-shadow:0 4px 24px #0000000a; margin:14px 0 4px; }
  .fbfd-title { font-family:'Plus Jakarta Sans',sans-serif; font-size:0.858rem; font-weight:800; letter-spacing:.04em; color:var(--ink); padding-bottom:9px; border-bottom:1px solid #f0eaf4; margin-bottom:14px; }
  .fbfd-grid { display:grid; grid-template-columns:1fr 1fr; gap:18px; }
  .fbfd-col { display:flex; flex-direction:column; gap:8px; padding:12px 14px; border-radius:10px; background:#faf6fd; }
  .fbfd-col-header { font-family:'Plus Jakarta Sans',sans-serif; font-size:0.682rem; font-weight:800; letter-spacing:.1em; text-transform:uppercase; color:var(--soft); text-align:center; }
  .fbfd-col-sub { font-size:.7rem; color:var(--soft); font-weight:400; text-align:center; font-style:italic; margin-top:-4px; margin-bottom:4px; }
  .fbfd-row { display:flex; justify-content:space-between; align-items:baseline; padding:6px 8px; border-bottom:1px solid #ece5f3; }
  .fbfd-row:last-child { border-bottom:none; }
  .fbfd-row-label { font-size:.78rem; color:var(--ink); font-weight:500; }
  .fbfd-row-val { font-family:'Plus Jakarta Sans',sans-serif; font-size:1.43rem; font-weight:800; color:#ED1C7C; font-variant-numeric:tabular-nums; }
  .fbfd-baseline { font-size:.72rem; color:var(--soft); font-weight:300; font-style:italic; text-align:center; margin-top:10px; }
  @media(max-width:560px) { .fbfd-grid { grid-template-columns:1fr; } .rating-heatmap th.player-col { min-width:74px; } }

  .content em { font-style:italic; }
  .content a { color:var(--ink); font-weight:400; }
  .content a:hover { opacity:.7; }
  footer { position:relative; z-index:1; text-align:center; padding:24px; color:var(--soft); font-size:.75rem; font-weight:300; }
  @keyframes fadeUp{from{opacity:0;transform:translateY(24px)}to{opacity:1;transform:translateY(0)}}
  .page { animation:fadeUp .6s ease both; }
  /* Larger body text on the fixed-width mobile viewport (fires at width=820). */
  @media (max-width:1000px){
    .content p, .content ul li, .content ol.numbered li { font-size:1.3rem; }
  }
  /* ── Mobile (phone) ── */
  @media (max-width:600px){
    .page { padding:24px 16px 56px; }
    .content p, .content ul li, .content ol.numbered li { font-size:.94rem; }
    .data-table-wrap { padding:14px 12px; }
    .data-table { font-size:.8rem; min-width:480px; }
    .data-table th, .data-table td { padding:7px 6px; }
    .data-table .team-logo { width:18px; height:18px; }
  }
</style>
</head>
<body>
<nav class="toc">
  <div class="toc-title">Sections</div>
  <a href="#paperrex">Paper Rex&rsquo;s (Un)inevitability</a>
  <a href="#power-rankings">Power Rankings</a>
  <a class="toc-sub" href="#s-tier">S Tier</a>
  <a class="toc-sub" href="#a-tier">A Tier</a>
  <a class="toc-sub" href="#b-tier">B Tier</a>
  <a class="toc-sub" href="#c-tier">C Tier</a>
  <a class="toc-sub" href="#d-tier">D Tier</a>
  <a href="#predictions">Predictions</a>
  <a href="#top-players">Top 15 Players</a>
</nav>
<div class="top-nav">
  <a href="/"><img src="/logo.svg" alt="Home" class="home-logo"></a>
</div>
<div class="page">
  <div class="article">
    <div class="label">Research / Opinion</div>
    <h1>Masters London Preview</h1>
    <div class="byline">Bobo &mdash; June 2026</div>
    <div class="cover">
      <img src="/static/PRXPacStage1Win-full.jpg" alt="Paper Rex win VCT Pacific Stage 1 in Ho Chi Minh City">
    </div>
    <p class="cover-caption">Paper Rex hoist the Pacific Stage 1 trophy in Ho Chi Minh City, cementing themselves as Masters London favorites.</p>
    <div class="content">
      <div class="section-bubble-wrap section-bubble-tight"><span class="section-bubble" id="paperrex"><span class="section-bubble-text">Paper Rex&rsquo;s (Un)inevitability</span></span></div>

      <p>Paper Rex are the favorites for Masters London, and it&rsquo;s not much of a secret. In fact, I believe that Paper Rex are the clearest favorites for an international in the history of VCT post-franchising, aside from one or two teams. This does not necessarily mean they&rsquo;re the strongest pre-international team, just the most consensus favorites. Other favorites often had flaws shown (e.g. before Tokyo, Fnatic lost their domestic grand final to Team Liquid) or had equally strong teams to compete with (e.g. before Bangkok, G2 looked dominant, but so did Vitality). Paper Rex have shown an aversion to slipping up in Stage 1, and no other teams in any region have looked nearly as good. The only other two teams that can compare to Paper Rex&rsquo;s current situation are Pre-Champions-LA Fnatic and Pre-Champions-Seoul Gen.G.</p>

      <p>Heading into Champions LA, Fnatic were the largest favorites we&rsquo;ve ever seen heading into an international. They won the two internationals prior to Champions, which is something we can&rsquo;t say of any other team in VCT history. I have to give Fnatic the nod here.</p>

      <p>Gen.G prior to Champions Seoul is a better comparison. Let&rsquo;s look it over:</p>

      <div class="comparison-chart">
        <div class="comparison-header">
          <div class="comparison-team">
            <img src="/logos/GEN.png" alt="Gen.G" onerror="this.style.display='none'">
            <div class="comparison-team-name">Gen.G</div>
            <div class="comparison-team-sub">pre&ndash;Champions Seoul</div>
          </div>
          <div></div>
          <div class="comparison-team">
            <img src="/logos/PRX.png" alt="Paper Rex" onerror="this.style.display='none'">
            <div class="comparison-team-name">Paper Rex</div>
            <div class="comparison-team-sub">pre&ndash;Masters London</div>
          </div>
        </div>
        <div class="comparison-row">
          <div class="comparison-value">1st &mdash; Masters Shanghai</div>
          <div class="comparison-label">Most Recent Intl.</div>
          <div class="comparison-value">2nd &mdash; Masters Santiago</div>
        </div>
        <div class="comparison-row">
          <div class="comparison-value num">67.0%</div>
          <div class="comparison-label">Stage Pythagorean</div>
          <div class="comparison-value num">73.9%</div>
        </div>
        <div class="comparison-row">
          <div class="comparison-value">Won via Upper Bracket</div>
          <div class="comparison-label">Domestic Trophy Run</div>
          <div class="comparison-value">Won via Lower Bracket</div>
        </div>
        <div class="comparison-row">
          <div class="comparison-value num">+3.61</div>
          <div class="comparison-label">BenPom Rating<br>(#1 World)</div>
          <div class="comparison-value num">+3.80</div>
        </div>
      </div>

      <p>You can go either way between the two in debating which team is the clearer favorite heading into their respective events. If you prioritize winning the most recent international above all else, then you&rsquo;d take Gen.G before Champions Seoul. If you prioritize having a greater performance in the more recent domestic event, then you&rsquo;d take Paper Rex before Masters London. I personally find Paper Rex to be the clearer favorite.</p>

      <p>Perhaps the most damning data point here is that all four members of Plat Chat predicted Paper Rex to win Masters London, something that (I think) has never happened before. For those curious, no one on Plat Chat picked Gen.G at Champions 2024, which settles the debate between the perceptions of pre-London Paper Rex and pre-Seoul Gen.G.</p>

      <p>Now, funnily enough, both of these two other teams (LA Fnatic and Seoul Gen.G) went on to lose their respective internationals. In this way, Masters London is just as much Paper Rex&rsquo;s event to lose as it is to win. However, the question of who they would even lose to is a hard question to answer. Let&rsquo;s look over all the teams through power rankings, seeing who could pose a threat to Paper Rex.</p>

      <div class="section-bubble-wrap"><span class="section-bubble" id="power-rankings"><span class="section-bubble-text">Power Rankings</span></span></div>

      <p class="section-note">Note: From C-tier onwards, I wrote slightly shorter sections on the teams.<br>(cough DRG cough)</p>

      <div class="tier-overview">
        <div class="tier-overview-label">Tier List &mdash; click a team to jump</div>
        <div class="tier-row">
          <div class="tier-badge s">S</div>
          <div class="tier-teams">
            <a class="tier-chip" href="#team-prx" title="Paper Rex"><img src="/logos/PRX.png" alt="Paper Rex" onerror="this.style.display='none'"></a>
          </div>
        </div>
        <div class="tier-row">
          <div class="tier-badge a">A</div>
          <div class="tier-teams">
            <a class="tier-chip" href="#team-g2" title="G2"><img src="/logos/G2.png" alt="G2" onerror="this.style.display='none'"></a>
          </div>
        </div>
        <div class="tier-row">
          <div class="tier-badge b">B</div>
          <div class="tier-teams">
            <a class="tier-chip" href="#team-nrg" title="NRG"><img src="/logos/NRG.png" alt="NRG" onerror="this.style.display='none'"></a>
            <a class="tier-chip" href="#team-edg" title="EDG"><img src="/logos/EDG.png" alt="EDG" onerror="this.style.display='none'"></a>
            <a class="tier-chip" href="#team-th" title="Team Heretics"><img src="/logos/TH.png" alt="Team Heretics" onerror="this.style.display='none'"></a>
            <a class="tier-chip" href="#team-fs" title="Full Sense"><img src="/logos/FS.png" alt="Full Sense" onerror="this.style.display='none'"></a>
            <a class="tier-chip" href="#team-lev" title="Leviat&aacute;n"><img src="/logos/LEV.png" alt="Leviat&aacute;n" onerror="this.style.display='none'"></a>
          </div>
        </div>
        <div class="tier-row">
          <div class="tier-badge c">C</div>
          <div class="tier-teams">
            <a class="tier-chip" href="#team-ge" title="Global Esports"><img src="/logos/GE.png" alt="Global Esports" onerror="this.style.display='none'"></a>
            <a class="tier-chip" href="#team-xlg" title="XLG"><img src="/logos/XLG.png" alt="XLG" onerror="this.style.display='none'"></a>
            <a class="tier-chip" href="#team-vit" title="Vitality"><img src="/logos/VIT.png" alt="Vitality" onerror="this.style.display='none'"></a>
          </div>
        </div>
        <div class="tier-row">
          <div class="tier-badge d">D</div>
          <div class="tier-teams">
            <a class="tier-chip" href="#team-fut" title="FUT"><img src="/logos/FUT.png" alt="FUT" onerror="this.style.display='none'"></a>
            <a class="tier-chip" href="#team-drg" title="Dragon Ranger Gaming"><img src="/logos/DRG.png" alt="Dragon Ranger Gaming" onerror="this.style.display='none'"></a>
          </div>
        </div>
      </div>

      <h2 id="s-tier" class="tier-section-header s">S Tier</h2>

      <div class="team-heading">
        <img src="/logos/PRX.png" alt="Paper Rex" onerror="this.style.display='none'">
        <h2 id="team-prx">Paper Rex</h2>
        <span class="tier-pill s">S Tier</span>
      </div>
      <div class="team-stat-card">
        <div class="team-stat-row">
          <div class="team-stat-block">
            <div class="team-stat-label">Pacific Stage 1 Pyth</div>
            <div class="team-stat-value big">73.9%</div>
          </div>
          <div class="team-stat-block">
            <div class="team-stat-label">BenPom (Current)</div>
            <div class="team-stat-value big pos">+3.80</div>
          </div>
          <div class="team-stat-block">
            <div class="team-stat-label">Pacific Stage 1 Finish</div>
            <div class="team-stat-value big">1st</div>
          </div>
        </div>
        <div class="roster-row">
          <div class="roster-row-label">Roster</div>
          <div class="roster-headshots">
            <div class="roster-player"><img class="roster-headshot" src="https://owcdn.net/img/69735f0889a6b.png" alt="Jinggg" onerror="this.style.visibility=&quot;hidden&quot;"><div class="roster-player-name">Jinggg</div></div>
            <div class="roster-player"><img class="roster-headshot" src="https://owcdn.net/img/69735f207c9bf.png" alt="d4v41" onerror="this.style.visibility=&quot;hidden&quot;"><div class="roster-player-name">d4v41</div></div>
            <div class="roster-player"><img class="roster-headshot" src="https://owcdn.net/img/69735f135cf21.png" alt="f0rsakeN" onerror="this.style.visibility=&quot;hidden&quot;"><div class="roster-player-name">f0rsakeN</div></div>
            <div class="roster-player"><img class="roster-headshot" src="https://owcdn.net/img/69735f4a21e3b.png" alt="invy" onerror="this.style.visibility=&quot;hidden&quot;"><div class="roster-player-name">invy</div></div>
            <div class="roster-player"><img class="roster-headshot" src="https://owcdn.net/img/69735f396861f.png" alt="something" onerror="this.style.visibility=&quot;hidden&quot;"><div class="roster-player-name">something</div></div>
          </div>
        </div>
      </div>
      <p class="team-lede">I&rsquo;ve already discussed them quite a bit, so I&rsquo;ll be quick here and rehash some of my previous points.</p>

      <div class="why-block">
        <span class="why-label win">Why they can win</span>
        <p>The narrative here is simple: Paper Rex are the best team in the strongest region. Pacific sent the 1st and 2nd place finishers at Masters Santiago, solidifying themselves as the premier region in VCT. Then, not only did Paper Rex win Stage 1, they won the grand final 3&ndash;0 <em>with map ban disadvantage</em>.</p>
        <p>Being the most dominant team in the strongest region is enough, but there are countless other metrics by which you could make an argument for Paper Rex.</p>
        <ul>
          <li>They had the highest Pyth% across all regions in Stage 1 (73.9%)</li>
          <li>They had the highest Win% across all regions in Stage 1 (74.1%)</li>
          <li>They have the highest BenPom rating across all regions right now (3.80)</li>
          <li>They are the most internationally successful + experienced core at Masters London</li>
        </ul>
      </div>

      <div class="why-block">
        <span class="why-label lose">Why they won&rsquo;t win</span>
        <p>As much as he&rsquo;s been their best player (or second best, if you&rsquo;re partial to Jinggg), I&rsquo;m worried Something could be the reason Paper Rex chokes again. Historically, Something is the player who falls short when it matters most for Paper Rex. Having someone like that be your best player can be dangerous. In fact, let&rsquo;s look at a similar situation in the past with Paper Rex.</p>
        <p>Before Champs 2025, Paper Rex were the consensus favorites to win the tournament, with Something being their best player and the second-highest-rated player in Pacific Stage 2. Ultimately, Paper Rex finished fourth. Something was the lowest-rated player once and second-lowest rated player twice in their final 3 matches.</p>
        <p>At Masters London, Paper Rex are the consensus favorites with Something being their best player and the second-highest-rated player in Pacific Stage 1. I don&rsquo;t like the direction this is going.</p>
        <p>Even at the most recent Masters Santiago, Something was the lowest-rated player in the grand final, dropping an abysmal 0.65 rating. It&rsquo;s worth noting that Paper Rex were actually favored pre-match.</p>
        <p>Alongside the aforementioned trend about being a heavy favorite before an international, historical trends are not in Paper Rex&rsquo;s favor.</p>
      </div>

      <div class="why-block">
        <span class="why-label watch">Something to watch for</span>
        <p>Paper Rex have what I consider to be the best map of any team at Masters London with their Split. Since losing on Split to Nongshim RedForce at Masters Santiago, they&rsquo;ve played Split 5 times in Stage 1 and won all 5 times, including against Nongshim. Their past 3 Split wins were 13&ndash;9 (T1), 13&ndash;1 (KRX), and 13&ndash;3 (GE). Jesus.</p>
        <p>Furthermore, their BenPom map rating of 4.55 is one of the highest map ratings we&rsquo;ve ever seen before an international (sample size &ge; 3), behind EG&rsquo;s Fracture before Tokyo (5.42), G2&rsquo;s Pearl heading into this same Masters London (5.06), 100 Thieves&rsquo; Bind before Shanghai (4.92), Nongshim&rsquo;s Haven before Santiago (4.87), and some miscellaneous maps from 2023 FNATIC.</p>
        <div class="data-table-wrap">
          <div class="data-table-label">Highest Pre-International Map BenPom Ratings (n &ge; 3) excluding 2023 Fnatic</div>
          <table class="data-table">
            <thead><tr><th>#</th><th>Team / Map</th><th>Pre-Intl</th><th class="num">BenPom</th></tr></thead>
            <tbody>
              <tr><td class="rank">1</td><td class="team">EG / Fracture</td><td>Masters Tokyo</td><td class="num">5.42</td></tr>
              <tr><td class="rank">2</td><td class="team">G2 / Pearl</td><td>Masters London</td><td class="num">5.06</td></tr>
              <tr><td class="rank">3</td><td class="team">100T / Bind</td><td>Masters Shanghai</td><td class="num">4.92</td></tr>
              <tr><td class="rank">4</td><td class="team">NS / Haven</td><td>Masters Santiago</td><td class="num">4.87</td></tr>
              <tr class="highlight"><td class="rank">5</td><td class="team">PRX / Split</td><td>Masters London</td><td class="num">4.55</td></tr>
            </tbody>
          </table>
        </div>
      </div>

      <h2 id="a-tier" class="tier-section-header a">A Tier</h2>

      <div class="team-heading">
        <img src="/logos/G2.png" alt="G2" onerror="this.style.display='none'">
        <h2 id="team-g2">G2</h2>
        <span class="tier-pill a">A Tier</span>
      </div>
      <div class="team-stat-card">
        <div class="team-stat-row">
          <div class="team-stat-block">
            <div class="team-stat-label">Americas Stage 1 Pyth</div>
            <div class="team-stat-value big">63.1%</div>
          </div>
          <div class="team-stat-block">
            <div class="team-stat-label">BenPom (Current)</div>
            <div class="team-stat-value big pos">+3.35</div>
          </div>
          <div class="team-stat-block">
            <div class="team-stat-label">Americas Stage 1 Finish</div>
            <div class="team-stat-value big">1st</div>
          </div>
        </div>
        <div class="roster-row">
          <div class="roster-row-label">Roster</div>
          <div class="roster-headshots">
            <div class="roster-player"><img class="roster-headshot" src="https://owcdn.net/img/6224a67aabf26.png" alt="BABYBAY" onerror="this.style.visibility=&quot;hidden&quot;"><div class="roster-player-name">BABYBAY</div></div>
            <div class="roster-player"><img class="roster-headshot" src="https://owcdn.net/img/64169400bfed7.png" alt="jawgemo" onerror="this.style.visibility=&quot;hidden&quot;"><div class="roster-player-name">jawgemo</div></div>
            <div class="roster-player"><img class="roster-headshot" src="https://owcdn.net/img/6613103dd51a4.png" alt="leaf" onerror="this.style.visibility=&quot;hidden&quot;"><div class="roster-player-name">leaf</div></div>
            <div class="roster-player"><img class="roster-headshot" src="https://owcdn.net/img/65e466071ca19.png" alt="trent" onerror="this.style.visibility=&quot;hidden&quot;"><div class="roster-player-name">trent</div></div>
            <div class="roster-player"><img class="roster-headshot" src="https://owcdn.net/img/65e4660ee20e5.png" alt="valyn" onerror="this.style.visibility=&quot;hidden&quot;"><div class="roster-player-name">valyn</div></div>
          </div>
        </div>
      </div>
      <div class="why-block">
        <span class="why-label win">Why they can win</span>
        <p>Personally, I think G2 are so so so&hellip; good. Not great, but good. In this way, I think they&rsquo;re perfectly poised to win this tournament, conditional upon one or two teams faltering. I&rsquo;ll explain.</p>
        <p>No player on G2 is in insane form, unlike past times (Leaf before Bangkok, Trent before Toronto, Valyn before Champs Paris). Some people would say Trent is, but I disagree. No G2 player is consistently going into the server and dominating.</p>
        <p>In G2&rsquo;s past 3 matches, each one came down to the final map. They haven&rsquo;t been as domestically dominant as previous iterations, and I think they&rsquo;re quite honestly a tad overrated by BenPom.</p>
        <p>How do they win, then? The one metric where they&rsquo;ll win is experience. This core has been dominating domestically and playing well enough at internationals for years now. Meanwhile, London is filled with internationally inexperienced teams (e.g. Global Esports, FUT, LEV, etc.) and relatively new rosters (e.g. Vitality). In fact, other than NRG, PRX, and EDG, every other team at London has at least one player who&rsquo;s never played at an international.</p>
        <p>While the mid-tier teams may be flustered with the international experience and struggle to adapt to a post-Neon meta, G2&rsquo;s core will be comfortable on the international stage and with shifting their team comps as they&rsquo;ve done many times throughout the years.</p>
        <p>G2 has two other reasons for optimism. Firstly, Babybay has hopefully gotten his &ldquo;first international&rdquo; jitters out of the way when he was the weak link at Santiago. Secondly, with the strength of their Pearl, they can pretty much guarantee a map as long as that gets through.</p>
        <p>This isn&rsquo;t the best G2 has looked coming into an international, but it&rsquo;d be stupid to doubt them. They have a bye into playoffs, they&rsquo;ll be calm when other teams would be nervous, and they&rsquo;re fresh off of another Americas trophy. So, as much as they might not dominate, if PRX can choke at the end (which is, as I&rsquo;ve mentioned, extremely possible) and NRG can continue to be unable to beat G2, I like G2&rsquo;s chances.</p>
      </div>

      <div class="why-block">
        <span class="why-label lose">Why they won&rsquo;t win</span>
        <p>For the same reason that they might win. They&rsquo;re good, but not great. If G2 doesn&rsquo;t win the last 3 rounds against Kr&uuml; in Playoffs and they lose Lotus, they go on to Haven as map 3 &mdash; a map which they&rsquo;ve lost the past two times they played it. They probably lose the match, go to Lowers, and maybe they wouldn&rsquo;t even be at London altogether. The round differentials in their last three matches are +1, +4, and &minus;10. Again, they&rsquo;re not that convincing.</p>
        <p>If one of PRX, NRG, or EDG can elevate their form, I don&rsquo;t see G2 being able to beat them.</p>
      </div>

      <div class="why-block">
        <span class="why-label watch">Something to watch for</span>
        <p>One of Babybay or Leaf needs to step up. Or both of them! Trent and Valyn have been carrying too much of the firepower recently. Babybay and Leaf have been playing high-fragging roles like Chamber and Phoenix, but they&rsquo;ve been lackluster. Leaf, we know, has the potential to be a top-5 player in the world, but he&rsquo;s been only decent in Stage 1 (1.01 rating). Keep in mind his most played agent has been Phoenix.</p>
      </div>

      <h2 id="b-tier" class="tier-section-header b">B Tier</h2>

      <div class="team-heading">
        <img src="/logos/NRG.png" alt="NRG" onerror="this.style.display='none'">
        <h2 id="team-nrg">NRG</h2>
        <span class="tier-pill b">B Tier</span>
      </div>
      <div class="team-stat-card">
        <div class="team-stat-row">
          <div class="team-stat-block">
            <div class="team-stat-label">Americas Stage 1 Pyth</div>
            <div class="team-stat-value big">61.9%</div>
          </div>
          <div class="team-stat-block">
            <div class="team-stat-label">BenPom (Current)</div>
            <div class="team-stat-value big pos">+2.81</div>
          </div>
          <div class="team-stat-block">
            <div class="team-stat-label">Americas Stage 1 Finish</div>
            <div class="team-stat-value big">3rd</div>
          </div>
        </div>
        <div class="roster-row">
          <div class="roster-row-label">Roster</div>
          <div class="roster-headshots">
            <div class="roster-player"><img class="roster-headshot" src="https://owcdn.net/img/6974068bad561.png" alt="Ethan" onerror="this.style.visibility=&quot;hidden&quot;"><div class="roster-player-name">Ethan</div></div>
            <div class="roster-player"><img class="roster-headshot" src="https://owcdn.net/img/697426e1f3308.png" alt="brawk" onerror="this.style.visibility=&quot;hidden&quot;"><div class="roster-player-name">brawk</div></div>
            <div class="roster-player"><img class="roster-headshot" src="https://owcdn.net/img/697406c5ecbcd.png" alt="keiko" onerror="this.style.visibility=&quot;hidden&quot;"><div class="roster-player-name">keiko</div></div>
            <div class="roster-player"><img class="roster-headshot" src="https://owcdn.net/img/697406cf6b086.png" alt="mada" onerror="this.style.visibility=&quot;hidden&quot;"><div class="roster-player-name">mada</div></div>
            <div class="roster-player"><img class="roster-headshot" src="https://owcdn.net/img/6974067806f34.png" alt="skuba" onerror="this.style.visibility=&quot;hidden&quot;"><div class="roster-player-name">skuba</div></div>
          </div>
        </div>
      </div>
      <div class="why-block">
        <span class="why-label win">Why they can win</span>
        <p>In Weeks 1 and 2, NRG lost to Furia and Kr&uuml; with Keiko playing Killjoy and Omen through those 4 maps. Since then, Keiko hasn&rsquo;t gone a match without playing duelist at least once and NRG has won those next 6 games until losing 2&ndash;3 to G2, a perfectly respectable loss. With Keiko on duelist, this team is ridiculously hot. So long as that continues, they can go as far as he takes them. It also helps that Brawk is looking like the best initiator in the world.</p>
        <p>NRG has the experience and are a proven roster, something that will be valuable at Masters London just as I pointed out with G2. The individuals are in form, even to the point where Ethan dropped a match-high 1.31 rating in their EWC qualifier. If player form continues to be good and role issues are figured out (i.e. Keiko/Mada), NRG have enough momentum to go all the way to the grand final. Whether they can beat a top-level team with their new roster remains to be seen.</p>
      </div>

      <div class="why-block">
        <span class="why-label lose">Why they won&rsquo;t win</span>
        <p>Transitioning from that previous section, it feels like this NRG team lacks a certain W streamer when it comes to these big matches against high-level opponents who put the onus on NRG to step up. NRG barely beat MIBR in Kickoff to qualify for Santiago after failing multiple times to qualify. At Santiago, when they reached top 3, their performance dropped drastically as they lost to NS and PRX (who they previously 2&ndash;0&rsquo;d a week ago). Similarly, NRG barely squeaked by 100 Thieves for the last spot to qualify for London.</p>
        <p>This new roster refuses to be irrelevant, but they fail to prove their relevancy when it matters most. Why should London be any different?</p>
      </div>

      <div class="why-block">
        <span class="why-label watch">Something to watch for</span>
        <p>Keiko and Mada playing hot potato with their roles. In some matches, Keiko plays Jett and Mada plays Harbor. In some matches, Keiko plays Omen and Mada plays Neon. In fact, Keiko played THREE DIFFERENT ROLES alongside Mada&rsquo;s Neon in their most recent Americas match. ONE match. What makes it even worse is that, during EWC qualifiers after Stage 1, Keiko was playing Vyse &mdash; yet another agent for his agent pool. I don&rsquo;t understand what&rsquo;s happening with Keiko&rsquo;s role, and I honestly don&rsquo;t know if Keiko knows himself.</p>
      </div>

      <div class="team-heading">
        <img src="/logos/EDG.png" alt="EDG" onerror="this.style.display='none'">
        <h2 id="team-edg">EDward Gaming</h2>
        <span class="tier-pill b">B Tier</span>
      </div>
      <div class="team-stat-card">
        <div class="team-stat-row">
          <div class="team-stat-block">
            <div class="team-stat-label">China Stage 1 Pyth</div>
            <div class="team-stat-value big">71.8%</div>
          </div>
          <div class="team-stat-block">
            <div class="team-stat-label">BenPom (Current)</div>
            <div class="team-stat-value big pos">+0.95</div>
          </div>
          <div class="team-stat-block">
            <div class="team-stat-label">China Stage 1 Finish</div>
            <div class="team-stat-value big">1st</div>
          </div>
        </div>
        <div class="roster-row">
          <div class="roster-row-label">Roster</div>
          <div class="roster-headshots">
            <div class="roster-player"><img class="roster-headshot" src="https://owcdn.net/img/677fe4289bb34.png" alt="CHICHOO" onerror="this.style.visibility=&quot;hidden&quot;"><div class="roster-player-name">CHICHOO</div></div>
            <div class="roster-player"><img class="roster-headshot" src="https://owcdn.net/img/67e2e8846d5a5.png" alt="Jieni7" onerror="this.style.visibility=&quot;hidden&quot;"><div class="roster-player-name">Jieni7</div></div>
            <div class="roster-player"><img class="roster-headshot" src="https://owcdn.net/img/677fe435edc57.png" alt="Smoggy" onerror="this.style.visibility=&quot;hidden&quot;"><div class="roster-player-name">Smoggy</div></div>
            <div class="roster-player"><img class="roster-headshot" src="https://owcdn.net/img/677fe40edf9da.png" alt="ZmjjKK" onerror="this.style.visibility=&quot;hidden&quot;"><div class="roster-player-name">ZmjjKK</div></div>
            <div class="roster-player"><img class="roster-headshot" src="https://owcdn.net/img/677fe4210de6e.png" alt="nobody" onerror="this.style.visibility=&quot;hidden&quot;"><div class="roster-player-name">nobody</div></div>
          </div>
        </div>
      </div>
      <div class="why-block">
        <span class="why-label win">Why they can win</span>
        <p>I honestly believe that EDG have the strongest case for winning Masters London after Paper Rex, for very simple reasons:</p>
        <ul>
          <li>They&rsquo;re the only one-seed to win their domestic title through the upper bracket.</li>
          <li>This EDG core is proven to be of international-winning caliber and has the experience to perform well at an international + in a new meta.</li>
          <li>Speaking of, ZmjjKK has already been playing a copious amount of Jett. Having the historically greatest Chinese Valorant player with his comfort agent coming back into the meta is a huge advantage. Neon did not treat EDG well (see Masters Santiago).</li>
          <li>Their domestic performance was dominant, only losing once to XLG who they proceeded to beat twice in playoffs.</li>
        </ul>
        <p>3 of the players on EDG have the capability to be top 15 players in the world, and if they can prove that they&rsquo;ve been doing more than just farming CN teams, this team meets every criteria to win Masters London.</p>
      </div>

      <div class="why-block">
        <span class="why-label lose">Why they won&rsquo;t win</span>
        <p>CN, right now, looks the worst that it&rsquo;s ever been. Somehow worse than 2023. At the previous international, both group stage Chinese teams went out 0&ndash;2. In playoffs, AG went out 1&ndash;2. The entire Chinese region won 1 match and suffered 6 losses. I can&rsquo;t emphasize enough how the issue of EDG is simply their region. They truly meet every criteria, but how can you have faith in a team that&rsquo;s playing in a region whose best teams wouldn&rsquo;t win a T2 Pacific tournament? You can&rsquo;t. It just won&rsquo;t be surprising if EDG are the real deal.</p>
      </div>

      <div class="why-block">
        <span class="why-label watch">Something to watch for</span>
        <p>Amidst all of the discussion of ZmjjKK&rsquo;s return, I think it&rsquo;s important to remember that CHICHOO is (I believe) better than KK, and has been for the past two years. The last time EDG looked legit (Masters Bangkok), CHICHOO was making everyone else look silly. For instance, in the last match that EDG won in Bangkok, look at the performance CHICHOO put up in a do-or-die Map 3 victory:</p>
        <div class="player-statline">
          <div class="player-statline-head">
            <img class="headshot" src="https://owcdn.net/img/677fe4289bb34.png" alt="CHICHOO" onerror="this.style.visibility='hidden'">
            <div>
              <div class="player-statline-name">CHICHOO &middot; EDG</div>
              <div class="player-statline-sub">Masters Bangkok 2025, UB Semifinals vs T1 &mdash; Map 3 (Pearl)</div>
            </div>
          </div>
          <div class="player-statline-stats">
            <div class="player-statline-stat"><div class="player-statline-stat-label">R2.0</div><div class="player-statline-stat-value">1.85</div></div>
            <div class="player-statline-stat"><div class="player-statline-stat-label">ACS</div><div class="player-statline-stat-value">357</div></div>
            <div class="player-statline-stat"><div class="player-statline-stat-label">K / D / A</div><div class="player-statline-stat-value">32/16/10</div></div>
            <div class="player-statline-stat"><div class="player-statline-stat-label">KAST</div><div class="player-statline-stat-value">88%</div></div>
            <div class="player-statline-stat"><div class="player-statline-stat-label">ADR</div><div class="player-statline-stat-value">221</div></div>
            <div class="player-statline-stat"><div class="player-statline-stat-label">FK / FD</div><div class="player-statline-stat-value">2 / 0</div></div>
          </div>
        </div>
        <p>CHICHOO ended up as the highest-rated player at Masters Bangkok.</p>
        <p>Smoggy is also seemingly entering his prime based on stats alone. I haven&rsquo;t watched much CN Valorant, but I know this team is filled with more talent than just ZmjjKK.</p>
      </div>

      <div class="team-heading">
        <img src="/logos/TH.png" alt="Team Heretics" onerror="this.style.display='none'">
        <h2 id="team-th">Team Heretics</h2>
        <span class="tier-pill b">B Tier</span>
      </div>
      <div class="team-stat-card">
        <div class="team-stat-row">
          <div class="team-stat-block">
            <div class="team-stat-label">EMEA Stage 1 Pyth</div>
            <div class="team-stat-value big">65.4%</div>
          </div>
          <div class="team-stat-block">
            <div class="team-stat-label">BenPom (Current)</div>
            <div class="team-stat-value big pos">+2.49</div>
          </div>
          <div class="team-stat-block">
            <div class="team-stat-label">EMEA Stage 1 Finish</div>
            <div class="team-stat-value big">1st</div>
          </div>
        </div>
        <div class="roster-row">
          <div class="roster-row-label">Roster</div>
          <div class="roster-headshots">
            <div class="roster-player"><img class="roster-headshot" src="https://owcdn.net/img/69778b1c2192b.png" alt="Boo" onerror="this.style.visibility=&quot;hidden&quot;"><div class="roster-player-name">Boo</div></div>
            <div class="roster-player"><img class="roster-headshot" src="https://owcdn.net/img/69778b71c9366.png" alt="RieNs" onerror="this.style.visibility=&quot;hidden&quot;"><div class="roster-player-name">RieNs</div></div>
            <div class="roster-player"><img class="roster-headshot" src="https://owcdn.net/img/69778b5885054.png" alt="Wo0t" onerror="this.style.visibility=&quot;hidden&quot;"><div class="roster-player-name">Wo0t</div></div>
            <div class="roster-player"><img class="roster-headshot" src="https://owcdn.net/img/69778b2bf3e20.png" alt="benjyfishy" onerror="this.style.visibility=&quot;hidden&quot;"><div class="roster-player-name">benjyfishy</div></div>
            <div class="roster-player"><div class="roster-headshot"></div><div class="roster-player-name">koshmaras</div></div>
          </div>
        </div>
      </div>
      <div class="why-block">
        <span class="why-label win">Why they can win</span>
        <p>They&rsquo;re a 1-seed who beat a team with Chronicle, Derke, and Sayonara in the EMEA grand final <em>while having map-ban disadvantage</em>. That alone qualifies them to be contenders.</p>
        <p>Outside of that, I would highlight their ability to midround. In watching some Heretics matches, the calling of every player on TH has been so beautiful during the mid-round fights and pauses. Whether or not they have a man advantage, they figure out the correct percentage play. For instance, watch this clip of perfect calling from benjyfishy and RieNs:</p>
        <video src="/static/heretics_clip.mp4" controls playsinline style="width:100%;height:auto;border-radius:10px;display:block;margin:16px 0 32px;background:#000;"></video>
        <p>Benjy&rsquo;s orchestration of caging and playing counterflash, calling RieNs to hold A link, calling the peek off of the ult, and then RieNs taking over the calling for the 2v1 is amazing.</p>
      </div>

      <div class="why-block">
        <span class="why-label lose">Why they won&rsquo;t win</span>
        <p>This is much easier to argue. Firstly, Neon getting nerfed affects this team more than, or as much as, any other team. I am not joking when I say that <strong>Heretics have not played a single map in Stage 1 without Neon in their comp</strong>.</p>
        <p>Secondly, they shouldn&rsquo;t have even qualified to London. What if RieNs doesn&rsquo;t win the 1v2 vs. Fnatic on Ascent? Heretics lose and go out 1&ndash;2 in playoffs. What if one of the two close 13&ndash;10 map wins against Eternal Fire turns into a loss? They lose that Bo3 and don&rsquo;t qualify to London. Even in the EMEA Stage 1 grand finals, they lost more rounds than they won overall.</p>
        <p>Heretics are extremely lucky to be a 1-seed let alone qualified. More so, all of this mediocrity and luckiness is occurring in the second-worst region currently. That doesn&rsquo;t count for much. Combine the Neon nerfs, and you have one of the worst non-CN one seeds I&rsquo;ve ever seen in VCT.</p>
      </div>

      <div class="why-block">
        <span class="why-label watch">Something to watch for</span>
        <p>What will Koshmaras play? The last time he played someone other than Neon, it was Raze, and that was over two months ago. It wouldn&rsquo;t surprise me if Team Heretics stuck to their guns with Neon despite the nerfs. At some point, it&rsquo;s less of a question of what you should play and what you <em>can</em> play.</p>
      </div>

      <div class="team-heading">
        <img src="/logos/FS.png" alt="Full Sense" onerror="this.style.display='none'">
        <h2 id="team-fs">Full Sense</h2>
        <span class="tier-pill b">B Tier</span>
      </div>
      <div class="team-stat-card">
        <div class="team-stat-row">
          <div class="team-stat-block">
            <div class="team-stat-label">Pacific Stage 1 Pyth</div>
            <div class="team-stat-value big">63.5%</div>
          </div>
          <div class="team-stat-block">
            <div class="team-stat-label">BenPom (Current)</div>
            <div class="team-stat-value big pos">+2.47</div>
          </div>
          <div class="team-stat-block">
            <div class="team-stat-label">Pacific Stage 1 Finish</div>
            <div class="team-stat-value big">2nd</div>
          </div>
        </div>
        <div class="roster-row">
          <div class="roster-row-label">Roster</div>
          <div class="roster-headshots">
            <div class="roster-player"><img class="roster-headshot" src="https://owcdn.net/img/6944049622e2a.png" alt="Crws" onerror="this.style.visibility=&quot;hidden&quot;"><div class="roster-player-name">Crws</div></div>
            <div class="roster-player"><img class="roster-headshot" src="https://owcdn.net/img/694404f04f64d.png" alt="JitBoyS" onerror="this.style.visibility=&quot;hidden&quot;"><div class="roster-player-name">JitBoyS</div></div>
            <div class="roster-player"><img class="roster-headshot" src="https://owcdn.net/img/6944048d3e33c.png" alt="Killua" onerror="this.style.visibility=&quot;hidden&quot;"><div class="roster-player-name">Killua</div></div>
            <div class="roster-player"><div class="roster-headshot"></div><div class="roster-player-name">Leviathan</div></div>
            <div class="roster-player"><img class="roster-headshot" src="https://owcdn.net/img/6944047b31bc9.png" alt="primmie" onerror="this.style.visibility=&quot;hidden&quot;"><div class="roster-player-name">primmie</div></div>
          </div>
        </div>
      </div>
      <div class="why-block">
        <span class="why-label win">Why they can win</span>
        <p>Their argument is as follows:</p>
        <ul>
          <li>They have the best player in the tournament in Primmie. Switching from a Neon meta to a Jett meta rewards star players with great aim, reducing chaos and prioritizing aim duels.</li>
          <li>For all of the talk about how great the Pacific region is currently, it&rsquo;s important to remember that &mdash; up until the Playoffs Grand Final &mdash; Full Sense looked like they might be the best APAC team. In Playoffs, they beat DRX (#12 in the world) 2&ndash;0, T1 (#3 in the world) 2&ndash;0, and GE (#13 in the world) 2&ndash;0, not dropping a single map until the Grand Final.</li>
        </ul>
      </div>

      <div class="why-block">
        <span class="why-label lose">Why they won&rsquo;t win</span>
        <p>They&rsquo;re an easy team to counter-strat. Just as much as Primmie is the reason for this team&rsquo;s success, his level of importance can be a reason for concern. For instance, in the two matches prior to the Pacific Grand Final (when they were on a tear), Primmie took 41% of his team&rsquo;s first interactions against GE and 46.5% against T1. These numbers are insanely high. Then, in the Grand Final, PRX intentionally avoided him unless there were team fights. There, Primmie only took 31% of his team&rsquo;s first interactions as Full Sense lost 3&ndash;0. Importantly, Primmie was still winning his first interactions at a good rate (12/9 in FK/FD), but Paper Rex avoided giving him a high volume of these first duels.</p>
        <p>This strategy has proven consistent.</p>
        <div class="primmie-chart">
          <div class="primmie-title">Full Sense match win rate by Primmie&rsquo;s share of team first interactions &mdash; 2026 Pacific Stage 1</div>
          <div class="primmie-grid">
            <div class="primmie-col high">
              <div class="primmie-pct">100%</div>
              <div class="primmie-label">when Primmie has <strong>&ge;40%</strong> of FS first interactions</div>
              <div class="primmie-record">3W &middot; 0L</div>
            </div>
            <div class="primmie-divider">vs.</div>
            <div class="primmie-col low">
              <div class="primmie-pct">50%</div>
              <div class="primmie-label">when Primmie has <strong>&lt;40%</strong> of FS first interactions</div>
              <div class="primmie-record">3W &middot; 3L</div>
            </div>
          </div>
        </div>
      </div>

      <div class="why-block">
        <span class="why-label watch">Something to watch for</span>
        <p>Just watch Primmie. This team is much more than Primmie (I specifically rate Killua&rsquo;s aim and CRWS&rsquo;s IGLing highly), but Primmie is, at the end of the day, this team&rsquo;s human highlight reel.</p>
      </div>

      <div class="team-heading">
        <img src="/logos/LEV.png" alt="Leviat&aacute;n" onerror="this.style.display='none'">
        <h2 id="team-lev">Leviat&aacute;n</h2>
        <span class="tier-pill b">B Tier</span>
      </div>
      <div class="team-stat-card">
        <div class="team-stat-row">
          <div class="team-stat-block">
            <div class="team-stat-label">Americas Stage 1 Pyth</div>
            <div class="team-stat-value big">66.0%</div>
          </div>
          <div class="team-stat-block">
            <div class="team-stat-label">BenPom (Current)</div>
            <div class="team-stat-value big pos">+2.32</div>
          </div>
          <div class="team-stat-block">
            <div class="team-stat-label">Americas Stage 1 Finish</div>
            <div class="team-stat-value big">2nd</div>
          </div>
        </div>
        <div class="roster-row">
          <div class="roster-row-label">Roster</div>
          <div class="roster-headshots">
            <div class="roster-player"><img class="roster-headshot" src="https://owcdn.net/img/69234b3d0a58d.png" alt="Neon" onerror="this.style.visibility=&quot;hidden&quot;"><div class="roster-player-name">Neon</div></div>
            <div class="roster-player"><img class="roster-headshot" src="https://owcdn.net/img/69234b72ce3db.png" alt="Sato" onerror="this.style.visibility=&quot;hidden&quot;"><div class="roster-player-name">Sato</div></div>
            <div class="roster-player"><img class="roster-headshot" src="https://owcdn.net/img/69234aef86f4b.png" alt="blowz" onerror="this.style.visibility=&quot;hidden&quot;"><div class="roster-player-name">blowz</div></div>
            <div class="roster-player"><img class="roster-headshot" src="https://owcdn.net/img/69234b213a609.png" alt="kiNgg" onerror="this.style.visibility=&quot;hidden&quot;"><div class="roster-player-name">kiNgg</div></div>
            <div class="roster-player"><img class="roster-headshot" src="https://owcdn.net/img/69234b7d30781.png" alt="spike" onerror="this.style.visibility=&quot;hidden&quot;"><div class="roster-player-name">spike</div></div>
          </div>
        </div>
      </div>
      <div class="why-block">
        <span class="why-label win">Why they can win</span>
        <p>Leviat&aacute;n are one of the more interesting cases. If you want to believe they can win, the only thing you should look at is their recent match results in Stage 1 Playoffs. They were up 10&ndash;8 in map 5 against G2 in the Americas Grand Final. If they closed that map out, they&rsquo;d come into London as the 1-seed from the second-best region after having beaten G2 twice in a row. However, they didn&rsquo;t close out that map, and that&rsquo;s important to acknowledge. The point here is that, whether it be because of inexperience, nerves, or a shallow map pool, Leviat&aacute;n were just barely unable to come into London with such a resume. The good news is how close they were. If they can shore up whatever shortcomings they suffered from, they can be right there with the best teams at London. They&rsquo;ve demonstrated that potential.</p>
        <p>Leviat&aacute;n, other than that Grand Final, went unbeaten in Stage 1&rsquo;s Playoffs after stumbling at the end of Group Stage. Granted these matches were close, but they&rsquo;ve shown that they can lock in for high-stakes scenarios. Also, since Ascent was added to the map pool, they haven&rsquo;t lost it once.</p>
        <p>Stage 1 Playoffs, from a macro perspective, offers reason for optimism in Leviat&aacute;n.</p>
      </div>

      <div class="why-block">
        <span class="why-label lose">Why they won&rsquo;t win</span>
        <p>However, the more you look into the micro, the more Leviat&aacute;n&rsquo;s case falls apart. Just like Team Heretics, Leviat&aacute;n has <strong>played Neon in every single map of Stage 1</strong>. Spikezin has, quite literally, not played a different agent since he played Viper in February (where he dropped a 0.63 rating). Adding Neon (the player) helped Leviat&aacute;n a lot, but adding Neon (the agent) also helped them immensely. Leviat&aacute;n will either look worse because the agent their comps are centered around is worse, or they&rsquo;ll look worse because they&rsquo;re switching to new comps.</p>
        <p>Furthermore, Leviat&aacute;n is heavily centered around set plays. Why do you think G2 was able to beat them in a Bo5 (with map-ban disadvantage) after losing 2&ndash;1 in their previous meeting? Because they&rsquo;re easy to counter-strat and adapt to.</p>
        <p>The more you dissect Leviat&aacute;n, the worse they look. It doesn&rsquo;t help that they went 0&ndash;5 in maps for EWC qualifiers after Stage 1 Playoffs.</p>
      </div>

      <div class="why-block">
        <span class="why-label watch">Something to watch for</span>
        <p>If Spikezin falls back to other roles, Sato will likely have to shoulder even more responsibility on the duelist role. Spikezin has often been the player to initiate site hits and first contacts, while Sato and Neon come in after him. With Spikezin potentially being on Deadlock or Viper, Sato will be the key to any success Leviat&aacute;n has at London. Sato has been touted as a rising star for about a year now, and this would be his opportunity to prove it, with London being his first international and a potentially increased burden on duelist.</p>
      </div>

      <h2 id="c-tier" class="tier-section-header c">C Tier</h2>

      <div class="team-heading">
        <img src="/logos/GE.png" alt="Global Esports" onerror="this.style.display='none'">
        <h2 id="team-ge">Global Esports</h2>
        <span class="tier-pill c">C Tier</span>
      </div>
      <div class="team-stat-card">
        <div class="team-stat-row">
          <div class="team-stat-block">
            <div class="team-stat-label">Pacific Stage 1 Pyth</div>
            <div class="team-stat-value big">46.5%</div>
          </div>
          <div class="team-stat-block">
            <div class="team-stat-label">BenPom (Current)</div>
            <div class="team-stat-value big pos">+1.64</div>
          </div>
          <div class="team-stat-block">
            <div class="team-stat-label">Pacific Stage 1 Finish</div>
            <div class="team-stat-value big">3rd</div>
          </div>
        </div>
        <div class="roster-row">
          <div class="roster-row-label">Roster</div>
          <div class="roster-headshots">
            <div class="roster-player"><img class="roster-headshot" src="https://owcdn.net/img/6780cc5e726c5.png" alt="Autumn" onerror="this.style.visibility=&quot;hidden&quot;"><div class="roster-player-name">Autumn</div></div>
            <div class="roster-player"><img class="roster-headshot" src="https://owcdn.net/img/677901eba6d1f.png" alt="Kr1stal" onerror="this.style.visibility=&quot;hidden&quot;"><div class="roster-player-name">Kr1stal</div></div>
            <div class="roster-player"><img class="roster-headshot" src="https://owcdn.net/img/67c6d1461c5cf.png" alt="PatMen" onerror="this.style.visibility=&quot;hidden&quot;"><div class="roster-player-name">PatMen</div></div>
            <div class="roster-player"><img class="roster-headshot" src="https://owcdn.net/img/677901e39cef9.png" alt="UdoTan" onerror="this.style.visibility=&quot;hidden&quot;"><div class="roster-player-name">UdoTan</div></div>
            <div class="roster-player"><img class="roster-headshot" src="https://owcdn.net/img/6846808f529c9.png" alt="xavi8k" onerror="this.style.visibility=&quot;hidden&quot;"><div class="roster-player-name">xavi8k</div></div>
          </div>
        </div>
      </div>
      <div class="why-block">
        <span class="why-label win">Why they can win</span>
        <p>Of all that there is to say about Global Esports, the most convincing argument is that they managed to beat Paper Rex 2&ndash;1 in Pacific Playoffs, which is about the best win you could tally before Masters London. Though they lost to Paper Rex 0&ndash;3 in their later rematch, that&rsquo;s partially driven by the fact that GE (in my opinion) messed up the pick/ban. They didn&rsquo;t pick Breeze when it was offered to them despite beating Paper Rex on Breeze the last time they played and being unbeaten on Breeze in Stage 1 at the time. Also, they&rsquo;re an APAC team in 2026, so they&rsquo;re automatically afforded some level of legitimacy.</p>
        <p>If they can beat the best team at Masters London, surely they can win the event? Right?</p>
      </div>

      <div class="why-block">
        <span class="why-label lose">Why they won&rsquo;t win</span>
        <p>Ever since beating Team Secret at the beginning of Stage 1, every match since then has been:</p>
        <ul>
          <li>A loss (4)</li>
          <li>A 2&ndash;1 win (2)</li>
          <li>A 2&ndash;0 win with a round differential of 5 or less (1)</li>
        </ul>
        <p>The fact that they managed to get here involves luck heavily. Their wins are (clearly) not convincing.</p>
        <p>Though they beat Paper Rex, they were only able to do so because PatMen could immediately read his former team.</p>
        <video src="/static/ge_patmen_clip.mp4" controls playsinline style="width:100%;height:auto;border-radius:10px;display:block;margin:16px 0 32px;background:#000;"></video>
      </div>

      <div class="why-block">
        <span class="why-label watch">Something to watch for</span>
        <p>The last time Autumn was at an international was at Champions Seoul, where he was FPX&rsquo;s best player. I&rsquo;d like to see if he can step up again for GE.</p>
      </div>

      <div class="team-heading">
        <img src="/logos/XLG.png" alt="XLG" onerror="this.style.display='none'">
        <h2 id="team-xlg">XLG</h2>
        <span class="tier-pill c">C Tier</span>
      </div>
      <div class="team-stat-card">
        <div class="team-stat-row">
          <div class="team-stat-block">
            <div class="team-stat-label">China Stage 1 Pyth</div>
            <div class="team-stat-value big">69.0%</div>
          </div>
          <div class="team-stat-block">
            <div class="team-stat-label">BenPom (Current)</div>
            <div class="team-stat-value big pos">+0.58</div>
          </div>
          <div class="team-stat-block">
            <div class="team-stat-label">China Stage 1 Finish</div>
            <div class="team-stat-value big">2nd</div>
          </div>
        </div>
        <div class="roster-row">
          <div class="roster-row-label">Roster</div>
          <div class="roster-headshots">
            <div class="roster-player"><img class="roster-headshot" src="https://owcdn.net/img/677d1df2d375a.png" alt="Lysoar" onerror="this.style.visibility=&quot;hidden&quot;"><div class="roster-player-name">Lysoar</div></div>
            <div class="roster-player"><div class="roster-headshot"></div><div class="roster-player-name">NoMan</div></div>
            <div class="roster-player"><img class="roster-headshot" src="https://owcdn.net/img/6782528608fd1.png" alt="Rarga" onerror="this.style.visibility=&quot;hidden&quot;"><div class="roster-player-name">Rarga</div></div>
            <div class="roster-player"><img class="roster-headshot" src="https://owcdn.net/img/6585b7ad2cb28.png" alt="WsLeo" onerror="this.style.visibility=&quot;hidden&quot;"><div class="roster-player-name">WsLeo</div></div>
            <div class="roster-player"><img class="roster-headshot" src="https://owcdn.net/img/6782528e85d4c.png" alt="happywei" onerror="this.style.visibility=&quot;hidden&quot;"><div class="roster-player-name">happywei</div></div>
          </div>
        </div>
      </div>
      <div class="why-block">
        <span class="why-label win">Why they can win</span>
        <p>Between their player quality (i.e. Happywei and Rarga) and their domestic results (e.g. losing 3&ndash;2 in the CN final against EDG), this team is firmly the second-best team in China. If EDG show up despite China&rsquo;s previous performance at Santiago, there are no excuses for XLG not to, at least, qualify for playoffs. As I said previously, I firmly believe EDG have a high chance of winning this tournament, so XLG should follow closely behind them.</p>
      </div>

      <div class="why-block">
        <span class="why-label lose">Why they won&rsquo;t win</span>
        <p>However, if China does make a resurgence, historical precedent will not be XLG&rsquo;s friend. At Masters Santiago they went out 0&ndash;2. At Champions Paris, they went out 1&ndash;2. At Masters Toronto they went out 0&ndash;2. Despite EDG and XLG&rsquo;s (relatively) close matches, the difference in ranking is based on how they historically perform at internationals. It will be hard for China&rsquo;s current status and XLG&rsquo;s historical record at internationals to both flip on their heads. It will be hard for XLG to win.</p>
      </div>

      <div class="why-block">
        <span class="why-label watch">Something to watch for</span>
        <p>Both Rarga and Happywei are capable of putting up a memorable performance at Masters London. By pure firepower, there aren&rsquo;t many that come close to them in VCT CN. Hopefully, one of them does it.</p>
      </div>

      <div class="team-heading">
        <img src="/logos/VIT.png" alt="Team Vitality" onerror="this.style.display='none'">
        <h2 id="team-vit">Team Vitality</h2>
        <span class="tier-pill c">C Tier</span>
      </div>
      <div class="team-stat-card">
        <div class="team-stat-row">
          <div class="team-stat-block">
            <div class="team-stat-label">EMEA Stage 1 Pyth</div>
            <div class="team-stat-value big">61.3%</div>
          </div>
          <div class="team-stat-block">
            <div class="team-stat-label">BenPom (Current)</div>
            <div class="team-stat-value big pos">+1.87</div>
          </div>
          <div class="team-stat-block">
            <div class="team-stat-label">EMEA Stage 1 Finish</div>
            <div class="team-stat-value big">2nd</div>
          </div>
        </div>
        <div class="roster-row">
          <div class="roster-row-label">Roster</div>
          <div class="roster-headshots">
            <div class="roster-player"><img class="roster-headshot" src="https://owcdn.net/img/6977a6d8e354a.png" alt="Chronicle" onerror="this.style.visibility=&quot;hidden&quot;"><div class="roster-player-name">Chronicle</div></div>
            <div class="roster-player"><img class="roster-headshot" src="https://owcdn.net/img/6977a70c4ff1b.png" alt="Derke" onerror="this.style.visibility=&quot;hidden&quot;"><div class="roster-player-name">Derke</div></div>
            <div class="roster-player"><img class="roster-headshot" src="https://owcdn.net/img/6977a6f130128.png" alt="Jamppi" onerror="this.style.visibility=&quot;hidden&quot;"><div class="roster-player-name">Jamppi</div></div>
            <div class="roster-player"><img class="roster-headshot" src="https://owcdn.net/img/6977a6e4ea727.png" alt="PROFEK" onerror="this.style.visibility=&quot;hidden&quot;"><div class="roster-player-name">PROFEK</div></div>
            <div class="roster-player"><div class="roster-headshot"></div><div class="roster-player-name">Sayonara</div></div>
          </div>
        </div>
      </div>
      <div class="why-block">
        <span class="why-label win">Why they can win</span>
        <p>Names. If you were to ask 1,000 VCT fans which 5 players they&rsquo;d want on their team for an international run, Chronicle and Derke would come up for probably 200&ndash;300 times. Granted, these 200&ndash;300 fans would be wrong, but the point is that these players exist in a level of mythology where it&rsquo;s hard <em>not</em> to believe that they have some real chance of winning.</p>
        <p>The last time Derke was at an international, he was the best player on Vitality and the fourth-highest rated player at Bangkok. The event before that, Derke was the best player on Fnatic and the second-highest rated player at Champions Seoul.</p>
        <p>Chronicle has three trophies.</p>
        <p>On top of all of this, Sayonara looks like the best player on Vitality.</p>
        <p>You can have your reservations about Jamppi and PROFEK (though I think Jamppi has been playing and calling very well), but this <em>roster</em> seems like an international-winning roster.</p>
      </div>

      <div class="why-block">
        <span class="why-label lose">Why they won&rsquo;t win</span>
        <p>Don&rsquo;t look at the roster and look at their results. In Group Stage, they went 2&ndash;3 (only beating PCIFIC and GX). In Playoffs, when every match was do-or-die, they barely won against Liquid, beat Fnatic with a sub, and then SHOULD&rsquo;VE lost to FUT (they lost more rounds than they won in the 2&ndash;1 win, which is hard to do). After all of this, they lost to Heretics despite their pick/ban advantage.</p>
        <p>This team gets criminally overrated because of the players. Based on results, they shouldn&rsquo;t be at this tournament, let alone in discussions of winning London.</p>
      </div>

      <div class="why-block">
        <span class="why-label watch">Something to watch for</span>
        <p>Derke vocally hates playing Neon.</p>
        <div class="tweet-embeds">
          <blockquote class="twitter-tweet"><a href="https://twitter.com/Derke/status/2060011567459029117">View tweet</a></blockquote>
          <blockquote class="twitter-tweet"><a href="https://twitter.com/Derke/status/2055609143243776245">View tweet</a></blockquote>
        </div>
        <p>If you want to see a happy camper, watch Derke at this event. Maybe this is another reason for optimism in Vitality.</p>
      </div>

      <h2 id="d-tier" class="tier-section-header d">D Tier</h2>

      <div class="team-heading">
        <img src="/logos/FUT.png" alt="FUT Esports" onerror="this.style.display='none'">
        <h2 id="team-fut">FUT Esports</h2>
        <span class="tier-pill d">D Tier</span>
      </div>
      <div class="team-stat-card">
        <div class="team-stat-row">
          <div class="team-stat-block">
            <div class="team-stat-label">EMEA Stage 1 Pyth</div>
            <div class="team-stat-value big">46.9%</div>
          </div>
          <div class="team-stat-block">
            <div class="team-stat-label">BenPom (Current)</div>
            <div class="team-stat-value big pos">+0.61</div>
          </div>
          <div class="team-stat-block">
            <div class="team-stat-label">EMEA Stage 1 Finish</div>
            <div class="team-stat-value big">3rd</div>
          </div>
        </div>
        <div class="roster-row">
          <div class="roster-row-label">Roster</div>
          <div class="roster-headshots">
            <div class="roster-player"><img class="roster-headshot" src="https://owcdn.net/img/697aba0af3cd9.png" alt="KROSTALY" onerror="this.style.visibility=&quot;hidden&quot;"><div class="roster-player-name">KROSTALY</div></div>
            <div class="roster-player"><div class="roster-headshot"></div><div class="roster-player-name">s0pp</div></div>
            <div class="roster-player"><img class="roster-headshot" src="https://owcdn.net/img/680a926893d7b.png" alt="sociablEE" onerror="this.style.visibility=&quot;hidden&quot;"><div class="roster-player-name">sociablEE</div></div>
            <div class="roster-player"><img class="roster-headshot" src="https://owcdn.net/img/697ab97a4855f.png" alt="xeus" onerror="this.style.visibility=&quot;hidden&quot;"><div class="roster-player-name">xeus</div></div>
            <div class="roster-player"><img class="roster-headshot" src="https://owcdn.net/img/697ab948ca75e.png" alt="yetujey" onerror="this.style.visibility=&quot;hidden&quot;"><div class="roster-player-name">yetujey</div></div>
          </div>
        </div>
      </div>
      <div class="why-block">
        <span class="why-label win">Why they can win</span>
        <p>They have TMV on their side, who (based on Wolves) should guarantee them a top-3 finish. This is genuinely the only argument I can think of.</p>
      </div>

      <div class="why-block">
        <span class="why-label lose">Why they won&rsquo;t win</span>
        <p>They&rsquo;ve lost 3 of their past 4 matches, they come from the second-worst region, their best map in Stage 1 (Bind) isn&rsquo;t in the map pool anymore, and their best player (s0pp) has only played Neon in Stage 1, who&rsquo;s getting nerfed into the ground. Again, this team is hard to be optimistic about.</p>
      </div>

      <div class="why-block">
        <span class="why-label watch">Something to watch for</span>
        <p>TMV&rsquo;s costreams. That&rsquo;s about the best FUT can bring to Masters London.</p>
      </div>

      <div class="team-heading">
        <img src="/logos/DRG.png" alt="Dragon Ranger Gaming" onerror="this.style.display='none'">
        <h2 id="team-drg">Dragon Ranger Gaming</h2>
        <span class="tier-pill d">D Tier</span>
      </div>
      <div class="team-stat-card">
        <div class="team-stat-row">
          <div class="team-stat-block">
            <div class="team-stat-label">China Stage 1 Pyth</div>
            <div class="team-stat-value big">51.5%</div>
          </div>
          <div class="team-stat-block">
            <div class="team-stat-label">BenPom (Current)</div>
            <div class="team-stat-value big neg">−1.27</div>
          </div>
          <div class="team-stat-block">
            <div class="team-stat-label">China Stage 1 Finish</div>
            <div class="team-stat-value big">3rd</div>
          </div>
        </div>
        <div class="roster-row">
          <div class="roster-row-label">Roster</div>
          <div class="roster-headshots">
            <div class="roster-player"><img class="roster-headshot" src="https://owcdn.net/img/678260fd260da.png" alt="Flex1n" onerror="this.style.visibility=&quot;hidden&quot;"><div class="roster-player-name">Flex1n</div></div>
            <div class="roster-player"><img class="roster-headshot" src="https://owcdn.net/img/6780cc893eb8f.png" alt="Life" onerror="this.style.visibility=&quot;hidden&quot;"><div class="roster-player-name">Life</div></div>
            <div class="roster-player"><img class="roster-headshot" src="https://owcdn.net/img/6782611e6a470.png" alt="Nicc" onerror="this.style.visibility=&quot;hidden&quot;"><div class="roster-player-name">Nicc</div></div>
            <div class="roster-player"><img class="roster-headshot" src="https://owcdn.net/img/67826152e8ca0.png" alt="SpiritZ1" onerror="this.style.visibility=&quot;hidden&quot;"><div class="roster-player-name">SpiritZ1</div></div>
            <div class="roster-player"><img class="roster-headshot" src="https://owcdn.net/img/67826114df9ea.png" alt="vo0kashu" onerror="this.style.visibility='hidden'"><div class="roster-player-name">vo0kashu</div></div>
          </div>
        </div>
      </div>
      <div class="why-block">
        <span class="why-label win">Why they can win</span>
        <p>vo0kashu.</p>
      </div>

      <div class="why-block">
        <span class="why-label lose">Why they won&rsquo;t win</span>
        <p>The players on DRG not named vo0kashu.</p>
      </div>

      <div class="why-block">
        <span class="why-label watch">Something to watch for</span>
        <p>You won&rsquo;t believe this&hellip; vo0kashu.</p>
      </div>

      <div class="section-bubble-wrap"><span class="section-bubble" id="predictions"><span class="section-bubble-text">Predictions</span></span></div>

      <p>If PRX doesn&rsquo;t win this tournament, who will?</p>

      <p>G2 have shown their worst against top teams in Stage 1 Group Stage, and then managed to pull together a string of close wins to qualify for London through lowers. It&rsquo;ll be hard to recreate that level of luck (or &ldquo;locking in&rdquo; if you prefer to call it that) at Masters London. Again, believing in G2 feels more like doubting the other top teams. G2 won&rsquo;t put up a horrible performance, but I don&rsquo;t expect them to stomp either. NRG feel wishy-washy with their confusing comps. If I had a promise that NRG will put Keiko primarily on duelist, then I&rsquo;d pick them, but that doesn&rsquo;t seem to be happening. EDG have the aforementioned problem of simply being a Chinese team in 2026. It&rsquo;s hard to pick one after Masters Santiago.</p>

      <p>All of this is to say that I think NRG likely have the second-best chance of winning Masters London if Keiko plays duelist, but EDG are the most undervalued (i.e. the difference between their perceived odds and true odds).</p>

      <p>For the sake of being fun, I&rsquo;ll predict that, if it&rsquo;s not PRX, EDG win Masters London.</p>

      <p><strong>What about BenPom?</strong></p>

      <p>After 50,000 Monte Carlo simulations using current BenPom ratings:</p>

      <div class="data-table-wrap">
        <div class="data-table-label">BenPom Monte Carlo &mdash; 50,000 simulated tournaments</div>
        <table class="data-table">
          <thead><tr><th>#</th><th>Team</th><th class="num">Win London %</th><th class="num">Reach Playoffs via Swiss %</th></tr></thead>
          <tbody>
            <tr><td class="rank">1</td><td class="team"><div class="team-cell"><img class="team-logo" src="/logos/PRX.png" alt="PRX" onerror="this.style.display='none'">Paper Rex</div></td><td class="num">25.39%</td><td class="num">(bye)</td></tr>
            <tr><td class="rank">2</td><td class="team"><div class="team-cell"><img class="team-logo" src="/logos/G2.png" alt="G2" onerror="this.style.display='none'">G2</div></td><td class="num">20.54%</td><td class="num">(bye)</td></tr>
            <tr><td class="rank">3</td><td class="team"><div class="team-cell"><img class="team-logo" src="/logos/TH.png" alt="TH" onerror="this.style.display='none'">Team Heretics</div></td><td class="num">12.84%</td><td class="num">(bye)</td></tr>
            <tr><td class="rank">4</td><td class="team"><div class="team-cell"><img class="team-logo" src="/logos/NRG.png" alt="NRG" onerror="this.style.display='none'">NRG</div></td><td class="num">9.71%</td><td class="num">65.0%</td></tr>
            <tr><td class="rank">5</td><td class="team"><div class="team-cell"><img class="team-logo" src="/logos/FS.png" alt="FS" onerror="this.style.display='none'">Full Sense</div></td><td class="num">7.48%</td><td class="num">61.7%</td></tr>
            <tr><td class="rank">6</td><td class="team"><div class="team-cell"><img class="team-logo" src="/logos/LEV.png" alt="LEV" onerror="this.style.display='none'">Leviat&aacute;n</div></td><td class="num">6.97%</td><td class="num">59.6%</td></tr>
            <tr><td class="rank">7</td><td class="team"><div class="team-cell"><img class="team-logo" src="/logos/VIT.png" alt="VIT" onerror="this.style.display='none'">Team Vitality</div></td><td class="num">4.98%</td><td class="num">54.8%</td></tr>
            <tr><td class="rank">8</td><td class="team"><div class="team-cell"><img class="team-logo" src="/logos/EDG.png" alt="EDG" onerror="this.style.display='none'">EDward Gaming</div></td><td class="num">4.71%</td><td class="num">(bye)</td></tr>
            <tr><td class="rank">9</td><td class="team"><div class="team-cell"><img class="team-logo" src="/logos/GE.png" alt="GE" onerror="this.style.display='none'">Global Esports</div></td><td class="num">3.95%</td><td class="num">52.4%</td></tr>
            <tr><td class="rank">10</td><td class="team"><div class="team-cell"><img class="team-logo" src="/logos/FUT.png" alt="FUT" onerror="this.style.display='none'">FUT</div></td><td class="num">1.66%</td><td class="num">41.4%</td></tr>
            <tr><td class="rank">11</td><td class="team"><div class="team-cell"><img class="team-logo" src="/logos/XLG.png" alt="XLG" onerror="this.style.display='none'">XLG</div></td><td class="num">1.55%</td><td class="num">41.3%</td></tr>
            <tr><td class="rank">12</td><td class="team"><div class="team-cell"><img class="team-logo" src="/logos/DRG.png" alt="DRG" onerror="this.style.display='none'">Dragon Ranger Gaming</div></td><td class="num">0.21%</td><td class="num">23.8%</td></tr>
          </tbody>
        </table>
      </div>

      <br>

      <p>It&rsquo;s important to remember here that BenPom is operating off of Neon-meta data and international calibration we got from Masters Santiago (hence EDG&rsquo;s low, low odds). It&rsquo;s not necessarily wrong, I just write this to explain causation. Beyond China, BenPom has higher stock on Full Sense than I would&rsquo;ve guessed, given that they&rsquo;re starting in Swiss.</p>

      <div class="section-bubble-wrap"><span class="section-bubble" id="top-players"><span class="section-bubble-text">Top 15 Players</span></span></div>

      <p>To end off this preview, I&rsquo;ll quickly rank my top 15 players at Masters London.</p>

      <div class="player-rank-list">
        <div class="player-rank-card">
          <div class="player-rank-header">
            <div class="player-rank-num">1</div>
            <img class="player-rank-img" src="https://owcdn.net/img/6944047b31bc9.png" alt="Primmie" onerror="this.style.visibility='hidden'">
            <div class="player-rank-name">Primmie</div>
            <div class="player-rank-team"><img src="/logos/FS.png" alt="FS" onerror="this.style.display='none'">FS</div>
          </div>
          <p class="player-rank-desc">Not only did he end Pacific as the highest-rated player, but he did so on a team that went 3&ndash;2 in Group Stage and didn&rsquo;t win Playoffs. He&rsquo;s doing all of this while playing extremely selflessly (refer back to his first-interaction stats in the Full Sense preview) and often not being set up by his team to succeed, doing the heavy lifting alone.</p>
        </div>
        <div class="player-rank-card">
          <div class="player-rank-header">
            <div class="player-rank-num">2</div>
            <img class="player-rank-img" src="https://owcdn.net/img/697426e1f3308.png" alt="brawk" onerror="this.style.visibility='hidden'">
            <div class="player-rank-name">brawk</div>
            <div class="player-rank-team"><img src="/logos/NRG.png" alt="NRG" onerror="this.style.display='none'">NRG</div>
          </div>
          <p class="player-rank-desc">NRG&rsquo;s success runs entirely through brawk. In their run, it felt that almost every kill is done by brawk or because of his scan, which is proven by him having a 79% KAST (the highest in Americas). He was also the second-highest-rated player in Americas. Besides, I wouldn&rsquo;t bet against the Champions MVP to perform at an international.</p>
        </div>
        <div class="player-rank-card">
          <div class="player-rank-header">
            <div class="player-rank-num">3</div>
            <img class="player-rank-img" src="https://owcdn.net/img/69735f0889a6b.png" alt="Jinggg" onerror="this.style.visibility='hidden'">
            <div class="player-rank-name">Jinggg</div>
            <div class="player-rank-team"><img src="/logos/PRX.png" alt="PRX" onerror="this.style.display='none'">PRX</div>
          </div>
          <p class="player-rank-desc">He has been the main motor in this Paper Rex iteration, flexing across two or three roles in matches, stepping up in the biggest moments, and creating the space that players like f0rsakeN and something can steamroll off of. His impact is so insanely high for the best team in VCT, having the highest KAST% in Pacific.</p>
        </div>
        <div class="player-rank-card">
          <div class="player-rank-header">
            <div class="player-rank-num">4</div>
            <img class="player-rank-img" src="https://owcdn.net/img/677fe4289bb34.png" alt="CHICHOO" onerror="this.style.visibility='hidden'">
            <div class="player-rank-name">CHICHOO</div>
            <div class="player-rank-team"><img src="/logos/EDG.png" alt="EDG" onerror="this.style.display='none'">EDG</div>
          </div>
          <p class="player-rank-desc">With the highest K/D and second-highest KAST% in China, CHICHOO has been able to play his role both efficiently and selflessly &mdash; a hard duo to balance. It&rsquo;s been too long since we&rsquo;ve had CHICHOO play at an international with a good team. He&rsquo;ll remind people why he deserves to be in these discussions of top players in VCT.</p>
        </div>
        <div class="player-rank-card">
          <div class="player-rank-header">
            <div class="player-rank-num">5</div>
            <img class="player-rank-img" src="https://owcdn.net/img/69735f396861f.png" alt="something" onerror="this.style.visibility='hidden'">
            <div class="player-rank-name">something</div>
            <div class="player-rank-team"><img src="/logos/PRX.png" alt="PRX" onerror="this.style.display='none'">PRX</div>
          </div>
          <p class="player-rank-desc">something has been on a tear, being the second-highest rated player in the strongest region in the world. What&rsquo;s more, he&rsquo;s averaging a 1.4 K/D while also having 0.18 FKPR (the fifth-highest in Pacific). He&rsquo;s winning duels at a historically high rate. If it wasn&rsquo;t for him consistently dropping off when it matters most, I&rsquo;d put him higher.</p>
        </div>
        <div class="player-rank-card">
          <div class="player-rank-header">
            <div class="player-rank-num">6</div>
            <img class="player-rank-img" src="https://owcdn.net/img/697406c5ecbcd.png" alt="keiko" onerror="this.style.visibility='hidden'">
            <div class="player-rank-name">keiko</div>
            <div class="player-rank-team"><img src="/logos/NRG.png" alt="NRG" onerror="this.style.display='none'">NRG</div>
          </div>
          <p class="player-rank-desc">As discussed previously, since playing more duelist, he&rsquo;s looked better and better. His stats don&rsquo;t do him justice, as he has only shown his true potential toward the end of Stage 1. With himself being the highest-rated NRG player at Santiago, we have no reason not to expect him to put up a similar performance in London.</p>
        </div>
        <div class="player-rank-card">
          <div class="player-rank-header">
            <div class="player-rank-num">7</div>
            <img class="player-rank-img" src="https://owcdn.net/img/69735f135cf21.png" alt="f0rsakeN" onerror="this.style.visibility='hidden'">
            <div class="player-rank-name">f0rsakeN</div>
            <div class="player-rank-team"><img src="/logos/PRX.png" alt="PRX" onerror="this.style.display='none'">PRX</div>
          </div>
          <p class="player-rank-desc">While f0rsakeN has been quieter in Stage 1, it&rsquo;s partially because Jinggg and something have been killing all the players before he can see them. He continues to flex and support at a high level (76% KAST). Besides, why would you ever bet against f0rsakeN when the stakes matter most? PRX have played 3 grand finals this year, and f0rsakeN has been the highest-rated player in the server two times.</p>
        </div>
        <div class="player-rank-card">
          <div class="player-rank-header">
            <div class="player-rank-num">8</div>
            <img class="player-rank-img" src="https://owcdn.net/img/69778b71c9366.png" alt="RieNs" onerror="this.style.visibility='hidden'">
            <div class="player-rank-name">RieNs</div>
            <div class="player-rank-team"><img src="/logos/TH.png" alt="TH" onerror="this.style.display='none'">TH</div>
          </div>
          <p class="player-rank-desc">With Heretics qualifying to an event, you can&rsquo;t not include RieNs. As the highest-rated player from the EMEA 1-seed Team Heretics, he&rsquo;s notched the highest KAST% in EMEA in supporting his roster&rsquo;s deep run to London. His calling and utility show up in ways that stats can&rsquo;t fully detail.</p>
        </div>
        <div class="player-rank-card">
          <div class="player-rank-header">
            <div class="player-rank-num">9</div>
            <img class="player-rank-img" src="https://owcdn.net/img/69234b3d0a58d.png" alt="Neon" onerror="this.style.visibility='hidden'">
            <div class="player-rank-name">Neon</div>
            <div class="player-rank-team"><img src="/logos/LEV.png" alt="LEV" onerror="this.style.display='none'">LEV</div>
          </div>
          <p class="player-rank-desc">Neon has the highest rating and K/D across all regions in Stage 1. What&rsquo;s more, he&rsquo;s doing this while flexing across multiple roles (duelist/sentinel/controller). Why so low? A lot of his kills come at the end of the round when he&rsquo;s last alive, and it&rsquo;s dangerous to rank a rookie so highly before his first international. Still, one doesn&rsquo;t get these numbers based on a poor playstyle. He&rsquo;s insane, and could easily be the best player at the tournament.</p>
        </div>
        <div class="player-rank-card">
          <div class="player-rank-header">
            <div class="player-rank-num">10</div>
            <img class="player-rank-img" src="https://owcdn.net/img/677fe40edf9da.png" alt="ZmjjKK" onerror="this.style.visibility='hidden'">
            <div class="player-rank-name">ZmjjKK</div>
            <div class="player-rank-team"><img src="/logos/EDG.png" alt="EDG" onerror="this.style.display='none'">EDG</div>
          </div>
          <p class="player-rank-desc">He was the only player to show up for EDG at Masters Santiago. With his team&rsquo;s newfound level of success, he should be able to level up as well.</p>
        </div>
        <div class="player-rank-card">
          <div class="player-rank-header">
            <div class="player-rank-num">11</div>
            <img class="player-rank-img" src="https://owcdn.net/img/65e466071ca19.png" alt="trent" onerror="this.style.visibility='hidden'">
            <div class="player-rank-name">trent</div>
            <div class="player-rank-team"><img src="/logos/G2.png" alt="G2" onerror="this.style.display='none'">G2</div>
          </div>
          <p class="player-rank-desc">G2 continue their Americas dominance, and trent continues to play extremely well. Calm, clutch, and consistent &mdash; there&rsquo;s not much else to say.</p>
        </div>
        <div class="player-rank-card">
          <div class="player-rank-header">
            <div class="player-rank-num">12</div>
            <img class="player-rank-img" src="https://owcdn.net/img/69234b72ce3db.png" alt="Sato" onerror="this.style.visibility='hidden'">
            <div class="player-rank-name">Sato</div>
            <div class="player-rank-team"><img src="/logos/LEV.png" alt="LEV" onerror="this.style.display='none'">LEV</div>
          </div>
          <p class="player-rank-desc">The pressure will be on Sato with the recent Neon nerfs, and I expect him to rise. Sato showed us what he can do in 2025 as the primary duelist. While he&rsquo;s been quieter with Neon and Spikezin now on his team, he should step back into the spotlight at London. Based on the eye test alone, his fragging is as clean as any other player on this chart.</p>
        </div>
        <div class="player-rank-card">
          <div class="player-rank-header">
            <div class="player-rank-num">13</div>
            <img class="player-rank-img" src="https://owcdn.net/img/6977a7018811e.png" alt="Sayonara" onerror="this.style.visibility='hidden'">
            <div class="player-rank-name">Sayonara</div>
            <div class="player-rank-team"><img src="/logos/VIT.png" alt="VIT" onerror="this.style.display='none'">VIT</div>
          </div>
          <p class="player-rank-desc">Vitality&rsquo;s greatest-performing player, the rookie is the one carrying the veterans on Vitality. While flexing and gaining experience, he&rsquo;s been the perfect support fragger for Derke&rsquo;s entries. He&rsquo;s the highest-rated player from EMEA coming into the tournament, but his inexperience and EMEA&rsquo;s power level makes him hard to rate higher.</p>
        </div>
        <div class="player-rank-card">
          <div class="player-rank-header">
            <div class="player-rank-num">14</div>
            <img class="player-rank-img" src="https://owcdn.net/img/677fe435edc57.png" alt="Smoggy" onerror="this.style.visibility='hidden'">
            <div class="player-rank-name">Smoggy</div>
            <div class="player-rank-team"><img src="/logos/EDG.png" alt="EDG" onerror="this.style.display='none'">EDG</div>
          </div>
          <p class="player-rank-desc">Based on stats alone, Smoggy is the best player on the best team in China. Furthermore, he dropped a 1.41 rating in the Grand Final while playing 4 agents across 5 maps. His job on EDG is often just to shoot, and he does his job better than almost anyone else.</p>
        </div>
        <div class="player-rank-card">
          <div class="player-rank-header">
            <div class="player-rank-num">15</div>
            <img class="player-rank-img" src="https://owcdn.net/img/65e4660ee20e5.png" alt="valyn" onerror="this.style.visibility='hidden'">
            <div class="player-rank-name">valyn</div>
            <div class="player-rank-team"><img src="/logos/G2.png" alt="G2" onerror="this.style.display='none'">G2</div>
          </div>
          <p class="player-rank-desc">While we&rsquo;ve seen better forms from him, his calling and shooting have been at a level consistent enough to drive G2 to another domestic title. Unless you count f0rsakeN as an IGL, there&rsquo;s no other IGL playing like him right now, and there hasn&rsquo;t been one like him for the past two years.</p>
        </div>
      </div>

    </div>

    </div>
  </div>
</div>
<footer>
  Data sourced from VLR.gg
  <div style="margin-top:8px;">Like my work? Tips are appreciated! <a href="https://ko-fi.com/bobovct" target="_blank" rel="noopener" style="color:inherit;text-decoration:underline;">ko-fi.com/bobovct</a></div>
</footer>
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
<script async src="https://platform.twitter.com/widgets.js"></script>
</body>
</html>
"""


@article_masters_london_bp.route("/")
def index():
    return render_template_string(PAGE_HTML)
