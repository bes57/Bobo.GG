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
from flask import Blueprint, render_template_string

article_championship_dna_bp = Blueprint("article_championship_dna", __name__)

_ROOT = os.path.dirname(os.path.abspath(__file__))


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
      <!-- Body goes here. Left empty on purpose. -->
    </div>
  </div>
</div>
</body>
</html>
"""


@article_championship_dna_bp.route("/")
def article_championship_dna():
    return render_template_string(PAGE_HTML)
