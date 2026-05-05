import os
from flask import Flask, render_template_string, send_from_directory
from EventLeaderboards import vct_bp
from AllTimeHighs import highs_bp
from IdentifyingOverUnderPerformers import article_overunder_bp

app = Flask(__name__)
app.register_blueprint(vct_bp, url_prefix="/vct")
app.register_blueprint(highs_bp, url_prefix="/highs")
app.register_blueprint(article_overunder_bp, url_prefix="/articles/over-underperformers")

HOME_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Bobo's VCT Database</title>
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
  .page { position:relative; z-index:1; flex:1; display:flex; flex-direction:column; align-items:center; justify-content:center; padding:60px 32px; text-align:center; }
  h1 { font-family:'Syne',sans-serif; font-size:clamp(3rem,8vw,6rem); font-weight:700; letter-spacing:-2px; line-height:1; }
  .tagline { margin-top:16px; color:var(--soft); font-size:1rem; font-weight:300; line-height:1.6; white-space:nowrap; }
  .sections { display:flex; flex-direction:column; gap:40px; margin-top:52px; width:100%; max-width:900px; }
  .section-title { font-family:'Syne',sans-serif; font-size:1.5rem; font-weight:800; color:var(--ink); margin-bottom:20px; text-align:left; cursor:pointer; display:flex; align-items:center; gap:10px; user-select:none; letter-spacing:-0.5px; }
  .section-chevron { font-size:1rem; color:var(--soft); transition:transform .25s ease; display:inline-block; }
  .section.collapsed .section-chevron { transform:rotate(-90deg); }
  .cards-wrap { display:grid; grid-template-rows:1fr; transition:grid-template-rows .3s ease, opacity .3s ease; opacity:1; overflow:hidden; }
  .section.collapsed .cards-wrap { grid-template-rows:0fr; opacity:0; }
  .cards-inner { min-height:0; }
  .cards { display:flex; gap:20px; flex-wrap:wrap; justify-content:flex-start; padding:8px 0 8px; }
  .nav-card { background:white; border-radius:24px; padding:32px 24px; width:275px; text-decoration:none; color:var(--ink); box-shadow:0 4px 24px #0000000a; transition:transform .2s,box-shadow .2s; text-align:left; }
  .nav-card:hover { transform:translateY(-6px); box-shadow:0 16px 40px #00000014; }
  .nav-card-title { font-family:'Syne',sans-serif; font-size:1.05rem; font-weight:800; margin-bottom:6px; overflow-wrap:anywhere; }
  .nav-card-desc { font-size:.82rem; color:var(--soft); font-weight:300; line-height:1.5; }
  .nav-card-arrow { margin-top:18px; font-size:.8rem; color:#ccc; }
  footer { position:relative; z-index:1; text-align:center; padding:24px; color:var(--soft); font-size:.75rem; font-weight:300; }
  @keyframes fadeUp{from{opacity:0;transform:translateY(24px)}to{opacity:1;transform:translateY(0)}}
  .page { animation:fadeUp .6s ease both; }
</style>
</head>
<body>
<div class="page">
  <h1><img src="/logo.svg" alt="B" style="height:1.65em;width:auto;vertical-align:-0.2em;margin-right:-0.2em;object-fit:contain;cursor:pointer;" onclick="easterEgg()">obo's VCT Database</h1>
  <p class="tagline" id="tagline">Misceallneous analyses in the competitive Valorant space</p>
  <div class="sections">
    <div class="section">
      <div class="section-title">General Statistics and Databases<span class="section-chevron">▾</span></div>
      <div class="cards-wrap"><div class="cards-inner">
      <div class="cards">
        <a class="nav-card" href="/vct/">
          <div class="nav-card-title">Event Leaderboards</div>
          <div class="nav-card-desc">Sift through leaderboards by events, highlighting indivdual performances and percentiles.</div>
          <div class="nav-card-arrow">Explore &rarr;</div>
        </a>
        <a class="nav-card" href="/highs/">
          <div class="nav-card-title">All-Time Highs (and Lows)</div>
          <div class="nav-card-desc">The best and worst individual performances across all VCT franchised events.</div>
          <div class="nav-card-arrow">Explore &rarr;</div>
        </a>
      </div>
      </div></div>
    </div>
    <div class="section">
      <div class="section-title">Research / Opinion Articles <span class="section-chevron">▾</span></div>
      <div class="cards-wrap"><div class="cards-inner">
      <div class="cards">
        <a class="nav-card" href="/articles/over-underperformers/">
          <div class="nav-card-title">Overperforming in VCT: who's doing it?</div>
          <div class="nav-card-desc">Using VCT stats to surface players who are outperforming (or underperforming) their team.</div>
          <div class="nav-card-arrow">Read &rarr;</div>
        </a>
      </div>
      </div></div>
    </div>
  </div>
</div>
<footer>Data sourced from VLR.gg</footer>
<script>
function easterEgg() {
  document.getElementById('tagline').textContent = "Uxie is N0te's dada";
}
document.querySelectorAll('.section-title').forEach(function(title) {
  title.addEventListener('click', function() {
    this.closest('.section').classList.toggle('collapsed');
  });
});
</script>
</body>
</html>
"""

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory(STATIC_DIR, filename)

@app.route("/favicon.svg")
def favicon():
    return send_from_directory(STATIC_DIR, "BoboLogo-cropped.svg", mimetype="image/svg+xml")

@app.route("/logo.svg")
def logo():
    return send_from_directory(STATIC_DIR, "BoboLogo.svg", mimetype="image/svg+xml")

@app.route("/")
def home():
    return render_template_string(HOME_HTML)


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5001)
