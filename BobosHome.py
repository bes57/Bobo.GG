import os
import json as _json
import re as _re
import math as _math
import datetime as _datetime
from flask import Flask, render_template_string, send_from_directory
from flask_compress import Compress
from EventLeaderboards import vct_bp
from AllTimeHighs import highs_bp
from IdentifyingOverUnderPerformers import article_overunder_bp
from AmericasStage1Playoffs import article_americas_stage1_bp
from MastersLondonPreview import article_masters_london_bp
from MastersLondonPlayoffsPreview import article_masters_london_playoffs_bp
from AspasGreatestPrime import article_aspas_prime_bp
from MapElo import mapelo_bp
from InternationalEvents import intl_bp
from MatchDataExplorer import match_data_bp
from TestingLab import testing_bp

app = Flask(__name__)
Compress(app)
app.register_blueprint(vct_bp, url_prefix="/vct")
app.register_blueprint(highs_bp, url_prefix="/highs")
app.register_blueprint(article_overunder_bp, url_prefix="/articles/over-underperformers")
app.register_blueprint(article_americas_stage1_bp, url_prefix="/articles/americas-stage1-playoffs-preview")
app.register_blueprint(article_masters_london_bp, url_prefix="/articles/masters-london-preview")
app.register_blueprint(article_masters_london_playoffs_bp, url_prefix="/articles/masters-london-playoffs-preview")
app.register_blueprint(article_aspas_prime_bp, url_prefix="/articles/greatest-prime")
app.register_blueprint(mapelo_bp, url_prefix="/mapelo")
app.register_blueprint(intl_bp, url_prefix="/intl")
app.register_blueprint(match_data_bp, url_prefix="/match-data")
app.register_blueprint(testing_bp, url_prefix="/testing")

# ── Alpha UI data layer ──────────────────────────────────────────────────────
# The alpha dashboard reuses the Modern Hub's read-only data builder
# (_mhub_load: power ratings, recent matches with pre-match odds, event bands,
# upcoming) plus the Event Leaderboards' player data. No new scraping — it just
# reads the same files the rest of the site already produces.
_BASE = os.path.dirname(__file__)


def _parse_team_colors():
    """Single source of truth for team brand colors: the canonical
    one-color-per-team `var TEAM_COLORS` dict in MapElo.py (the user-curated
    palette). Parsed at import so the alpha UI never drifts from the rest of
    the site. Falls back to an empty dict (region colors used instead)."""
    try:
        src = open(os.path.join(_BASE, "MapElo.py"), encoding="utf-8").read()
        m = _re.search(r"\nvar TEAM_COLORS = \{(.*?)\n\};", src, _re.S)
        block = m.group(1) if m else ""
        return {k: v for k, v in _re.findall(
            r"['\"]?([^'\":\s]+)['\"]?\s*:\s*'(#[0-9A-Fa-f]{3,8})'", block)}
    except Exception:
        return {}


ALPHA_TEAM_COLORS = _parse_team_colors()
try:
    ALPHA_LOGOS = _json.load(open(os.path.join(_BASE, "static", "logos", "logos.json")))
except Exception:
    ALPHA_LOGOS = {}

# ── v6 site model (data/site_model.json) ─────────────────────────────────────
# Import-light mtime-cached loader — the single source of truth for every
# probability this file renders (β, cross-region offsets, region priors,
# gf_upper_logit). Reference math: trading_model/predict.py.
_SITE_MODEL_PATH = os.path.join(_BASE, "data", "site_model.json")
_site_model_state = {"m": None, "mtime": 0.0}


def _site_model():
    try:
        mt = os.path.getmtime(_SITE_MODEL_PATH)
    except OSError:
        mt = 0.0
    if _site_model_state["m"] is None or mt > _site_model_state["mtime"]:
        with open(_SITE_MODEL_PATH) as f:
            _site_model_state["m"] = _json.load(f)
        _site_model_state["mtime"] = mt
    return _site_model_state["m"]


def _v6_series_wp(model, r_a, r_b, reg_a, reg_b, fmt, upper_is_a=None):
    """predict.py series_probability on explicit ratings/regions — fallback
    only; the primary path consumes the hub payload's precomputed win_prob_a
    (MapElo._mhub_load, same snapshot)."""
    adj = 0.0
    if reg_a and reg_b and reg_a != reg_b:
        off = model.get("xregion_offsets") or {}
        adj = off.get(reg_a, 0.0) - off.get(reg_b, 0.0)
    p = 1.0 / (1.0 + _math.exp(-model["beta"] * (r_a - r_b + adj)))
    if fmt == "bo1":
        ps = p
    elif fmt in ("bo5", "bo5_gf"):
        q = 1.0 - p
        ps = p ** 3 * (1 + 3 * q + 6 * q * q)
    else:
        ps = p * p * (3 - 2 * p)
    if fmt == "bo5_gf" and upper_is_a is not None:
        delta = model["gf_upper_logit"] if upper_is_a else -model["gf_upper_logit"]
        ps = min(max(ps, 1e-9), 1 - 1e-9)
        ps = 1.0 / (1.0 + _math.exp(-(_math.log(ps / (1 - ps)) + delta)))
    return ps


def _alpha_days_between(a, b):
    try:
        return (_datetime.date.fromisoformat(b) - _datetime.date.fromisoformat(a)).days
    except Exception:
        return None


def _alpha_event_context(bands, today):
    bands = sorted([b for b in bands if b.get("start")], key=lambda b: b["start"])
    live = next((b for b in bands if b.get("start", "") <= today <= b.get("end", "")), None)
    nxt = next((b for b in bands if b.get("start", "") > today), None)
    past = [b for b in bands if b.get("end", "") < today]
    return live, nxt, (past[-1] if past else None)


def _alpha_data_version():
    """A cheap CONTENT signature of the data the home page depends on. Uses
    signals that only move when there's genuinely new data (latest ratings date,
    match/upcoming counts, newest match id) — NOT file mtimes, so a background
    re-scrape that finds nothing doesn't falsely flag an update. mhub is cached,
    so this is a few ms."""
    try:
        from MapElo import _mhub_load
        hub = _mhub_load()
        pm = hub.get("past_matches") or []
        up = hub.get("upcoming") or []
        return "|".join([str(hub.get("as_of_date")), str(len(pm)), str(len(up)),
                         str(pm[0].get("match_id")) if pm else ""])
    except Exception:
        return ""


def _build_alpha_data():
    """Assemble the compact payload the alpha dashboard renders from."""
    from MapElo import _mhub_load
    hub = _mhub_load()
    lb = hub.get("leaderboard") or {}
    # v6 site-model β (the hub payload carries it; fall back to reading
    # data/site_model.json directly).
    beta = lb.get("beta") or _site_model()["beta"]
    teams = lb.get("teams", [])
    today = _datetime.date.today().isoformat()

    # Power rankings — rating + an intuitive "expected map win vs an average
    # VCT team" percentage (v6 β sigmoid of the rating), region, and last-5
    # form.
    rankings = []
    for t in teams:
        rating = t.get("rating", 0.0)
        rankings.append({
            "rank": t.get("rank"),
            "org": t["org"],
            "region": t.get("region", ""),
            "rating": round(rating, 2),
            "w": t.get("w", 0), "l": t.get("l", 0),
            "winpct": round(100.0 / (1.0 + _math.exp(-beta * rating))),
            # Full last-5 match objects (newest first) so each dot can show the
            # same BenPom match-hover card the chart uses. Reversed + padded in UI.
            "form": (t.get("recent_matches") or [])[:5],
        })

    # Recent matches — already carry pre-match (morning-of) series odds + result.
    recent = []
    # Sort by full timestamp when known so same-day matches order by actual
    # kickoff time (not match_id, which doesn't reliably track start time).
    _past_sorted = sorted((hub.get("past_matches") or []),
                          key=lambda x: (x.get("time") or x.get("date") or "", x.get("match_id") or 0),
                          reverse=True)
    for m in _past_sorted[:140]:
        recent.append({k: m.get(k) for k in (
            "org_a", "org_b", "date", "time", "event", "format", "region",
            "rating_a", "rating_b",
            "win_prob_a", "win_prob_b", "actual_winner", "actual_score", "gf_upper")})

    # Upcoming matches — show everything scheduled within ~a month ahead
    # (soonest first, no count cap). Series win prob = the hub payload's
    # precomputed v6 closed form (win_prob_a from data/site_model.json —
    # snapshot β + cross-region offsets + gf_upper_logit; computed in
    # MapElo._mhub_load's upcoming loop). Cards link to the Modern Hub for
    # the veto/map breakdown.
    try:
        _horizon = (_datetime.date.fromisoformat(today) + _datetime.timedelta(days=31)).isoformat()
    except Exception:
        _horizon = "9999-12-31"
    try:
        from MapElo import ORG_REGIONS as _OREG
    except Exception:
        _OREG = {}
    upcoming = []
    for m in (hub.get("upcoming") or []):
        md = m.get("date") or ""
        if md and md > _horizon:        # beyond the one-month window
            continue
        ra, rb = m.get("rating_a"), m.get("rating_b")
        wp = m.get("win_prob_a")
        if wp is None and ra is not None and rb is not None:
            fmt = m.get("format") or "bo3"
            wp = _v6_series_wp(
                _site_model(), ra, rb,
                _OREG.get(m.get("org_a", "")), _OREG.get(m.get("org_b", "")),
                fmt,
                True if (fmt == "bo5_gf" and m.get("gf_upper")) else None)
        upcoming.append({
            "org_a": m.get("org_a"), "org_b": m.get("org_b"),
            "date": m.get("date"), "datetime": m.get("datetime"),
            "event": m.get("event"), "format": m.get("format"),
            "region": m.get("region"),
            "rating_a": ra, "rating_b": rb,
            "win_prob_a": round(wp, 3) if wp is not None else None,
            "match_name": m.get("match_name"),
        })
    upcoming.sort(key=lambda x: ((x.get("date") or "9999"), x.get("datetime") or ""))

    # International events (Masters/Champions) span every region, so their matches
    # shouldn't inherit a single team's region — tag them "International".
    try:
        from MoreTestingMaybeFiles import ALL_EVENTS as _AE2
        _intl_labels = {e.get("label", e["id"]) for e in _AE2
                        if "International" in (e.get("regions") or {})}
    except Exception:
        _intl_labels = set()
    for _m in recent:
        if _m.get("event") in _intl_labels:
            _m["region"] = "International"
    for _m in upcoming:
        if _m.get("event") in _intl_labels:
            _m["region"] = "International"

    live, nxt, last = _alpha_event_context(hub.get("event_bands") or [], today)
    event = None
    if live:
        event = {"status": "live", "label": live["label"],
                 "start": live["start"], "end": live["end"]}
    elif nxt:
        event = {"status": "upcoming", "label": nxt["label"], "start": nxt["start"],
                 "end": nxt["end"], "days": _alpha_days_between(today, nxt["start"])}
    elif last:
        event = {"status": "recent", "label": last["label"],
                 "start": last["start"], "end": last["end"]}
    last_event = {"label": last["label"], "end": last["end"]} if last else None
    next_event = ({"label": nxt["label"], "start": nxt["start"],
                   "days": _alpha_days_between(today, nxt["start"])} if nxt else None)

    # Player leaderboard — top by VLR rating at the most recent event with data.
    player_stats, players_event, players_event_id = [], None, None
    try:
        from EventLeaderboards import (load_event, _most_recent_event_with_data,
                                       get_all, _ensure_headshots_loaded,
                                       _live_split_has_data, LIVE_EVENT_ID)
        from MoreTestingMaybeFiles import ALL_EVENTS as _ALL_EVENTS
        _ensure_headshots_loaded()
        # Focus the leaders on the current/live split once it has scraped data;
        # otherwise fall back to the most recent completed event.
        ev = None
        if _live_split_has_data():
            ev = next((e for e in _ALL_EVENTS if e["id"] == LIVE_EVENT_ID), None)
        if ev is None:
            ev = _most_recent_event_with_data()
        cache = load_event(ev)

        def _rnd(p):
            try:
                return int(float(p.get("Rnd", 0)))
            except Exception:
                return 0
        # Top players across a few different stats (mini-leaderboard each).
        # Ship a deep sorted list (with round counts) instead of a pre-filtered
        # top 5, so the min-rounds slider can re-filter client-side instantly.
        for _col, _label in (("R2.0", "VLR Rating"), ("KAST", "KAST"),
                             ("HS%", "Headshot %"), ("FIWR", "First Duel Win %")):
            try:
                allp = get_all(cache, _col)
                leaders = [{"name": p.get("Player"), "org": p.get("Org", ""),
                            "region": p.get("Region", ""),
                            "headshot": p.get("HeadshotURL", ""),
                            "profile": p.get("ProfileURL", ""),
                            "value": p.get(_col),
                            "rnd": _rnd(p)} for p in allp[:60]]
                if leaders:
                    player_stats.append({"stat": _col, "label": _label, "leaders": leaders})
            except Exception:
                pass
        players_event = ev.get("label")
        players_event_id = ev.get("id")
    except Exception:
        pass

    # 2026 season timeline — every non-CN-only 2026 event, tagged done/live/next.
    season = []
    try:
        from MoreTestingMaybeFiles import ALL_EVENTS
        seen = set()
        for e in ALL_EVENTS:
            if e.get("year") != 2026 or list(e.get("regions", {}).keys()) == ["CN"]:
                continue
            lbl = (e.get("label", "") or "").replace("2026 ", "")
            if lbl in seen:
                continue
            seen.add(lbl)
            st, en = e.get("start", ""), e.get("end", "")
            status = "done" if en and en < today else (
                "live" if (st <= today <= en) else "upcoming")
            season.append({"label": lbl, "start": st, "end": en, "status": status,
                           "intl": list(e.get("regions", {}).keys()) == ["International"]})
        season.sort(key=lambda x: x["start"])
    except Exception:
        season = []

    # event_id → label, for the match-hover card event tag.
    event_labels = {}
    try:
        from MoreTestingMaybeFiles import ALL_EVENTS as _AE
        event_labels = {e["id"]: e.get("label", e["id"]) for e in _AE}
    except Exception:
        pass

    # Recent all-time records (performances from the latest event that cracked an
    # all-time top-50) — previewed beneath the player leaders.
    try:
        from AllTimeHighs import build_recent_records
        records = build_recent_records(8)
    except Exception:
        records = []

    orgs = ({r["org"] for r in rankings}
            | {x for m in recent for x in (m["org_a"], m["org_b"])}
            | {x for m in upcoming for x in (m["org_a"], m["org_b"])}
            | {l["org"] for s in player_stats for l in s["leaders"]}
            | {r.get("org") for r in records} | {r.get("opp") for r in records}
            | {x for r in rankings for m in r["form"] for x in (m.get("winner"), m.get("loser"))})
    colors = {o: ALPHA_TEAM_COLORS.get(o, "#8a8a8a") for o in orgs if o}
    logos = {o: ALPHA_LOGOS.get(o) for o in orgs if o and ALPHA_LOGOS.get(o)}

    return {
        "version": _alpha_data_version(),
        "as_of": hub.get("as_of_date"), "today": today, "beta": beta,
        "event": event, "last_event": last_event, "next_event": next_event,
        "season": season,
        "rankings": rankings, "recent": recent, "upcoming": upcoming,
        "player_stats": player_stats, "players_event": players_event,
        "players_event_id": players_event_id,
        "records": records,
        "event_labels": event_labels,
        "colors": colors, "logos": logos,
    }


def _build_team_profile(org):
    """Fast team-profile payload (no scrape) for /team/<org>: BenPom rating,
    global rank, record, recent matches (with maps), best/worst maps, roster."""
    from MapElo import _mhub_load
    hub = _mhub_load()
    lb = hub.get("leaderboard") or {}
    teams = lb.get("teams", [])
    t = next((x for x in teams if x.get("org") == org), None)
    if not t:
        return None
    try:
        from MoreTestingMaybeFiles import ALL_EVENTS as _AE
        elabels = {e["id"]: e.get("label", e["id"]) for e in _AE}
    except Exception:
        elabels = {}
    # Season rating trajectory for this org (one point per timeline checkpoint).
    traj = []
    for cp in (hub.get("chart") or {}).get("checkpoints", []):
        rr = (cp.get("ratings") or {}).get(org)
        if rr is not None:
            traj.append({"d": cp.get("date", ""), "r": round(float(rr), 2)})
    # Every match this org played (for the per-map game breakdown — same source the
    # Modern Hub's map drill-down uses). Keep just the fields the breakdown needs.
    org_events = []
    for me in (hub.get("chart") or {}).get("match_events", []):
        if me.get("winner") == org or me.get("loser") == org:
            org_events.append({k: me.get(k) for k in
                ("date", "event_id", "winner", "loser", "match_id", "maps")})
    opp_orgs = {(me["loser"] if me.get("winner") == org else me.get("winner"))
                for me in org_events}
    # Every event relevant to this team's region (its own region's events + ALL
    # internationals), with official start/end dates — so the season-trajectory
    # graph can mark each event's start AND end, INCLUDING internationals the team
    # didn't attend (e.g. a team that skipped Masters still gets the Masters band).
    team_region = t.get("region", "")
    season_events = []
    for e in _AE:
        regs = e.get("regions") or {}
        if ("International" in regs) or (team_region and team_region in regs):
            season_events.append({
                "id": e["id"], "label": e.get("label", e["id"]),
                "start": e.get("start", ""), "end": e.get("end", ""),
            })
    season_events.sort(key=lambda x: x.get("start", ""))
    # Upcoming matches for this org (compact left-rail list) — same Modern Hub
    # source the alpha dashboard uses. Projected series win % for THIS team =
    # the hub payload's precomputed v6 closed form (win_prob_a from
    # data/site_model.json, flipped when this org is slot B); local v6 closed
    # form only as a fallback.
    try:
        from MapElo import ORG_REGIONS as _OREG
    except Exception:
        _OREG = {}
    upcoming = []
    for m in (hub.get("upcoming") or []):
        a, b = m.get("org_a"), m.get("org_b")
        if org not in (a, b):
            continue
        is_a = (a == org)
        opp = b if is_a else a
        ra, rb = m.get("rating_a"), m.get("rating_b")
        wp = m.get("win_prob_a")
        if wp is not None:
            wp = wp if is_a else (1.0 - wp)
        elif ra is not None and rb is not None:
            rme, rop = (ra, rb) if is_a else (rb, ra)
            fmt = m.get("format") or "bo3"
            wp = _v6_series_wp(
                _site_model(), rme, rop,
                _OREG.get(org), _OREG.get(opp), fmt,
                # slot A is the upper-bracket team for bo5_gf
                is_a if (fmt == "bo5_gf" and m.get("gf_upper")) else None)
        upcoming.append({
            "opponent": opp, "date": m.get("date"), "time": m.get("datetime"),
            "event": m.get("event"), "region": m.get("region"),
            "format": m.get("format"),
            "win_prob": round(wp, 3) if wp is not None else None,
        })
    upcoming.sort(key=lambda x: ((x.get("date") or "9999"), x.get("time") or ""))
    # Colors/logos for this team + every opponent it has faced or will face
    orgs = ({org} | {m.get("opponent") for m in (t.get("recent_matches") or [])}
            | opp_orgs | {u["opponent"] for u in upcoming})
    return {
        "org": org, "region": t.get("region", ""),
        "rating": round(t.get("rating", 0.0), 2), "rank": t.get("rank"),
        "n_teams": len(teams), "w": t.get("w", 0), "l": t.get("l", 0),
        "season": (lb.get("as_of_date") or "")[:4],
        "beta": lb.get("beta") or _site_model()["beta"],
        "all_maps": t.get("all_maps") or [],
        "best_maps": (t.get("best_maps") or [])[:3],
        "worst_maps": (t.get("worst_maps") or [])[:3],
        "recent": (t.get("recent_matches") or [])[:4],
        "upcoming": upcoming,
        "form": (t.get("recent_matches") or [])[:5],
        "roster": (t.get("roster") or [])[:6],
        "traj": traj,
        "events": org_events,
        "season_events": season_events,
        "event_labels": elabels,
        "colors": {o: ALPHA_TEAM_COLORS.get(o, "#8a8a8a") for o in orgs if o},
        "logos": {o: ALPHA_LOGOS.get(o) for o in orgs if o and ALPHA_LOGOS.get(o)},
    }

HOME_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Bobo's VCT Database — Classic</title>
<link rel="icon" type="image/svg+xml" href="/favicon.svg">
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@700;800&family=DM+Sans:wght@300;400;500;700;800&display=swap" rel="stylesheet">
<link rel="stylesheet" href="/static/base.css">
<style>
  .page { position:relative; z-index:1; flex:1; display:flex; flex-direction:column; align-items:center; justify-content:center; padding:60px 32px; text-align:center; }
  h1 { font-family:'DM Sans',sans-serif; font-size:clamp(3rem,8vw,6rem); font-weight:400; letter-spacing:normal; line-height:1.15; padding-bottom:.12em; overflow:visible; }
  .nav-card-cover { width:calc(100% + 48px); margin:-32px -24px 20px; height:140px; object-fit:cover; object-position:center top; display:block; border-radius:24px 24px 0 0; }
  .tagline { margin-top:16px; color:#111; font-size:1rem; font-weight:300; line-height:1.6; white-space:nowrap; transition:opacity .28s ease, transform .28s ease; }
  .sections { display:flex; flex-direction:column; gap:40px; margin-top:20px; width:100%; max-width:900px; }
  .section-title { font-family:'Plus Jakarta Sans',sans-serif; font-size:1.75rem; font-weight:800; color:var(--ink); margin-bottom:28px; text-align:left; cursor:pointer; display:flex; align-items:center; gap:10px; user-select:none; letter-spacing:-0.5px; }
  .section-chevron { font-size:1rem; color:var(--soft); transition:transform .25s ease; display:inline-block; }
  .section.collapsed .section-chevron { transform:rotate(-90deg); }
  .cards-wrap { display:grid; grid-template-rows:1fr; transition:grid-template-rows .3s ease, opacity .3s ease; opacity:1; overflow:hidden; }
  .section.collapsed .cards-wrap { grid-template-rows:0fr; opacity:0; }
  .cards-inner { min-height:0; }
  .cards { display:flex; gap:20px; flex-wrap:wrap; justify-content:flex-start; padding:8px 0 8px; }
  .nav-card { background:white; border-radius:24px; padding:32px 24px 26px; width:275px; text-decoration:none; color:var(--ink); box-shadow:0 4px 24px #0000000a; transition:transform .2s,box-shadow .2s; text-align:center; display:flex; flex-direction:column; }
  .nav-card:hover { transform:translateY(-6px); box-shadow:0 16px 40px #00000014; }
  .nav-card-title { font-family:'Plus Jakarta Sans',sans-serif; font-size:1.08rem; font-weight:800; margin-bottom:8px; letter-spacing:-.01em; overflow-wrap:anywhere; }
  .nav-card-desc { font-size:.82rem; color:var(--soft); line-height:1.55; }
  .nav-card-arrow { margin-top:auto; padding-top:20px; font-size:.85rem; color:#9a7ab4; font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; letter-spacing:.04em; }
  .nav-card-date { margin-top:10px; font-size:.7rem; color:var(--soft); font-weight:500; letter-spacing:.04em; text-transform:uppercase; }
  footer { position:relative; z-index:1; text-align:center; padding:24px; color:var(--soft); font-size:.75rem; font-weight:300; }
  .ai-disclosure { margin-top:10px; margin-bottom:4px; }
  .ai-disclosure summary { list-style:none; cursor:pointer; font-size:.82rem; font-weight:600; color:#111; user-select:none; display:inline-flex; align-items:center; gap:5px; }
  .ai-disclosure summary::-webkit-details-marker { display:none; }
  .ai-disclosure summary::before { content:'▸'; font-size:.7rem; transition:transform .25s ease; display:inline-block; }
  .ai-disclosure[open] summary::before { transform:rotate(90deg); }
  .ai-disclosure-body { margin-top:6px; font-size:.82rem; font-weight:500; color:#111; white-space:nowrap; line-height:1.5; overflow:hidden; height:0; opacity:0; transition:height .24s ease, opacity .24s ease; }
  @keyframes fadeUp{from{opacity:0;transform:translateY(24px)}to{opacity:1;transform:translateY(0)}}
  .page { animation:fadeUp .6s ease both; }
  .benpom-hero { display:flex; flex-direction:column; width:100%; max-width:440px; margin:24px 0 24px; border-radius:22px; overflow:hidden; background:white; text-decoration:none; box-shadow:0 6px 24px #0000001a; transition:transform .2s,box-shadow .2s; }
  .benpom-hero:hover { transform:translateY(-5px); box-shadow:0 16px 38px #00000026; }
  .benpom-hero-banner { position:relative; height:235px; overflow:hidden; background:#1a0f24; }
  .benpom-hero-img { position:absolute; inset:0; background-size:cover; background-position:center; z-index:0; opacity:0; transition:transform .35s ease, opacity 1.2s ease; }
  .benpom-hero:hover .benpom-hero-img { transform:scale(1.06); }
  .benpom-hero-banner::after { content:''; position:absolute; inset:0; background:linear-gradient(180deg,#1a0f2455 0%,#1a0f24d0 100%); z-index:1; }
  .benpom-hero-content { position:absolute; inset:0; z-index:2; display:flex; flex-direction:column; align-items:center; justify-content:center; gap:7px; }
  .benpom-hero-title { font-family:'Plus Jakarta Sans',sans-serif; font-size:clamp(2.2rem,5.5vw,3.1rem); font-weight:800; color:#fff; letter-spacing:-1px; line-height:1; text-shadow:0 4px 22px #0e0a14cc; }
  .benpom-hero-desc { padding:18px 24px 20px; text-align:center; text-wrap:balance; }
  .benpom-hero-desc-body { font-family:'DM Sans',sans-serif; font-size:.82rem; color:var(--soft); line-height:1.55; }
  /* ── Alpha/Classic toggle — styled to MATCH the injected Alpha nav switch
        exactly (bare, same 38x21 purple track, same top-right position) so the
        control doesn't change size/shape/position between pages. Default knob is
        on the right (Classic active here); .alpha slides it left toward Alpha. ── */
  .uiswitch{position:fixed;top:12px;right:18px;z-index:60;display:inline-flex;align-items:center;gap:8px;
            cursor:pointer;user-select:none}
  .uiswitch .lbl{font-size:.74rem;font-weight:700;color:#9a93a6;transition:color .2s}
  .uiswitch .lbl.on{color:#16121d}
  .uitrack{position:relative;width:38px;height:21px;border-radius:999px;background:#7c4dd6}
  .uiknob{position:absolute;top:2px;left:2px;width:17px;height:17px;border-radius:50%;background:#fff;
          box-shadow:0 1px 4px #0003;transform:translateX(17px);transition:transform .25s cubic-bezier(.34,1.4,.5,1)}
  .uitrack.alpha .uiknob{transform:translateX(0)}
  @media (max-width:600px){ .uiswitch{top:10px;right:10px} }
  /* ── Mobile ─────────────────────────────────────────────── */
  @media (max-width:600px){
    .page { padding:32px 16px; }
    .tagline { white-space:normal; }
    .ai-disclosure-body { white-space:normal; }
    .sections { gap:28px; }
    .cards { gap:13px; justify-content:center; }
    .nav-card { width:100%; max-width:300px; padding:24px 20px 22px; }
    .nav-card-cover { height:100px; margin:-24px -20px 16px; width:calc(100% + 40px); }
    .nav-card-title { font-size:.96rem; }
    .nav-card-desc { font-size:.78rem; }
    .nav-card-arrow { font-size:.8rem; padding-top:16px; }
    .nav-card-date { font-size:.66rem; }
    .benpom-hero { max-width:320px; }
    .benpom-hero-banner { height:170px; }
    .benpom-hero-title { font-size:clamp(1.8rem,7vw,2.4rem); }
  }
</style>
</head>
<body>
<div class="uiswitch" onclick="goAlpha()" title="Back to the Alpha layout (now the default)">
  <span class="lbl">Alpha</span>
  <div class="uitrack"><div class="uiknob"></div></div>
  <span class="lbl on">Classic</span>
</div>
<div class="page">
  <h1><img src="/logo.svg" alt="B" style="height:1.65em;width:auto;vertical-align:-0.2em;margin-left:-0.3em;margin-right:-0.2em;object-fit:contain;cursor:pointer;" onclick="easterEgg()">obo gg</h1>
  <p class="tagline" id="tagline">Misceallneous analyses in the competitive Valorant space</p>
  <details class="ai-disclosure">
    <summary>AI Disclosure</summary>
    <div class="ai-disclosure-body">All narrative, text, mathematical equations, and ideas are my own creation. AI was/is only used for writing code.</div>
  </details>
  <a class="benpom-hero" href="/mapelo/">
    <div class="benpom-hero-banner">
      <div class="benpom-hero-img"></div>
      <div class="benpom-hero-content">
        <div class="benpom-hero-title">BenPom</div>
      </div>
    </div>
    <div class="benpom-hero-desc">
      <span class="benpom-hero-desc-body">A statistical rating system for VCT teams, both past and present.</span>
    </div>
  </a>
  <div class="sections">
    <div class="section">
      <div class="section-title">Research / Opinion Articles <span class="section-chevron">▾</span></div>
      <div class="cards-wrap"><div class="cards-inner">
      <div class="cards">
        <a class="nav-card" href="/articles/greatest-prime/">
          <img class="nav-card-cover" src="/aspas25corrode.jpg" alt="Aspas at Champions 2025">
          <div class="nav-card-title">The Greatest Prime in<br>VCT History Isn&rsquo;t a Debate</div>
          <div class="nav-card-desc">Aspas at Champions Paris towers over VCT history, including your favorite player.</div>
          <div class="nav-card-date">June 28, 2026</div>
          <div class="nav-card-arrow">Read &rarr;</div>
        </a>
        <a class="nav-card" href="/articles/masters-london-playoffs-preview/">
          <img class="nav-card-cover" src="/chronlondon.jpg" alt="Masters London">
          <div class="nav-card-title">Masters London<br>Playoffs Preview</div>
          <div class="nav-card-desc">A brief statistical glimpse into the final stage of Masters London.</div>
          <div class="nav-card-date">June 10, 2026</div>
          <div class="nav-card-arrow">Read &rarr;</div>
        </a>
        <a class="nav-card" href="/articles/masters-london-preview/">
          <img class="nav-card-cover" src="/prxpacstage1win.jpg" alt="Paper Rex win VCT Pacific Stage 1">
          <div class="nav-card-title">Masters London<br>Tournament Preview</div>
          <div class="nav-card-desc">Paper Rex's (un)inevitability, Neon nerfs, China's resurgence, and other bold predictions.</div>
          <div class="nav-card-date">June 2, 2026</div>
          <div class="nav-card-arrow">Read &rarr;</div>
        </a>
        <a class="nav-card" href="/articles/americas-stage1-playoffs-preview/">
          <img class="nav-card-cover" src="/loudlev26.jpg" alt="LOUD vs Leviatán">
          <div class="nav-card-title">Americas Stage 1<br>Playoffs Preview</div>
          <div class="nav-card-desc">A quick discussion after a wild Split 1: LOUD's resurgence, Leviatan's Bind, the ubiquitous question of 100 Thieves, and BenPom's final say.</div>
          <div class="nav-card-date">May 12, 2026</div>
          <div class="nav-card-arrow">Read &rarr;</div>
        </a>
        <a class="nav-card" href="/articles/over-underperformers/">
          <img class="nav-card-cover" src="/patmen.jpg" alt="Patmen">
          <div class="nav-card-title">Overperforming in VCT: Who's Doing It?</div>
          <div class="nav-card-desc">Using VCT stats to surface players who are outperforming (or underperforming) their team.</div>
          <div class="nav-card-date">May 4, 2026</div>
          <div class="nav-card-arrow">Read &rarr;</div>
        </a>
      </div>
      </div></div>
    </div>
    <div class="section">
      <div class="section-title">Statistics and Databases<span class="section-chevron">▾</span></div>
      <div class="cards-wrap"><div class="cards-inner">
      <div class="cards">
        <a class="nav-card" href="/mapelo/pythagorean/">
          <div class="nav-card-title">VCT's Pythagorean Rating</div>
          <div class="nav-card-desc">A pythagorean win% model hand-tuned for VCT, ranking teams by how dominant they've been and their true domestic strength levels.</div>
          <div class="nav-card-arrow">Explore &rarr;</div>
        </a>
        <a class="nav-card" href="/match-data/">
          <div class="nav-card-title">Match Data Explorer</div>
          <div class="nav-card-desc">Browse deeper per-map data from VLR match pages: round-by-round outcomes (side &amp; win condition), economy, clutches &amp; multikills, and the kill matrix.</div>
          <div class="nav-card-arrow">Explore &rarr;</div>
        </a>
        <a class="nav-card" href="/vct/">
          <div class="nav-card-title">Event Leaderboards</div>
          <div class="nav-card-desc">Sift through leaderboards by events, highlighting indivdual performances and percentiles.</div>
          <div class="nav-card-arrow">Explore &rarr;</div>
        </a>
        <a class="nav-card" href="/highs/">
          <div class="nav-card-title">All-Time Highs<br>(and Lows)</div>
          <div class="nav-card-desc">The best and worst individual map/match performances across all VCT franchised events.</div>
          <div class="nav-card-arrow">Explore &rarr;</div>
        </a>
      </div>
      </div></div>
    </div>
  </div>
</div>
<footer>
  Data sourced from VLR.gg
  <div style="margin-top:8px;">Like my work? Tips are appreciated! <a href="https://ko-fi.com/bobovct" target="_blank" rel="noopener" style="color:inherit;text-decoration:underline;">ko-fi.com/bobovct</a></div>
</footer>
<script>
function goAlpha(){
  try{localStorage.setItem('bobo_ui','alpha');}catch(e){}
  var t=document.querySelector('.uiswitch .uitrack');
  var labels=document.querySelectorAll('.uiswitch .lbl');
  if(t)t.classList.add('alpha');
  if(labels[0])labels[0].classList.add('on');     // Alpha (now the left label)
  if(labels[1])labels[1].classList.remove('on');  // Classic (right label)
  setTimeout(function(){location.href='/';},240);
}
var EGG_TEXT = "Uxie is N0te's dada";
var ORIG_TAGLINE = null;
var eggTimer = null;
function swapTagline(newText) {
  var el = document.getElementById('tagline');
  el.style.opacity = '0';
  el.style.transform = 'translateY(-8px)';
  setTimeout(function() {
    el.textContent = newText;
    el.style.transform = 'translateY(8px)';
    el.offsetHeight;
    el.style.opacity = '1';
    el.style.transform = 'translateY(0)';
  }, 280);
}
function easterEgg() {
  var el = document.getElementById('tagline');
  if (ORIG_TAGLINE === null) ORIG_TAGLINE = el.textContent;
  if (eggTimer) clearTimeout(eggTimer);
  swapTagline(EGG_TEXT);
  eggTimer = setTimeout(function() {
    swapTagline(ORIG_TAGLINE);
    eggTimer = null;
  }, 5000);
}
(function(){
  var details = document.querySelector('.ai-disclosure');
  var body = details && details.querySelector('.ai-disclosure-body');
  if (!details || !body) return;
  details.addEventListener('click', function(e) {
    e.preventDefault();
    if (!details.open) {
      details.open = true;
      var h = body.scrollHeight;
      body.style.height = '0px';
      body.style.opacity = '0';
      body.offsetHeight;
      body.style.height = h + 'px';
      body.style.opacity = '1';
      body.addEventListener('transitionend', function done(ev) {
        if (ev.propertyName !== 'height') return;
        body.style.height = 'auto';
        body.removeEventListener('transitionend', done);
      });
    } else {
      body.style.height = body.scrollHeight + 'px';
      body.offsetHeight;
      body.style.height = '0px';
      body.style.opacity = '0';
      body.addEventListener('transitionend', function done(ev) {
        if (ev.propertyName !== 'height') return;
        details.open = false;
        body.removeEventListener('transitionend', done);
      });
    }
  });
})();
document.querySelectorAll('.section-title').forEach(function(title) {
  title.addEventListener('click', function() {
    this.closest('.section').classList.toggle('collapsed');
  });
});
(function(){
  var heroImg = document.querySelector('.benpom-hero-img');
  if (!heroImg) return;
  var src = '/static/MastersShanghaiFinal.jpg';
  var shown = false;
  function show() {
    if (shown) return;
    shown = true;
    heroImg.style.backgroundImage = 'url(' + src + ')';
    requestAnimationFrame(function() { requestAnimationFrame(function() {
      heroImg.style.opacity = '1';
    }); });
  }
  var img = new Image();
  img.onload = show;
  img.onerror = show;
  img.src = src;
  setTimeout(show, 3000);
})();
</script>
</body>
</html>
"""

ALPHA_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Bobo gg</title>
<link rel="icon" type="image/svg+xml" href="/favicon.svg">
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@600;700;800&family=DM+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
<link rel="stylesheet" href="/static/base.css">
<style>
  /* Kill the perpetual background animation on this view — it repaints the whole
     viewport every frame and makes scrolling the long dashboard janky. The
     static gradient (body::before) stays for the look. */
  body::after{animation:none !important;}
  :root{
    --card:#ffffff; --line:#eceef2; --ink:#16121d; --soft:#6b6478; --faint:#9a93a6;
    --good:#1f9d55; --bad:#d23b3b; --accent:#7c4dd6;
    /* Canonical VCT region colors (match .lb-region across the site):
       EMEA=green, Americas=orange, Pacific=blue, CN=pink/magenta. */
    --r-emea:#15803d; --r-amer:#c2410c; --r-pac:#1d4ed8; --r-cn:#be185d; --r-int:#666;
  }
  *{box-sizing:border-box}
  body{font-family:'DM Sans',sans-serif;color:var(--ink);}
  a{color:inherit;text-decoration:none}
  .wrap{width:100%;max-width:1180px;margin:0 auto;padding:0 22px 64px;position:relative;z-index:1}

  /* ── Top bar + nav ── */
  .atop{display:flex;align-items:center;justify-content:space-between;gap:16px;padding:16px 4px 8px}
  .abrand{display:flex;align-items:center;gap:9px;font-family:'Plus Jakarta Sans',sans-serif;font-weight:800;font-size:1.18rem;letter-spacing:-.02em}
  .abrand img{height:1.5em;width:auto}
  .abeta{font-size:.6rem;font-weight:800;letter-spacing:.12em;text-transform:uppercase;color:#fff;background:var(--accent);padding:3px 7px;border-radius:6px}
  .uiswitch{position:fixed;top:16px;right:18px;z-index:60;display:inline-flex;align-items:center;gap:9px;cursor:pointer;user-select:none;background:#fff;border:1px solid #e7e2ee;border-radius:999px;padding:6px 12px;box-shadow:0 3px 14px #0000000f;transition:border-color .2s,box-shadow .2s}
  .uiswitch:hover{border-color:#d9d2e6}
  .uiswitch .lbl{font-size:.78rem;font-weight:700;color:var(--faint)}
  .uiswitch .lbl.on{color:var(--ink)}
  .uitrack{position:relative;width:42px;height:23px;border-radius:999px;background:var(--accent)}
  .uiknob{position:absolute;top:2px;left:2px;width:19px;height:19px;border-radius:50%;background:#fff;box-shadow:0 1px 4px #0003;transform:translateX(19px)}
  .anav{display:flex;gap:7px;overflow-x:auto;padding:4px 4px 14px;-webkit-overflow-scrolling:touch;scrollbar-width:none}
  .anav::-webkit-scrollbar{display:none}
  .anav a{flex:0 0 auto;font-size:.82rem;font-weight:700;color:var(--soft);background:#fff;border:1px solid var(--line);border-radius:999px;padding:7px 14px;transition:color .16s,border-color .16s,background .16s;white-space:nowrap}
  .anav a:hover{color:var(--ink);border-color:#d9d2e6;background:#faf8ff}
  .anav a.cta{color:#fff;background:var(--accent);border-color:var(--accent)}
  .anav a.cta:hover{background:#6c3fc6}
  /* ── Home title (the fixed top nav carries the brand; this is the page's own
        hero header so /alpha doesn't start cramped right under the nav bar) ── */
  .ahome{padding:16px 4px 22px;text-align:center}
  .ahome-brand{font-family:'Plus Jakarta Sans',sans-serif;font-weight:800;font-size:2.6rem;letter-spacing:-.03em;color:var(--ink);line-height:1}
  .ahome-logo{height:1.65em;width:auto;vertical-align:-0.2em;margin-left:-0.3em;margin-right:-0.2em;object-fit:contain}
  @media (max-width:600px){.ahome{padding:10px 4px 16px}.ahome-brand{font-size:1.95rem}}

  /* ── Banner + season timeline ── */
  .ebanner{background:linear-gradient(135deg,#1d1330 0%,#2a1c44 55%,#3a1f55 100%);border-radius:22px;padding:24px 28px;color:#fff;position:relative;overflow:hidden;box-shadow:0 10px 34px #1d133033;margin-bottom:24px;display:grid;grid-template-columns:1fr 1fr;gap:26px;align-items:stretch}
  .ebanner-l{min-width:0;display:flex;flex-direction:column}
  .ebanner-l .timeline{margin-top:auto}
  .ebanner-r{position:relative;border-radius:16px;overflow:hidden;min-height:240px;display:flex;flex-direction:column;justify-content:flex-end;padding:22px 22px 17px;text-decoration:none;color:#fff;background:#1a0f24;transition:transform .18s,box-shadow .18s}
  .ebanner-r:hover{transform:translateY(-3px);box-shadow:0 16px 38px #00000040}
  .ebanner-r img{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;object-position:center 22%}
  .ebanner-r::after{content:'';position:absolute;inset:0;background:linear-gradient(180deg,#1a0f2400 0%,#1a0f2452 36%,#1a0f24dd 66%,#1a0f24 100%)}
  .ebanner-r>*{position:relative;z-index:1}
  .ead-tag{font-size:.62rem;font-weight:800;letter-spacing:.14em;text-transform:uppercase;color:#e7dcff;margin-bottom:8px;text-shadow:0 1px 7px #1a0f24}
  .ead-title{font-family:'Plus Jakarta Sans',sans-serif;font-weight:800;font-size:1.5rem;line-height:1.13;letter-spacing:-.01em;text-shadow:0 2px 10px #1a0f24cc}
  .ead-link{margin-top:13px;font-size:.86rem;font-weight:800;color:#fff}
  @media(max-width:760px){.ebanner{grid-template-columns:1fr}.ebanner-r{min-height:170px}}
  .ebanner::after{content:'';position:absolute;right:-60px;top:-70px;width:260px;height:260px;border-radius:50%;background:radial-gradient(circle,#a87bff2e,transparent 70%);pointer-events:none}
  .epill{display:inline-flex;align-items:center;gap:7px;font-size:.66rem;font-weight:800;letter-spacing:.14em;text-transform:uppercase;padding:5px 11px;border-radius:999px;background:#ffffff1c;margin-bottom:12px}
  .epill .dot{width:7px;height:7px;border-radius:50%;background:#ffd56b}
  .epill.live .dot{background:#41f59a;animation:lpulse 1.6s infinite}
  @keyframes lpulse{0%{box-shadow:0 0 0 0 #41f59a99}70%{box-shadow:0 0 0 7px #41f59a00}100%{box-shadow:0 0 0 0 #41f59a00}}
  .etitle{font-family:'Plus Jakarta Sans',sans-serif;font-weight:800;font-size:clamp(1.9rem,4vw,2.7rem);letter-spacing:-.02em;line-height:1.04}
  .esub{margin-top:7px;font-size:.92rem;color:#e6dcf5;font-weight:500}
  .ebtns{margin-top:16px;display:flex;gap:10px;flex-wrap:wrap}
  .ebtn{display:inline-flex;align-items:center;gap:7px;font-size:.82rem;font-weight:700;padding:9px 15px;border-radius:11px;background:#fff;color:#241636;transition:opacity .2s}
  .ebtn:hover{opacity:.88}
  .ebtn.ghost{background:#ffffff1f;color:#fff}
  /* Side padding is deliberate: a long single-word label (e.g. "Champions") is
     wider than its node and spills a few px past the last node — the padding box
     is the clip edge, so that spill renders instead of being cut off. */
  .timeline{display:flex;align-items:flex-start;margin-top:22px;position:relative;overflow-x:auto;padding:5px 8px 4px;scrollbar-width:none}
  .timeline::-webkit-scrollbar{display:none}
  .tnode{flex:1 1 0;min-width:44px;display:flex;flex-direction:column;align-items:center;text-align:center;position:relative}
  .tnode::before{content:'';position:absolute;top:8px;left:-50%;width:100%;height:2px;background:#ffffff22}
  .tnode:first-child::before{display:none}
  .tnode.done::before,.tnode.live::before{background:#a98bff}
  .tdot{width:17px;height:17px;border-radius:50%;background:#ffffff2e;z-index:1;display:flex;align-items:center;justify-content:center;font-size:.58rem;color:#3a1f55;font-weight:800}
  .tnode.done .tdot{background:#a98bff}
  .tnode.live .tdot,.tnode.next .tdot{background:#fff;box-shadow:0 0 0 4px #ffffff30}
  .tlbl{margin-top:7px;font-size:.61rem;font-weight:700;color:#cdbfe6;line-height:1.18;width:100%;padding:0 2px;box-sizing:border-box}
  .tnode.next .tlbl,.tnode.live .tlbl{color:#fff}
  .tdate{font-size:.54rem;color:#9d8fbb;font-weight:600;margin-top:2px}

  /* ── Panels / grid ── */
  .agrid{display:grid;grid-template-columns:1.35fr 1fr;gap:22px;align-items:start}
  #rankings-panel{position:sticky;top:14px}
  .panel{background:var(--card);border:1px solid var(--line);border-radius:20px;padding:20px 20px 14px;box-shadow:0 4px 22px #0000000a}
  .phead{display:flex;align-items:center;justify-content:space-between;gap:10px;margin-bottom:4px}
  .ptitle{font-family:'Plus Jakarta Sans',sans-serif;font-weight:800;font-size:1.18rem;letter-spacing:-.01em}
  .plink{font-size:.82rem;font-weight:800;color:var(--accent);background:#f1ebfb;border:1px solid #e4d9f6;padding:7px 15px;border-radius:999px;white-space:nowrap;flex-shrink:0;transition:background .15s,transform .12s,box-shadow .15s}
  .plink:hover{background:#e7dbfa;transform:translateY(-1px);box-shadow:0 4px 13px rgba(124,77,214,.2)}
  .psub{font-size:.74rem;color:var(--faint);font-weight:600;margin:0 0 12px}
  .seg-row{display:flex;align-items:center;justify-content:space-between;gap:10px;margin-top:14px;margin-bottom:14px;flex-wrap:wrap}
  .seg{display:inline-flex;background:#f1eef6;border-radius:10px;padding:3px;gap:2px}
  .mregion{font-family:inherit;font-size:.78rem;font-weight:700;color:var(--soft);background:#fff;border:1px solid var(--line);border-radius:9px;padding:6px 11px;cursor:pointer;transition:border-color .15s}
  .mregion:hover{border-color:#d6cce8}
  .seg button{border:0;background:transparent;font-family:inherit;font-size:.8rem;font-weight:700;color:var(--soft);padding:6px 14px;border-radius:8px;cursor:pointer;transition:color .16s,background .16s}
  .seg button.on{background:#fff;color:var(--ink);box-shadow:0 1px 5px #00000012}

  /* ── Match cards ── */
  .mcard{border:1px solid var(--line);border-radius:15px;padding:13px 15px;margin-bottom:11px;transition:border-color .16s,background .16s;background:#fff}
  .mcard:hover{border-color:#e0d8ee;background:#fcfbff}
  .mc-meta{display:flex;align-items:center;gap:8px;font-size:.7rem;color:var(--faint);font-weight:600;margin-bottom:9px;flex-wrap:wrap}
  .mc-meta .mtag{background:#f3eefb;color:#6a4caf;padding:2px 7px;border-radius:6px;font-weight:700}
  .mc-row{display:grid;grid-template-columns:1fr auto 1fr;align-items:center;gap:10px}
  .mc-team{display:flex;align-items:center;gap:9px;min-width:0;color:inherit;text-decoration:none;border-radius:9px;transition:background .15s;justify-self:start}
  a.mc-team{padding:3px 6px;margin:-3px -6px}
  a.mc-team:hover{background:rgba(124,77,214,.09)}   /* clear "clickable team" affordance, no stray underline */
  .mc-team.b{flex-direction:row-reverse;text-align:right;justify-self:end}   /* shrink link to its content, not the whole column */
  .mc-logo{width:30px;height:30px;border-radius:7px;object-fit:contain;flex-shrink:0}
  .mc-init{width:30px;height:30px;border-radius:7px;display:flex;align-items:center;justify-content:center;font-size:.6rem;font-weight:800;color:#fff;flex-shrink:0}
  .mc-name{font-weight:800;font-size:.95rem;letter-spacing:-.01em;line-height:1.1}
  .mc-rat{font-size:.68rem;color:var(--soft);font-weight:600}
  .mc-win{font-size:1.18rem;font-weight:800;font-family:'Plus Jakarta Sans',sans-serif;text-align:center;min-width:54px}
  .mc-win .vs{font-size:.6rem;color:var(--faint);font-weight:700;display:block;letter-spacing:.1em}
  .mc-win .mc-vs{color:#b9b1c6;letter-spacing:.04em}
  .mc-bar{height:22px;border-radius:7px;display:flex;overflow:hidden;margin:11px 0 7px;background:#eee;font-family:'Plus Jakarta Sans',sans-serif;font-weight:800;font-size:.72rem}
  .mc-seg{display:flex;align-items:center;justify-content:center;height:100%;white-space:nowrap;min-width:0;padding:0 6px}
  .mc-seg.a{background:linear-gradient(90deg,#9b7be6,#7c4dd6);color:#fff}
  .mc-seg.b{background:#e2dcec;color:#6b6478}
  .mc-foot{display:flex;align-items:center;justify-content:center;gap:9px;font-size:.8rem;font-weight:700}
  a.mc-simlink{color:inherit;text-decoration:none}
  a.mc-simlink:hover .mc-final{color:var(--accent)}
  .mc-pager{display:flex;align-items:center;justify-content:center;gap:14px;padding:8px 0 4px;color:var(--soft);font-size:.74rem;font-weight:700}
  .mc-pager button{width:28px;height:28px;border-radius:8px;border:1px solid var(--line);background:#fff;color:var(--ink);font-size:1rem;font-weight:700;cursor:pointer;line-height:1;transition:border-color .15s,background .15s}
  .mc-pager button:hover:not(:disabled){border-color:#d6cce8;background:#faf8ff}
  .mc-pager button:disabled{opacity:.35;cursor:default}
  .mc-score{font-weight:800}
  .mc-res{font-size:.62rem;font-weight:800;letter-spacing:.06em;text-transform:uppercase;padding:3px 8px;border-radius:6px}
  .mc-res.win{background:#e7f6ec;color:var(--good)}
  .mc-res.upset{background:#fdeaea;color:var(--bad)}
  .mc-final{font-size:.6rem;letter-spacing:.12em;text-transform:uppercase;color:var(--faint);font-weight:800}
  /* ── Recent-card redesign: result is the focal point, pre-match odds sink to the bottom ── */
  .mc-result{display:grid;grid-template-columns:1fr auto 1fr;align-items:center;gap:10px;margin:4px 0 2px}
  .mc-rteam{display:flex;align-items:center;gap:9px;min-width:0;color:inherit;text-decoration:none;border-radius:9px;padding:3px 6px;margin:-3px -6px;transition:background .15s;justify-self:start}
  .mc-rteam:hover{background:rgba(124,77,214,.09)}
  .mc-rteam.b{flex-direction:row-reverse;text-align:right;justify-self:end}
  .mc-wl{flex:0 0 auto;font-size:.78rem;font-weight:800;letter-spacing:.03em;padding:3px 9px;border-radius:6px}
  .mc-wl.w{background:#e7f6ec;color:var(--good)}
  .mc-wl.l{background:#fdeaea;color:var(--bad)}
  .mc-rname{font-weight:800;font-size:1.02rem;letter-spacing:-.01em;line-height:1.15;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  .mc-rscore{font-family:'Plus Jakarta Sans',sans-serif;font-weight:800;font-size:1.72rem;line-height:1;letter-spacing:-.02em;text-align:center;white-space:nowrap}
  .mc-pre{border-top:1px solid var(--line);margin-top:11px;padding-top:9px}
  .mc-pre-lbl{font-size:.58rem;letter-spacing:.12em;text-transform:uppercase;color:var(--faint);font-weight:800;margin-bottom:6px;text-align:center}
  .mc-prow{display:flex;align-items:center;gap:9px}
  .mc-prow .mc-bar{flex:1;margin:0}
  .mc-prat{font-size:.72rem;font-weight:800;color:var(--soft);font-family:'Plus Jakarta Sans',sans-serif;min-width:44px;text-align:center}
  .empty{padding:26px 12px;text-align:center;color:var(--faint);font-size:.86rem;font-weight:600;line-height:1.7}
  /* match-body fade cap so the panel never grows too tall when a card expands */
  #match-body{overflow-y:auto;scrollbar-width:thin}
  #match-body.capped{-webkit-mask-image:linear-gradient(to bottom,#000 calc(100% - 22px),transparent);mask-image:linear-gradient(to bottom,#000 calc(100% - 22px),transparent)}
  #match-body::-webkit-scrollbar{width:6px}#match-body::-webkit-scrollbar-thumb{background:#dcd5e8;border-radius:6px}
  /* upcoming card expand */
  .mcard.upc{cursor:pointer}
  .mc-expand-hint{text-align:center;font-size:.6rem;color:#c3bcd0;margin-top:8px;letter-spacing:.06em;font-weight:700;text-transform:uppercase;cursor:pointer}
  .mcard.upc:hover .mc-expand-hint{color:var(--accent)}
  .mc-details{display:grid;grid-template-rows:0fr;transition:grid-template-rows .32s cubic-bezier(.22,1,.36,1)}
  .mc-details-inner{overflow:hidden;min-height:0;transition:padding-top .32s ease,margin-top .32s ease}
  .mcard.open .mc-details{grid-template-rows:1fr}
  .mcard.open .mc-details-inner{padding-top:13px;margin-top:11px;border-top:1px solid rgba(0,0,0,.07)}
  .h2h-load{text-align:center;color:var(--faint);font-size:.78rem;font-weight:600;padding:10px}
  .h2h-head{text-align:center;font-size:.92rem;font-weight:700;margin-bottom:12px}
  .h2h-head b{font-weight:800}
  .h2h-sub{font-size:.62rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:var(--faint);margin-top:2px}
  .h2h-grid{display:grid;grid-template-columns:1fr auto 1fr;gap:10px;align-items:start}
  .h2h-col{text-align:center;min-width:0}
  .h2h-col.b{}
  .h2h-vs{align-self:center;font-family:'Plus Jakarta Sans',sans-serif;font-weight:800;font-size:.72rem;color:var(--faint)}
  .h2h-org{display:inline-flex;align-items:center;gap:6px;font-weight:800;font-size:.95rem;color:inherit;text-decoration:none}
  .h2h-org:hover span{text-decoration:underline}
  .h2h-logo{width:24px;height:24px;object-fit:contain;border-radius:5px}
  .h2h-init{width:24px;height:24px;border-radius:5px;display:inline-flex;align-items:center;justify-content:center;font-size:.55rem;font-weight:800;color:#fff}
  .h2h-meta{display:flex;align-items:center;justify-content:center;gap:7px;margin:5px 0 6px}
  .h2h-rank{font-size:.7rem;font-weight:700;color:var(--soft)}
  .h2h-rat{font-family:'Plus Jakarta Sans',sans-serif;font-weight:800;font-size:1.25rem;line-height:1}
  .h2h-rat span{display:block;font-family:'DM Sans',sans-serif;font-size:.56rem;font-weight:700;letter-spacing:.06em;text-transform:uppercase;color:var(--faint)}
  .h2h-rat.pos{color:#16a34a}.h2h-rat.neg{color:#c0392b}
  .h2h-form{display:flex;justify-content:center;gap:4px;margin:8px 0}
  .h2h-dot{width:8px;height:8px;border-radius:50%}.h2h-dot.w{background:var(--good)}.h2h-dot.l{background:#e6b0b0}
  .h2h-mtitle{font-size:.74rem;font-weight:800;letter-spacing:.03em;color:var(--ink);margin:11px 0 6px}
  .h2h-map{display:flex;align-items:center;justify-content:space-between;gap:6px;font-size:.74rem;font-weight:600;padding:2px 0}
  .h2h-map b{font-family:'Plus Jakarta Sans',sans-serif}.h2h-map b.pos{color:#16a34a}.h2h-map b.neg{color:#c0392b}
  .h2h-na{font-size:.72rem;color:var(--faint);font-weight:600}
  .h2h-simlink{display:block;text-align:center;margin-top:12px;font-size:.76rem;font-weight:800;color:var(--accent)}
  .h2h-simlink:hover{text-decoration:underline}

  /* ── Rankings — aligned columns ── */
  .rhead,.rrow{display:grid;grid-template-columns:26px 32px minmax(0,1fr) 64px 44px 58px;align-items:center;gap:10px}
  .rhead{padding:0 4px 8px;border-bottom:1px solid #eee;font-size:.6rem;font-weight:800;letter-spacing:.06em;text-transform:uppercase;color:var(--faint)}
  .rhead .rh-r{text-align:center}.rhead .rh-form{text-align:center}.rhead .rh-pct,.rhead .rh-rat{text-align:right}
  .rrow{padding:9px 4px;border-bottom:1px solid #f4f2f8}
  .rrow:last-child{border-bottom:0}
  .rrow:first-child .rrank{color:#d9a300}
  .rrank{font-family:'Plus Jakarta Sans',sans-serif;font-weight:800;font-size:.92rem;color:var(--faint);text-align:center}
  .rlogo{width:32px;height:32px;border-radius:7px;object-fit:contain}
  .rinit{width:32px;height:32px;border-radius:7px;display:flex;align-items:center;justify-content:center;font-size:.6rem;font-weight:800;color:#fff}
  .rteam{display:flex;align-items:center;gap:8px;min-width:0}
  .rorg{font-weight:800;font-size:.95rem;letter-spacing:-.01em}
  .rbadge{font-size:.56rem;font-weight:800;letter-spacing:.03em;padding:2px 7px;border-radius:5px;flex:0 0 auto}
  .rbadge.emea{background:rgba(22,163,74,.14);color:#15803d}
  .rbadge.americas{background:rgba(234,88,12,.14);color:#c2410c}
  .rbadge.pacific{background:rgba(37,99,235,.14);color:#1d4ed8}
  .rbadge.cn{background:rgba(219,39,119,.14);color:#be185d}
  .rbadge.int{background:rgba(0,0,0,.06);color:#666}
  .rcell-link{display:inline-flex;align-items:center;justify-content:center;justify-self:center}
  .rteam-link{display:inline-flex;align-items:center;gap:8px;min-width:0;color:inherit;text-decoration:none;justify-self:start}
  .rteam-link:hover .rorg{text-decoration:underline}
  .rform{display:flex;gap:5px;justify-content:center;align-items:center}
  .fdot{width:9px;height:9px;border-radius:50%;display:block;flex:0 0 9px;padding:0;box-sizing:border-box;transition:transform .12s}
  a.fdot:hover{transform:scale(1.35)}
  .fdot.w{background:var(--good)}
  .fdot.l{background:#e6b0b0}
  .fdot.empty{background:#edebf2}
  .rpct{font-size:.82rem;color:var(--soft);font-weight:700;text-align:right}
  .rrat{font-family:'Plus Jakarta Sans',sans-serif;font-weight:800;font-size:1.0rem;text-align:right}
  .rlegend{font-size:.66rem;color:var(--faint);font-weight:600;padding:11px 4px 2px;display:flex;gap:14px;flex-wrap:wrap;align-items:center}
  /* Match-hover card — copied from the Modern Hub chart dot tooltip */
  #dotTooltip{position:fixed;z-index:200;pointer-events:none;min-width:240px;max-width:340px;background:#f3edfc;border:1px solid #e1d6f4;border-radius:14px;padding:16px 20px;box-shadow:0 18px 50px rgba(40,20,70,.22);opacity:0;transform:translateY(8px);transition:opacity .15s ease,transform .15s ease;left:0;top:0}
  #dotTooltip.visible{opacity:1;transform:translateY(0)}
  #dotTooltip .popup-inner{text-align:center}
  #dotTooltip .popup-event-label{font-size:.64rem;font-weight:800;color:#6a35b8;text-transform:uppercase;letter-spacing:.08em;margin-bottom:8px}
  #dotTooltip .popup-teams{display:flex;align-items:center;justify-content:center;gap:14px;margin-bottom:6px}
  #dotTooltip .popup-team-block{display:flex;flex-direction:column;align-items:center;gap:4px;min-width:54px}
  #dotTooltip .popup-logo{width:38px;height:38px;object-fit:contain}
  #dotTooltip .popup-team-name{font-size:.68rem;color:#241a2e;font-weight:700}
  #dotTooltip .popup-score-block{display:flex;flex-direction:column;align-items:center;gap:2px}
  #dotTooltip .popup-score{font-size:1.7rem;font-weight:800;font-family:'Plus Jakarta Sans',sans-serif;line-height:1}
  #dotTooltip .popup-score.w{color:#16a34a}#dotTooltip .popup-score.l{color:#dc2626}
  #dotTooltip .popup-vs-label{font-size:.6rem;color:#6b6478;font-weight:600}
  #dotTooltip .popup-date{color:#544c63;font-size:.66rem;font-weight:600;margin-bottom:3px}
  #dotTooltip .popup-delta{font-size:.8rem;font-weight:700;margin-bottom:10px;color:#241a2e}
  #dotTooltip .popup-delta .pos{color:#16a34a}#dotTooltip .popup-delta .neg{color:#c0392b}
  #dotTooltip .popup-maps-table{width:100%;border-collapse:collapse;margin-top:2px}
  #dotTooltip .popup-maps-table th{font-size:.58rem;font-weight:800;color:#6a35b8;text-transform:uppercase;letter-spacing:.07em;padding:0 6px 5px;text-align:center}
  #dotTooltip .popup-maps-table td{padding:4px 6px;font-size:.74rem;color:#2a1f2d;border-top:1px solid #e1d6f4;text-align:center}
  #dotTooltip .popup-map-name{font-weight:700;color:#2a1f2d;text-align:center}
  #dotTooltip .popup-map-score{text-align:center;font-variant-numeric:tabular-nums;font-weight:700}
  #dotTooltip .popup-map-score.w{color:#16a34a}#dotTooltip .popup-map-score.l{color:#dc2626}
  #dotTooltip .popup-map-diff{text-align:center;font-size:.7rem;font-weight:700;color:#4a4357}
  .rlegend i{width:8px;height:8px;border-radius:50%;display:inline-block;margin-right:3px;vertical-align:middle}
  .rlegend i.w{background:var(--good)}

  /* ── Player leaders — one mini-leaderboard per stat ── */
  .players{margin-top:22px}
  .pl-minrnd{display:flex;align-items:center;gap:8px;margin-left:auto;flex-shrink:0}
  .pl-minrnd-lab{font-size:.66rem;font-weight:800;letter-spacing:.07em;text-transform:uppercase;color:var(--faint);white-space:nowrap}
  .pl-minrnd-val{font-family:'Plus Jakarta Sans',sans-serif;font-weight:800;font-size:.8rem;color:var(--accent);min-width:36px;text-align:right;font-variant-numeric:tabular-nums}
  #minRndSlider{-webkit-appearance:none;appearance:none;width:110px;height:5px;border-radius:99px;background:#e4d9f6;outline:none;cursor:pointer}
  #minRndSlider::-webkit-slider-thumb{-webkit-appearance:none;appearance:none;width:15px;height:15px;border-radius:50%;background:var(--accent);border:2px solid #fff;box-shadow:0 1px 5px rgba(124,77,214,.45);cursor:pointer}
  #minRndSlider::-moz-range-thumb{width:15px;height:15px;border-radius:50%;background:var(--accent);border:2px solid #fff;box-shadow:0 1px 5px rgba(124,77,214,.45);cursor:pointer}
  @media (max-width:700px){.pl-minrnd-lab{display:none}#minRndSlider{width:80px}}
  .pl-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:14px}
  .pl-card{border:1px solid var(--line);border-radius:15px;padding:13px 13px 9px;background:#fff}
  #records-panel{margin-top:22px}
  .rec-wrap{display:flex;align-items:center;gap:8px}
  .rec-nav{flex:0 0 30px;width:30px;height:30px;border-radius:50%;border:1px solid var(--line);background:#fff;color:#7c4dd6;font-family:'Plus Jakarta Sans',sans-serif;font-size:1.1rem;font-weight:800;line-height:1;cursor:pointer;box-shadow:0 3px 12px #00000014;display:flex;align-items:center;justify-content:center;transition:background .15s,transform .15s,box-shadow .15s}
  .rec-nav:hover{background:#f1ebfb;transform:scale(1.1);box-shadow:0 5px 16px #7c4dd633}
  .rec-vp{flex:1;min-width:0;overflow:hidden;position:relative;padding:4px 0;-webkit-mask-image:linear-gradient(90deg,transparent 0,#000 3%,#000 97%,transparent 100%);mask-image:linear-gradient(90deg,transparent 0,#000 3%,#000 97%,transparent 100%)}
  .rec-track{display:flex;align-items:stretch;width:max-content;will-change:transform}
  .rec-card{position:relative;flex:0 0 300px;margin-right:13px;display:flex;align-items:center;gap:12px;border:1px solid var(--line);border-radius:15px;padding:12px 14px;background:#fff;text-decoration:none;color:inherit;transition:transform .15s,box-shadow .15s,border-color .15s}
  .rec-card:hover{transform:translateY(-3px);box-shadow:0 12px 28px #0000000f;border-color:#e4d9f6}
  .rec-rankbadge{position:absolute;top:8px;right:12px;font-family:'Plus Jakarta Sans',sans-serif;font-weight:800;font-size:.8rem;color:#bcaadd}
  .rec-av,.rec-av-ph{width:46px;height:46px;border-radius:50%;object-fit:cover;object-position:top center;flex:0 0 46px}
  .rec-av-ph{display:flex;align-items:center;justify-content:center;font-family:'Plus Jakarta Sans',sans-serif;font-weight:800;color:#fff;font-size:15px}
  .rec-info{display:flex;flex-direction:column;min-width:0;flex:1}
  .rec-name{display:flex;align-items:center;gap:6px;font-family:'Plus Jakarta Sans',sans-serif;font-weight:800;font-size:.95rem;letter-spacing:-.01em}
  .rec-tlogo{height:15px;width:auto;object-fit:contain;flex:0 0 auto}
  .rec-tinit{height:15px;min-width:15px;padding:0 3px;border-radius:4px;display:inline-flex;align-items:center;justify-content:center;font-size:.5rem;font-weight:800;color:#fff;flex:0 0 auto}
  .rec-desc{font-size:.74rem;font-weight:700;color:#7c4dd6;margin-top:2px;line-height:1.3}
  .rec-foot{display:flex;flex-direction:column;align-items:flex-start;gap:3px;margin-top:4px;font-size:.7rem;color:var(--soft);font-weight:600}
  .rec-foot .rec-vs{display:inline-flex;align-items:center;gap:4px}
  .rec-foot .rec-vs img{height:13px;width:auto}
  .rec-date{color:var(--soft);font-size:.68rem;font-weight:600;white-space:nowrap}
  .rec-ev{color:#9a93a6;white-space:nowrap}
  .rec-map{background:#f0ecf4;border-radius:99px;padding:1px 8px;color:var(--soft)}
  .rec-val{font-family:'Plus Jakarta Sans',sans-serif;font-weight:800;font-size:1.32rem;color:var(--ink);flex:0 0 auto;padding-left:6px}
  .pl-stat{display:flex;align-items:center;gap:6px;font-family:'Plus Jakarta Sans',sans-serif;font-weight:800;font-size:.82rem;letter-spacing:-.01em;color:var(--ink);text-decoration:none;padding:0 4px 9px;border-bottom:1px solid var(--line);margin-bottom:5px;transition:color .14s}
  a.pl-stat:hover{color:var(--accent)}
  .pl-arrow{margin-left:auto;color:var(--accent);opacity:0;transform:translateX(-3px);transition:opacity .14s,transform .14s}
  a.pl-stat:hover .pl-arrow{opacity:1;transform:translateX(0)}
  .plr{display:flex;align-items:center;gap:9px;padding:6px 5px;border-radius:9px;color:inherit;text-decoration:none;transition:background .14s}
  .plr:hover{background:#faf8ff}
  .plr-n{font-family:'Plus Jakarta Sans',sans-serif;font-size:.74rem;font-weight:800;color:var(--faint);width:14px;text-align:center;flex:0 0 auto}
  .plr-av{width:30px;height:30px;border-radius:50%;object-fit:cover;object-position:top center;background:#efeaf6;flex:0 0 auto;box-shadow:0 0 0 2px var(--ring,#eee)}
  .plr-av-ph{width:30px;height:30px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:.64rem;font-weight:800;color:#fff;flex:0 0 auto}
  .plr-info{display:flex;flex-direction:column;min-width:0;flex:1}
  .plr-name{font-weight:700;font-size:.83rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .plr-meta{font-size:.65rem;color:var(--soft);font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .plr-val{font-family:'Plus Jakarta Sans',sans-serif;font-weight:800;font-size:.92rem;flex:0 0 auto;font-variant-numeric:tabular-nums}

  /* ── Explore ── */
  #explore{margin-top:30px}
  .sec-title{font-family:'Plus Jakarta Sans',sans-serif;font-weight:800;font-size:1.3rem;letter-spacing:-.01em;margin:0 2px 14px}
  /* Match the classic home nav-cards exactly: centered, 24px radius, full-bleed
     cover, generous padding, bottom action link. */
  .acards{display:grid;grid-template-columns:repeat(auto-fill,minmax(255px,1fr));gap:20px;margin-bottom:34px}
  .acard,.dbtile{background:#fff;border-radius:24px;padding:32px 24px 26px;text-decoration:none;color:var(--ink);box-shadow:0 4px 24px #0000000a;transition:transform .2s,box-shadow .2s;text-align:center;display:flex;flex-direction:column}
  .acard:hover,.dbtile:hover{transform:translateY(-6px);box-shadow:0 16px 40px #00000014}
  .acard img{width:calc(100% + 48px);margin:-32px -24px 20px;height:140px;object-fit:cover;object-position:center top;display:block;border-radius:24px 24px 0 0;background:#1a0f24}
  .acard .ab{display:flex;flex-direction:column;flex:1}
  .acard .at,.dbtile .dt{font-family:'Plus Jakarta Sans',sans-serif;font-weight:800;font-size:1.08rem;margin-bottom:8px;letter-spacing:-.01em;overflow-wrap:anywhere}
  .acard .ad,.dbtile .dd{font-size:.82rem;color:var(--soft);line-height:1.55}
  .acard .adate{margin-top:10px;font-size:.7rem;color:var(--soft);font-weight:500;letter-spacing:.04em;text-transform:uppercase}
  .acard .ab::after{content:'Read →';margin-top:auto;padding-top:20px;font-size:.85rem;color:#9a7ab4;font-family:'Plus Jakarta Sans',sans-serif;font-weight:800;letter-spacing:.04em}
  .dbtiles{display:grid;grid-template-columns:repeat(auto-fill,minmax(255px,1fr));gap:20px}
  .dbtile .dd{flex:1}
  .dbtile .darrow{margin-top:auto;padding-top:20px;font-size:.85rem;color:#9a7ab4;font-weight:800;letter-spacing:.04em;font-family:'Plus Jakarta Sans',sans-serif}

  footer{text-align:center;padding:30px 18px;color:var(--faint);font-size:.74rem;font-weight:500;position:relative;z-index:1}

  @media (max-width:860px){
    .agrid{grid-template-columns:1fr}
    #rankings-panel{position:static}
    .wrap{padding:0 14px 48px}
  }
  @media (max-width:560px){
    .uiswitch{top:10px;right:10px;padding:5px 9px;gap:6px}
    .uiswitch .lbl{display:none}
    .podium{grid-template-columns:1fr}
    .plist{grid-template-columns:1fr}
    .mc-name{font-size:.88rem}
  }
  /* Phones: the team column gets too narrow for the region badge, which is
     flex:0 0 auto and overflows onto the form-dots column. Drop the badge
     (logos already signal region; it's on the full rankings page) and tighten
     the fixed columns so names + last-5 dots get breathing room. */
  @media (max-width:480px){
    .rhead,.rrow{grid-template-columns:22px 30px minmax(0,1fr) 64px 40px 52px;gap:7px}
    .rrow .rbadge{display:none}
    .rform{gap:4px}
  }

  /* ── Bottom data-refresh widget (mirrors the Modern Hub progress bar) ── */
  .refresh-sec{margin:46px auto 4px;text-align:center;max-width:560px}
  .refresh-divider{height:1px;background:var(--line);margin:0 0 26px}
  .refresh-btn{display:inline-flex;align-items:center;gap:9px;font-family:'DM Sans',sans-serif;font-size:.92rem;font-weight:700;color:#fff;background:linear-gradient(135deg,#1d1330,#3a1f55);border:none;border-radius:13px;padding:13px 22px;cursor:pointer;box-shadow:0 8px 26px rgba(29,19,48,.22);transition:transform .14s,box-shadow .2s,opacity .2s}
  .refresh-btn:hover:not(:disabled){transform:translateY(-2px);box-shadow:0 12px 30px rgba(29,19,48,.3)}
  .refresh-btn:disabled{opacity:.55;cursor:default}
  .refresh-btn .ricon{font-size:1.1rem;line-height:1}
  .refresh-sub{font-size:.76rem;color:var(--faint);font-weight:600;margin-top:11px;line-height:1.5}
  .rfp-card{background:#1a0a2e;border-radius:20px;padding:28px 30px;margin:20px auto 0;max-width:520px;text-align:center;display:none}
  .rfp-card.show{display:block}
  .rfp-label{color:rgba(232,213,245,.95);font-family:'Plus Jakarta Sans',sans-serif;font-weight:800;font-size:1.1rem;margin-bottom:5px}
  .rfp-msg{color:rgba(232,213,245,.55);font-size:.8rem;margin-bottom:20px;font-variant-numeric:tabular-nums}
  .rfp-track{height:9px;background:rgba(255,255,255,.08);border-radius:6px;overflow:hidden;margin-bottom:9px}
  .rfp-fill{height:100%;border-radius:6px;background:linear-gradient(90deg,#5b21b6,#7c3aed,#a78bfa,#c4b5fd);background-size:200% 100%;transition:width .6s cubic-bezier(.4,0,.2,1);width:0%;animation:rfpShimmer 1.8s linear infinite}
  @keyframes rfpShimmer{0%{background-position:200% 0}100%{background-position:-200% 0}}
  .rfp-fill.done{animation:none;width:100%!important;background:#c4b5fd}
  .rfp-card.err .rfp-fill{animation:none;background:#e06666}
  .rfp-pct{color:rgba(232,213,245,.5);font-size:.72rem;font-variant-numeric:tabular-nums}
  .rfp-log{margin-top:14px;text-align:left;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.7rem;color:rgba(232,213,245,.42);line-height:1.7}
  .rfp-log .ple{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  /* Floating "new data" pill — shown when a background auto-refresh finds new
     matches, so the page updates without a jarring auto-reload. */
  .update-pill{position:fixed;left:50%;bottom:22px;transform:translateX(-50%);z-index:500;display:inline-flex;align-items:center;gap:8px;font-family:'DM Sans',sans-serif;font-size:.86rem;font-weight:700;color:#fff;background:linear-gradient(135deg,#7c3aed,#5b21b6);border:none;border-radius:999px;padding:11px 20px;cursor:pointer;box-shadow:0 10px 30px rgba(124,58,237,.42);animation:pillIn .42s cubic-bezier(.22,1,.36,1)}
  .update-pill:hover{filter:brightness(1.08)}
  .update-pill:disabled{opacity:.7;cursor:default}
  .update-pill .ricon{font-size:1rem;line-height:1}
  @keyframes pillIn{from{opacity:0;transform:translate(-50%,22px)}to{opacity:1;transform:translate(-50%,0)}}
</style>
</head>
<body>
<div class="wrap">
  <!-- Top nav is the shared, app-wide Alpha bar injected by _inject_alpha_nav,
       so it is byte-identical on every page. Do not add a bespoke nav here.
       (Do not name the injected script file here — the injector guards on that
       string and would skip injection if it appeared in this page's body.) -->

  <header class="ahome">
    <div class="ahome-brand"><img class="ahome-logo" src="/logo.svg" alt="B">obo gg</div>
  </header>

  <div id="banner"></div>

  <div class="agrid">
    <div class="panel" id="matches-panel">
      <div class="phead"><div class="ptitle">Matches</div><a class="plink" href="/mapelo/modern/">Live hub &rarr;</a></div>
      <div class="seg-row">
        <div class="seg" id="match-seg"><button data-tab="upcoming" class="on">Upcoming</button><button data-tab="recent">Recent</button></div>
        <select id="match-region" class="mregion" title="Filter by region">
          <option value="All">All Regions</option>
          <option value="EMEA">EMEA</option>
          <option value="Americas">Americas</option>
          <option value="Pacific">Pacific</option>
          <option value="CN">China</option>
          <option value="International">International</option>
        </select>
      </div>
      <div id="match-body"></div>
    </div>
    <div class="panel" id="rankings-panel">
      <div class="phead"><div class="ptitle">BenPom Power Rankings</div><a class="plink" href="/mapelo/modern/">Full current rankings &rarr;</a></div>
      <div class="psub">Net rating &middot; top 15 currently</div>
      <div class="rhead"><div class="rh-r">#</div><div></div><div>Team</div><div class="rh-form">Last 5</div><div class="rh-pct">Win%</div><div class="rh-rat">Rating</div></div>
      <div id="rankings-body"></div>
      <div class="rlegend"><span><i class="w"></i><i class="l" style="background:#e6b0b0"></i> last 5 &middot; oldest&rarr;newest &middot; click a dot for the match</span><span>Win% = expected map win vs an average team</span></div>
    </div>
  </div>

  <div class="panel players" id="players-panel">
    <div class="phead"><div class="ptitle">Player Leaders</div>
      <div class="pl-minrnd" title="Minimum rounds played to qualify">
        <span class="pl-minrnd-lab">Min rounds</span>
        <input type="range" id="minRndSlider" min="0" max="200" step="10" value="50" aria-label="Minimum rounds played">
        <span class="pl-minrnd-val" id="minRndVal">50+</span>
      </div>
      <a class="plink" id="players-full-link" href="/vct/">Full Leaderboards &rarr;</a></div>
    <div class="psub" id="players-sub"></div>
    <div id="players-body"></div>
  </div>

  <div class="panel" id="records-panel">
    <div class="phead"><div class="ptitle">Recent VCT Records</div><a class="plink" href="/highs/">Full VCT Records &rarr;</a></div>
    <div class="psub" id="records-sub"></div>
    <div id="records-body"></div>
  </div>

  <div id="explore">
    <div class="phead" style="margin-bottom:14px"><div class="sec-title" style="margin:0">Recent Articles</div><a class="plink" href="/articles/">View all articles &rarr;</a></div>
    <div class="acards">
      <a class="acard" href="/articles/greatest-prime/"><img src="/aspas25corrode.jpg" alt=""><div class="ab"><div class="at">The Greatest Prime in VCT History Isn't a Debate</div><div class="ad">Aspas at Champions Paris towers over VCT history, including your favorite player.</div><div class="adate">Jun 28, 2026</div></div></a>
      <a class="acard" href="/articles/masters-london-playoffs-preview/"><img src="/chronlondon.jpg" alt=""><div class="ab"><div class="at">Masters London Playoffs Preview</div><div class="ad">A brief statistical glimpse into the final stage of Masters London.</div><div class="adate">Jun 10, 2026</div></div></a>
      <a class="acard" href="/articles/masters-london-preview/"><img src="/prxpacstage1win.jpg" alt=""><div class="ab"><div class="at">Masters London Tournament Preview</div><div class="ad">Paper Rex's (un)inevitability, Neon nerfs, China's resurgence, and other bold predictions.</div><div class="adate">Jun 2, 2026</div></div></a>
      <a class="acard" href="/articles/over-underperformers/"><img src="/patmen.jpg" alt=""><div class="ab"><div class="at">Overperforming in VCT: Who's Doing It?</div><div class="ad">Surfacing the players outperforming (or underperforming) their team.</div><div class="adate">May 4, 2026</div></div></a>
    </div>
    <div class="sec-title">Stats &amp; Databases</div>
    <div class="dbtiles">
      <a class="dbtile" href="/highs/"><div class="dt">All-Time Highs &amp; Lows</div><div class="dd">The best and worst individual performances across VCT.</div><div class="darrow">Open &rarr;</div></a>
      <a class="dbtile" href="/vct/"><div class="dt">Event Leaderboards</div><div class="dd">Per-event player leaderboards, percentiles, and best matches.</div><div class="darrow">Open &rarr;</div></a>
      <a class="dbtile" href="/mapelo/pythagorean/"><div class="dt">VCT Pythagorean</div><div class="dd">A Pythagorean win% model hand-tuned for VCT's domestic strength.</div><div class="darrow">Open &rarr;</div></a>
      <a class="dbtile" href="/match-data/"><div class="dt">Match Data Explorer</div><div class="dd">Round-by-round outcomes, economy, clutches, and kill matrices from VLR match pages.</div><div class="darrow">Open &rarr;</div></a>
    </div>
  </div>

  <!-- Bottom: manual data refresh. Runs the SAME RefreshLiveData pipeline as the
       Modern Hub (new matches -> BenPom ratings -> win probabilities -> records
       -> leaders), shows its live progress, then reloads with the fresh data. -->
  <div class="refresh-sec" id="refreshSec">
    <div class="refresh-divider"></div>
    <button class="refresh-btn" id="refreshBtn" type="button" onclick="startRefresh()">
      <span class="ricon">&#x21bb;</span> Check for new matches &amp; refresh
    </button>
    <div class="refresh-sub">Scrapes new results and recomputes BenPom ratings, win probabilities, recent records &amp; leaders.</div>
    <div class="rfp-card" id="rfpCard">
      <div class="rfp-label" id="rfpLabel">Refreshing VCT data</div>
      <div class="rfp-msg" id="rfpMsg">Starting&hellip;</div>
      <div class="rfp-track"><div class="rfp-fill" id="rfpFill"></div></div>
      <div class="rfp-pct" id="rfpPct">0%</div>
      <div class="rfp-log" id="rfpLog"></div>
    </div>
  </div>
</div>
<div id="dotTooltip"><div id="dotTooltipContent" class="popup-inner"></div></div>
<footer>Like my work? Tips are appreciated! <a href="https://ko-fi.com/bobovct" target="_blank" rel="noopener" style="color:inherit;text-decoration:underline">ko-fi.com/bobovct</a></footer>
<script>
var DATA = {{ data_json | safe }};
var COL = DATA.colors || {}, LOGOS = DATA.logos || {};
var REGION_COLOR = {EMEA:'var(--r-emea)',Americas:'var(--r-amer)',Pacific:'var(--r-pac)',CN:'var(--r-cn)',International:'var(--r-int)'};

function esc(s){return String(s==null?'':s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');}
function col(org){return COL[org]||'#8a8a8a';}
function regColor(r){return REGION_COLOR[r]||'var(--r-int)';}
function regClass(r){return ({EMEA:'emea',Americas:'americas',Pacific:'pacific',CN:'cn'})[r]||'int';}
function fmtR(r){if(r==null||r==='')return '';var n=Number(r);return (n>=0?'+':'')+n.toFixed(2);}
function fmtFmt(f){return f==='bo5_gf'?'Bo5 GF':String(f||'').toUpperCase().replace('BO','Bo');}
var MO=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
function shortDate(d){if(!d)return '';var p=String(d).split('-');if(p.length<3)return d;return MO[+p[1]-1]+' '+(+p[2]);}
function logoOrInit(org,logoCls,initCls){
  var f=LOGOS[org];
  if(f)return '<img class="'+logoCls+'" src="/logos/'+esc(f)+'" alt="'+esc(org)+'" loading="lazy">';
  return '<div class="'+initCls+'" style="background:'+col(org)+'">'+esc(String(org||'').slice(0,3))+'</div>';
}
function imgFail(img){img.style.display='none';var n=img.nextElementSibling;if(n)n.style.display='flex';}
function goClassic(){try{localStorage.setItem('bobo_ui','classic');}catch(e){}location.href='/';}

/* ── Banner + season timeline ── */
function renderBanner(){
  var e=DATA.event, le=DATA.last_event;
  var pillCls='', pillTxt='Upcoming', title='VCT 2026', sub='Season in progress';
  if(e && e.status==='live'){pillCls='live';pillTxt='Live now';title=e.label;sub='In progress &middot; '+shortDate(e.start)+' – '+shortDate(e.end);}
  else if(e && e.status==='upcoming'){pillTxt=(e.days!=null?('Starts in '+e.days+' day'+(e.days===1?'':'s')):'Upcoming');title=e.label;sub='Begins '+shortDate(e.start)+(le?' &middot; last: '+esc(le.label)+' ('+shortDate(le.end)+')':'');}
  else if(e && e.status==='recent'){pillTxt='Most recent';title=e.label;sub='Ended '+shortDate(e.end);}
  var tl=(DATA.season||[]).map(function(s){
    var mark=s.status==='done'?'&#10003;':'';
    return '<div class="tnode '+s.status+'"><span class="tdot">'+mark+'</span><span class="tlbl">'+esc(s.label)+'</span><span class="tdate">'+shortDate(s.start)+'</span></div>';
  }).join('');
  var pillHtml=(e && e.status==='live')?'':'<div class="epill '+pillCls+'"><span class="dot"></span>'+pillTxt+'</div>';
  document.getElementById('banner').innerHTML='<div class="ebanner">'
    +'<div class="ebanner-l">'
    +pillHtml
    +'<div class="etitle">'+esc(title)+'</div>'
    +'<div class="esub">'+sub+'</div>'
    +'<div class="ebtns"><a class="ebtn" href="/mapelo/modern/">Open live hub &rarr;</a></div>'
    +(tl?'<div class="timeline">'+tl+'</div>':'')
    +'</div>'
    +'<a class="ebanner-r" href="/articles/greatest-prime/">'
    +'<img src="/aspas25corrode.jpg" alt="">'
    +'<div class="ead-tag">Latest Article</div>'
    +'<div class="ead-title">The Greatest Prime in VCT History Isn&rsquo;t a Debate</div>'
    +'<div class="ead-link">Read &rarr;</div>'
    +'</a>'
    +'</div>';
}

/* ── Matches (paginated, 4 per page) ── */
var PER_PAGE=4, PAGE={upcoming:0,recent:0}, CUR_TAB='upcoming', MATCH_REGION='All';
var TEAM_HREF=function(org){return '/team/'+encodeURIComponent(org);};
function probBar(pa){pa=Math.max(0,Math.min(1,pa||0));
  var a=Math.round(pa*100), b=100-a;
  return '<div class="mc-bar"><span class="mc-seg a" style="width:'+(pa*100)+'%">'+a+'%</span>'
    +'<span class="mc-seg b" style="width:'+((1-pa)*100)+'%">'+b+'%</span></div>';}
function teamSide(org,rating,side,link){
  var inner=logoOrInit(org,'mc-logo','mc-init')+'<div><div class="mc-name">'+esc(org)+'</div><div class="mc-rat">'+fmtR(rating)+'</div></div>';
  if(link)return '<a class="mc-team '+side+'" href="'+TEAM_HREF(org)+'" title="'+esc(org)+' profile">'+inner+'</a>';
  return '<div class="mc-team '+side+'">'+inner+'</div>';}   // non-link (upcoming card expands instead)
function _scoreParts(s){var g=String(s==null?'':s).match(/(\\d+)\\D+(\\d+)/);return g?[+g[1],+g[2]]:null;}
// UTC "YYYY-MM-DD HH:MM:SS" -> viewer's local time with tz label, e.g. "1:00 PM EDT".
function fmtLocalTime(utc){
  if(!utc)return '';
  var iso=String(utc).trim().replace(' ','T');
  if(!/[zZ]|[+-]\\d\\d:?\\d\\d$/.test(iso))iso+='Z';
  var d=new Date(iso);
  if(isNaN(d.getTime()))return '';
  return d.toLocaleTimeString([],{hour:'numeric',minute:'2-digit',timeZoneName:'short'});
}
function _resTeam(org,side,win,wcol){
  var nm='<span class="mc-rname">'+esc(org)+'</span>';   // names stay default (black); winner shown via weight, loser dimmed
  var wl='<span class="mc-wl '+(win?'w':'l')+'">'+(win?'W':'L')+'</span>';
  // DOM order stays [logo, name, badge] for both sides — .mc-rteam.b's
  // existing flex-direction:row-reverse mirrors it visually (same pattern
  // teamSide() already relies on), so the badge lands on the outer edge.
  return '<a class="mc-rteam '+side+(win?' win':'')+'" href="'+TEAM_HREF(org)+'" title="'+esc(org)+' profile">'
    +logoOrInit(org,'mc-logo','mc-init')+nm+wl+'</a>';}
function recentCard(m){
  var pa=(m.win_prob_a!=null?m.win_prob_a:0), aWon=m.actual_winner==='a', favA=pa>=0.5;
  var favWon=(favA&&aWon)||(!favA&&!aWon), resCls=favWon?'win':'upset', resTxt=favWon?'Final':'Upset';
  // Winner first: the completed result is the focal point of the card.
  var wOrg=aWon?m.org_a:m.org_b, lOrg=aWon?m.org_b:m.org_a, wCol=col(wOrg);
  var sp=_scoreParts(m.actual_score);
  var scoreTxt=sp?(Math.max(sp[0],sp[1])+'&ndash;'+Math.min(sp[0],sp[1])):esc(m.actual_score||'');
  // Pre-match ratings/odds must follow the SAME winner-left ordering as the
  // result row above — swap them (and the win-prob) whenever org_b won,
  // otherwise the numbers below silently belong to the opposite team from
  // what's shown on top.
  var wRat=aWon?m.rating_a:m.rating_b, lRat=aWon?m.rating_b:m.rating_a;
  var wProb=aWon?pa:(1-pa);
  return '<div class="mcard rec">'
    +'<div class="mc-meta"><span class="mtag">'+esc(m.event||'')+'</span><span>'+fmtFmt(m.format)+'</span><span>&middot;</span><span>'+shortDate(m.date)+'</span>'+(m.time?'<span>&middot;</span><span>'+fmtLocalTime(m.time)+'</span>':'')
    +'<span class="mc-res '+resCls+'" style="margin-left:auto">'+resTxt+'</span></div>'
    +'<div class="mc-result">'+_resTeam(wOrg,'a',true,wCol)
    +'<div class="mc-rscore">'+scoreTxt+'</div>'
    +_resTeam(lOrg,'b',false,wCol)+'</div>'
    +'<div class="mc-pre"><div class="mc-pre-lbl">Pre-match &middot; '+esc(wOrg)+' vs '+esc(lOrg)+'</div>'
    +'<div class="mc-prow"><span class="mc-prat">'+fmtR(wRat)+'</span>'+probBar(wProb)+'<span class="mc-prat">'+fmtR(lRat)+'</span></div></div></div>';}
function upcomingCard(m,i){
  var pa=m.win_prob_a;
  return '<div class="mcard upc" data-ua="'+esc(m.org_a)+'" data-ub="'+esc(m.org_b)+'" data-up="'+(pa!=null?pa:'')+'" data-fmt="'+esc(m.format||'')+'" data-i="'+i+'" data-date="'+esc(m.date||'')+'">'
    +'<div class="mc-meta"><span class="mtag">'+esc(m.event||'')+'</span><span>'+fmtFmt(m.format)+'</span>'+(m.date?'<span>&middot;</span><span>'+shortDate(m.date)+'</span>':'')+(m.datetime?'<span>&middot;</span><span>'+fmtLocalTime(m.datetime)+'</span>':'')+'</div>'
    +'<div class="mc-row">'+teamSide(m.org_a,m.rating_a,'a',true)
    +'<div class="mc-win"><span class="mc-vs">'+(pa!=null?'VS':'–')+'</span><span class="vs">PROJ</span></div>'
    +teamSide(m.org_b,m.rating_b,'b',true)+'</div>'+(pa!=null?probBar(pa):'')
    +'<div class="mc-details"><div class="mc-details-inner" id="ud'+i+'"></div></div>'
    +'<div class="mc-expand-hint">&#9656; tap for analysis</div></div>';}
function pager(tab,total){
  var pages=Math.ceil(total/PER_PAGE); if(pages<=1)return '';
  var p=PAGE[tab];
  return '<div class="mc-pager"><button '+(p<=0?'disabled':'')+' onclick="matchPage(-1)">&lsaquo;</button>'
    +'<span>'+(p+1)+' / '+pages+'</span>'
    +'<button '+(p>=pages-1?'disabled':'')+' onclick="matchPage(1)">&rsaquo;</button></div>';}
function _matchList(tab){
  var list=tab==='upcoming'?(DATA.upcoming||[]):(DATA.recent||[]);
  if(MATCH_REGION!=='All') list=list.filter(function(m){return m.region===MATCH_REGION;});
  return list;}
function matchPage(d){
  var list=_matchList(CUR_TAB);
  var pages=Math.max(1,Math.ceil(list.length/PER_PAGE));
  PAGE[CUR_TAB]=Math.max(0,Math.min(pages-1,PAGE[CUR_TAB]+d));
  renderMatches(CUR_TAB);
  var mp=document.getElementById('matches-panel'); if(mp)mp.scrollIntoView({block:'nearest'});}
function renderMatches(tab){
  CUR_TAB=tab;
  var body=document.getElementById('match-body');
  var list=_matchList(tab);
  if(!list.length){
    if(MATCH_REGION!=='All'){
      body.innerHTML='<div class="empty">No '+esc(MATCH_REGION)+' matches '+(tab==='upcoming'?'scheduled':'recently')+'.<br>Try <b>All Regions</b>.</div>';
    } else if(tab==='upcoming'){var ne=DATA.next_event;
      body.innerHTML='<div class="empty">No matches scheduled right now.'+(ne?'<br><b>'+esc(ne.label)+'</b> starts '+shortDate(ne.start)+(ne.days!=null?' ('+ne.days+' days)':'')+'.':'')+'<br>Check <b>Recent</b> for the latest results.</div>';}
    else body.innerHTML='<div class="empty">No recent matches.</div>';
    return;}
  var p=Math.min(PAGE[tab], Math.max(0,Math.ceil(list.length/PER_PAGE)-1));
  PAGE[tab]=p;
  var start=p*PER_PAGE, card=(tab==='upcoming')?upcomingCard:recentCard;
  body.innerHTML=list.slice(start,start+PER_PAGE).map(card).join('')+pager(tab,list.length);
  body.scrollTop=0; if(typeof _updateMatchCap==='function')_updateMatchCap();}   // fresh page: no card open → clear cap

/* ── Upcoming-card in-place analysis (accordion) ── */
var TEAM_CACHE={};
function _fetchTeam(org){
  if(TEAM_CACHE[org])return Promise.resolve(TEAM_CACHE[org]);
  return fetch('/api/team/'+encodeURIComponent(org)).then(function(r){return r.json();})
    .then(function(j){TEAM_CACHE[org]=j;return j;}).catch(function(){return null;});
}
function _h2hCol(t,org,side){
  if(!t)return '<div class="h2h-col '+side+'"><div class="h2h-org">'+esc(org)+'</div><div class="h2h-na">no data</div></div>';
  var best=(t.best_maps||[]).slice(0,3).map(function(mp){
    return '<div class="h2h-map"><span>'+esc(mp.map)+'</span><b class="'+(mp.rating>=0?'pos':'neg')+'">'+fmtR(mp.rating)+'</b></div>';}).join('');
  return '<div class="h2h-col '+side+'">'
    +'<a class="h2h-org" href="/team/'+encodeURIComponent(org)+'">'+logoOrInit(org,'h2h-logo','h2h-init')+'<span>'+esc(org)+'</span></a>'
    +'<div class="h2h-meta"><span class="rbadge '+regClass(t.region)+'">'+esc(t.region||'')+'</span><span class="h2h-rank">#'+t.rank+'</span></div>'
    +'<div class="h2h-rat '+(t.rating>=0?'pos':'neg')+'">'+fmtR(t.rating)+'<span>BenPom</span></div>'
    +'<div class="h2h-form">'+formDots(t.form)+'</div>'
    +'<div class="h2h-mtitle">Best maps</div>'+best+'</div>';
}
function renderH2H(id,org_a,org_b,pa,date){
  var el=document.getElementById(id); if(!el)return;
  el.innerHTML='<div class="h2h-load">Loading analysis…</div>';
  // Deep-link straight to this match's card on the Modern Hub's Upcoming
  // Matches tab — see the matching #panel=b deep-link IIFE at the bottom of
  // MapElo.py's Modern Hub <script> block.
  var simHref='/mapelo/modern/#panel=b&a='+encodeURIComponent(org_a)+'&b='+encodeURIComponent(org_b)+(date?'&date='+encodeURIComponent(date):'');
  Promise.all([_fetchTeam(org_a),_fetchTeam(org_b)]).then(function(res){
    var A=res[0],B=res[1], pct=(pa!=null)?Math.round(pa*100):null;
    el.innerHTML=(pct!=null?'<div class="h2h-head"><b>'+esc(org_a)+'</b> '+pct+'%&nbsp;&middot;&nbsp;'+(100-pct)+'% <b>'+esc(org_b)+'</b><div class="h2h-sub">projected series win</div></div>':'')
      +'<div class="h2h-grid">'+_h2hCol(A,org_a,'a')+'<div class="h2h-vs">VS</div>'+_h2hCol(B,org_b,'b')+'</div>'
      +'<a class="h2h-simlink" href="'+simHref+'">Full veto sim &amp; per-map odds &rarr;</a>';
    // Make the form dots interactable like Power Rankings (hover = BenPom match card).
    el.addEventListener('mouseover',function(e){var d=e.target.closest&&e.target.closest('.fdot[data-mi]');if(d)_showDotTip(d);});
    el.addEventListener('mouseout',function(e){var d=e.target.closest&&e.target.closest('.fdot[data-mi]');if(d)_hideDotTip();});
  });
}
function _onMatchClick(e){
  if(e.target.closest('a'))return;                 // team links / sim link navigate
  var card=e.target.closest('.mcard.upc'); if(!card)return;
  var open=card.classList.toggle('open');
  var hint=card.querySelector('.mc-expand-hint');
  if(hint)hint.innerHTML=open?'&#9662; close analysis':'&#9656; tap for analysis';
  if(open && !card.dataset.loaded){
    card.dataset.loaded='1';
    var up=card.dataset.up; renderH2H(card.querySelector('.mc-details-inner').id, card.dataset.ua, card.dataset.ub, up!==''?parseFloat(up):null, card.dataset.date);
  }
  if(open)setTimeout(function(){card.scrollIntoView({block:'nearest',behavior:'smooth'});},60);
  setTimeout(_updateMatchCap,380);   // cap + fade only while expanded
}

/* ── Rankings ── */
var DOT_MATCHES=[];   // registry: index -> full match object, for hover cards
function formDots(form,nopad){
  // form is newest-first; show oldest→newest. Padded to 5 (empty on the old
  // side) for the rankings grid alignment, but NOT when nopad is set (h2h, where
  // an empty slot would leave a stray grey dot). Hover shows the BenPom card.
  var arr=(form||[]).slice().reverse();
  if(!nopad){ while(arr.length<5)arr.unshift(null); arr=arr.slice(-5); }
  return '<div class="rform">'+arr.map(function(m,i){
    var newest=(i===arr.length-1);
    if(!m)return '<span class="fdot empty"></span>';
    var idx=DOT_MATCHES.push(m)-1;
    var cls='fdot '+(m.result==='W'?'w':'l')+(newest?' new':'');
    if(m.match_id)return '<a class="'+cls+'" data-mi="'+idx+'" href="https://www.vlr.gg/'+esc(m.match_id)+'" target="_blank" rel="noopener"></a>';
    return '<span class="'+cls+'" data-mi="'+idx+'"></span>';
  }).join('')+'</div>';}
// BenPom match-hover card — copied from the Modern Hub chart's dot tooltip.
function _matchTooltipHTML(m, won){
  var org=won?m.winner:m.loser, opp=won?m.loser:m.winner;
  var d=won?m.winner_delta:m.loser_delta, rat=won?m.winner_after:m.loser_after;
  var dStr=((d||0)>=0?'+':'')+Number(d||0).toFixed(2);
  var evt=(DATA.event_labels||{})[m.event_id]||m.event_id||'';
  var raw=String(m.series_score||m.score||'0-0').split('-');
  var disp=won?(m.series_score||m.score):(raw[1]+'-'+raw[0]);
  var rows=(m.maps||[]).map(function(mp){
    var mw=mp.winner===org, a=mw?mp.wr:mp.lr, b=mw?mp.lr:mp.wr, diff=a-b;
    return '<tr><td class="popup-map-name">'+esc(mp.map)+'</td>'
      +'<td class="popup-map-score '+(mw?'w':'l')+'">'+a+'</td>'
      +'<td class="popup-map-score '+(mw?'l':'w')+'">'+b+'</td>'
      +'<td class="popup-map-diff">'+(diff>=0?'+':'')+diff+'</td></tr>';
  }).join('');
  return (evt?'<div class="popup-event-label">'+esc(evt)+'</div>':'')
    +'<div class="popup-teams"><div class="popup-team-block">'
    +'<img class="popup-logo" src="/static/logos/'+esc(org)+'.png" onerror="this.style.display=\\'none\\'" alt="'+esc(org)+'">'
    +'<span class="popup-team-name">'+esc(org)+'</span></div>'
    +'<div class="popup-score-block"><span class="popup-score '+(won?'w':'l')+'">'+esc(disp)+'</span><span class="popup-vs-label">series</span></div>'
    +'<div class="popup-team-block"><img class="popup-logo" src="/static/logos/'+esc(opp)+'.png" onerror="this.style.display=\\'none\\'" alt="'+esc(opp)+'">'
    +'<span class="popup-team-name">'+esc(opp)+'</span></div></div>'
    +'<div class="popup-date">'+esc(m.date)+'</div>'
    +'<div class="popup-delta">BenPom '+Number(rat||0).toFixed(2)+' &nbsp;(<span class="'+((d||0)>=0?'pos':'neg')+'">'+dStr+'</span>)</div>'
    +(rows?'<table class="popup-maps-table"><thead><tr><th>Map</th><th>'+esc(org)+'</th><th>'+esc(opp)+'</th><th>Diff</th></tr></thead><tbody>'+rows+'</tbody></table>':'');
}
function _showDotTip(el){
  var idx=el.getAttribute('data-mi'); if(idx==null)return;
  var m=DOT_MATCHES[+idx]; if(!m)return;
  var tt=document.getElementById('dotTooltip');
  document.getElementById('dotTooltipContent').innerHTML=_matchTooltipHTML(m, m.result==='W');
  tt.style.visibility='hidden'; tt.classList.add('visible');
  // Consistent placement: to the RIGHT of the dot (into the page margin),
  // vertically centered. Only flips left if it can't fit on the right. Never
  // flips above/below, so it doesn't jump around between dots.
  var r=el.getBoundingClientRect(), w=tt.offsetWidth, h=tt.offsetHeight, gap=14;
  var left=r.right+gap;
  if(left+w>window.innerWidth-6) left=r.left-w-gap;
  var top=r.top+r.height/2-h/2;
  top=Math.max(6, Math.min(top, window.innerHeight-h-6));
  tt.style.left=left+'px'; tt.style.top=top+'px'; tt.style.visibility='';
}
function _hideDotTip(){document.getElementById('dotTooltip').classList.remove('visible');}
function rankRow(t){
  var href=TEAM_HREF(t.org);
  return '<div class="rrow"><div class="rrank">'+t.rank+'</div>'
    +'<a class="rcell-link" href="'+href+'" title="'+esc(t.org)+' profile">'+logoOrInit(t.org,'rlogo','rinit')+'</a>'
    +'<a class="rteam-link" href="'+href+'" title="'+esc(t.org)+' profile"><span class="rorg">'+esc(t.org)+'</span><span class="rbadge '+regClass(t.region)+'">'+esc(t.region||'')+'</span></a>'
    +formDots(t.form)
    +'<div class="rpct">'+t.winpct+'%</div>'
    +'<div class="rrat" style="color:'+(t.rating>=0?'#16121d':'#c0392b')+'">'+fmtR(t.rating)+'</div></div>';}
function renderRankings(){
  DOT_MATCHES=[];
  var body=document.getElementById('rankings-body');
  body.innerHTML=(DATA.rankings||[]).slice(0,15).map(rankRow).join('');
  body.addEventListener('mouseover',function(e){var d=e.target.closest&&e.target.closest('.fdot[data-mi]');if(d)_showDotTip(d);});
  body.addEventListener('mouseout',function(e){var d=e.target.closest&&e.target.closest('.fdot[data-mi]');if(d)_hideDotTip();});
}

/* ── Players (podium + list) ── */
function avatar(p,cls,phcls){
  var img=p.headshot?'<img class="'+cls+'" style="--ring:'+col(p.org)+'" src="'+esc(p.headshot)+'" loading="lazy" onerror="imgFail(this)">':'';
  var ph='<div class="'+phcls+'" style="background:'+col(p.org)+(p.headshot?';display:none':'')+'">'+esc(String(p.name||'').slice(0,2))+'</div>';
  return img+ph;}
function plRow(p,i,stat){
  var href='/vct/player?profile='+encodeURIComponent(p.profile||'')
    +'&stat='+encodeURIComponent(stat||'')
    +'&event='+encodeURIComponent(DATA.players_event_id||'');
  return '<a class="plr" href="'+href+'" title="'+esc(p.name)+' player card">'
    +'<span class="plr-n">'+(i+1)+'</span>'+avatar(p,'plr-av','plr-av-ph')
    +'<span class="plr-info"><span class="plr-name">'+esc(p.name)+'</span><span class="plr-meta">'+esc(p.org)+' &middot; '+esc(p.region)+'</span></span>'
    +'<span class="plr-val">'+esc(p.value)+'</span></a>';}
var MIN_RND=50;   // min-rounds slider value; leaders re-filter client-side
function renderPlayers(){
  document.getElementById('players-sub').textContent=DATA.players_event?('Leaders · '+DATA.players_event):'';
  var pfl=document.getElementById('players-full-link');
  if(pfl)pfl.href='/vct/'+(DATA.players_event_id?('?event='+encodeURIComponent(DATA.players_event_id)):'');
  var ss=DATA.player_stats||[];
  document.getElementById('players-body').innerHTML = ss.length
    ? '<div class="pl-grid">'+ss.map(function(s){
        var lbHref='/vct/ranking/'+encodeURIComponent(s.stat)
          +'?event='+encodeURIComponent(DATA.players_event_id||'')+'&region=All';
        var all=s.leaders||[];
        var picked=all.filter(function(p){return p.rnd==null||p.rnd>=MIN_RND;}).slice(0,5);
        if(!picked.length)picked=all.slice(0,5);   // nobody qualifies yet — show unfiltered
        return '<div class="pl-card"><a class="pl-stat" href="'+lbHref+'" title="Full '+esc(s.label)+' leaderboard">'
          +esc(s.label)+'<span class="pl-arrow">&rarr;</span></a>'
          +picked.map(function(p,i){return plRow(p,i,s.stat);}).join('')+'</div>';
      }).join('')+'</div>'
    : '<div class="empty">No player data.</div>';}
(function(){
  var s=document.getElementById('minRndSlider'),v=document.getElementById('minRndVal');
  if(!s)return;
  function upd(){
    MIN_RND=+s.value; v.textContent=s.value+'+';
    var pct=(s.value-s.min)/(s.max-s.min)*100;
    s.style.background='linear-gradient(90deg,#7c4dd6 '+pct+'%,#e4d9f6 '+pct+'%)';
  }
  s.addEventListener('input',function(){upd();renderPlayers();});
  upd();
})();
function recHref(r){
  return '/highs/?direction='+encodeURIComponent(r.direction||'high')
    +'&stat='+encodeURIComponent(r.stat||'')
    +'&format='+encodeURIComponent(r.fmt||'')
    +'&context='+encodeURIComponent(r.context||'all')
    +'&year=all';
}
function recCard(r){
  var pa={headshot:r.headshot,org:r.org,name:r.player};
  var vs=r.opp?'<span class="rec-vs">vs '+logoOrInit(r.opp,'rec-tlogo','rec-tinit')+esc(r.opp)+'</span>':'';
  var mp=r.map_name?'<span class="rec-map">'+esc(r.map_name)+'</span>':'';
  var ev=r.event?'<span class="rec-ev">'+esc(r.event)+(r.date?' &middot; '+esc(shortDate(r.date)):'')+'</span>'
       :(r.date?'<span class="rec-ev">'+esc(shortDate(r.date))+'</span>':'');
  return '<a class="rec-card" href="'+esc(recHref(r))+'" target="_blank" rel="noopener" title="'+esc(r.player)+' — '+esc(r.desc)+'">'
    +'<span class="rec-rankbadge">#'+r.rank+'</span>'
    +avatar(pa,'rec-av','rec-av-ph')
    +'<span class="rec-info">'
      +'<span class="rec-name">'+esc(r.player)+(r.org?logoOrInit(r.org,'rec-tlogo','rec-tinit'):'')+'</span>'
      +'<span class="rec-desc">'+esc(r.desc)+'</span>'
      +'<span class="rec-foot">'+vs+ev+mp+'</span>'
    +'</span>'
    +'<span class="rec-val">'+esc(r.value)+'</span>'
  +'</a>';}
var _recTimer=null;
function renderRecords(){
  var rs=DATA.records||[];
  var sub=document.getElementById('records-sub');
  if(sub)sub.textContent=rs.length?'Latest performances that cracked an all-time top 50':'';
  var body=document.getElementById('records-body');
  if(!rs.length){body.innerHTML='<div class="empty">No recent record-setting performances.</div>';return;}
  // Gallery: cards rendered twice so we glide one step at a time and loop
  // seamlessly. Steady auto-advance + prev/next controls, pauses on hover.
  var cards=rs.map(recCard).join('');
  body.innerHTML='<div class="rec-wrap">'
    +'<button class="rec-nav rec-prev" type="button" aria-label="Previous record">&#8249;</button>'
    +'<div class="rec-vp"><div class="rec-track" id="rec-track">'+cards+cards+cards+'</div></div>'
    +'<button class="rec-nav rec-next" type="button" aria-label="Next record">&#8250;</button>'
    +'</div>';
  _recSlideshow(rs.length);}
function _recSlideshow(n){
  if(_recTimer){clearTimeout(_recTimer);_recTimer=null;}
  var track=document.getElementById('rec-track'); if(!track)return;
  // The track holds 3 copies of the cards (indices 0..3n-1); we keep the visible
  // index in the MIDDLE copy [n, 2n) so there's always a full copy of runway in
  // either direction. Wrapping is a synchronous, no-transition jump to the
  // identical-looking middle position — so manual clicks and the auto timer can
  // never drift the target or fight a half-finished transition.
  var wrap=track.closest('.rec-wrap'), i=n, STEP=313, hover=false;
  function measure(){var c=track.children[0];if(c){var cs=getComputedStyle(c);
    STEP=Math.round(c.getBoundingClientRect().width+(parseFloat(cs.marginRight)||0));}}
  function apply(anim){
    track.style.transition=anim?'transform .55s cubic-bezier(.4,.02,.2,1)':'none';
    track.style.transform='translateX('+(-i*STEP)+'px)';}
  function move(dir){                       // dir>0 slides cards LEFT (right-to-left, auto dir)
    measure();
    if(i<n){ i+=n; apply(false); void track.offsetWidth; }
    else if(i>=2*n){ i-=n; apply(false); void track.offsetWidth; }
    i+=dir; apply(true);
  }
  function schedule(){ if(_recTimer)clearTimeout(_recTimer);
    _recTimer=window.setTimeout(function(){ if(!hover)move(1); schedule(); }, 3800); }  // slides right-to-left
  if(wrap){
    wrap.addEventListener('mouseenter',function(){hover=true;});
    wrap.addEventListener('mouseleave',function(){hover=false;});
    var nb=wrap.querySelector('.rec-next'), pb=wrap.querySelector('.rec-prev');
    if(n<2){ if(nb)nb.style.display='none'; if(pb)pb.style.display='none'; }
    else{
      // Every manual step restarts the auto timer from full, so a click never
      // triggers an immediate auto-advance on top of it.
      if(nb)nb.addEventListener('click',function(e){e.preventDefault();e.stopPropagation();move(1);schedule();});
      if(pb)pb.addEventListener('click',function(e){e.preventDefault();e.stopPropagation();move(-1);schedule();});
    }
  }
  measure(); apply(false);
  if(n>=2) schedule();}

/* ── Init ── */
try{localStorage.setItem('bobo_ui','alpha');}catch(e){}   // remember the choice
renderBanner();renderRankings();renderPlayers();renderRecords();
document.getElementById('match-body').addEventListener('click',_onMatchClick);  // upcoming expand
var startTab='upcoming';   // matches default to Upcoming
document.querySelectorAll('#match-seg button').forEach(function(b){
  b.classList.toggle('on',b.dataset.tab===startTab);
  b.addEventListener('click',function(){document.querySelectorAll('#match-seg button').forEach(function(x){x.classList.remove('on');});b.classList.add('on');PAGE[b.dataset.tab]=0;renderMatches(b.dataset.tab);});
});
var _mrSel=document.getElementById('match-region');
if(_mrSel)_mrSel.addEventListener('change',function(){MATCH_REGION=this.value;PAGE.upcoming=0;PAGE.recent=0;renderMatches(CUR_TAB);});
renderMatches(startTab);

// Size the Matches pages so the panel is the same height as Power Rankings
// (desktop side-by-side only). Measures a real card + the panel chrome, then
// picks how many cards fit the rankings height.
// Only cap + fade the match-body when a card is EXPANDED (so the panel doesn't
// grow past the rankings). On the normal paginated view there's NO cap/fade.
function _updateMatchCap(){
  var mb=document.getElementById('match-body'); if(!mb)return;
  var open=mb.querySelector('.mcard.open');
  if(!(open && window.innerWidth>860)){ mb.style.maxHeight=''; mb.classList.remove('capped'); return; }
  var rp=document.getElementById('rankings-panel'); if(!rp)return;
  var avail=Math.max(260, rp.getBoundingClientRect().bottom - mb.getBoundingClientRect().top);
  mb.style.maxHeight=Math.round(avail)+'px';
  mb.classList.toggle('capped', mb.scrollHeight-mb.clientHeight>4);
}
function fitMatchesToRankings(){
  var mb=document.getElementById('match-body');
  if(window.innerWidth<=860){ if(mb)mb.style.maxHeight=''; return; }
  var rp=document.getElementById('rankings-panel'), mp=document.getElementById('matches-panel');
  if(!rp||!mp||!mb) return;
  var card=mb.querySelector('.mcard'); if(!card) return;
  var cardH=card.getBoundingClientRect().height+11;            // + margin
  var chrome=mp.getBoundingClientRect().height-mb.getBoundingClientRect().height;
  var avail=rp.getBoundingClientRect().height-chrome;
  var n=Math.max(3,Math.round(avail/cardH));
  if(n!==PER_PAGE){PER_PAGE=n;PAGE[CUR_TAB]=0;renderMatches(CUR_TAB);}
}
setTimeout(fitMatchesToRankings,60);
var _fitT;window.addEventListener('resize',function(){clearTimeout(_fitT);_fitT=setTimeout(fitMatchesToRankings,150);});

// ── Bottom data-refresh widget ──────────────────────────────────────────────
// Reuses the Modern Hub pipeline: /mapelo/modern/refresh kicks off the same
// RefreshLiveData scrape (new matches -> BenPom -> probabilities -> records),
// and /mapelo/modern/progress is its live progress file. On completion we clear
// the home page's sticky leaders cache and reload with the fresh data.
var _rfSeen={}, _rfBaseTs=0;
function _rfSet(pct,msg,log){
  document.getElementById('rfpFill').style.width=(pct||0)+'%';
  document.getElementById('rfpPct').textContent=(pct||0)+'%';
  if(msg) document.getElementById('rfpMsg').textContent=msg;
  var logEl=document.getElementById('rfpLog');
  (log||[]).forEach(function(line){
    if(_rfSeen[line]) return; _rfSeen[line]=1;
    var d=document.createElement('div'); d.className='ple'; d.textContent=line;
    logEl.appendChild(d);
  });
  var es=logEl.querySelectorAll('.ple');
  for(var i=0;i<es.length-4;i++) es[i].remove();
}
var _manualRefreshing=false;   // suppresses the "new matches" pill while the
                               // bottom-button refresh runs (it reloads anyway)
function startRefresh(){
  _manualRefreshing=true;
  var old=document.getElementById('updatePill'); if(old) old.remove();
  var btn=document.getElementById('refreshBtn'); btn.disabled=true;
  _rfSeen={};
  var card=document.getElementById('rfpCard');
  card.classList.add('show'); card.classList.remove('err');
  document.getElementById('rfpLabel').textContent='Refreshing VCT data';
  document.getElementById('rfpFill').classList.remove('done');
  _rfSet(3,'Starting refresh…',[]);
  // Baseline the current progress timestamp so a STALE 'done' from a previous
  // run isn't mistaken for this run completing instantly.
  fetch('/mapelo/modern/progress').then(function(r){return r.json();})
    .then(function(d){ _rfBaseTs=(d.progress&&d.progress.ts)||0; })
    .catch(function(){ _rfBaseTs=0; })
    .then(function(){
      fetch('/mapelo/modern/refresh').catch(function(){})
        .then(function(){ _rfPoll(0); });
    });
}
function _rfPoll(n){
  if(n>400){ _rfFail('Timed out — please try again later.'); return; }
  fetch('/mapelo/modern/progress').then(function(r){return r.json();}).then(function(d){
    var p=d.progress||{}, phase=p.phase||'', fresh=((p.ts||0)>_rfBaseTs);
    // Only reflect progress once THIS run starts writing (ts past the baseline);
    // otherwise a stale 'done' (pct 100) from a previous run flashes the bar to
    // 100% before the real run drops it back to ~2%.
    if(fresh) _rfSet(p.pct||0, p.message||'Working…', p.log||[]);
    if(fresh && phase==='error'){ _rfFail(p.message||'Refresh failed.'); return; }
    if(fresh && phase==='done'){
      _rfSet(100,'All data refreshed!',p.log||[]);
      document.getElementById('rfpFill').classList.add('done');
      document.getElementById('rfpLabel').textContent='Done';
      fetch('/alpha/bust-cache').catch(function(){}).then(function(){
        setTimeout(function(){
          // Reload at the TOP — restoring the bottom scroll position makes the
          // page visibly jump as the matches panel re-fits its height on load.
          try { if('scrollRestoration' in history) history.scrollRestoration='manual'; } catch(e){}
          window.scrollTo(0,0);
          location.reload();
        }, 1100);
      });
      return;
    }
    setTimeout(function(){ _rfPoll(n+1); }, 2000);
  }).catch(function(){ setTimeout(function(){ _rfPoll(n+1); }, 2500); });
}
function _rfFail(msg){
  _manualRefreshing=false;   // refresh ended without a reload — pill may resume
  document.getElementById('rfpCard').classList.add('err');
  document.getElementById('rfpLabel').textContent='Refresh failed';
  document.getElementById('rfpMsg').textContent=msg;
  var btn=document.getElementById('refreshBtn'); btn.disabled=false;
  btn.innerHTML='<span class="ricon">↻</span> Try again';
}

// ── Auto background refresh (stale-while-revalidate) ─────────────────────────
// The page renders instantly from cache; ~2s after load we quietly ask the
// server to check VLR for new matches (throttled server-side). We then poll the
// content version — if it actually changes, a gentle "new matches" pill appears.
// No auto-reload: the user taps to update when they're ready.
(function autoRefresh(){
  var baseVersion = (typeof DATA!=='undefined' && DATA && DATA.version) || '';
  if(!baseVersion) return;
  setTimeout(function(){
    fetch('/alpha/auto-refresh').catch(function(){});
    var tries=0;
    (function poll(){
      if(tries++>14) return;                        // ~3.5 min, then stop
      setTimeout(function(){
        fetch('/alpha/version').then(function(r){return r.json();}).then(function(d){
          if(d && d.version && d.version!==baseVersion){ _showUpdatePill(); return; }
          poll();
        }).catch(poll);
      }, 14000);
    })();
  }, 2200);
})();
function _showUpdatePill(){
  if(_manualRefreshing) return;   // user is already refreshing via the button
  if(document.getElementById('updatePill')) return;
  var pill=document.createElement('button');
  pill.id='updatePill'; pill.className='update-pill'; pill.type='button';
  pill.innerHTML='<span class="ricon">&#x21bb;</span> New matches &mdash; tap to update';
  pill.onclick=function(){
    pill.disabled=true; pill.innerHTML='<span class="ricon">&#x21bb;</span> Updating&hellip;';
    fetch('/alpha/bust-cache').catch(function(){}).then(function(){
      try{ if('scrollRestoration' in history) history.scrollRestoration='manual'; }catch(e){}
      window.scrollTo(0,0); location.reload();
    });
  };
  document.body.appendChild(pill);
}
</script>
</body>
</html>
"""

ARTICLES_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Articles — Bobo gg</title>
<link rel="icon" type="image/svg+xml" href="/favicon.svg">
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@700;800&family=DM+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<link rel="stylesheet" href="/static/base.css">
<style>
  *{box-sizing:border-box}
  body{font-family:'DM Sans',sans-serif;color:#16121d}
  a{color:inherit;text-decoration:none}
  .wrap{width:100%;max-width:1180px;margin:0 auto;padding:30px 22px 64px;position:relative;z-index:1}
  h1{font-family:'Plus Jakarta Sans',sans-serif;font-weight:800;font-size:clamp(2rem,5vw,2.8rem);letter-spacing:-.02em;margin:6px 2px 22px;color:#16121d}
  .alist{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:20px}
  .asub{font-size:.95rem;color:#6b6478;font-weight:500;margin:-12px 2px 22px;line-height:1.5}
  .afilter{display:flex;gap:8px;flex-wrap:wrap;margin:0 2px 26px}
  .afbtn{font-family:'DM Sans',sans-serif;font-size:.82rem;font-weight:700;color:#6b6478;background:#fff;border:1.5px solid #eceef2;border-radius:99px;padding:8px 17px;cursor:pointer;transition:background .15s,color .15s,border-color .15s}
  .afbtn:hover{border-color:#d8cdee;color:#16121d}
  .afbtn.on{background:#16121d;color:#fff;border-color:#16121d}
  .acard{position:relative;background:#fff;border-radius:20px;overflow:hidden;display:flex;flex-direction:column;text-align:center;box-shadow:0 4px 24px #0000000a;transition:transform .16s,box-shadow .16s}
  .acard:hover{transform:translateY(-5px);box-shadow:0 16px 40px #2a224018}
  /* featured (latest) story spans full width, image beside the text */
  .afeat{flex-direction:row;align-items:stretch;margin-bottom:22px}
  .acard.afeat img{width:50%;height:auto;min-height:308px;flex-shrink:0}
  .afeat .ab{padding:36px 40px;justify-content:center}
  .afeat .at{font-size:1.85rem;line-height:1.16}
  .afeat .ad{font-size:1.02rem;margin-top:13px;flex:0}
  .afeat .adate{margin-top:18px}
  .afeat .acat{top:16px;left:16px;font-size:.64rem;padding:6px 13px}
  .seclabel{font-family:'Plus Jakarta Sans',sans-serif;font-weight:800;font-size:.72rem;letter-spacing:.1em;text-transform:uppercase;color:#9a93a6;margin:6px 2px 14px}
  @media(max-width:680px){.afeat{flex-direction:column}.acard.afeat img{width:100%;min-height:0;height:200px}.afeat .ab{padding:24px}.afeat .at{font-size:1.45rem}}
  .acat{position:absolute;top:13px;left:13px;z-index:2;font-family:'Plus Jakarta Sans',sans-serif;font-size:.6rem;font-weight:800;letter-spacing:.09em;text-transform:uppercase;padding:5px 11px;border-radius:99px;color:#fff;box-shadow:0 3px 10px #0000003d}
  .acats{position:absolute;top:13px;left:13px;z-index:2;display:flex;flex-direction:column;gap:6px;align-items:flex-start}
  .acats .acat{position:static;top:auto;left:auto;box-shadow:0 3px 10px #0000003d}
  .acat.preview{background:#7c4dd6}
  .acat.opinion{background:#e07b39}
  .acat.research{background:#1f9d8a}
  .aempty{text-align:center;color:#9a93a6;font-weight:600;font-size:.9rem;padding:48px 0}
  .acard img{width:100%;height:168px;object-fit:cover;object-position:center top;display:block;background:#1a0f24}
  .ab{padding:20px 22px 22px;display:flex;flex-direction:column;flex:1}
  .at{font-family:'Plus Jakarta Sans',sans-serif;font-weight:800;font-size:1.16rem;line-height:1.22;color:#16121d}
  .ad{font-size:.85rem;color:#6b6478;font-weight:500;margin-top:8px;line-height:1.55;flex:1}
  .adate{font-size:.66rem;text-transform:uppercase;letter-spacing:.04em;color:#9a93a6;font-weight:700;margin-top:14px}
  .ab::after{content:'Read →';margin-top:12px;font-size:.82rem;font-weight:800;color:#7c4dd6;font-family:'Plus Jakarta Sans',sans-serif}
</style>
</head>
<body>
<div class="wrap">
  <h1>Articles</h1>
  <p class="asub">Previews, research, and opinions from across the VCT season.</p>
  <div class="afilter">
    <button class="afbtn on" data-f="all" type="button">All</button>
    <button class="afbtn" data-f="preview" type="button">Preview</button>
    <button class="afbtn" data-f="opinion" type="button">Opinion</button>
    <button class="afbtn" data-f="research" type="button">Research</button>
  </div>
  <div class="alist">
    <a class="acard" data-cat="research opinion" href="/articles/greatest-prime/">
      <span class="acats"><span class="acat research">Research</span><span class="acat opinion">Opinion</span></span>
      <img src="/aspas25corrode.jpg" alt="">
      <div class="ab">
        <div class="at">The Greatest Prime in VCT History Isn't a Debate</div>
        <div class="ad">Aspas at Champions Paris towers over VCT history, including your favorite player.</div>
        <div class="adate">Jun 28, 2026</div>
      </div>
    </a>
    <a class="acard" data-cat="preview" href="/articles/masters-london-playoffs-preview/">
      <span class="acat preview">Preview</span>
      <img src="/chronlondon.jpg" alt="">
      <div class="ab">
        <div class="at">Masters London Playoffs Preview</div>
        <div class="ad">A brief statistical glimpse into the final stage of Masters London.</div>
        <div class="adate">Jun 10, 2026</div>
      </div>
    </a>
    <a class="acard" data-cat="preview" href="/articles/masters-london-preview/">
      <span class="acat preview">Preview</span>
      <img src="/prxpacstage1win.jpg" alt="">
      <div class="ab">
        <div class="at">Masters London Tournament Preview</div>
        <div class="ad">Paper Rex's (un)inevitability, Neon nerfs, China's resurgence, and other bold predictions.</div>
        <div class="adate">Jun 2, 2026</div>
      </div>
    </a>
    <a class="acard" data-cat="preview" href="/articles/americas-stage1-playoffs-preview/">
      <span class="acat preview">Preview</span>
      <img src="/loudlev26.jpg" alt="">
      <div class="ab">
        <div class="at">Americas Stage 1 Playoffs Preview</div>
        <div class="ad">LOUD's resurgence, Leviatán's Bind, the 100T question, and BenPom's final say.</div>
        <div class="adate">May 12, 2026</div>
      </div>
    </a>
    <a class="acard" data-cat="research" href="/articles/over-underperformers/">
      <span class="acat research">Research</span>
      <img src="/patmen.jpg" alt="">
      <div class="ab">
        <div class="at">Overperforming in VCT: Who's Doing It?</div>
        <div class="ad">Surfacing the players outperforming (or underperforming) their team.</div>
        <div class="adate">May 4, 2026</div>
      </div>
    </a>
  </div>
  <div class="aempty" hidden>No articles in this category yet.</div>
</div>
<script>
(function(){
  var btns=document.querySelectorAll('.afbtn'),
      cards=document.querySelectorAll('.acard'),
      empty=document.querySelector('.aempty');
  btns.forEach(function(b){
    b.addEventListener('click',function(){
      btns.forEach(function(x){x.classList.remove('on');});
      b.classList.add('on');
      var f=b.getAttribute('data-f'), shown=0;
      cards.forEach(function(c){
        var ok=(f==='all'||(' '+(c.getAttribute('data-cat')||'')+' ').indexOf(' '+f+' ')!==-1);
        c.style.display=ok?'':'none';
        if(ok)shown++;
      });
      empty.hidden=shown>0;
    });
  });
})();
</script>
</body></html>
"""


TEAM_PROFILE_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{ org }} — Bobo gg</title>
<link rel="icon" type="image/svg+xml" href="/favicon.svg">
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@600;700;800&family=DM+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
<link rel="stylesheet" href="/static/base.css">
<style>
  body::after{animation:none !important;}
  :root{--card:#fff;--line:#eceef2;--ink:#16121d;--soft:#6b6478;--faint:#9a93a6;--good:#1f9d55;--bad:#d23b3b;--accent:#7c4dd6;}
  *{box-sizing:border-box}
  body{font-family:'DM Sans',sans-serif;color:var(--ink)}
  a{color:inherit;text-decoration:none}
  .wrap{max-width:1080px;margin:0 auto;padding:18px 22px 64px;position:relative;z-index:1}

  /* ── hero (team-color themed) ── */
  .tp-hero{position:relative;display:flex;align-items:center;justify-content:space-between;gap:22px;flex-wrap:wrap;
    background:linear-gradient(140deg,#fbf9ff,#f1ecf9);border-radius:22px;padding:24px 28px;color:var(--ink);overflow:hidden;
    box-shadow:0 8px 26px #2a224012;border:1px solid var(--line);border-left:5px solid var(--tc,#7c4dd6)}
  .tp-glow{position:absolute;right:-60px;top:-70px;width:260px;height:260px;border-radius:50%;
    background:radial-gradient(circle,var(--tc,#7c4dd6) 0%,transparent 68%);opacity:.1;pointer-events:none}
  .tp-hl{display:flex;align-items:center;gap:18px;position:relative;z-index:1;min-width:0}
  .tp-logo{width:72px;height:72px;border-radius:16px;background:#f4f2f8;object-fit:contain;padding:8px;flex-shrink:0;box-shadow:0 0 0 2px var(--tc,#7c4dd6)}
  .tp-logo-ph{width:72px;height:72px;border-radius:16px;display:flex;align-items:center;justify-content:center;font-weight:800;font-size:1.4rem;color:#fff;flex-shrink:0;box-shadow:0 0 0 2px var(--tc,#7c4dd6)}
  .tp-name{font-family:'Plus Jakarta Sans',sans-serif;font-weight:800;font-size:clamp(1.8rem,4vw,2.5rem);letter-spacing:-.02em;line-height:1}
  .tp-meta{display:flex;align-items:center;gap:12px;margin-top:10px;flex-wrap:wrap}
  .tp-reg{font-size:.62rem;font-weight:800;letter-spacing:.05em;padding:3px 9px;border-radius:6px;text-transform:uppercase}
  .fdots{display:flex;gap:5px}
  .fdot{width:13px;height:13px;border-radius:50%;display:block;border:2px solid transparent;transition:transform .12s}
  .fdot:hover{transform:scale(1.28)}
  .fdot.w{background:#19a85e}.fdot.l{background:#e8536a}
  .tp-hr{display:flex;align-items:center;gap:24px;flex-wrap:wrap;position:relative;z-index:1}
  .tp-stat{text-align:center}
  .tp-stat .v{font-family:'Plus Jakarta Sans',sans-serif;font-weight:800;font-size:1.5rem;line-height:1}
  .tp-stat .k{font-size:.6rem;font-weight:700;letter-spacing:.07em;text-transform:uppercase;color:#8a8296;margin-top:4px}
  .tp-spark{display:flex;flex-direction:column;align-items:center;gap:4px}
  .tp-spark svg{display:block}
  .tp-spark-k{font-size:.58rem;font-weight:700;letter-spacing:.07em;text-transform:uppercase;color:#8a8296}

  /* ── columns: 2 by default (Recent | right stack); 3 when the team has upcoming
     matches (Upcoming | Recent | right stack) ── */
  .tp-cols{display:grid;grid-template-columns:minmax(390px,1fr) minmax(0,1.1fr);gap:18px;margin-top:18px;align-items:stretch}
  .tp-cols.with-up{grid-template-columns:minmax(230px,.82fr) minmax(390px,1fr) minmax(0,1.2fr)}
  /* Recent panel keeps its natural height — only the right column stretches to
     fill when it is shorter; expanding a map must NOT extend recent matches. */
  .tp-cols > .panel{align-self:start}
  .tp-right{display:flex;flex-direction:column;gap:18px;min-width:0}
  .mapwrap{display:flex;flex-direction:column;flex:1}
  #maps{flex:1;display:flex;flex-direction:column;justify-content:space-between;gap:3px}
  .panel{background:var(--card);border:1px solid var(--line);border-radius:18px;padding:18px 18px 14px;box-shadow:0 4px 22px #0000000a}
  .ptitle{font-family:'Plus Jakarta Sans',sans-serif;font-weight:800;font-size:1.05rem;margin-bottom:13px;display:flex;align-items:baseline;gap:9px}
  .ptit-sub{font-family:'DM Sans',sans-serif;font-weight:600;font-size:.64rem;letter-spacing:.04em;text-transform:uppercase;color:var(--faint)}

  /* upcoming matches (left rail) — JS (syncUpcomingHeights) stretches each
     card to match its same-row .rm recent-match card exactly; box-sizing so
     the JS-set height includes padding/border like getBoundingClientRect(). */
  .uc{box-sizing:border-box;display:flex;flex-direction:column;justify-content:center;gap:11px;border:1px solid var(--line);border-left:4px solid var(--tc,#7c4dd6);border-radius:11px;padding:14px 15px;margin-bottom:10px;transition:box-shadow .15s,border-color .15s}
  .uc:hover{box-shadow:0 3px 14px #0000000a;border-color:#d6cce8}
  .uc-top{display:flex;align-items:center;justify-content:space-between;gap:8px;font-size:.66rem;color:var(--faint);font-weight:700}
  .uc-date{white-space:nowrap;flex:0 0 auto}
  .uc-evt{background:#f3eefb;color:#6a4caf;padding:2px 7px;border-radius:6px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .uc-opp{display:flex;align-items:center;gap:10px;min-width:0}
  .uc-logo{width:30px;height:30px;object-fit:contain;border-radius:6px;background:#f6f4fa;flex:0 0 auto}
  .uc-ph{width:30px;height:30px;border-radius:6px;display:flex;align-items:center;justify-content:center;font-size:.58rem;font-weight:800;color:#fff;flex:0 0 auto}
  .uc-nm{font-weight:700;font-size:.98rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;min-width:0}
  .uc-wp{margin-left:auto;flex:0 0 auto;font-family:'Plus Jakarta Sans',sans-serif;font-weight:800;font-size:.95rem;color:var(--soft)}
  .uc-wp.fav{color:var(--good)}
  .uc-wp small{display:block;font-family:'DM Sans',sans-serif;font-size:.5rem;font-weight:700;letter-spacing:.06em;text-transform:uppercase;color:var(--faint);text-align:right;line-height:1;margin-top:1px}

  /* recent matches (expandable + filterable) */
  .rm{border:1px solid var(--line);border-left:4px solid var(--line);border-radius:11px;padding:10px 13px;margin-bottom:10px;transition:box-shadow .15s}
  .rm.win{border-left-color:var(--good)}.rm.loss{border-left-color:var(--bad)}
  .rm:hover{box-shadow:0 3px 14px #0000000a}
  .rm-top{display:flex;align-items:center;justify-content:space-between;gap:10px;font-size:.68rem;color:var(--faint);font-weight:600;margin-bottom:4px}
  .rm-tag{background:#f3eefb;color:#6a4caf;padding:2px 7px;border-radius:6px;font-weight:700}
  .rm-right{display:flex;align-items:center;gap:9px}
  .rm-wl{font-weight:800;font-size:.8rem;letter-spacing:.08em;padding:3px 13px;border-radius:7px;color:#fff;box-shadow:0 2px 9px #00000022}
  .rm-wl.w{background:var(--good)}.rm-wl.l{background:var(--bad)}
  /* VLR-style scoreboard: ONE grid shared by the header + both team rows (rows
     are display:contents) so every map column lines up exactly. */
  .rm-board{display:grid;grid-template-columns:var(--gtc);gap:6px 10px;align-items:center;margin-top:7px;overflow-x:auto}
  .rb-row{display:contents}
  .rb-head{font-size:.6rem;color:var(--faint);font-weight:700;letter-spacing:.01em}
  .rb-team{display:flex;align-items:center;gap:8px;min-width:0}
  .rb-logo{width:22px;height:22px;object-fit:contain;border-radius:5px;background:#f6f4fa;flex:0 0 auto}
  .rb-ph{width:22px;height:22px;border-radius:5px;display:flex;align-items:center;justify-content:center;font-size:.5rem;font-weight:800;color:#fff;flex:0 0 auto}
  .rb-nm{font-weight:700;font-size:.88rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;color:var(--soft)}
  .rb-row.wn .rb-nm{font-weight:800;color:var(--ink)}
  .rb-mh{text-align:center;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .rb-c{text-align:center;font-size:.8rem;font-variant-numeric:tabular-nums;color:var(--faint)}
  .rb-c.win{color:var(--ink);font-weight:800}
  .rb-tot{text-align:center;font-family:'Plus Jakarta Sans',sans-serif;font-weight:800;font-size:.98rem;font-variant-numeric:tabular-nums;color:var(--faint)}
  .rb-row.wn .rb-tot{color:var(--ink)}

  /* ── map-performance diverging bars (KenPom net rating per map) ── */
  .mapblk{border-radius:9px}
  .mapbar{display:flex;align-items:center;gap:9px;padding:8px 6px;border-radius:9px;cursor:pointer;transition:background .12s}
  .mapbar:hover{background:#faf8ff}
  .mapblk.open .mapbar{background:#f3eefb}
  .mb-name{flex:0 0 60px;font-weight:700;font-size:.82rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .mb-wl{flex:0 0 38px;font-size:.66rem;color:var(--faint);font-weight:600;font-variant-numeric:tabular-nums}
  .mb-track{position:relative;flex:1;height:17px;border-radius:0;background:#f4f2f8}
  .mb-zero{position:absolute;left:50%;top:-2px;bottom:-2px;width:2px;background:#dcd6e6;transform:translateX(-1px)}
  .mb-seg{position:absolute;top:0;bottom:0;border-radius:0}
  .mb-seg.pos{background:linear-gradient(90deg,#34c47a,#1f9d55);border-radius:0 9px 9px 0}
  .mb-seg.neg{background:linear-gradient(270deg,#e06464,#d23b3b);border-radius:9px 0 0 9px}
  .mb-val{flex:0 0 42px;text-align:right;font-family:'Plus Jakarta Sans',sans-serif;font-weight:800;font-size:.84rem;font-variant-numeric:tabular-nums}
  .mb-val.pos{color:#16a34a}.mb-val.neg{color:#c0392b}
  .mb-chev{flex:0 0 auto;color:var(--faint);font-size:.62rem;transition:transform .18s}
  .mapblk.open .mb-chev{transform:rotate(180deg)}
  /* per-map game breakdown (every time the map was played) */
  .mapgames{display:grid;grid-template-rows:0fr;transition:grid-template-rows .28s cubic-bezier(.4,0,.2,1)}
  .mapblk.open .mapgames{grid-template-rows:1fr}
  .mapgames-in{overflow:hidden;min-height:0;padding-left:4px}
  .mg{display:flex;align-items:center;gap:8px;padding:5px 8px;border-radius:7px;font-size:.72rem}
  .mg+.mg{margin-top:3px}
  .mg:first-child{margin-top:5px}
  .mg.win{background:#f3faf5}.mg.loss{background:#fdf4f4}
  .mg-res{font-weight:800;font-size:.62rem;width:15px;text-align:center;flex:0 0 auto}
  .mg.win .mg-res{color:var(--good)}.mg.loss .mg-res{color:var(--bad)}
  .mg-logo{width:18px;height:18px;object-fit:contain;border-radius:4px;flex:0 0 auto}
  .mg-logoph{width:18px;height:18px;border-radius:4px;display:flex;align-items:center;justify-content:center;font-size:.46rem;font-weight:800;color:#fff;flex:0 0 auto}
  .mg-opp{font-weight:700;flex:0 0 auto}
  .mg-score{font-family:'Plus Jakarta Sans',sans-serif;font-weight:800;font-variant-numeric:tabular-nums;color:var(--soft)}
  .mg-score.wn{color:var(--ink)}
  .mg-dash{color:var(--faint);margin:0 -2px}
  .mg-diff{font-weight:700;font-variant-numeric:tabular-nums}
  .mg-diff.pos{color:#16a34a}.mg-diff.neg{color:#c0392b}
  .mg-meta{margin-left:auto;color:var(--faint);font-size:.63rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;padding-left:6px}
  .mg-empty{color:var(--faint);font-size:.72rem;padding:6px 8px}

  /* ── roster (right column, under map performance) — one even row ── */
  .roster{display:flex;gap:10px}
  .pl{display:flex;flex-direction:column;align-items:center;text-align:center;gap:7px;flex:1 1 0;min-width:0;padding:12px 6px;border:1px solid var(--line);border-radius:13px;transition:border-color .15s,transform .15s}
  .pl:hover{border-color:#d6cce8;transform:translateY(-2px)}
  .pl-ring{display:inline-flex;border-radius:50%;padding:2px;background:var(--tc,#7c4dd6)}
  .pl-ring img{width:52px;height:52px;border-radius:50%;object-fit:cover;object-position:top center;background:#efeaf6;display:block;border:2px solid #fff}
  .pl-ring .ph{width:52px;height:52px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:800;color:#fff;border:2px solid #fff}
  .pl .nm{font-weight:700;font-size:.82rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:100%}
  .empty{color:var(--faint);font-size:.84rem;font-weight:600;padding:14px 4px;text-align:center}
  footer{text-align:center;padding:28px;color:var(--faint);font-size:.74rem;font-weight:500;position:relative;z-index:1}
  html.inmodal footer{display:none}
  /* Compact the profile when shown inside the team modal so it fits with no scroll. */
  html.inmodal .wrap{padding:10px 18px 14px}
  html.inmodal .tp-hero{padding:16px 22px}
  html.inmodal .tp-cols{gap:14px;margin-top:14px}
  html.inmodal .panel{padding:14px 16px 11px}
  html.inmodal .tp-roster{margin-top:14px}
  html.inmodal .rm{margin-bottom:8px}
  /* hover popup: match card on form dots, label on trajectory event lines */
  #tpop{position:fixed;z-index:99999;pointer-events:none;background:linear-gradient(160deg,#241839,#19102a);color:#fff;border:1px solid #3a2a5a;border-radius:13px;padding:11px 13px;box-shadow:0 14px 38px #0007;font-size:.74rem;max-width:232px;opacity:0;visibility:hidden;transition:opacity .12s}
  #tpop.on{opacity:1;visibility:visible}
  #tpop .tp-evt{font-size:.56rem;font-weight:800;letter-spacing:.08em;text-transform:uppercase;color:#a78bfa;margin-bottom:6px}
  #tpop .tp-h{display:flex;align-items:center;gap:7px;font-weight:800;font-size:.84rem}
  #tpop .tp-h b{margin-left:auto;font-variant-numeric:tabular-nums}
  #tpop .tp-res{font-size:.6rem;font-weight:800;padding:1px 6px;border-radius:5px}
  #tpop .tp-res.w{background:#1f6f43;color:#b6f0cd}#tpop .tp-res.l{background:#7a2230;color:#ffc7cf}
  #tpop .tp-date{color:#b9a9d6;font-size:.64rem;margin-top:4px}
  #tpop .tp-delta{font-weight:700;font-size:.66rem;margin-top:6px}
  #tpop .tp-delta.pos{color:#41f59a}#tpop .tp-delta.neg{color:#ff8b8b}
  /* event tooltip: start/end date each paired with the team's BenPom then */
  #tpop .tp-erow{display:flex;align-items:baseline;gap:9px;font-size:.66rem;margin-top:5px;font-variant-numeric:tabular-nums}
  #tpop .tp-erow .l{color:#8d80ad;font-weight:800;font-size:.54rem;letter-spacing:.07em;text-transform:uppercase;width:30px;flex:0 0 auto}
  #tpop .tp-erow .dt{color:#cdbfe6}
  #tpop .tp-erow b{margin-left:auto;color:#fff;font-weight:800;padding-left:10px}
  #tpop .tp-maps{width:100%;border-collapse:collapse;margin-top:7px}
  #tpop .tp-maps td{font-size:.66rem;padding:3px 2px;border-top:1px solid #ffffff14;color:#cdbfe6}
  #tpop .tp-ms{text-align:right;font-variant-numeric:tabular-nums;font-weight:700}
  #tpop .tp-ms.mw{color:#41f59a}#tpop .tp-ms.ml{color:#ff8b8b}
  /* mid widths: Upcoming | Recent on top, right stack spans full width below */
  @media (max-width:960px){.tp-cols.with-up{grid-template-columns:minmax(180px,.72fr) 1fr}.tp-cols.with-up > .tp-right{grid-column:1 / -1}}
  @media (max-width:780px){.tp-cols,.tp-cols.with-up{grid-template-columns:1fr}.tp-hero{gap:16px}.tp-hr{gap:16px}}
</style>
</head>
<body>
<div class="wrap">
  <div id="profile"></div>
</div>
<footer>Team profile &middot; BenPom &middot; <a href="/mapelo/" style="text-decoration:underline">full ratings</a></footer>
<script>
var D = {{ data_json | safe }};
var COL=D.colors||{}, LOGOS=D.logos||{}, EL=D.event_labels||{};
var RC={EMEA:['#15803d','rgba(22,163,74,.14)'],Americas:['#c2410c','rgba(234,88,12,.14)'],Pacific:['#1d4ed8','rgba(37,99,235,.14)'],CN:['#be185d','rgba(219,39,119,.14)']};
function esc(s){return String(s==null?'':s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');}
function col(o){return COL[o]||'#8a8a8a';}
function logo(o,cls,phcls,big){var f=LOGOS[o];if(f)return '<img class="'+cls+'" src="/logos/'+esc(f)+'" alt="'+esc(o)+'">';return '<div class="'+phcls+'" style="background:'+col(o)+'">'+esc(String(o||'').slice(0,big?3:2))+'</div>';}
function fmtR(r){if(r==null)return '';var n=Number(r);return (n>=0?'+':'')+n.toFixed(2);}
var MO=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
function sd(d){if(!d)return '';var p=String(d).split('-');if(p.length<3)return d;return MO[+p[1]-1]+' '+(+p[2]);}
function winVsAvg(r){return Math.round(100/(1+Math.exp(-(D.beta||0)*r)));}
function vlrUrl(id){return id?('https://www.vlr.gg/'+id):'';}

function evAbbr(label){
  var s=String(label||'').trim();
  // drop a leading 4-digit year, then keep capitals + digits (Stage 1 -> S1,
  // Masters London -> ML, China Kickoff -> CK, Champions -> C).
  if(s.length>=4 && s.charCodeAt(0)>=48 && s.charCodeAt(0)<=57){
    var k=0; while(k<s.length && s[k]>='0' && s[k]<='9')k++; s=s.slice(k).trim();
  }
  var out='';
  for(var i=0;i<s.length;i++){ var c=s[i]; if((c>='A'&&c<='Z')||(c>='0'&&c<='9'))out+=c; }
  return (out||s).slice(0,3);
}
function sparkline(traj,color,events){
  if(!traj||traj.length<2)return '';
  var rs=traj.map(function(p){return p.r;});
  var mn=Math.min.apply(null,rs), mx=Math.max.apply(null,rs);
  // Headroom/footroom so the line never touches the top/bottom edge.
  var rng=(mx-mn)||1, marg=Math.max(rng*0.28,0.15);
  mn-=marg; mx+=marg;
  var W=230,H=66,pad=5,top=15,span=(mx-mn);   // `top` band holds the event labels
  // Date-based x-axis so event boundary lines line up with the rating line.
  var t0=+new Date(traj[0].d), t1=+new Date(traj[traj.length-1].d), tspan=(t1-t0)||1;
  function xRaw(d){return pad+((+new Date(d))-t0)/tspan*(W-2*pad);}
  function xCl(d){return Math.max(pad,Math.min(W-pad,xRaw(d)));}
  function yOf(r){return top+(1-(r-mn)/span)*(H-pad-top);}
  var pts=traj.map(function(p){ return xCl(p.d).toFixed(1)+','+yOf(p.r).toFixed(1); });
  var area='M'+pts.join(' L')+' L'+(W-pad).toFixed(1)+','+(H-pad)+' L'+pad+','+(H-pad)+' Z';
  // For every event on this team's timeline — including internationals it didn't
  // attend — draw a labelled bracket spanning the event whose two ends drop down
  // as dashed boundary lines into the chart.
  var ev='', braY=11;
  function dline(x,y0){return '<line x1="'+x.toFixed(1)+'" y1="'+y0+'" x2="'+x.toFixed(1)+'" y2="'+(H-pad)+'" stroke="#9a93a6" stroke-opacity="0.45" stroke-width="1" stroke-dasharray="2 3"></line>';}
  if(events&&events.length){
    events.forEach(function(e){
      if(!e.start)return;
      var rs0=xRaw(e.start), re0=xRaw(e.end||e.start);
      if(re0<pad-1 || rs0>W-pad+1) return;             // event outside this team's span
      var xs=Math.max(pad,Math.min(W-pad,rs0)), xe=Math.max(pad,Math.min(W-pad,re0));
      var mid=Math.max(10,Math.min((xs+xe)/2,W-10)), lbl=esc(e.label||'');
      if(xe-xs>3){
        // bracket: top bar + short feet at each end, then dashed lines descend
        ev+='<path d="M'+xs.toFixed(1)+','+(braY+3)+' L'+xs.toFixed(1)+','+braY+' L'+xe.toFixed(1)+','+braY+' L'+xe.toFixed(1)+','+(braY+3)+'" fill="none" stroke="#9a93a6" stroke-opacity="0.6" stroke-width="1"></path>'
          +dline(xs,braY+3)+dline(xe,braY+3);
      } else {
        ev+=dline(xs,braY);
      }
      ev+='<text x="'+mid.toFixed(1)+'" y="8" text-anchor="middle" font-size="7.4" font-weight="700" fill="#8a8296" fill-opacity="0.95">'+esc(evAbbr(e.label||''))+'</text>'
        +'<rect x="'+(xs-5).toFixed(1)+'" y="0" width="'+Math.max(xe-xs+10,12).toFixed(1)+'" height="'+H+'" fill="transparent" data-ev="'+lbl+'" data-d="'+esc(e.start)+'" data-d2="'+esc(e.end||e.start)+'" style="pointer-events:all;cursor:pointer"></rect>';
    });
  }
  return '<svg viewBox="0 0 '+W+' '+H+'" width="'+W+'" height="'+H+'">'
    +'<path d="'+area+'" fill="'+color+'" opacity="0.16"/>'+ev
    +'<polyline points="'+pts.join(' ')+'" fill="none" stroke="'+color+'" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>'
    +'</svg>';
}
var FORMM=[];          // registry: form-dot index -> full match object
function formDots(){
  var r=(D.recent||[]).slice(0,5).slice().reverse();
  if(!r.length)return '';
  return '<div class="fdots">'+r.map(function(m){
    var won=m.result==='W', idx=FORMM.push(m)-1;
    return '<a class="fdot '+(won?'w':'l')+'" data-mi="'+idx+'" href="'+vlrUrl(m.match_id)+'" target="_blank" rel="noopener"></a>';
  }).join('')+'</div>';
}
// Shared hover popup (BenPom match card on form dots, event label on trajectory lines).
var POP=null;
function _pop(){ if(!POP){POP=document.createElement('div');POP.id='tpop';document.body.appendChild(POP);} return POP; }
function _showPop(html,el){
  var p=_pop(); p.innerHTML=html; p.classList.add('on');
  var r=el.getBoundingClientRect(), pr=p.getBoundingClientRect(), gap=12;
  var left=r.right+gap; if(left+pr.width>window.innerWidth-8)left=r.left-pr.width-gap;
  var top=r.top+r.height/2-pr.height/2; top=Math.max(8,Math.min(top,window.innerHeight-pr.height-8));
  p.style.left=left+'px'; p.style.top=top+'px';
}
function _hidePop(){ if(POP)POP.classList.remove('on'); }
function matchCardHTML(m){
  var won=m.result==='W', rat=won?m.winner_after:m.loser_after, d=m.delta;
  var evt=EL[m.event_id]||m.event_id||'';
  var rows=(m.maps||[]).map(function(mp){
    var mw=mp.winner===D.org, a=mw?mp.wr:mp.lr, b=mw?mp.lr:mp.wr;
    return '<tr><td class="tp-mn">'+esc(mp.map)+'</td><td class="tp-ms '+(mw?'mw':'ml')+'">'+a+'–'+b+'</td></tr>';
  }).join('');
  return (evt?'<div class="tp-evt">'+esc(evt)+'</div>':'')
    +'<div class="tp-h"><span class="tp-res '+(won?'w':'l')+'">'+(won?'W':'L')+'</span><span>vs '+esc(m.opponent)+'</span><b>'+esc(m.score||'')+'</b></div>'
    +'<div class="tp-date">'+esc(sd(m.date))+'</div>'
    +(d!=null?'<div class="tp-delta '+(d>=0?'pos':'neg')+'">BenPom '+Number(rat||0).toFixed(2)+' &nbsp;('+(d>=0?'+':'')+Number(d).toFixed(2)+')</div>':'')
    +(rows?'<table class="tp-maps">'+rows+'</table>':'');
}

// VLR-style scoreboard: both teams stacked, each with its per-map rounds in
// aligned columns + the series total. Series winner is bold/highlighted; a clear
// WIN/LOSS badge + green/red left edge identify this team's result.
function recentCard(m){
  var won=m.result==='W', opp=m.opponent, maps=m.maps||[];
  var aWins=0,bWins=0; maps.forEach(function(mp){ if(mp.winner===D.org)aWins++; else bWins++; });
  var n=maps.length;
  if(!n){ var raw=String(m.score||'0-0').split('-'), ws=+raw[0]||0, ls=+raw[1]||0; aWins=won?ws:ls; bWins=won?ls:ws; }
  var evt=EL[m.event_id]||m.event_id||'';
  var mw=n>3?Math.max(28,Math.min(52,Math.floor((305-74-26-(n+2)*10)/n))):52;
  var gtc='minmax(74px,auto) 26px 1fr '+new Array(n+1).join(mw+'px ');
  function teamRow(org,tot,winner){
    var cells=maps.map(function(mp){ var w=(mp.winner===org); return '<div class="rb-c'+(w?' win':'')+'">'+(w?mp.wr:mp.lr)+'</div>'; }).join('');
    return '<div class="rb-row'+(winner?' wn':'')+'"><div class="rb-team">'+logo(org,'rb-logo','rb-ph','')+'<span class="rb-nm">'+esc(org)+'</span></div><div class="rb-tot">'+tot+'</div><div></div>'+cells+'</div>';
  }
  var head=n?('<div class="rb-row rb-head"><div></div><div></div><div></div>'+maps.map(function(mp){return '<div class="rb-mh">'+esc(mp.map)+'</div>';}).join('')+'</div>'):'';
  return '<div class="rm '+(won?'win':'loss')+'">'
    +'<div class="rm-top"><span class="rm-tag">'+esc(evt)+'</span><span class="rm-right"><span>'+sd(m.date)+'</span><span class="rm-wl '+(won?'w':'l')+'">'+(won?'WIN':'LOSS')+'</span></span></div>'
    +'<div class="rm-board" style="--gtc:'+gtc+'">'+head+teamRow(D.org,aWins,won)+teamRow(opp,bWins,!won)+'</div>'
    +'</div>';
}
// UTC "YYYY-MM-DD HH:MM:SS" -> viewer-local time + tz label, e.g. "1:00 PM EDT".
function tlocal(utc){
  if(!utc)return '';
  var iso=String(utc).trim().replace(' ','T');
  if(!/[zZ]|[+-]\\d\\d:?\\d\\d$/.test(iso))iso+='Z';
  var d=new Date(iso);
  return isNaN(d.getTime())?'':d.toLocaleTimeString([],{hour:'numeric',minute:'2-digit',timeZoneName:'short'});
}
// Compact upcoming-match row: opponent (logo), date, event, projected win%.
// Links to the opponent's profile (upcoming feed carries no VLR id).
function upcomingRow(u){
  var wp=(u.win_prob!=null)?Math.round(u.win_prob*100):null;
  return '<a class="uc" href="/team/'+encodeURIComponent(u.opponent)+'" title="'+esc(u.opponent)+' profile">'
    +'<div class="uc-top"><span class="uc-date">'+esc(sd(u.date))+(u.time?' &middot; '+tlocal(u.time):'')+'</span>'
    +(u.event?'<span class="uc-evt">'+esc(u.event)+'</span>':'')+'</div>'
    +'<div class="uc-opp">'+logo(u.opponent,'uc-logo','uc-ph','')
    +'<span class="uc-nm">vs '+esc(u.opponent)+'</span>'
    +(wp!=null?'<span class="uc-wp'+(wp>=50?' fav':'')+'">'+wp+'%<small>win proj</small></span>':'')
    +'</div></a>';
}
function mapBar(mp,maxAbs){
  var r=mp.rating, frac=Math.min(1,Math.abs(r)/(maxAbs||1)), pos=r>=0, w=(frac*46).toFixed(1);
  var seg=pos?('<span class="mb-seg pos" style="left:50%;width:'+w+'%"></span>')
             :('<span class="mb-seg neg" style="right:50%;width:'+w+'%"></span>');
  return '<div class="mapblk" data-map="'+esc(mp.map)+'">'
    +'<div class="mapbar">'
    +'<span class="mb-name">'+esc(mp.map)+'</span>'
    +'<span class="mb-wl">'+mp.w+'-'+mp.l+'</span>'
    +'<span class="mb-track"><span class="mb-zero"></span>'+seg+'</span>'
    +'<span class="mb-val '+(pos?'pos':'neg')+'">'+fmtR(r)+'</span>'
    +'<span class="mb-chev">&#9662;</span>'
    +'</div><div class="mapgames"><div class="mapgames-in"></div></div></div>';
}
// Every game this team played on `map` (newest first) — same drill-down as the
// BenPom hub's map breakdown: W/L is the MAP outcome, with rounds + round diff.
function mapGamesHTML(map){
  var games=(D.events||[]).filter(function(me){
    return (me.maps||[]).some(function(m){return m.map===map;});
  }).slice().sort(function(a,b){return (b.match_id||0)-(a.match_id||0);});
  if(!games.length)return '<div class="mg-empty">No recorded games.</div>';
  return games.map(function(me){
    var mi=(me.maps||[]).filter(function(m){return m.map===map;})[0];
    var won=mi?(mi.winner===D.org):(me.winner===D.org);
    var opp=(me.winner===D.org)?me.loser:me.winner;
    var orgRd=mi?(mi.winner===D.org?mi.wr:mi.lr):'?';
    var oppRd=mi?(mi.winner===D.org?mi.lr:mi.wr):'?';
    var diff=(typeof orgRd==='number'&&typeof oppRd==='number')?(orgRd-oppRd):null;
    var diffStr=diff!=null?((diff>=0?'+':'')+diff):'';
    var evt=EL[me.event_id]||me.event_id||'';
    return '<div class="mg '+(won?'win':'loss')+'">'
      +'<span class="mg-res">'+(won?'W':'L')+'</span>'
      +logo(D.org,'mg-logo','mg-logoph','')
      +'<span class="mg-score'+(won?' wn':'')+'">'+orgRd+'</span>'
      +'<span class="mg-dash">&ndash;</span>'
      +'<span class="mg-score'+(won?'':' wn')+'">'+oppRd+'</span>'
      +logo(opp,'mg-logo','mg-logoph','')
      +'<span class="mg-opp">'+esc(opp)+'</span>'
      +(diffStr?'<span class="mg-diff '+(diff>=0?'pos':'neg')+'">'+diffStr+'</span>':'')
      +'<span class="mg-meta">'+sd(me.date)+(evt?' &middot; '+esc(evt):'')+'</span>'
      +'</div>';
  }).join('');
}
function pf(img){img.style.display='none';var n=img.nextElementSibling;if(n)n.style.display='flex';}
function playerCard(p){
  var img=p.headshot?'<img src="'+esc(p.headshot)+'" loading="lazy" onerror="pf(this)">':'';
  var ph='<div class="ph" style="background:'+col(D.org)+(p.headshot?';display:none':'')+'">'+esc(String(p.player||'').slice(0,2))+'</div>';
  return '<a class="pl" href="'+esc(p.url||'#')+'" target="_blank" rel="noopener"><span class="pl-ring" style="--tc:'+col(D.org)+'">'+img+ph+'</span><span class="nm">'+esc(p.player)+'</span></a>';
}

(function(){
  var root=document.getElementById('profile');
  if(!D || !D.org){root.innerHTML='<div class="empty" style="padding:60px">Team not found.</div>';return;}
  var rc=RC[D.region]||['#666','rgba(0,0,0,.06)'];
  var rec=(D.region==='International')?'':((D.w||0)+'-'+(D.l||0));
  var C=col(D.org);

  var hero='<div class="tp-hero" style="--tc:'+C+'"><div class="tp-glow" style="--tc:'+C+'"></div>'
    +'<div class="tp-hl">'+logo(D.org,'tp-logo','tp-logo-ph',true)
    +'<div><div class="tp-name">'+esc(D.org)+'</div>'
    +'<div class="tp-meta"><span class="tp-reg" style="color:'+rc[0]+';background:'+rc[1]+'">'+esc(D.region||'')+'</span>'+formDots()+'</div></div></div>'
    +'<div class="tp-hr">'
    +'<div class="tp-stat"><div class="v">'+fmtR(D.rating)+'</div><div class="k">BenPom</div></div>'
    +'<div class="tp-stat"><div class="v">#'+D.rank+'</div><div class="k">of '+D.n_teams+' VCT teams</div></div>'
    +(rec?'<div class="tp-stat"><div class="v">'+rec+'</div><div class="k">Map W-L'+(D.season?'<br>'+D.season+' season':'')+'</div></div>':'')
    +'<div class="tp-stat" title="Expected chance to win a single map against an average VCT team"><div class="v">'+winVsAvg(D.rating)+'%</div><div class="k">Map win vs avg<br>VCT team</div></div>'
    +((D.traj&&D.traj.length>1)?('<div class="tp-spark">'+sparkline(D.traj,'#7c4dd6',D.season_events)+'<div class="tp-spark-k">Season trajectory</div></div>'):'')
    +'</div></div>';

  var recentHTML=(D.recent&&D.recent.length)?D.recent.map(recentCard).join(''):'<div class="empty">No recent matches.</div>';
  var maps=D.all_maps||[];
  var maxAbs=maps.reduce(function(a,m){return Math.max(a,Math.abs(m.rating));},0)||1;
  var mapsHTML=maps.length?maps.map(function(m){return mapBar(m,maxAbs);}).join(''):'<div class="empty">No map data yet.</div>';
  var rosterHTML=(D.roster&&D.roster.length)?D.roster.map(playerCard).join(''):'<div class="empty">No roster data.</div>';
  // Upcoming matches sit to the LEFT of Recent — but only when the team actually
  // has some, so off-season profiles fall back to the original 2-column layout.
  var hasUp=D.upcoming&&D.upcoming.length;
  var upcomingPanel=hasUp
    ?'<section class="panel"><div class="ptitle">Upcoming matches <span class="ptit-sub">proj. series win</span></div><div id="upcoming">'+D.upcoming.map(upcomingRow).join('')+'</div></section>'
    :'';

  root.innerHTML=hero
    +'<div class="tp-cols'+(hasUp?' with-up':'')+'">'
    +upcomingPanel
    +'<section class="panel"><div class="ptitle">Recent matches</div><div id="recent">'+recentHTML+'</div></section>'
    +'<div class="tp-right">'
    +'<section class="panel mapwrap"><div class="ptitle">Map performance <span class="ptit-sub">net rating &middot; click a map for its games</span></div><div id="maps">'+mapsHTML+'</div></section>'
    +'<section class="panel"><div class="ptitle">Roster</div><div class="roster">'+rosterHTML+'</div></section>'
    +'</div>'
    +'</div>';

  // Upcoming cards must match Recent cards' height row-by-row — Recent's
  // per-map scoreboard grid makes it inherently taller than a CSS min-height
  // guess can track. Measure each rendered .rm card and stretch the
  // same-index .uc card to match (post-layout, so fonts/logos are accounted
  // for); re-measure on resize since wrapping can change card heights.
  function syncUpcomingHeights(){
    var ups=document.querySelectorAll('#upcoming .uc');
    var recs=document.querySelectorAll('#recent .rm');
    if(!ups.length||!recs.length)return;
    ups.forEach(function(u,i){
      u.style.height='';
      var r=recs[i]||recs[recs.length-1];
      var h=r.getBoundingClientRect().height;
      if(h)u.style.height=h+'px';
    });
  }
  if(hasUp){
    syncUpcomingHeights();
    // Re-sync once web fonts finish swapping in (a font-metric shift after the
    // first measurement is the only thing that can still throw the two
    // columns off by a few px) and once more next frame for full precision.
    if(document.fonts&&document.fonts.ready)document.fonts.ready.then(syncUpcomingHeights).catch(function(){});
    requestAnimationFrame(function(){requestAnimationFrame(syncUpcomingHeights);});
    if(!window._tpHeightSync){
      window._tpHeightSync=true;
      window.addEventListener('resize',function(){clearTimeout(window._tpHSt);window._tpHSt=setTimeout(syncUpcomingHeights,120);});
    }
  }

  // Click a map → expand the game-by-game breakdown underneath it (lazy-built).
  document.getElementById('maps').addEventListener('click',function(e){
    var blk=e.target.closest('.mapblk'); if(!blk)return;
    var open=blk.classList.toggle('open');
    if(open && !blk.dataset.loaded){
      blk.dataset.loaded='1';
      blk.querySelector('.mapgames-in').innerHTML=mapGamesHTML(blk.getAttribute('data-map'));
    }
  });
  // Form dots → BenPom match card on hover.
  var fd=document.querySelector('.fdots');
  if(fd){
    fd.addEventListener('mouseover',function(e){var d=e.target.closest('.fdot[data-mi]');if(d){var m=FORMM[+d.getAttribute('data-mi')];if(m)_showPop(matchCardHTML(m),d);}});
    fd.addEventListener('mouseout',function(e){if(e.target.closest('.fdot[data-mi]'))_hidePop();});
  }
  // Trajectory event lines → event label + the team's BenPom at the event's
  // start and end (read off the trajectory) on hover.
  // BenPom as of an ISO date = the last checkpoint on or before it; null for a
  // date past the latest checkpoint (a future/unfinished event has no data yet).
  function _ratingAt(traj,date){
    if(!traj||!traj.length||!date) return null;
    var lastD=traj[traj.length-1].d;
    if(lastD && date>lastD) return null;
    var best=null;
    for(var i=0;i<traj.length;i++){ var p=traj[i];
      if(p.d && p.d<=date && (best===null || p.d>=best.d)) best=p; }
    return best?best.r:null;
  }
  var sv=document.querySelector('.tp-spark svg');
  if(sv){
    sv.addEventListener('mouseover',function(e){
      var t=e.target.closest('[data-ev]'); if(!t) return;
      var raw1=t.getAttribute('data-d'), raw2=t.getAttribute('data-d2');
      var r1=_ratingAt(D.traj,raw1), r2=_ratingAt(D.traj,raw2);
      var same=(!raw2||raw2===raw1);
      function row(lbl,raw,r){
        return '<div class="tp-erow"><span class="l">'+lbl+'</span><span class="dt">'+esc(sd(raw))+'</span>'
          +(r!=null?'<b>'+Number(r).toFixed(2)+'</b>':'')+'</div>';
      }
      var body = same
        ? '<div class="tp-date">'+esc(sd(raw1))+(r1!=null?' &middot; BenPom '+Number(r1).toFixed(2):'')+'</div>'
        : row('Start',raw1,r1)+row('End',raw2,r2);
      _showPop('<div class="tp-h"><span>'+esc(t.getAttribute('data-ev'))+'</span></div>'+body,t);
    });
    sv.addEventListener('mouseout',function(e){if(e.target.closest('[data-ev]'))_hidePop();});
  }
})();
(function(){
  if(window.self===window.top) return;            // only when embedded in the modal
  document.documentElement.classList.add('inmodal');
  function postH(){ try{ parent.postMessage({__teamH: Math.ceil(document.documentElement.scrollHeight)}, '*'); }catch(e){} }
  postH();
  window.addEventListener('load', postH);
  window.addEventListener('resize', postH);
  setTimeout(postH, 250); setTimeout(postH, 800); setTimeout(postH, 1500);
})();
</script>
</body>
</html>
"""

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory(STATIC_DIR, filename, max_age=31536000)

@app.route("/favicon.svg")
def favicon():
    return send_from_directory(STATIC_DIR, "BoboLogo-cropped.svg", mimetype="image/svg+xml")

@app.route("/logo.svg")
def logo():
    return send_from_directory(STATIC_DIR, "BoboLogo.svg", mimetype="image/svg+xml")

@app.route("/patmen.jpg")
def patmen():
    return send_from_directory(os.path.dirname(__file__), "Patmen.jpg", mimetype="image/jpeg")

@app.route("/aspas25corrode.jpg")
def aspas25corrode():
    return send_from_directory(os.path.dirname(__file__), "Aspas25CorrodeChamps.jpg", mimetype="image/jpeg")

@app.route("/loudlev26.jpg")
def loudlev26():
    return send_from_directory(os.path.dirname(__file__), "LoudLev26.jpg", mimetype="image/jpeg")

@app.route("/prxpacstage1win.jpg")
def prxpacstage1win():
    return send_from_directory(os.path.dirname(__file__), "PRXPacStage1Win.jpg", mimetype="image/jpeg")

@app.route("/krustage1.png")
def krustage1():
    return send_from_directory(os.path.dirname(__file__), "KruStage1.png", mimetype="image/png")

@app.route("/mapelo.png")
def mapelo_img():
    return send_from_directory(os.path.dirname(__file__), "MapElo.png", mimetype="image/png")

@app.route("/chronlondon.jpg")
def chronlondon_jpg():
    return send_from_directory(os.path.dirname(__file__), "ChronLondon.jpg", mimetype="image/jpeg")

@app.route("/edgchamps.jpg")
def edgchamps_img():
    return send_from_directory(os.path.dirname(__file__), "EDGCHamps.jpg", mimetype="image/jpeg")

@app.route("/maps/<filename>")
def map_img(filename):
    return send_from_directory(os.path.join(os.path.dirname(__file__), "static/maps"), filename)

@app.route("/logos/<filename>")
def team_logo(filename):
    return send_from_directory(os.path.join(os.path.dirname(__file__), "static/logos"), filename)

# Alpha is the default home now; /classic is the alternative. /alpha stays as an
# explicit alias so existing links/bookmarks keep working.
@app.route("/")
@app.route("/alpha")
def alpha_home():
    try:
        data = _build_alpha_data()
    except Exception as e:
        data = {"error": str(e), "event": None, "next_event": None, "last_event": None,
                "rankings": [], "recent": [], "upcoming": [], "player_stats": [],
                "players_event": None, "colors": {}, "logos": {}}
    return render_template_string(ALPHA_HTML, data_json=_json.dumps(data))


@app.route("/classic")
def classic_home():
    return render_template_string(HOME_HTML)


@app.route("/alpha/version")
def alpha_version():
    """Current content signature — the home page polls this after kicking a
    background refresh to detect whether new data actually arrived."""
    return {"version": _alpha_data_version()}


@app.route("/alpha/auto-refresh")
def alpha_auto_refresh():
    """Kick a background data refresh IF one is due. Non-forcing, so the
    scraper's own cooldown throttles it — many page loads still produce at most
    one scrape every couple minutes. The home page calls this on load so it
    self-updates for new matches without the user clicking anything."""
    try:
        from MapElo import _mhub_trigger_build
        _mhub_trigger_build(force=False)
    except Exception:
        pass
    return {"ok": True}


@app.route("/alpha/bust-cache")
def alpha_bust_cache():
    """Clear the home page's in-process data caches so a reload right after a
    manual refresh shows freshly-scraped data. BenPom (mhub) and recent-records
    already self-invalidate on file mtime; the event-leaderboard cache that feeds
    the player leaders does NOT, so clear it explicitly."""
    cleared = []
    try:
        from MapElo import _mhub_cache, _mhub_cache_lock
        with _mhub_cache_lock:
            _mhub_cache["ts"] = 0.0
        cleared.append("benpom")
    except Exception:
        pass
    try:
        import EventLeaderboards as _EL
        _EL._event_cache.clear()
        cleared.append("leaders")
    except Exception:
        pass
    # Warm (recompute + repersist) the recent-records cache when the data has
    # changed, so the reload after a refresh doesn't pay the cold ~1.3s build —
    # and the disk JSON is fresh for every other worker. No-op/fast if unchanged.
    try:
        from AllTimeHighs import build_recent_records
        build_recent_records(8)
        cleared.append("records")
    except Exception:
        pass
    return {"cleared": cleared}


@app.route("/articles/")
def articles_index():
    return render_template_string(ARTICLES_HTML)


@app.route("/team/<org>")
def team_profile(org):
    try:
        data = _build_team_profile(org)
    except Exception:
        data = None
    return render_template_string(TEAM_PROFILE_HTML,
                                  org=org, data_json=_json.dumps(data))


@app.route("/api/team/<org>")
def api_team(org):
    try:
        data = _build_team_profile(org)
    except Exception:
        data = None
    from flask import Response
    return Response(_json.dumps(data), mimetype="application/json")


def _anav_ver():
    """Cache-buster for alpha-nav.js: its file mtime. /static/ is served with a
    1-year max-age, so without this the browser would keep an old cached copy of
    the nav script forever (it did — a stale copy that still skipped /alpha)."""
    try:
        return str(int(os.path.getmtime(os.path.join(STATIC_DIR, "alpha-nav.js"))))
    except Exception:
        return "0"


@app.after_request
def _inject_alpha_nav(resp):
    """Inject the persistent Alpha top-nav script into every HTML page. The
    script (static/alpha-nav.js) is the single source of truth for the bar: it
    renders always on /alpha and /team/*, in Alpha mode on every other page, and
    never on the classic home (/). Runs before flask-compress (registered later
    → called first), and bails if the body is already compressed or is a
    passthrough/static response."""
    try:
        if resp.direct_passthrough or resp.headers.get("Content-Encoding"):
            return resp
        if "text/html" not in resp.headers.get("Content-Type", ""):
            return resp
        body = resp.get_data(as_text=True)
        # Match the actual injected <script> tag, not a bare substring — pages
        # that mention "alpha-nav.js" in their own inline-script comments (e.g.
        # MatchDataExplorer.py) were tripping this as a false positive and
        # silently skipping injection, so the nav bar never appeared on them.
        if 'src="/static/alpha-nav.js' in body:
            return resp
        # Run the bar builder at the TOP of <body> (not deferred at the end) so the
        # bar is in place before the page's content renders. Deferred-at-end meant
        # long/slow pages (e.g. Articles, Pythagorean) painted first, then the bar
        # popped in a beat later — a visible blip.
        m = _re.search(r"<body[^>]*>", body)
        if m:
            tag = m.group(0)
            resp.set_data(body.replace(
                tag,
                tag + '<script src="/static/alpha-nav.js?v=%s"></script>'
                % _anav_ver(), 1))
    except Exception:
        pass
    return resp


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5001)
