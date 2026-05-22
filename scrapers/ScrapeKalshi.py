"""
Scrape Kalshi closed Valorant per-match markets and match them to VLR
matches + our model's prediction.

Output: data/kalshi_valorant.json
  list of dicts: {event_ticker, kalshi_team_a, kalshi_team_b, match_date,
                  pre_match_price_a, pre_match_price_b, actual_winner,
                  pre_match_window_minutes, n_trades_window, total_volume,
                  model_org_a, model_org_b, model_p_a, model_match_date,
                  vlr_match_id}

Strategy:
  - Walk every settled event in series KXVALORANTGAME
  - Each event has 2 sub-markets (one per team)
  - Trade endpoint gives tick history; we capture the volume-weighted
    average price across the 4-hour window before match start
  - Then map to our data/match_results.csv by team-name aliasing + date
"""
import json, os, sys, time, datetime, urllib.request, urllib.parse
import pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

API = "https://api.elections.kalshi.com/trade-api/v2"

# Map Kalshi team names to our model's org codes.
# Kalshi uses full team names; we use short codes in match_results.csv.
TEAM_ALIAS = {
    # EMEA
    "Sentinels": "SEN", "NRG": "NRG", "Evil Geniuses": "EG",
    "100 Thieves": "100T", "Cloud9": "C9", "G2": "G2", "G2 Esports": "G2",
    "MIBR": "MIBR", "FURIA": "FUR", "FURIA Esports": "FUR",
    "Leviatán Esports": "LEV", "Leviatan Esports": "LEV", "Leviatán": "LEV",
    "LOUD": "LOUD", "KRÜ Esports": "KRÜ", "KRU Esports": "KRÜ",
    "Team Envy": "ENVY",
    # EMEA
    "Fnatic": "FNC", "Team Liquid": "TL", "Natus Vincere": "NAVI", "NAVI": "NAVI",
    "Vitality": "VIT", "Team Vitality": "VIT", "BBL Esports": "BBL",
    "GIANTX": "GX", "Karmine Corp": "KC", "Team Heretics": "TH",
    "FUT Esports": "FUT", "Gentle Mates": "M8", "Gentle Mates Karmine Corp": "M8",
    "GIANTX Pride": "GX", "Movistar KOI": "MKOI", "M8": "M8",
    "Eternal Fire": "EF", "Pcific Esports": "PCF", "Pacific Esports": "PCF",
    # Pacific
    "Paper Rex": "PRX", "T1": "T1", "Gen.G Esports": "GEN", "Gen.G": "GEN",
    "DRX": "DRX", "DetonatioN FocusMe": "DFM", "ZETA DIVISION": "ZETA",
    "Rex Regum Qeon": "RRQ", "Talon Esports": "TS", "Team Secret": "TS",
    "Global Esports": "GE", "Nongshim RedForce": "NS", "Bleed Esports": "BLD",
    "FNATIC": "FNC", "TALON": "TS", "BOOM Esports": "BME", "BME": "BME",
    "FS": "FS", "FullSense": "FS", "VL": "VL", "Vasanta Hue": "VL",
    "KRX": "KRX", "KRU": "KRÜ",
    # CN
    "EDward Gaming": "EDG", "Bilibili Gaming": "BLG", "TYLOO": "TYLOO",
    "Trace Esports": "TE", "Dragon Ranger Gaming": "DRG",
    "All Gamers": "AG", "Xi Lai Gaming": "XLG", "FunPlus Phoenix": "FPX",
    "Nova Esports": "NOVA", "Wolves Esports": "WOL", "JD Gaming": "JDG",
    "TEC.GG": "TEC", "Titan Esports Club": "TEC", "Titans Esports Club": "TEC",
}


def fetch(path, params=None):
    url = API + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    })
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())
    except Exception as e:
        return {"error": str(e)}


def all_settled_events():
    events = []
    cursor = ""
    page = 0
    while True:
        page += 1
        params = {"series_ticker": "KXVALORANTGAME", "status": "settled", "limit": 200}
        if cursor:
            params["cursor"] = cursor
        data = fetch("/events", params)
        ev = data.get("events", [])
        if not ev:
            break
        events.extend(ev)
        cursor = data.get("cursor", "")
        if not cursor or page >= 30:
            break
        time.sleep(0.2)
    return events


def event_markets(event_ticker):
    data = fetch("/markets", {"event_ticker": event_ticker})
    return data.get("markets", [])


def market_trades(ticker, max_trades=1500):
    """Trades are returned newest-first. Walk back via cursor until we've got
    enough, then return a chronological list of tuples (datetime_utc, yes_price)."""
    out = []
    cursor = ""
    for _ in range(15):
        params = {"ticker": ticker, "limit": 1000}
        if cursor:
            params["cursor"] = cursor
        data = fetch("/markets/trades", params)
        trades = data.get("trades", [])
        if not trades:
            break
        for t in trades:
            try:
                ts = datetime.datetime.fromisoformat(t["created_time"].replace("Z", "+00:00"))
                out.append((ts, float(t["yes_price_dollars"]), float(t.get("count_fp", 0))))
            except Exception:
                continue
        cursor = data.get("cursor", "")
        if not cursor or len(out) >= max_trades:
            break
        time.sleep(0.15)
    return out


def parse_event_ticker(ticker):
    """KXVALORANTGAME-26MAY102000NRGSEN → date + team1 + team2 markers."""
    try:
        suffix = ticker.split("-", 1)[1]
        # "26MAY102000NRGSEN" — year(2) + month(3) + day(2) + hhmm(4) + teams
        year = 2000 + int(suffix[:2])
        month_str = suffix[2:5]
        day = int(suffix[5:7])
        hour = int(suffix[7:9])
        mnt = int(suffix[9:11])
        teams = suffix[11:]
        month_map = {"JAN":1,"FEB":2,"MAR":3,"APR":4,"MAY":5,"JUN":6,
                     "JUL":7,"AUG":8,"SEP":9,"OCT":10,"NOV":11,"DEC":12}
        month = month_map.get(month_str)
        if not month:
            return None
        # EDT/EST? Kalshi shows match times in EDT/EST. Assume EDT (UTC-4).
        dt = datetime.datetime(year, month, day, hour, mnt, tzinfo=datetime.timezone(datetime.timedelta(hours=-4)))
        dt_utc = dt.astimezone(datetime.timezone.utc)
        return {"date_utc": dt_utc, "date_str": dt_utc.date().isoformat(), "teams_blob": teams}
    except Exception:
        return None


def pre_match_price(trades, match_start_utc, window_minutes=240):
    """Volume-weighted average yes_price across trades in the [match_start - window, match_start - 5min] interval."""
    cutoff_end = match_start_utc - datetime.timedelta(minutes=5)
    cutoff_start = match_start_utc - datetime.timedelta(minutes=window_minutes)
    in_window = [(ts, p, c) for (ts, p, c) in trades
                  if cutoff_start <= ts <= cutoff_end]
    if not in_window:
        # fall back to the last trade before match start
        before = sorted([(ts, p, c) for (ts, p, c) in trades if ts <= cutoff_end],
                        key=lambda x: x[0])
        if before:
            ts, p, c = before[-1]
            return {"price": p, "n_trades": 1, "volume": c, "method": "fallback_last"}
        return None
    total_vol = sum(c for _, _, c in in_window)
    if total_vol <= 0:
        avg_price = sum(p for _, p, _ in in_window) / len(in_window)
    else:
        avg_price = sum(p * c for _, p, c in in_window) / total_vol
    return {"price": avg_price, "n_trades": len(in_window), "volume": total_vol, "method": "vwap"}


def main():
    print("Fetching all settled Valorant events…", flush=True)
    events = all_settled_events()
    print(f"  got {len(events)} events")

    # Cache
    cache_path = os.path.join(ROOT, "data", "kalshi_valorant.json")
    rows = []
    for i, e in enumerate(events, 1):
        ticker = e.get("event_ticker")
        title = e.get("title", "")
        parsed = parse_event_ticker(ticker)
        if not parsed:
            continue
        # Only 2026
        if parsed["date_utc"].year != 2026:
            continue
        if i % 25 == 0:
            print(f"  [{i}/{len(events)}] {ticker}", flush=True)
        markets = event_markets(ticker)
        if len(markets) != 2:
            continue
        # Build a base row
        ms = []
        for m in markets:
            sub = m.get("yes_sub_title") or m.get("no_sub_title")
            trades = market_trades(m["ticker"])
            pmp = pre_match_price(trades, parsed["date_utc"])
            ms.append({
                "ticker": m["ticker"],
                "team": sub,
                "result": m.get("result"),
                "volume_total": float(m.get("volume_fp", 0)),
                "pre_match": pmp,
                "n_trades_total": len(trades),
            })
        # Identify winner
        winner = None
        for mm in ms:
            if mm["result"] == "yes":
                winner = mm["team"]
        row = {
            "event_ticker": ticker,
            "title": title,
            "match_start_utc": parsed["date_utc"].isoformat(),
            "date": parsed["date_str"],
            "team_a_kalshi": ms[0]["team"],
            "team_b_kalshi": ms[1]["team"],
            "team_a_pre_yes": ms[0]["pre_match"]["price"] if ms[0]["pre_match"] else None,
            "team_b_pre_yes": ms[1]["pre_match"]["price"] if ms[1]["pre_match"] else None,
            "team_a_n_trades_window": ms[0]["pre_match"]["n_trades"] if ms[0]["pre_match"] else 0,
            "team_b_n_trades_window": ms[1]["pre_match"]["n_trades"] if ms[1]["pre_match"] else 0,
            "team_a_n_trades_total": ms[0]["n_trades_total"],
            "team_b_n_trades_total": ms[1]["n_trades_total"],
            "team_a_volume_total": ms[0]["volume_total"],
            "team_b_volume_total": ms[1]["volume_total"],
            "winner_kalshi": winner,
        }
        rows.append(row)
        if i % 25 == 0:
            with open(cache_path, "w") as f:
                json.dump(rows, f, indent=2)
        time.sleep(0.15)

    with open(cache_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved {len(rows)} settled 2026 matches to {cache_path}")


if __name__ == "__main__":
    main()
