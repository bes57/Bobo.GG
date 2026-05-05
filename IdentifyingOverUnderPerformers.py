import os, re, time, json
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from flask import Blueprint, render_template_string

article_overunder_bp = Blueprint("article_overunder", __name__)

ROOT               = os.path.dirname(os.path.abspath(__file__))
DATA_DIR           = os.path.join(ROOT, "data")
CACHE_PATH         = os.path.join(DATA_DIR, "article_kickoff_roles.json")
ALL_ROLES_DATA_PATH = os.path.join(DATA_DIR, "article_all_roles_data.json")
HEADERS  = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"}

AGENT_ROLES = {
    "jett": "Duelist", "reyna": "Duelist", "raze": "Duelist",
    "neon": "Duelist", "yoru": "Duelist", "iso": "Duelist", "waylay": "Duelist",
    "sova": "Initiator", "breach": "Initiator", "fade": "Initiator",
    "kayo": "Initiator", "gekko": "Initiator", "tejo": "Initiator",
    "omen": "Controller", "brimstone": "Controller", "viper": "Controller",
    "astra": "Controller", "harbor": "Controller", "clove": "Controller",
    "killjoy": "Sentinel", "cypher": "Sentinel", "sage": "Sentinel",
    "chamber": "Sentinel", "deadlock": "Sentinel", "vyse": "Sentinel",
}


def get_role_chart_data():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH) as f:
            return json.load(f)

    maps_df = pd.read_csv(os.path.join(DATA_DIR, "maps", "2026_kickoff.csv"))
    maps_df = maps_df[maps_df["Region"].isin(["Americas", "EMEA", "Pacific"])]
    maps_df = maps_df[maps_df["MapNum"].astype(str) != "all"]
    maps_df["R2.0"]   = pd.to_numeric(maps_df["R2.0"], errors="coerce")
    maps_df["MatchID"] = maps_df["MatchID"].astype(str)
    maps_df["MapNum"]  = maps_df["MapNum"].astype(str)
    maps_df = maps_df.dropna(subset=["R2.0"])

    # Scrape agent per (match, game_id, player) from VLR match pages
    agent_map = {}
    for match_id in maps_df["MatchID"].unique():
        url = f"https://www.vlr.gg/{match_id}/"
        try:
            res  = requests.get(url, headers=HEADERS, timeout=20)
            soup = BeautifulSoup(res.text, "html.parser")
            for game_div in soup.select("div.vm-stats-game"):
                game_id = game_div.get("data-game-id", "")
                if game_id == "all":
                    continue
                for table in game_div.select("table.wf-table-inset.mod-overview"):
                    for tr in table.select("tbody tr"):
                        tds = tr.find_all("td")
                        if len(tds) < 3:
                            continue
                        a = tds[0].find("a", href=True)
                        if not a:
                            continue
                        profile_url = "https://www.vlr.gg" + a["href"]
                        for td in tds:
                            for img in td.find_all("img"):
                                m = re.search(r'/agents/([^.]+)\.png', img.get("src", ""))
                                if m:
                                    agent_map[(match_id, game_id, profile_url)] = m.group(1).lower()
                                    break
                            if (match_id, game_id, profile_url) in agent_map:
                                break
            time.sleep(0.5)
        except Exception as e:
            print(f"Error scraping match {match_id}: {e}")

    maps_df["agent"] = maps_df.apply(
        lambda r: agent_map.get((r["MatchID"], r["MapNum"], r["ProfileURL"])), axis=1
    )
    maps_df["Role"] = maps_df["agent"].map(AGENT_ROLES)
    maps_df = maps_df.dropna(subset=["Role"])

    roles   = ["Duelist", "Initiator", "Controller", "Sentinel"]
    regions = ["Americas", "EMEA", "Pacific"]
    grouped = maps_df.groupby(["Region", "Role"])["R2.0"].mean()

    data = {}
    for region in regions:
        data[region] = []
        for role in roles:
            try:
                data[region].append(round(float(grouped.loc[region, role]), 3))
            except KeyError:
                data[region].append(None)

    result = {"roles": roles, "regions": regions, "data": data}
    with open(CACHE_PATH, "w") as f:
        json.dump(result, f)
    return result


def get_scatter_data():
    event_df = pd.read_csv(os.path.join(DATA_DIR, "2026_kickoff.csv"))
    event_df = event_df[event_df["Region"].isin(["Americas", "EMEA", "Pacific"])]
    event_df["R2.0"] = pd.to_numeric(event_df["R2.0"], errors="coerce")
    event_df["Rnd"]  = pd.to_numeric(event_df["Rnd"],  errors="coerce")
    event_df = event_df.dropna(subset=["R2.0"])
    event_df = event_df[event_df["Rnd"] > 75]

    maps_df = pd.read_csv(os.path.join(DATA_DIR, "maps", "2026_kickoff.csv"))
    maps_df = maps_df[maps_df["MapNum"].astype(str) != "all"]
    maps_df["MatchID"] = maps_df["MatchID"].astype(str)
    maps_df["MapNum"]  = maps_df["MapNum"].astype(str)

    results_df = pd.read_csv(os.path.join(DATA_DIR, "match_results.csv"))
    results_df = results_df[results_df["MapNum"].astype(str) != "all"]
    results_df["MatchID"] = results_df["MatchID"].astype(str)
    results_df["MapNum"]  = results_df["MapNum"].astype(str)

    # Filter match_results to kickoff match IDs only
    kickoff_ids = maps_df["MatchID"].unique()
    results_df  = results_df[results_df["MatchID"].isin(kickoff_ids)]

    map_orgs = maps_df.groupby(["MatchID", "MapNum"])["Org"].apply(lambda x: list(x.unique())).reset_index()
    merged   = results_df.merge(map_orgs, on=["MatchID", "MapNum"], how="inner")

    org_rounds = {}
    for _, row in merged.iterrows():
        winner_org = row["WinnerOrg"]
        try:
            w, l = [int(x) for x in str(row["Score"]).split("-")]
        except Exception:
            continue
        loser_orgs = [o for o in row["Org"] if o != winner_org]
        if not loser_orgs:
            continue
        loser_org = loser_orgs[0]
        for org, won, lost in [(winner_org, w, l), (loser_org, l, w)]:
            if org not in org_rounds:
                org_rounds[org] = [0, 0]
            org_rounds[org][0] += won
            org_rounds[org][1] += lost

    org_ratio = {
        org: round(won / (won + lost), 4)
        for org, (won, lost) in org_rounds.items()
        if (won + lost) > 0
    }

    event_df["round_ratio"] = event_df["Org"].map(org_ratio)
    event_df = event_df.dropna(subset=["round_ratio"])

    points = {"Americas": [], "EMEA": [], "Pacific": []}
    for _, row in event_df.iterrows():
        region = row["Region"]
        if region in points:
            points[region].append({
                "x": row["round_ratio"],
                "y": round(float(row["R2.0"]), 3),
                "player": row["Player"],
                "org": row["Org"],
                "profile": row["ProfileURL"],
            })
    return points


def get_all_roles_scatter_data():
    if not os.path.exists(ALL_ROLES_DATA_PATH):
        return {}
    with open(ALL_ROLES_DATA_PATH) as f:
        return json.load(f)


def get_overall_by_role(role_data):
    overall = []
    for i in range(len(role_data["roles"])):
        vals = [role_data["data"][r][i] for r in role_data["regions"] if role_data["data"][r][i] is not None]
        overall.append(round(sum(vals) / len(vals), 3) if vals else None)
    return overall


def get_residuals_tables(all_roles_data, model_params):
    all_points = []
    for role, pts in all_roles_data.items():
        m = model_params.get(role)
        if not m:
            continue
        for pt in pts:
            expected = m["intercept"] + m["slope"] * pt["x"]
            residual  = pt["y"] - expected
            all_points.append({
                "player":   pt["player"],
                "org":      pt["org"],
                "profile":  pt["profile"],
                "event":    pt["event"],
                "role":     role,
                "rating":   pt["y"],
                "expected": round(expected, 3),
                "residual": round(residual, 4),
            })

    std_dev = float(np.std([p["residual"] for p in all_points]))
    for p in all_points:
        p["z"] = round(p["residual"] / std_dev, 2)

    def top(lst, reverse=True):
        return sorted(lst, key=lambda p: p["residual"], reverse=reverse)[:10]

    kickoff = [p for p in all_points if p["event"] == "2026 Kickoff"]
    return {
        "alltime_best":  top(all_points, reverse=True),
        "alltime_worst": top(all_points, reverse=False),
        "kickoff_best":  top(kickoff,    reverse=True),
        "kickoff_worst": top(kickoff,    reverse=False),
    }


_DOMESTIC_EVENT_IDS = [
    "2023_league", "2024_kickoff", "2024_stage1", "2024_stage2",
    "2025_kickoff", "2025_stage1", "2025_stage2", "2026_kickoff",
]
_DOMESTIC_EVENT_LABELS = {
    "2023_league": "2023 League", "2024_kickoff": "2024 Kickoff",
    "2024_stage1": "2024 Stage 1", "2024_stage2": "2024 Stage 2",
    "2025_kickoff": "2025 Kickoff", "2025_stage1": "2025 Stage 1",
    "2025_stage2": "2025 Stage 2", "2026_kickoff": "2026 Kickoff",
}

def get_alltime_no_baiters(all_roles_data, model_params):
    omitted_keys = set()
    omitted = []
    for eid in _DOMESTIC_EVENT_IDS:
        path = os.path.join(DATA_DIR, f"{eid}.csv")
        if not os.path.exists(path):
            continue
        label = _DOMESTIC_EVENT_LABELS[eid]
        df = pd.read_csv(path)
        df = df[df["Region"].isin(["Americas", "EMEA", "Pacific"])]
        df["FDPR"] = pd.to_numeric(df["FDPR"], errors="coerce")
        df["Rnd"]  = pd.to_numeric(df["Rnd"],  errors="coerce")
        df = df[df["Rnd"] > 75].dropna(subset=["FDPR"])
        for _, row in df.loc[df.groupby("Org")["FDPR"].idxmin()].iterrows():
            omitted_keys.add((row["ProfileURL"], label))
            omitted.append({"org": row["Org"], "player": row["Player"],
                            "event": label, "fdpr": round(float(row["FDPR"]), 3)})
    omitted.sort(key=lambda x: x["fdpr"])

    all_res = []
    for role, pts in all_roles_data.items():
        m = model_params.get(role)
        if not m:
            continue
        for pt in pts:
            all_res.append(pt["y"] - (m["intercept"] + m["slope"] * pt["x"]))
    std_dev = float(np.std(all_res))

    all_pts = []
    for role, pts in all_roles_data.items():
        m = model_params.get(role)
        if not m:
            continue
        for pt in pts:
            if (pt["profile"], pt["event"]) in omitted_keys:
                continue
            expected = m["intercept"] + m["slope"] * pt["x"]
            residual = pt["y"] - expected
            all_pts.append({
                "player":   pt["player"],
                "org":      pt["org"],
                "profile":  pt["profile"],
                "event":    pt["event"],
                "rating":   pt["y"],
                "expected": round(expected, 3),
                "residual": round(residual, 4),
                "z":        round(residual / std_dev, 2),
            })

    return {
        "top10":   sorted(all_pts, key=lambda p: p["residual"], reverse=True)[:10],
        "omitted": omitted,
    }


def get_fd_proportions():
    fd_map = {}
    for eid in _DOMESTIC_EVENT_IDS:
        path = os.path.join(DATA_DIR, f"{eid}.csv")
        if not os.path.exists(path):
            continue
        label = _DOMESTIC_EVENT_LABELS[eid]
        df = pd.read_csv(path)
        df = df[df["Region"].isin(["Americas", "EMEA", "Pacific"])]
        df["FD"]  = pd.to_numeric(df["FD"],  errors="coerce")
        df["Rnd"] = pd.to_numeric(df["Rnd"], errors="coerce")
        df = df[df["Rnd"] > 75].dropna(subset=["FD"])
        team_fd = df.groupby("Org")["FD"].sum()
        for _, row in df.iterrows():
            total = float(team_fd.get(row["Org"], 0))
            if total > 0:
                fd_map[(row["ProfileURL"], label)] = round(float(row["FD"]) / total, 4)
    return fd_map


def get_model_with_fd(all_roles_data, fd_proportions):
    roles   = ["Initiator", "Controller", "Duelist", "Sentinel", "Flex"]
    ref     = "Initiator"
    non_ref = [r for r in roles if r != ref]

    xs, ys, fds, lbls = [], [], [], []
    for role in roles:
        for pt in all_roles_data.get(role, []):
            fd = fd_proportions.get((pt["profile"], pt["event"]))
            if fd is None:
                continue
            xs.append(pt["x"]); ys.append(pt["y"]); fds.append(fd); lbls.append(role)
    if not xs:
        return {}

    xs   = np.array(xs,  dtype=float)
    ys   = np.array(ys,  dtype=float)
    fds  = np.array(fds, dtype=float)
    lbls = np.array(lbls)

    cols = [np.ones(len(xs)), xs, fds]
    for role in non_ref:
        d = (lbls == role).astype(float)
        cols.extend([d, d * xs])
    X = np.column_stack(cols)

    coeffs, _, _, _ = np.linalg.lstsq(X, ys, rcond=None)
    b0, b1, b2 = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])

    role_params = {}
    for role in roles:
        if role == ref:
            intercept, slope = b0, b1
        else:
            i2 = 3 + non_ref.index(role) * 2
            intercept = b0 + float(coeffs[i2])
            slope     = b1 + float(coeffs[i2 + 1])
        role_params[role] = {"intercept": round(intercept, 4), "slope": round(slope, 4)}

    y_hat   = X @ coeffs
    res_arr = ys - y_hat
    std_dev = float(np.std(res_arr))

    all_pts = []
    idx = 0
    for role in roles:
        for pt in all_roles_data.get(role, []):
            fd = fd_proportions.get((pt["profile"], pt["event"]))
            if fd is None:
                continue
            all_pts.append({
                "player":   pt["player"],
                "org":      pt["org"],
                "profile":  pt["profile"],
                "event":    pt["event"],
                "rating":   pt["y"],
                "expected": round(float(y_hat[idx]), 3),
                "residual": round(float(res_arr[idx]), 4),
                "z":        round(float(res_arr[idx]) / std_dev, 2),
            })
            idx += 1

    return {
        "fd_coeff": round(b2, 4),
        "top10":    sorted(all_pts, key=lambda p: p["residual"], reverse=True)[:10],
    }


def get_ptd_values():
    ptd_map = {}
    for eid in _DOMESTIC_EVENT_IDS:
        path = os.path.join(DATA_DIR, f"{eid}.csv")
        if not os.path.exists(path):
            continue
        label = _DOMESTIC_EVENT_LABELS[eid]
        df = pd.read_csv(path)
        df = df[df["Region"].isin(["Americas", "EMEA", "Pacific"])]
        df["D"]   = pd.to_numeric(df["D"],   errors="coerce")
        df["Rnd"] = pd.to_numeric(df["Rnd"], errors="coerce")
        df = df[df["Rnd"] > 75].dropna(subset=["D", "Rnd"])
        df["death_rate"] = df["D"] / df["Rnd"]
        team_rate  = df.groupby("Org")["death_rate"].sum()
        team_count = df.groupby("Org").size()
        for _, row in df.iterrows():
            total = float(team_rate.get(row["Org"], 0))
            n     = int(team_count.get(row["Org"], 5))
            if total > 0:
                ptd = float(row["death_rate"]) / total * (n / 5)
                ptd_map[(row["ProfileURL"], label)] = round(ptd, 4)
    return ptd_map


def get_model_with_ptd(all_roles_data, model_params, ptd_values):
    roles   = ["Initiator", "Controller", "Duelist", "Sentinel", "Flex"]
    ref     = "Initiator"
    non_ref = [r for r in roles if r != ref]

    xs, ys, ptds, lbls, pt_info = [], [], [], [], []
    for role in roles:
        for pt in all_roles_data.get(role, []):
            ptd = ptd_values.get((pt["profile"], pt["event"]))
            if ptd is None:
                continue
            xs.append(pt["x"]); ys.append(pt["y"]); ptds.append(ptd)
            lbls.append(role); pt_info.append(pt)
    if not xs:
        return {}

    xs   = np.array(xs,   dtype=float)
    ys   = np.array(ys,   dtype=float)
    ptds = np.array(ptds, dtype=float)
    lbls = np.array(lbls)

    def role_cols(x_arr, l_arr):
        cols = [np.ones(len(x_arr)), x_arr]
        for role in non_ref:
            d = (l_arr == role).astype(float)
            cols.extend([d, d * x_arr])
        return np.column_stack(cols)

    X_old = role_cols(xs, lbls)
    X_new = np.column_stack([np.ones(len(xs)), xs, ptds] +
                            [c for role in non_ref
                             for c in [(lbls == role).astype(float),
                                       (lbls == role).astype(float) * xs]])

    coeffs_old, _, _, _ = np.linalg.lstsq(X_old, ys, rcond=None)
    coeffs_new, _, _, _ = np.linalg.lstsq(X_new, ys, rcond=None)
    y_hat_old = X_old @ coeffs_old
    y_hat_new = X_new @ coeffs_new

    ss_tot = float(np.sum((ys - np.mean(ys)) ** 2))
    r2_old = round(1 - float(np.sum((ys - y_hat_old) ** 2)) / ss_tot, 4)
    r2_new = round(1 - float(np.sum((ys - y_hat_new) ** 2)) / ss_tot, 4)

    res_new = ys - y_hat_new
    std_dev = float(np.std(res_new))

    all_pts = []
    for i, pt in enumerate(pt_info):
        all_pts.append({
            "player":   pt["player"],
            "org":      pt["org"],
            "profile":  pt["profile"],
            "event":    pt["event"],
            "rating":   pt["y"],
            "expected": round(float(y_hat_new[i]), 3),
            "residual": round(float(res_new[i]), 4),
            "z":        round(float(res_new[i]) / std_dev, 2),
        })

    non_ref_order = non_ref
    role_eqs = {}
    # X_new cols: intercept, x, ptd, then for each non_ref: dummy, dummy*x
    b = coeffs_new
    b0, b1, b_ptd = float(b[0]), float(b[1]), float(b[2])
    role_eqs[ref] = {"intercept": round(b0, 4), "slope": round(b1, 4), "ptd": round(b_ptd, 4)}
    for k, role in enumerate(non_ref_order):
        base_col = 3 + k * 2
        role_eqs[role] = {
            "intercept": round(b0 + float(b[base_col]), 4),
            "slope":     round(b1 + float(b[base_col + 1]), 4),
            "ptd":       round(b_ptd, 4),
        }

    kickoff = [p for p in all_pts if p["event"] == "2026 Kickoff"]
    by_res  = sorted(all_pts, key=lambda p: p["residual"], reverse=True)
    patmen  = next((p for p in all_pts if p["profile"] == "https://www.vlr.gg/player/13744/patmen"
                    and p["event"] == "2026 Kickoff"), None)

    return {
        "r2_old":        r2_old,
        "r2_new":        r2_new,
        "role_eqs":      role_eqs,
        "alltime_best":  by_res[:10],
        "alltime_worst": by_res[-10:][::-1],
        "kickoff_best":  sorted(kickoff, key=lambda p: p["residual"], reverse=True)[:10],
        "kickoff_worst": sorted(kickoff, key=lambda p: p["residual"])[:10],
        "patmen":        patmen,
    }


def get_ptd_all_scatter(all_roles_data, ptd_values):
    seen, result = set(), []
    for role, pts in all_roles_data.items():
        for pt in pts:
            key = (pt["profile"], pt["event"])
            if key in seen:
                continue
            ptd = ptd_values.get(key)
            if ptd is None:
                continue
            seen.add(key)
            result.append({"player": pt["player"], "org": pt["org"], "profile": pt["profile"],
                           "event": pt["event"], "x": ptd, "y": pt["y"]})
    return result


def get_oxy_ptd_chart_data(ptd_values):
    oxy_profile = "https://www.vlr.gg/player/18796/oxy"
    oxy_ptd = ptd_values.get((oxy_profile, "2024 Stage 1"))
    if oxy_ptd is None:
        return {}
    arr = np.array(list(ptd_values.values()), dtype=float)
    mean = float(arr.mean())
    std  = float(arr.std())
    pct  = round(float(np.mean(arr < oxy_ptd)) * 100, 1)
    return {"ptd": round(oxy_ptd, 4), "mean": round(mean, 4), "std": round(std, 4), "percentile": pct}


def get_ptd_kickoff_analysis(all_roles_data, ptd_values):
    kickoff_label = "2026 Kickoff"
    seen, players = set(), []
    for role, pts in all_roles_data.items():
        for pt in pts:
            if pt["event"] != kickoff_label or pt["profile"] in seen:
                continue
            ptd = ptd_values.get((pt["profile"], kickoff_label))
            if ptd is None:
                continue
            seen.add(pt["profile"])
            players.append({"player": pt["player"], "org": pt["org"], "profile": pt["profile"],
                            "region": pt["region"], "rating": pt["y"], "ptd": ptd})

    scatter = {"Americas": [], "EMEA": [], "Pacific": []}
    for p in players:
        if p["region"] in scatter:
            scatter[p["region"]].append({"x": p["ptd"], "y": p["rating"],
                                         "player": p["player"], "org": p["org"], "profile": p["profile"]})

    # All-time bottom 10 PTD across all domestic events
    alltime_seen, alltime_players = set(), []
    for role, pts in all_roles_data.items():
        for pt in pts:
            key = (pt["profile"], pt["event"])
            if key in alltime_seen:
                continue
            ptd = ptd_values.get(key)
            if ptd is None:
                continue
            alltime_seen.add(key)
            alltime_players.append({"player": pt["player"], "org": pt["org"], "profile": pt["profile"],
                                    "region": pt["region"], "event": pt["event"], "rating": pt["y"], "ptd": ptd})

    return {
        "top10":           sorted(players, key=lambda p: p["ptd"], reverse=True)[:10],
        "bottom10":        sorted(alltime_players, key=lambda p: p["ptd"])[:10],
        "scatter":         scatter,
    }


def get_model_coefficients(all_roles_data):
    roles   = ["Initiator", "Controller", "Duelist", "Sentinel", "Flex"]
    ref     = "Initiator"
    non_ref = [r for r in roles if r != ref]

    xs, ys, lbls = [], [], []
    for role in roles:
        for pt in all_roles_data.get(role, []):
            xs.append(pt["x"]); ys.append(pt["y"]); lbls.append(role)
    if not xs:
        return {}

    xs   = np.array(xs,   dtype=float)
    ys   = np.array(ys,   dtype=float)
    lbls = np.array(lbls)

    cols = [np.ones(len(xs)), xs]
    for role in non_ref:
        d = (lbls == role).astype(float)
        cols.extend([d, d * xs])
    X = np.column_stack(cols)

    coeffs, _, _, _ = np.linalg.lstsq(X, ys, rcond=None)
    b0, b1 = float(coeffs[0]), float(coeffs[1])

    result = {}
    for role in roles:
        if role == ref:
            intercept, slope = b0, b1
        else:
            i2 = 2 + non_ref.index(role) * 2
            intercept = b0 + float(coeffs[i2])
            slope     = b1 + float(coeffs[i2 + 1])
        mask  = lbls == role
        rx, ry = xs[mask], ys[mask]
        if len(rx) > 1:
            y_hat  = intercept + slope * rx
            ss_res = float(np.sum((ry - y_hat) ** 2))
            ss_tot = float(np.sum((ry - np.mean(ry)) ** 2))
            r2     = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        else:
            r2 = 0.0
        result[role] = {
            "intercept": round(intercept, 4),
            "slope":     round(slope,     4),
            "r2":        round(r2,        4),
        }
    return result


PAGE_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Overperforming in VCT: who's doing it? — Bobo's VCT Database</title>
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
  .page { position:relative; z-index:1; flex:1; display:flex; flex-direction:column; align-items:center; padding:60px 32px 80px; }
  .back { align-self:flex-start; font-size:.82rem; color:var(--soft); text-decoration:none; font-weight:400; margin-bottom:40px; }
  .back:hover { color:var(--ink); }
  .toc { position:fixed; top:32px; right:32px; background:white; border-radius:16px; padding:20px 24px; box-shadow:0 4px 24px #0000000f; display:flex; flex-direction:column; gap:6px; z-index:100; max-width:220px; }
  .toc-title { font-family:'Syne',sans-serif; font-size:.7rem; font-weight:800; letter-spacing:.1em; text-transform:uppercase; color:var(--soft); margin-bottom:4px; }
  .toc a { font-size:.78rem; color:var(--soft); text-decoration:none; font-weight:400; transition:color .15s; line-height:1.4; }
  .toc a:hover { color:var(--ink); }
  .toc a.active { color:var(--ink); font-weight:500; }
  @media(max-width:900px) { .toc { display:none; } }
  .article { max-width:860px; width:100%; }
  .label { font-family:'Syne',sans-serif; font-size:.7rem; font-weight:800; letter-spacing:.14em; text-transform:uppercase; color:var(--soft); margin-bottom:16px; }
  h1 { font-family:'Syne',sans-serif; font-size:clamp(2rem,5vw,3.2rem); font-weight:800; letter-spacing:-1px; line-height:1.1; margin-bottom:24px; }
  .byline { font-size:.82rem; color:var(--soft); font-weight:300; margin-bottom:48px; padding-bottom:32px; border-bottom:1px solid #e8e0ec; }
  .cover { width:100%; border-radius:16px; overflow:hidden; margin-bottom:12px; }
  .cover img { width:100%; height:auto; display:block; }
  .cover-caption { font-size:.75rem; color:var(--soft); font-weight:300; font-style:italic; margin-bottom:48px; }
  .content p { font-size:1rem; font-weight:300; line-height:1.8; color:var(--ink); margin-bottom:24px; }
  .content h2 { font-family:'Syne',sans-serif; font-size:1.4rem; font-weight:800; letter-spacing:-0.5px; margin:48px 0 20px; }
  .content a { color:var(--ink); font-weight:400; }
  .content a:hover { opacity:.7; }
  .chart-wrap { background:white; border-radius:20px; padding:28px 28px 20px; box-shadow:0 4px 24px #0000000a; margin:32px 0 32px; }
  .player-ref { cursor:pointer; text-decoration:underline; text-underline-offset:2px; }
  .chart-canvas-box { position:relative; height:360px; }
  .player-popup { position:fixed; background:white; border-radius:16px; padding:16px 18px; box-shadow:0 8px 40px #00000022; z-index:100; text-align:center; min-width:150px; pointer-events:none; opacity:0; transform:scale(0.92) translateY(4px); transition:opacity .15s ease, transform .15s ease; }
  .player-popup.visible { opacity:1; transform:scale(1) translateY(0); }
  .player-popup img { width:72px; height:72px; border-radius:50%; object-fit:cover; margin-bottom:10px; display:block; margin-left:auto; margin-right:auto; }
  .player-popup .popup-name { font-family:'Syne',sans-serif; font-size:.95rem; font-weight:800; color:var(--ink); pointer-events:auto; cursor:pointer; }
  .player-popup .popup-org  { font-size:.75rem; color:var(--soft); font-weight:300; margin-top:2px; margin-bottom:10px; }
  .player-popup .popup-stats { font-size:.75rem; color:var(--ink); font-weight:400; line-height:1.7; border-top:1px solid #f0eaf4; padding-top:8px; text-align:left; }
  .player-popup .popup-stats span { color:var(--soft); font-weight:300; }
  #rPopupAgentLabel { color:var(--ink); font-weight:400; }
  .chart-label { font-family:'Syne',sans-serif; font-size:.7rem; font-weight:800; letter-spacing:.1em; text-transform:uppercase; color:var(--soft); margin-bottom:16px; }
  footer { position:relative; z-index:1; text-align:center; padding:24px; color:var(--soft); font-size:.75rem; font-weight:300; }
  @keyframes fadeUp{from{opacity:0;transform:translateY(24px)}to{opacity:1;transform:translateY(0)}}
  .page { animation:fadeUp .6s ease both; }
  .clip-accordion { margin:16px 0 32px; }
  .clip-link { cursor:pointer; text-decoration:underline; text-underline-offset:2px; }
  .clip-toggle { background:none; border:1px solid #e0d8ea; border-radius:10px; padding:10px 18px; font-family:'DM Sans',sans-serif; font-size:.85rem; font-weight:400; color:var(--soft); cursor:pointer; transition:background .15s; }
  .clip-toggle:hover { background:#f0eaf4; }
  .clip-body { display:grid; grid-template-rows:0fr; transition:grid-template-rows .35s ease; overflow:hidden; }
  .clip-body.open { grid-template-rows:1fr; }
  .clip-inner { min-height:0; }
  .clip-frame-wrap { padding-top:16px; height:460px; position:relative; }
  .clip-frame-wrap iframe { width:100%; height:100%; border-radius:10px; border:none; }
  #clipVideo { width:100%; height:100%; border-radius:10px; display:block; }
  .side-notes { background:white; border-radius:16px; padding:24px 28px 16px; margin:-12px 0 40px; box-shadow:0 4px 24px #0000000a; }
  .side-notes-title { font-family:'Syne',sans-serif; font-size:.75rem; font-weight:800; letter-spacing:.12em; text-transform:uppercase; color:var(--soft); margin-bottom:16px; }
  .side-notes ul { list-style:none; margin:0; display:flex; flex-direction:column; gap:12px; }
  .side-notes ul li { font-size:.9rem; font-weight:300; line-height:1.7; color:var(--ink); padding-left:20px; position:relative; }
  .side-notes ul li::before { content:'—'; position:absolute; left:0; color:var(--soft); }
  .content ul { list-style:none; margin:-8px 0 24px; display:flex; flex-direction:column; gap:8px; }
  .content ul li { font-size:1rem; font-weight:300; line-height:1.8; padding-left:20px; position:relative; }
  .content ul li::before { content:'—'; position:absolute; left:0; color:var(--soft); }
  .content ul.bullet-list { list-style:disc; padding-left:24px; }
  .content ul.bullet-list li { padding-left:0; }
  .content ul.bullet-list li::before { content:none; }
  .content ol { list-style:decimal; margin:-8px 0 24px; display:flex; flex-direction:column; gap:8px; padding-left:24px; }
  .content ol li { font-size:1rem; font-weight:300; line-height:1.8; padding-left:8px; }
  .role-table { width:100%; border-collapse:collapse; font-size:.88rem; font-weight:300; margin-top:20px; }
  .role-table th { font-family:'Syne',sans-serif; font-size:.68rem; font-weight:800; text-transform:uppercase; letter-spacing:.08em; color:var(--soft); padding:8px 12px; text-align:left; border-bottom:2px solid #f0eaf4; }
  .role-table td { padding:8px 12px; border-bottom:1px solid #f0eaf4; color:var(--ink); }
  .model-equation { background:white; border-radius:16px; padding:20px 24px; margin:24px 0; box-shadow:0 4px 24px #0000000a; font-family:'Syne',sans-serif; font-size:1.1rem; font-weight:700; text-align:center; color:var(--ink); }
  .model-equation .eq-sub { font-size:.78rem; font-weight:300; font-family:'DM Sans',sans-serif; color:var(--soft); margin-top:8px; }
  .role-btns { display:flex; flex-wrap:wrap; gap:8px; margin-bottom:16px; }
  .role-btn { background:none; border:1.5px solid #e0d8ea; border-radius:10px; padding:7px 16px; font-family:'DM Sans',sans-serif; font-size:.82rem; font-weight:400; color:var(--soft); cursor:pointer; transition:background .15s,border-color .15s,color .15s; }
  .role-btn:hover { background:#f0eaf4; }
  .role-btn.active { background:var(--lavender); border-color:#b09ad4; color:var(--ink); font-weight:500; }
  .model-toggle { display:flex; align-items:center; gap:8px; margin-bottom:16px; font-size:.85rem; font-weight:300; color:var(--soft); cursor:pointer; user-select:none; }
  .model-toggle input { cursor:pointer; accent-color:var(--lavender); width:15px; height:15px; }
  .model-slope { font-size:.82rem; font-weight:300; color:var(--soft); margin-bottom:12px; }
  .res-table { width:100%; border-collapse:collapse; font-size:.85rem; font-weight:300; margin-top:16px; }
  .res-table th { font-family:'Syne',sans-serif; font-size:.65rem; font-weight:800; text-transform:uppercase; letter-spacing:.08em; color:var(--soft); padding:8px 10px; text-align:left; border-bottom:2px solid #f0eaf4; }
  .res-table td { padding:7px 10px; border-bottom:1px solid #f0eaf4; color:var(--ink); vertical-align:middle; }
  .res-table .td-img img { width:32px; height:32px; border-radius:50%; object-fit:cover; display:block; }
  .res-table .td-pos { color:#3aaa6e; font-weight:500; }
  .res-table .td-neg { color:#d45870; font-weight:500; }
  .omit-search { width:100%; border:1.5px solid #e0d8ea; border-radius:10px; padding:7px 14px; font-family:'DM Sans',sans-serif; font-size:.82rem; color:var(--ink); background:white; outline:none; margin-bottom:12px; }
  .interact-role { font-family:'DM Sans',sans-serif; font-size:.82rem; font-weight:500; padding:6px 14px; border-radius:20px; border:1.5px solid #e0d8ea; background:white; color:var(--ink); cursor:pointer; transition:all .15s; }
  .interact-role.active { background:#9060c4; border-color:#9060c4; color:white; }
  .interact-role:hover:not(.active) { border-color:#b09ad4; }
  .omit-search:focus { border-color:#b09ad4; }
  .omit-scroll { max-height:240px; overflow-y:auto; border-radius:8px; border:1px solid #f0eaf4; }
</style>
</head>
<body>
<nav class="toc">
  <div class="toc-title">Sections</div>
  <a href="#intro">Introduction</a>
  <a href="#tuning">Tuning the Definition</a>
  <a href="#mapping">Mapping the Definition</a>
  <a href="#model">Putting It All Together</a>
  <a href="#alltime">Preliminary Greatest Over/Underperformers</a>
  <a href="#baiting">The Baiting Problem</a>
  <a href="#ptd-model">Final PTD Model</a>
  <a href="#ptd-alltime">PTD: Updated Greatest Over/Underperformers</a>
  <a href="#conclusion">Conclusion</a>
</nav>
<div class="page">
  <a class="back" href="/">&larr; Back</a>
  <div class="article">
    <div class="label">Research / Opinion</div>
    <h1>Overperforming in VCT: who&rsquo;s doing it?</h1>
    <div class="byline">Bobo &mdash; May 2026</div>
    <div class="cover" id="intro">
      <img src="/static/Patmen.jpg" alt="Patmen at VCT Pacific Kickoff 2026">
    </div>
    <p class="cover-caption">Patmen at VCT Pacific Kickoff 2026, where GE went on to eventually finish 7th&ndash;8th &mdash; eliminated in the lower bracket by his former team, PRX.</p>
    <div class="content">
      <p>Earlier in VCT Pacific Kickoff 2026, Patmen pulled off one of the craziest VCT performances of all time, dropping 32 kills and 5 deaths on Breeze. Consequently, he had a 2.81 rating, which was (and is) the highest map rating of ALL TIME. Even crazier? Global Esports lost that match 1-2. Even crazier? They lost that match to DFM (the only team to end an entire VCT season without a single series win). Even crazier? He was playing OMEN. Even crazier? He didn&rsquo;t fist fight each and every one of his teammates (well, as far as we know).</p>
      <p>It seems clear that Patmen is better than the mediocrity that is Global Esports. However, it makes it even weirder to flash back to 6 months ago, when Patmen was dropped by PRX for being the weak link in that juggernaut of a team. The two realities seem contradictory. Can Patmen really exist in this void where he underperforms on the best teams, but he overperforms on the mediocre teams? However, in order to answer that question, we have to answer the question: what does it even mean to be an over/underperformer in VCT?</p>

      <h2 id="tuning">Tuning the Definition</h2>

      <p>Despite its detractions, the holistic approach underpinning VLR&rsquo;s rating system gives us as close to a good measure of player performance as we&rsquo;ve seen. Beyond just looking at simple (but important) numbers like kills and deaths, it weights kills based on the player count and the duel&rsquo;s subsequent weight on the round win percentage, accounts for economic differences between teams, looks at trades/tradability, and incorporates assists. For more details, you can read more about VLR rating <a href="https://www.vlr.gg/160667/vlr-gg-player-rating-explained" target="_blank">here</a>. At the end of the day, it emphasizes the basic yet important statistics while remaining wise to the fact that all kills and deaths are not made equal. Of course, there are still intangibles such as IGLing, midrounding, or &ldquo;vibes&rdquo; that have a real impact on the game, but I&rsquo;m yet to see a statistic that incorporates this as well. This is all to say that VLR rating will be my response variable for this article&rsquo;s sake.</p>

      <p>Next, I want to call back to a point I made in my sympathy rant for Patmen; he was playing Omen when he dropped that supercalifragilisticexpialidocious statline. For those who don&rsquo;t know much about Valorant or are ridiculously casual, Omen is a Controller (one of the four agent classes in this game), which means that his arsenal is centered around casting smokes for his team and other strategic/movement abilities (rather than offensive ones). This is noteworthy because that would, presumably, make it harder to rack up such a high rating on a passive and selfless role. Well, let&rsquo;s check if this initial presumption is even true by looking at the average VLR rating by role across all regions in 2026 Kickoff.</p>

      <div class="chart-wrap">
        <div class="chart-label">Avg. VLR Rating (in a Map) by Role &mdash; 2026 Kickoff (all regions)</div>
        <div class="chart-canvas-box"><canvas id="overallChart"></canvas></div>
      </div>

      <div class="chart-wrap">
        <div class="chart-label">Avg. VLR Rating (in a Map) by Role &mdash; 2026 Kickoff (by region)</div>
        <div class="chart-canvas-box"><canvas id="roleChart"></canvas></div>
      </div>


      <p>As you can see, there is a marginal, but notable difference in VLR rating by agent role. Surprisingly, it was the Sentinel role that had the highest rating. I postulate that Duelists are often dying first too frequently on site executes, Controllers and Initiators are forced to play with the pack, while Sentinels are fully able to both lurk and bait in whatever way suits their personal performance, but I digress. The important result here is the obvious difference by role (and for those who don&rsquo;t know, 0.03 is a non-negligible difference between rating averages by role).</p>

      <p>All that being said about roles, the larger point about Patmen was that his team wasn&rsquo;t winning at the rate you&rsquo;d expect from a player posting numbers like that <em>irrespective of the role he was playing</em>. Player rating and team success usually (and should) move together in Valorant. The theory here is that, when there&rsquo;s a wide gap between a player&rsquo;s rating and his team&rsquo;s round win %, it&rsquo;s a pretty clean signal that he&rsquo;s either hard carrying or getting hard carried. Let&rsquo;s see how this shakes out across Kickoff 2026 with a scatterplot of player VLR rating against round win %.</p>

      <div class="chart-wrap">
        <div class="chart-label">Player VLR Rating vs. Round Win % &mdash; 2026 Kickoff</div>
        <div class="chart-canvas-box"><canvas id="scatterChart"></canvas></div>
      </div>


      <p>So many fun observations to be had! First off, my initial assumption was true &mdash; there&rsquo;s a clear positive linear relationship between winning at a higher rate and higher VLR ratings, though any other result would be befuddling. Honestly, it&rsquo;s surprising that the trend isn&rsquo;t even more drastic. There are a decent number of players on the winning end of teams with sub-1.0 ratings (I&rsquo;m looking at you <span class="player-ref" onclick="highlightPlayer('jawgemo')">Jawgemo</span>), though the majority are healthily above 1. While we&rsquo;ll discuss this more explicitly later, you can hover your mouse over some of the notable datapoints in the top left/bottom right to see some of the biggest over/underperformers. Poor <span class="player-ref" onclick="highlightPlayer('lukxo')">Lukxo</span>, sitting there with a 1.13 rating at the lowest end of the Round Win % axis, a higher rating than any player on NS (the highest Round Win % team). At least he got this clip:</p>

      <div style="margin:16px 0 32px;height:460px;position:relative;">
        <video id="clipVideo" src="/static/lukxo_clip.mp4" controls playsinline style="width:100%;height:100%;border-radius:10px;display:block;"></video>
      </div>

      <aside class="side-notes">
        <div class="side-notes-title">Unrelated Side Notes on the Scatterplot</div>
        <ul>
          <li>The parity of EMEA compared to Americas and Pacific is visible in how the EMEA data points are more tightly spread around the 50% mark</li>
          <li>It&rsquo;s interesting to see which teams have players clustered around the same rating (e.g. PRX) and which teams have a wide range of player ratings (e.g. PCF or FNC)</li>
          <li>Will <span class="player-ref" onclick="highlightPlayer('primmie')">Primmie</span> have a team that&rsquo;s close to as good as him? That clearly didn&rsquo;t change this Kickoff.</li>
        </ul>
      </aside>

      <h2 id="mapping">Mapping the Definition</h2>

      <p>Now, what if we synthesize these two ideas: using roles and a team&rsquo;s round win % to observe trends in player performance? To do this, let&rsquo;s look at individual scatterplots, grouped by roles.</p>

      <p><em>A quick clarification of the methodology:</em></p>
      <ul>
        <li>To determine a player&rsquo;s &ldquo;role&rdquo;, I looked at their proportional breakdown of agent play counts in a given event and then assigned them to whichever role they played on a majority (&ge;50%) of their maps. In the case where there is no simple majority, as multiple roles occupy large proportions, they are classified as &ldquo;Flex&rdquo; and shown in a fifth chart. Additionally, as players are analyzed on a per-event basis, it&rsquo;s possible for a player to show up in multiple scatterplots if they switched roles.</li>
        <li>These graphs only use regional data, as international events introduce noise from varying competitiveness by region. Furthermore, they will skew the strongest teams&rsquo; numbers downward, since making a deep run at an international means you play the best teams from every other region. Meanwhile, in regional play, every team plays through a designated schedule, so a top team&rsquo;s round win % is buoyed by equal-level matchups.</li>
      </ul>

      <div class="chart-wrap">
        <div class="chart-label">Initiator Rating vs. Team Round Win % &mdash; All Domestic Events</div>
        <div class="chart-canvas-box"><canvas id="roleScatter_Initiator"></canvas></div>
      </div>

      <div class="chart-wrap">
        <div class="chart-label">Controller Rating vs. Team Round Win % &mdash; All Domestic Events</div>
        <div class="chart-canvas-box"><canvas id="roleScatter_Controller"></canvas></div>
      </div>

      <div class="chart-wrap">
        <div class="chart-label">Duelist Rating vs. Team Round Win % &mdash; All Domestic Events</div>
        <div class="chart-canvas-box"><canvas id="roleScatter_Duelist"></canvas></div>
      </div>

      <div class="chart-wrap">
        <div class="chart-label">Sentinel Rating vs. Team Round Win % &mdash; All Domestic Events</div>
        <div class="chart-canvas-box"><canvas id="roleScatter_Sentinel"></canvas></div>
      </div>

      <div class="chart-wrap">
        <div class="chart-label">Flex Player Rating vs. Team Round Win % &mdash; All Domestic Events</div>
        <div class="chart-canvas-box"><canvas id="roleScatter_Flex"></canvas></div>
      </div>

      <p>These graphs are treasure troves for highlights (and lowlights) of player performances throughout VCT history. But first, the important things to note:</p>
      <ul>
        <li>The main trend continues of a strong positive linear relationship between round win % and individual ratings. Just as before, different roles operate at different baseline rating scores and thus different expectations for what is &ldquo;overperforming&rdquo; and &ldquo;underperforming.&rdquo; Perhaps more interesting (and likely to come up later) is the fact that the slopes seem to be slightly different. For instance, the increase in ratings for the Duelist class is noticeably steeper than the Sentinel class. The implication here seems to be that a Duelist&rsquo;s individual performance is more closely bound by their team&rsquo;s performance than a Sentinel&rsquo;s, who has more skill expression allowed irrespective of team performance. Still, the positive correlation exists in both cases. Interesting!</li>
        <li>The scatterplots are clearly informative in the context of VCT. Players in the top-left quadrant of these scatterplots are players who are outperforming their team and deserve better. For instance, look at <span class="player-ref" onclick="highlightRolePlayer('invy', '2025 Stage 1')">Invy</span>&rsquo;s 2025 Stage 1 performance, where the scatterplot makes it obvious he has the pedigree to be a great player on a great team (hopefully one with a round win % higher than 45.8%). In fact, we now know this to be true, as he got scouted to PRX and led them to a second-place finish at Masters Santiago. In this same school of thought, it allows us to find other pros with hidden potential. For instance, a player like <span class="player-ref" onclick="highlightRolePlayer('UdoTan', '2025 Stage 1')">UdoTan</span> wouldn&rsquo;t stand out to you if you looked at VLR ratings in Pacific Stage 1; 1.04 is only moderately above average. However, given his team&rsquo;s horrendous performance, we know his rating is much more impressive than it may initially seem. Again, the scatterplots work.</li>
      </ul>
      <p>As for some personal thoughts, these graphs remind me of some of the notable individual forms we&rsquo;ve seen in VCT history. Trent&rsquo;s form in 2025 is something people seem to have moved past (probably because they choked at every international), but his datapoints for <span class="player-ref" onclick="highlightRolePlayer('trent', '2025 Kickoff')">Kickoff</span> and <span class="player-ref" onclick="highlightRolePlayer('trent', '2025 Stage 1')">Stage 1</span> are reminders of not only how insanely good he was, but how good he was even for a team that was as domestically dominant as G2 in 2025. People forget that he was in discussion for the greatest player in the world throughout 2025. What&rsquo;s more, he managed to put up those numbers without acing a single time in 2025 (weird but true).</p>
      <p>On the opposite end of the spectrum, if you go to the Controller scatterplot and just look at the lowest data points for high round win %, they&rsquo;re all Boaster. That checks out. However, this is a good reminder that these graphs are not gospel. Boaster is the backbone of Fnatic and does more for that organization&rsquo;s success than statistics will ever be able to show.</p>
      <p>Lastly, it seems like Gen.G were pretty warranted in dropping <span class="player-ref" onclick="highlightRolePlayer('Suggest', '2025 Stage 1')">Suggest</span>. There are more and more little anecdotes hidden amongst these scatterplots, but I&rsquo;ll leave that to you if you want to look around.</p>

      <h2 id="model">Putting It All Together</h2>
      <p>Finally, let&rsquo;s create a simple model that uses round win % relative to the role played to predict the expected VLR rating from a VCT pro.</p>

      <div class="model-equation">
        Rating = &alpha;<sub>role</sub> + &beta;<sub>role</sub> &times; Round Win %
      </div>

      <div class="chart-wrap">
        <div class="chart-label">Model Coefficients by Role</div>
        <table class="role-table">
          <thead><tr><th>Role</th><th>Intercept (&alpha;)</th><th>Slope (&beta;)</th></tr></thead>
          <tbody id="modelTableBody"></tbody>
        </table>
      </div>

      <div class="chart-wrap">
        <div class="chart-label">Line of Best Fit &mdash; Select a Role</div>
        <div class="role-btns" id="modelRoleBtns"></div>
        <div class="model-slope" id="modelSlopeLabel"></div>
        <label class="model-toggle"><input type="checkbox" id="showDotsCheck"> Show data points</label>
        <div class="chart-canvas-box"><canvas id="modelChart"></canvas></div>
      </div>

      <p>Adding an interaction effect between role and round win % reveals that among the four main roles, Duelist has the strongest positive relationship with round win % (&beta;&nbsp;&asymp;&nbsp;0.12) and Sentinel the weakest (&beta;&nbsp;&asymp;&nbsp;0.10), confirming my earlier observation.</p>

      <p>By looking at residuals (the difference between expected and true rating), we can get an initial sense for who the greatest over/underperformers are of both all-time and the current era!</p>

      <h2 id="alltime">Preliminary Greatest Over/Underperformers of All Time</h2>

      <p><em>Note: This model is not final, and thus neither are these rankings. The final rankings are further down.</em></p>

      <div class="chart-wrap">
        <div class="chart-label">Greatest Overperformers of All Time</div>
        <table class="res-table">
          <thead><tr><th></th><th>Player</th><th>Rating</th><th>Expected</th><th>Residual</th><th>Std. Devs.</th></tr></thead>
          <tbody id="resBody_alltime_best"></tbody>
        </table>
      </div>

      <div class="chart-wrap">
        <div class="chart-label">Greatest Underperformers of All Time</div>
        <table class="res-table">
          <thead><tr><th></th><th>Player</th><th>Rating</th><th>Expected</th><th>Residual</th><th>Std. Devs.</th></tr></thead>
          <tbody id="resBody_alltime_worst"></tbody>
        </table>
      </div>

      <h2 id="kickoff">For the Current Era (Kickoff 2026)</h2>

      <div class="chart-wrap">
        <div class="chart-label">Greatest Overperformers &mdash; Kickoff 2026</div>
        <table class="res-table">
          <thead><tr><th></th><th>Player</th><th>Rating</th><th>Expected</th><th>Residual</th><th>Std. Devs.</th></tr></thead>
          <tbody id="resBody_kickoff_best"></tbody>
        </table>
      </div>

      <div class="chart-wrap">
        <div class="chart-label">Greatest Underperformers &mdash; 2026 Kickoff</div>
        <table class="res-table">
          <thead><tr><th></th><th>Player</th><th>Rating</th><th>Expected</th><th>Residual</th><th>Std. Devs.</th></tr></thead>
          <tbody id="resBody_kickoff_worst"></tbody>
        </table>
      </div>

      <p>Awesome! Some quick thoughts:</p>
      <ul>
        <li>No Patmen in the top 10 overperformers of this Kickoff. Maybe I overreacted on his behalf.</li>
        <li>Every single member of the top 10 underperformers of all time is no longer in VCT, except for Derke (which makes sense as an all-time great) and Crws (who proceeded to get dropped and then, just recently, get picked back up by Full Sense). This initial analysis has some merit.</li>
        <li>2 of the 3 greatest &ldquo;overperformers&rdquo; of all time came from this most recent Kickoff &mdash; Johnqt and Hiro. Perhaps we should be more grateful for the quality of players that we&rsquo;re currently witnessing. I can&rsquo;t speak so much for Hiro as I didn&rsquo;t closely follow EMEA this Kickoff, but Johnqt did have a ridiculously good Kickoff. However, he was also accused of baiting and statfarming this Kickoff (hence the nickname JohnKD), which would potentially skew the validity of using rating as our outcome variable.</li>
      </ul>

      <h2 style="font-family:'Syne',sans-serif;font-weight:800;font-size:1.6rem;margin:40px 0 16px;letter-spacing:-0.5px;" id="baiting">The Baiting Problem</h2>

      <p>This is a genuine concern &mdash; if baiting inflates a player&rsquo;s rating beyond their actual contribution to the team, our model is rewarding them for it. What if we removed all players who had the lowest FDPR (First Deaths Per Round) on their team as a way of filtering out baiters? Let&rsquo;s see:</p>

      <div class="chart-wrap">
        <div class="chart-label">Greatest Overperformers of All Time (Excl. Lowest FDPR per Team per Event)</div>
        <table class="res-table">
          <thead><tr><th></th><th>Player</th><th>Rating</th><th>Expected</th><th>Residual</th><th>Std. Devs.</th></tr></thead>
          <tbody id="resBody_kickoff_nobaiters"></tbody>
        </table>
      </div>



      <p>Aaaaaaaaand there go Johnqt and Hiro. However, this feels a bit crude. I have no doubt that Hiro and Johnqt baited in order to get the statistics they got, but I also have no doubt that they still performed well <em>even after accounting for their baiting</em>. But how do you account for their baiting? I could just add FDPR into the model, but that comes with problems. Better teams will have fewer first deaths, so just plugging in this variable would be unwise, so it&rsquo;s not a &ldquo;fair&rdquo; variable per se. There&rsquo;s a difference between Zekken and Cauanzin having 0.11 FDPR in 2026 Kickoff (namely that Zekken is a selfless player who is just on a good team while Cauanzin is a massive baiter who&rsquo;s on a bad team).</p>


      <p>After thinking through this problem, I came to a solution &mdash; create a new variable that is the proportion of a player&rsquo;s deaths relative to their team&rsquo;s total deaths, namely:</p>

      <div id="ptdFormula" style="text-align:center;margin:24px 0;"></div>
      <div style="text-align:center;font-size:.85rem;color:var(--ink);margin-top:-12px;margin-bottom:24px;font-family:'KaTeX_Main',serif;">where <em>i</em> is the player and <em>j</em> &isin; {1, &hellip;, <em>n</em>} indexes all players on their team</div>

      <p>Using this, we can identify &ldquo;baiting&rdquo; while being fair to players on better teams. The name of this new variable, PTD, means Proportion of Team&rsquo;s Deaths.</p>

      <p>As a sanity check to make sure PTD was calculated correctly, let&rsquo;s make sure the PTDs for a given team add up to 100. Pun intended, here&rsquo;s 100 Thieves:</p>

      <div class="chart-wrap">
        <div class="chart-label">100 Thieves &mdash; PTD, Kickoff 2026</div>
        <table class="res-table">
          <thead><tr><th></th><th>Player</th><th>Rating</th><th>PTD</th></tr></thead>
          <tbody id="ptd100TBody"></tbody>
        </table>
      </div>

      <p>Checks out! Before moving on, let&rsquo;s see who the biggest baiters (lowest PTD) of all time are:</p>

      <div class="chart-wrap">
        <div class="chart-label">Lowest PTD &mdash; All Time (i.e. baiting)</div>
        <table class="res-table">
          <thead><tr><th></th><th>Player</th><th>Org</th><th>Event</th><th>Rating</th><th>PTD</th></tr></thead>
          <tbody id="ptdBottom10Body"></tbody>
        </table>
      </div>

      <p>Seems like we can move past &ldquo;accused of baiting&rdquo; and just say &ldquo;baiting&rdquo; for Johnqt.</p>

      <h2 style="font-family:'Syne',sans-serif;font-weight:800;font-size:1.6rem;margin:40px 0 16px;letter-spacing:-0.5px;" id="ptd-model">Final PTD Model</h2>

      <p>Now, let&rsquo;s add PTD as a variable to the model, which makes it now look like:</p>

      <div class="chart-wrap" id="ptdEqBox" style="font-family:'DM Sans',sans-serif;font-size:.9rem;line-height:2;"></div>

      <div class="chart-wrap">
        <div class="chart-label">Line of Best Fit &mdash; PTD Model &mdash; Select a Role &amp; Adjust PTD</div>
        <div class="role-btns" id="ptdModelRoleBtns"></div>
        <div style="display:flex;align-items:center;gap:12px;">
          <div class="model-slope" id="ptdModelSlopeLabel" style="flex:1;margin-bottom:0;"></div>
          <div style="display:flex;align-items:center;gap:6px;font-size:.78rem;font-weight:300;color:var(--soft);white-space:nowrap;">
            <span id="ptdBetaLabel"></span>
            <span style="margin:0 2px;">PTD</span>
            <input type="range" id="ptdSlider" min="10" max="30" value="20" step="1" style="width:80px;accent-color:var(--lavender);">
            <span id="ptdSliderLabel" style="min-width:28px;">20%</span>
          </div>
        </div>
        <label class="model-toggle" style="margin-top:10px;"><input type="checkbox" id="ptdShowDotsCheck"> Show data points</label>
        <div class="chart-canvas-box"><canvas id="ptdModelChart"></canvas></div>
      </div>

      <p>With this new model, let&rsquo;s look again at the greatest over/underperformers of all time.</p>

      <h3 style="font-family:'Syne',sans-serif;font-weight:800;font-size:1.2rem;margin:32px 0 12px;letter-spacing:-0.3px;" id="ptd-alltime">Updated Greatest Over/Underperformers of All Time</h3>

      <div class="chart-wrap">
        <div class="chart-label">Greatest Overperformers of All Time (PTD Model)</div>
        <table class="res-table">
          <thead><tr><th></th><th>Player</th><th>Rating</th><th>Expected</th><th>Residual</th><th>Std. Devs.</th><th>PTD</th></tr></thead>
          <tbody id="ptdAlltimeBestBody"></tbody>
        </table>
      </div>

      <div class="chart-wrap">
        <div class="chart-label">Greatest Underperformers of All Time (PTD Model)</div>
        <table class="res-table">
          <thead><tr><th></th><th>Player</th><th>Rating</th><th>Expected</th><th>Residual</th><th>Std. Devs.</th><th>PTD</th></tr></thead>
          <tbody id="ptdAlltimeWorstBody"></tbody>
        </table>
      </div>

      <h3 style="font-family:'Syne',sans-serif;font-weight:800;font-size:1.2rem;margin:32px 0 12px;letter-spacing:-0.3px;" id="ptd-kickoff">For the Current Era (Kickoff 2026)</h3>

      <div class="chart-wrap">
        <div class="chart-label">Greatest Overperformers &mdash; Kickoff 2026 (PTD Model)</div>
        <table class="res-table">
          <thead><tr><th></th><th>Player</th><th>Rating</th><th>Expected</th><th>Residual</th><th>Std. Devs.</th><th>PTD</th></tr></thead>
          <tbody id="ptdKickoffBestBody"></tbody>
        </table>
      </div>

      <div class="chart-wrap">
        <div class="chart-label">Greatest Underperformers &mdash; Kickoff 2026 (PTD Model)</div>
        <table class="res-table">
          <thead><tr><th></th><th>Player</th><th>Rating</th><th>Expected</th><th>Residual</th><th>Std. Devs.</th><th>PTD</th></tr></thead>
          <tbody id="ptdKickoffWorstBody"></tbody>
        </table>
      </div>

      <p>After all that work, we finally have a model that accounts for team performance, propensity to bait, and role played to see which players have vastly overperformed their team in a <em>meaningful manner</em> (re: accounting for baiting). Again, I find this fascinating. With these being my final findings, I have final thoughts:</p>

      <ol>
        <li>Congratulations to Oxy for winning my award for greatest performance of all time! I can&rsquo;t say I disagree with the model&rsquo;s assessment either. Oxy was dying at 23% (a ridiculously high rate) for a team that was composed of&hellip;
          <br><br>
          <ul class="bullet-list">
            <li>Runi (no longer in VCT)</li>
            <li>Moose (no longer in VCT)</li>
            <li>Vanity (no longer in VCT)</li>
            <li>Xeppaa (infamous paycheck stealer &mdash; for those who don&rsquo;t know, he&rsquo;s just straight up bad)</li>
          </ul>
          <div style="margin-top:16px;margin-bottom:8px;" class="chart-wrap">
            <div class="chart-label">Oxy &mdash; PTD vs. Every Domestic VCT Player in History</div>
            <div class="chart-canvas-box"><canvas id="oxyPtdNormalChart"></canvas></div>
          </div>
          <br>In fact, Oxy was taking more deaths for his team than 99.2% of players in domestic VCT history. Amidst playing this selflessly for teammates that we now know were middling (at best), he still managed to put up a 1.24 rating on a team that got bounced in the first round of Stage 1 playoffs.
        </li>
        <li>Florescent&rsquo;s 2025 Kickoff being in the top 10 greatest overperformances of all time is the kind of fascinating result that I hoped to discover. What&rsquo;s more, I appreciate this result because I&rsquo;ve been a strong believer that Florescent has what it takes to not just be a great player, but be a top-10 player in VCT. In fact, back in 2024, I predicted that Florescent would be a top 5 player in EMEA. Can I say I&rsquo;m right now? Probably not.<br><br>In any case, the model&rsquo;s result is a reminder of the squandered potential she consistently showed throughout the event. For instance:
          <div style="margin-top:16px;height:460px;position:relative;">
            <video id="floresentVideo" src="/static/florescent_clip.mp4" controls playsinline style="width:100%;height:100%;border-radius:10px;display:block;"></video>
          </div>
          <p style="margin-top:10px;margin-bottom:0;">The eye test is certainly passed. Recall that Florescent concluded 2025 EMEA Kickoff with a 1.17 rating (third-highest at the event) on a team that finished DEAD LAST, all while tanking an above-average amount of her team&rsquo;s deaths. A historically great performance on a historically horrible team.</p>
        </li>
        <li>Ironically, Aspas went from appearing twice in the &ldquo;Greatest Overperformers of All Time&rdquo; list to zero times after accounting for baiting. I&rsquo;m not saying anything, just noticing.</li>
        <li>Earlier, I wrote that &ldquo;I have no doubt that Hiro and Johnqt baited in order to get the statistics they got, but I also have no doubt that they still performed well even after accounting for their baiting.&rdquo; My new model affirms this. Johnqt&rsquo;s expected rating went from 0.96 to 1.01 while Hiro&rsquo;s went from 0.95 to 1.06. While it&rsquo;s clear neither of these performances was historically great enough to be in the top 10 for all time, they both still make the top 10 for Kickoff 2026. I&rsquo;m happy with this result, and it&rsquo;s as things should be. Overperformers are those who play better than their team while playing <em>with</em> them, not those who bait because their teams are subpar. Johnqt and Hiro are two great players who had great Kickoff performances, but it&rsquo;s less surprising when you consider the rate at which they were saving.</li>
        <li>Suggest may have had a 0.66 rating, but at least he wasn&rsquo;t baiting!</li>
        <li>Now, 9 (not 8) of the players in the all-time underperformers list are no longer in VCT. Even better!</li>
        <li>Based on the fact that the overperformers&rsquo; lists aren&rsquo;t just occupied with high PTD players and the inverse for the underperformers&rsquo; lists, this model seems fairly calibrated.</li>
        <li>If I&rsquo;m a team wanting to make changes, I&rsquo;d be looking at players like Seven or al0rante. Players like Primmie, Karon, and Lukxo are insane, but everyone already knows that.</li>
        <li>Based on this list for Kickoff 2026, I&rsquo;ll predict that at least 3 of C1ndeR, Okeanos, Eggster, and GLYPH will be dropped by the end of the year. The rest have already been dropped (thyy, d3mur, UNFAKE, and baha) or have too much historical credit (Boaster and Jawgemo).</li>
      </ol>

      <h2 style="font-family:'Syne',sans-serif;font-weight:800;font-size:1.6rem;margin:48px 0 20px;letter-spacing:-0.5px;" id="conclusion">Conclusion</h2>

      <p>Finally, we&rsquo;ve answered many questions: how do we understand being an &ldquo;overperformer&rdquo; or &ldquo;underperformer&rdquo;? Who are the greatest over/underperformers of all time? What about in current times? Yet, one question remains:</p><p>How much is Patmen truly &ldquo;better than the mediocrity that is Global Esports&rdquo;?</p>

      <div class="chart-wrap">
        <div class="chart-label">PatMen &mdash; 2026 Kickoff (Final PTD Model)</div>
        <table class="res-table">
          <thead><tr><th></th><th>Player</th><th>Rating</th><th>Expected</th><th>Residual</th><th>Std. Devs.</th><th>PTD</th></tr></thead>
          <tbody id="patmenBody"></tbody>
        </table>
      </div>

      <p>He&rsquo;s performing over a full standard-deviation&rsquo;s worth better than we&rsquo;d expect from a controller player on that Global Esports team. Also, he&rsquo;s doing it without baiting. It&rsquo;s not egregious, but he&rsquo;s certainly better than his team. No fist fighting is necessary, though.</p>

      <p>As a final gift, I&rsquo;ll leave an interactive version of this model to mess around with. Input values for a prospective VCT player and the model will give you their expected rating as well as the closest comparison we&rsquo;ve seen domestically, including their true values.</p>

      <div class="chart-wrap" id="interactiveModel">
        <div class="chart-label">Interactive Model</div>
        <div style="display:flex;flex-wrap:wrap;gap:24px;margin-bottom:16px;">
          <div style="flex:1;min-width:180px;">
            <div style="font-family:'DM Sans',sans-serif;font-size:.82rem;font-weight:500;margin-bottom:6px;">Round Win % &mdash; <span id="interactWinVal">50</span>%</div>
            <input type="range" id="interactWinSlider" min="30" max="70" value="50" step="1" style="width:100%;">
          </div>
          <div style="flex:1;min-width:180px;">
            <div style="font-family:'DM Sans',sans-serif;font-size:.82rem;font-weight:500;margin-bottom:6px;">PTD &mdash; <span id="interactPtdVal">20</span>%</div>
            <input type="range" id="interactPtdSlider" min="15" max="25" value="20" step="0.1" style="width:100%;">
          </div>
        </div>
        <div style="display:flex;flex-wrap:wrap;gap:8px;margin-bottom:20px;" id="interactRoleButtons">
          <button class="interact-role active" data-role="Initiator">Initiator</button>
          <button class="interact-role" data-role="Controller">Controller</button>
          <button class="interact-role" data-role="Duelist">Duelist</button>
          <button class="interact-role" data-role="Sentinel">Sentinel</button>
          <button class="interact-role" data-role="Flex">Flex</button>
        </div>
        <div style="font-family:'DM Sans',sans-serif;font-size:1rem;margin-bottom:16px;">
          Expected Rating: <strong id="interactExpected">&mdash;</strong>
        </div>
        <table class="res-table" id="interactMatchTable">
          <thead><tr><th>Closest Match</th><th>Org</th><th>Event</th><th>True Win%</th><th>True PTD</th><th>True Rating</th></tr></thead>
          <tbody id="interactMatchBody"></tbody>
        </table>
      </div>

    </div>
  </div>
</div>
<div class="player-popup" id="playerPopup">
  <img id="popupImg" src="" alt="">
  <div class="popup-name" id="popupName"></div>
  <div class="popup-org"  id="popupOrg"></div>
  <div class="popup-stats">
    <div>Rating <span id="popupRating"></span></div>
    <div>Team Round W% <span id="popupWin"></span></div>
  </div>
</div>
<div class="player-popup" id="rolePopup">
  <img id="rPopupImg" src="" alt="">
  <div class="popup-name" id="rPopupName" style="pointer-events:none;cursor:default;"></div>
  <div class="popup-org"  id="rPopupOrg"></div>
  <div class="popup-stats">
    <div>Event <span id="rPopupEvent"></span></div>
    <div><span id="rPopupAgentLabel">Most Played Agent</span> <span id="rPopupAgent"></span></div>
    <div>Rating <span id="rPopupRating"></span></div>
    <div>Team Round W% <span id="rPopupWin"></span></div>
    <div id="rPopupPtdRow" style="display:none;">PTD <span id="rPopupPtd"></span></div>
  </div>
</div>
<footer>Data sourced from VLR.gg</footer>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<script>
const roleData = {{ chart_json | safe }};
new Chart(document.getElementById('overallChart').getContext('2d'), {
  type: 'bar',
  data: {
    labels: roleData.roles,
    datasets: [{
      data: roleData.overall,
      backgroundColor: ['#f4b8c1','#b8d8f4','#b8e8d4','#d4b8f4'],
      borderColor:     ['#d4808f','#6ea8d4','#6ac4a0','#a07ac4'],
      borderWidth: 1.5,
      borderRadius: 6,
    }]
  },
  options: {
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: { callbacks: { label: function(c) { return 'Avg: ' + c.parsed.y.toFixed(2); } } }
    },
    scales: {
      x: { grid: { display: false }, ticks: { font: { family: 'DM Sans' }, color: '#7a6e7e' } },
      y: {
        min: 0.8, max: 1.2,
        ticks: { font: { family: 'DM Sans' }, color: '#7a6e7e', callback: function(v) { return v.toFixed(2); } },
        grid: { color: '#f0eaf4' }
      }
    }
  }
});
const ctx = document.getElementById('roleChart').getContext('2d');
new Chart(ctx, {
  type: 'bar',
  data: {
    labels: roleData.roles,
    datasets: roleData.regions.map(function(region) {
      const colors  = { Americas: '#f9a855', EMEA: '#82d49a', Pacific: '#6aaee8' };
      const borders = { Americas: '#c97820', EMEA: '#3a9e58', Pacific: '#2a78c4' };
      return {
        label: region,
        data: roleData.data[region],
        backgroundColor: colors[region],
        borderColor: borders[region],
        borderWidth: 1.5,
        borderRadius: 6,
      };
    })
  },
  options: {
    maintainAspectRatio: false,
    plugins: {
      legend: { labels: { font: { family: 'DM Sans', size: 12 }, color: '#2a1f2d' } },
      tooltip: { callbacks: { label: function(c) { return c.dataset.label + ': ' + c.parsed.y.toFixed(2); } } }
    },
    scales: {
      x: { grid: { display: false }, ticks: { font: { family: 'DM Sans' }, color: '#7a6e7e' } },
      y: {
        min: 0.8, max: 1.2,
        ticks: { font: { family: 'DM Sans' }, color: '#7a6e7e', callback: function(v) { return v.toFixed(2); } },
        grid: { color: '#f0eaf4' }
      }
    }
  }
});
const scatterData = {{ scatter_json | safe }};
const headshots   = {{ headshots_json | safe }};
const scatterColors = { Americas: '#f9a855', EMEA: '#82d49a', Pacific: '#6aaee8' };
const popup      = document.getElementById('playerPopup');
const popupImg    = document.getElementById('popupImg');
const popupName   = document.getElementById('popupName');
const popupOrg    = document.getElementById('popupOrg');
const popupRating = document.getElementById('popupRating');
const popupWin    = document.getElementById('popupWin');

const scatterChart = new Chart(document.getElementById('scatterChart').getContext('2d'), {
  type: 'scatter',
  data: {
    datasets: Object.keys(scatterData).map(function(region) {
      return {
        label: region,
        data: scatterData[region],
        backgroundColor: scatterColors[region] + 'bb',
        borderColor: scatterColors[region],
        borderWidth: 1,
        pointRadius: 6,
        pointHoverRadius: 13,
        hitRadius: 12,
      };
    })
  },
  options: {
    maintainAspectRatio: false,
    events: [],
    plugins: {
      legend: { labels: { font: { family: 'DM Sans', size: 12 }, color: '#2a1f2d' } },
      tooltip: { enabled: false }
    },
    scales: {
      x: {
        title: { display: true, text: 'Team Round Win %', font: { family: 'DM Sans' }, color: '#7a6e7e' },
        ticks: { font: { family: 'DM Sans' }, color: '#7a6e7e', callback: function(v) { return (v * 100).toFixed(0) + '%'; } },
        grid: { color: '#f0eaf4' }
      },
      y: {
        title: { display: true, text: 'VLR Rating', font: { family: 'DM Sans' }, color: '#7a6e7e' },
        ticks: { font: { family: 'DM Sans' }, color: '#7a6e7e', callback: function(v) { return v.toFixed(2); } },
        grid: { color: '#f0eaf4' }
      }
    }
  }
});
const scatterCanvas = document.getElementById('scatterChart');
let lastKey = null;
let hoverPt = null, hoverDi = -1, hoverPi = -1;
let pinnedKey = null, pinnedDi = -1, pinnedPi = -1;

function setChartActive() {
  const elems = [];
  const hoverKey = hoverPt !== null ? hoverDi + '-' + hoverPi : null;
  if (hoverPt !== null) elems.push({ datasetIndex: hoverDi, index: hoverPi });
  if (pinnedKey !== null && pinnedKey !== hoverKey) elems.push({ datasetIndex: pinnedDi, index: pinnedPi });
  const key = elems.map(function(e){ return e.datasetIndex+'-'+e.index; }).join('|');
  if (key !== lastKey) { lastKey = key; scatterChart.setActiveElements(elems); scatterChart.update('none'); }
}

function getPinnedPos() {
  const pt = scatterChart.data.datasets[pinnedDi].data[pinnedPi];
  const rect = scatterCanvas.getBoundingClientRect();
  const dotX = rect.left + scatterChart.scales.x.getPixelForValue(pt.x);
  const dotY = rect.top  + scatterChart.scales.y.getPixelForValue(pt.y);
  const onRight = dotX > window.innerWidth / 2;
  return {
    left: onRight ? dotX - (popup.offsetWidth || 170) - 16 : dotX + 16,
    top:  dotY - (popup.offsetHeight || 120) / 2
  };
}

function showPinnedPopup() {
  const pt = scatterChart.data.datasets[pinnedDi].data[pinnedPi];
  const hs = headshots[pt.profile] || '';
  popupImg.src = hs;
  popupImg.style.display = hs ? 'block' : 'none';
  popupName.textContent   = pt.player;
  popupOrg.textContent    = pt.org;
  popupRating.textContent = pt.y.toFixed(2);
  popupWin.textContent    = (pt.x * 100).toFixed(1) + '%';
  popupName.style.textDecoration = 'underline';
  const pos = getPinnedPos();
  popup.style.left = pos.left + 'px';
  popup.style.top  = pos.top  + 'px';
  popup.classList.add('visible');
}

var pinTimeout = null;
var popupTimer = null;

function smoothScrollCenter(el, duration) {
  const rect = el.getBoundingClientRect();
  const targetY = window.scrollY + rect.top + rect.height / 2 - window.innerHeight / 2;
  const startY  = window.scrollY;
  const dist    = targetY - startY;
  const t0      = performance.now();
  function ease(t) { return t < 0.5 ? 16*t*t*t*t*t : 1 - Math.pow(-2*t+2, 5)/2; }
  function step(now) {
    var p = Math.min((now - t0) / duration, 1);
    window.scrollTo(0, startY + dist * ease(p));
    if (p < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

function highlightPlayer(playerName) {
  let foundPt = null, foundDi = -1, foundPi = -1;
  scatterChart.data.datasets.forEach(function(ds, di) {
    ds.data.forEach(function(pt, pi) {
      if (pt.player === playerName) { foundPt = pt; foundDi = di; foundPi = pi; }
    });
  });
  if (!foundPt) return;
  const key = foundDi + '-' + foundPi;
  clearTimeout(pinTimeout);
  clearTimeout(popupTimer);
  popup.classList.remove('visible');
  if (pinnedKey === key) {
    pinnedKey = null; pinnedDi = -1; pinnedPi = -1;
    setChartActive();
    return;
  }
  pinnedKey = key; pinnedDi = foundDi; pinnedPi = foundPi;
  setChartActive();
  smoothScrollCenter(scatterCanvas, 700);
  popupTimer = setTimeout(showPinnedPopup, 350);
  pinTimeout = setTimeout(function() {
    pinnedKey = null; pinnedDi = -1; pinnedPi = -1;
    if (hoverPt === null) popup.classList.remove('visible');
    setChartActive();
  }, 3000);
}

popupName.addEventListener('click', function() {
  if (hoverPt !== null) {
    const hoverKey = hoverDi + '-' + hoverPi;
    if (pinnedKey === hoverKey) {
      pinnedKey = null; pinnedDi = -1; pinnedPi = -1;
      popupName.style.textDecoration = '';
    } else {
      pinnedKey = hoverKey; pinnedDi = hoverDi; pinnedPi = hoverPi;
      popupName.style.textDecoration = 'underline';
    }
  } else if (pinnedKey !== null) {
    pinnedKey = null; pinnedDi = -1; pinnedPi = -1;
    popup.classList.remove('visible');
  }
  setChartActive();
});

scatterCanvas.addEventListener('mousemove', function(e) {
  const rect   = scatterCanvas.getBoundingClientRect();
  const mouseX = e.clientX - rect.left;
  const mouseY = e.clientY - rect.top;
  const ca     = scatterChart.chartArea;
  if (mouseX < ca.left || mouseX > ca.right || mouseY < ca.top || mouseY > ca.bottom) {
    hoverPt = null; hoverDi = -1; hoverPi = -1;
    if (pinnedKey !== null) { showPinnedPopup(); } else { popup.classList.remove('visible'); }
    setChartActive();
    return;
  }
  const xScale = scatterChart.scales.x;
  const yScale = scatterChart.scales.y;
  const HIT    = 10;
  let nearest = null, nearestDi = -1, nearestPi = -1, minDist = Infinity;
  scatterChart.data.datasets.forEach(function(ds, di) {
    ds.data.forEach(function(pt, pi) {
      const px = xScale.getPixelForValue(pt.x);
      const py = yScale.getPixelForValue(pt.y);
      const dist = Math.hypot(mouseX - px, mouseY - py);
      if (dist < HIT && dist < minDist) { minDist = dist; nearest = pt; nearestDi = di; nearestPi = pi; }
    });
  });
  hoverPt = nearest; hoverDi = nearestDi; hoverPi = nearestPi;
  setChartActive();
  if (!nearest) { popup.classList.remove('visible'); return; }
  const hs = headshots[nearest.profile] || '';
  popupImg.src = hs;
  popupImg.style.display = hs ? 'block' : 'none';
  popupName.textContent   = nearest.player;
  popupOrg.textContent    = nearest.org;
  popupRating.textContent = nearest.y.toFixed(2);
  popupWin.textContent    = (nearest.x * 100).toFixed(1) + '%';
  const hoverKey = nearestDi + '-' + nearestPi;
  popupName.style.textDecoration = pinnedKey === hoverKey ? 'underline' : '';
  const onRight = e.clientX > window.innerWidth / 2;
  popup.style.left = onRight ? (e.clientX - popup.offsetWidth - 16) + 'px' : (e.clientX + 16) + 'px';
  popup.style.top  = (e.clientY - popup.offsetHeight / 2) + 'px';
  popup.classList.add('visible');
});
scatterCanvas.addEventListener('mouseleave', function() {
  hoverPt = null; hoverDi = -1; hoverPi = -1;
  if (pinnedKey !== null) { showPinnedPopup(); } else { popup.classList.remove('visible'); }
  setChartActive();
});
window.addEventListener('scroll', function() {
  if (pinnedKey === null || !popup.classList.contains('visible')) return;
  const pos = getPinnedPos();
  popup.style.left = pos.left + 'px';
  popup.style.top  = pos.top  + 'px';
}, { passive: true });

(function() {
  let xDev = 0;
  Object.values(scatterData).forEach(function(pts) {
    pts.forEach(function(p) { xDev = Math.max(xDev, Math.abs(p.x - 0.5)); });
  });
  const xPad = Math.ceil((xDev + 0.02) * 100) / 100;
  scatterChart.options.scales.x.min = parseFloat((0.5 - xPad).toFixed(2));
  scatterChart.options.scales.x.max = parseFloat((0.5 + xPad).toFixed(2));
  scatterChart.options.scales.y.min = 0.6;
  scatterChart.options.scales.y.max = 1.4;
  scatterChart.update();
})();

const allRolesData = {{ all_roles_json | safe }};
var roleCharts = {};
var rolePinned = { canvas: null, chart: null, pt: null, raf: null };
(function() {
  if (!allRolesData || !Object.keys(allRolesData).length) return;
  const rolePopup    = document.getElementById('rolePopup');
  const rPopupImg        = document.getElementById('rPopupImg');
  const rPopupName       = document.getElementById('rPopupName');
  const rPopupOrg        = document.getElementById('rPopupOrg');
  const rPopupEvent      = document.getElementById('rPopupEvent');
  const rPopupAgentLabel = document.getElementById('rPopupAgentLabel');
  const rPopupAgent      = document.getElementById('rPopupAgent');
  const rPopupRating     = document.getElementById('rPopupRating');
  const rPopupWin        = document.getElementById('rPopupWin');

  // Compute shared axis bounds across all roles
  var globalXDev = 0;
  Object.values(allRolesData).forEach(function(pts) {
    pts.forEach(function(p) {
      globalXDev = Math.max(globalXDev, Math.abs(p.x - 0.5));
    });
  });
  var globalXPad      = Math.ceil((globalXDev + 0.02) * 100) / 100;
  var globalXMin      = parseFloat((0.5 - globalXPad).toFixed(2));
  var globalXMax      = parseFloat((0.5 + globalXPad).toFixed(2));
  var globalYMinBound = 0.5;
  var globalYMaxBound = 1.5;

  ['Initiator','Controller','Duelist','Sentinel','Flex'].forEach(function(role) {
    const canvas = document.getElementById('roleScatter_' + role);
    if (!canvas || !allRolesData[role] || !allRolesData[role].length) return;

    const byRegion = { Americas: [], EMEA: [], Pacific: [] };
    allRolesData[role].forEach(function(pt) {
      if (byRegion[pt.region]) byRegion[pt.region].push(pt);
    });

    var chart = roleCharts[role] = new Chart(canvas.getContext('2d'), {
      type: 'scatter',
      data: {
        datasets: ['Americas','EMEA','Pacific'].map(function(region) {
          return {
            label: region,
            data: byRegion[region],
            backgroundColor: scatterColors[region] + 'aa',
            borderColor: scatterColors[region],
            borderWidth: 1,
            pointRadius: 4,
            pointHoverRadius: 10,
            hitRadius: 8,
          };
        })
      },
      options: {
        maintainAspectRatio: false,
        events: [],
        plugins: {
          legend: { labels: { font: { family: 'DM Sans', size: 12 }, color: '#2a1f2d' } },
          tooltip: { enabled: false }
        },
        scales: {
          x: {
            min: globalXMin, max: globalXMax,
            title: { display: true, text: 'Team Round Win %', font: { family: 'DM Sans' }, color: '#7a6e7e' },
            ticks: { font: { family: 'DM Sans' }, color: '#7a6e7e', callback: function(v) { return (v*100).toFixed(0)+'%'; } },
            grid: { color: '#f0eaf4' }
          },
          y: {
            min: globalYMinBound, max: globalYMaxBound,
            title: { display: true, text: 'VLR Rating', font: { family: 'DM Sans' }, color: '#7a6e7e' },
            ticks: { font: { family: 'DM Sans' }, color: '#7a6e7e', callback: function(v) { return v.toFixed(2); } },
            grid: { color: '#f0eaf4' }
          }
        }
      }
    });

    canvas.addEventListener('mousemove', function(e) {
      var rect   = canvas.getBoundingClientRect();
      var mouseX = e.clientX - rect.left;
      var mouseY = e.clientY - rect.top;
      var ca = chart.chartArea;
      if (mouseX < ca.left || mouseX > ca.right || mouseY < ca.top || mouseY > ca.bottom) {
        if (!rolePinned.canvas) rolePopup.classList.remove('visible');
        return;
      }
      var HIT = 8;
      var nearest = null, minDist = Infinity;
      chart.data.datasets.forEach(function(ds, di) {
        var meta = chart.getDatasetMeta(di);
        ds.data.forEach(function(pt, pi) {
          var el = meta.data[pi];
          var d  = Math.hypot(mouseX - el.x, mouseY - el.y);
          if (d < HIT && d < minDist) { minDist = d; nearest = pt; }
        });
      });
      if (!nearest) { if (!rolePinned.canvas) rolePopup.classList.remove('visible'); return; }
      if (rolePinned.canvas) return;
      var hs = headshots[nearest.profile] || '';
      rPopupImg.src = hs;
      rPopupImg.style.display = hs ? 'block' : 'none';
      document.getElementById('rPopupPtdRow').style.display = 'none';
      rPopupName.textContent   = nearest.player;
      rPopupOrg.textContent    = nearest.org;
      rPopupEvent.textContent  = nearest.event;
      if (nearest.roles) {
        rPopupAgentLabel.textContent = 'Roles';
        var rolesSorted = Object.entries(nearest.roles).sort(function(a,b){ return b[1]-a[1]; });
        rPopupAgent.textContent = rolesSorted.map(function(e){ return e[0] + ' ' + e[1] + '%'; }).join(' / ');
      } else {
        rPopupAgentLabel.textContent = 'Most Played Agent';
        var ag = nearest.agent;
        rPopupAgent.textContent = ag.charAt(0).toUpperCase() + ag.slice(1);
      }
      rPopupRating.textContent = nearest.y.toFixed(2);
      rPopupWin.textContent    = (nearest.x * 100).toFixed(1) + '%';
      var onRight = e.clientX > window.innerWidth / 2;
      rolePopup.style.left = onRight ? (e.clientX - rolePopup.offsetWidth - 16) + 'px' : (e.clientX + 16) + 'px';
      rolePopup.style.top  = (e.clientY - rolePopup.offsetHeight / 2) + 'px';
      rolePopup.classList.add('visible');
    });
    canvas.addEventListener('mouseleave', function() {
      if (!rolePinned.canvas) rolePopup.classList.remove('visible');
    });
  });

})();

function highlightRolePlayer(playerName, eventLabel) {
  var found = null, foundChart = null, foundCanvas = null;
  ['Initiator','Controller','Duelist','Sentinel','Flex'].forEach(function(role) {
    if (found) return;
    var chart = roleCharts[role];
    if (!chart) return;
    chart.data.datasets.forEach(function(ds, di) {
      if (found) return;
      ds.data.forEach(function(pt, pi) {
        if (found) return;
        if (pt.player.toLowerCase() === playerName.toLowerCase() &&
            (!eventLabel || pt.event === eventLabel)) {
          found = { pt: pt, di: di, pi: pi };
          foundChart = chart;
          foundCanvas = document.getElementById('roleScatter_' + role);
        }
      });
    });
  });
  if (!found) return;
  smoothScrollCenter(foundCanvas, 700);
  setTimeout(function() {
    var rp  = document.getElementById('rolePopup');
    var pt  = found.pt;
    var hs  = headshots[pt.profile] || '';
    document.getElementById('rPopupImg').src          = hs;
    document.getElementById('rPopupImg').style.display = hs ? 'block' : 'none';
    document.getElementById('rPopupPtdRow').style.display = 'none';
    document.getElementById('rPopupName').textContent  = pt.player;
    document.getElementById('rPopupOrg').textContent   = pt.org;
    document.getElementById('rPopupEvent').textContent = pt.event;
    if (pt.roles) {
      document.getElementById('rPopupAgentLabel').textContent = 'Roles';
      var rs = Object.entries(pt.roles).sort(function(a,b){ return b[1]-a[1]; });
      document.getElementById('rPopupAgent').textContent = rs.map(function(e){ return e[0]+' '+e[1]+'%'; }).join(' / ');
    } else {
      document.getElementById('rPopupAgentLabel').textContent = 'Most Played Agent';
      var ag = pt.agent || '';
      document.getElementById('rPopupAgent').textContent = ag.charAt(0).toUpperCase() + ag.slice(1);
    }
    document.getElementById('rPopupRating').textContent = pt.y.toFixed(2);
    document.getElementById('rPopupWin').textContent    = (pt.x * 100).toFixed(1) + '%';
    var rect  = foundCanvas.getBoundingClientRect();
    var dotX  = rect.left + foundChart.scales.x.getPixelForValue(pt.x);
    var dotY  = rect.top  + foundChart.scales.y.getPixelForValue(pt.y);
    var onRight = dotX > window.innerWidth / 2;
    rp.style.left = (onRight ? dotX - rp.offsetWidth - 16 : dotX + 16) + 'px';
    rp.style.top  = (dotY - rp.offsetHeight / 2) + 'px';
    rp.classList.add('visible');
    rolePinned = { canvas: foundCanvas, chart: foundChart, pt: found.pt, raf: null };
    setTimeout(function() {
      var ds = foundChart.data.datasets[found.di], pi = found.pi, start = null;
      function grow(ts) {
        if (!start) start = ts;
        var p = Math.min((ts - start) / 200, 1);
        ds.pointRadius = ds.data.map(function(_, j) { return j === pi ? 4 + 6 * p : 4; });
        foundChart.update('none');
        if (p < 1) rolePinned.raf = requestAnimationFrame(grow);
        else rolePinned.raf = null;
      }
      rolePinned.raf = requestAnimationFrame(grow);
    }, 150);
    setTimeout(function() {
      rp.classList.remove('visible');
      if (rolePinned.raf) cancelAnimationFrame(rolePinned.raf);
      var ds = foundChart.data.datasets[found.di], pi = found.pi, start = null;
      function shrink(ts) {
        if (!start) start = ts;
        var p = Math.min((ts - start) / 200, 1);
        ds.pointRadius = ds.data.map(function(_, j) { return j === pi ? 10 - 6 * p : 4; });
        foundChart.update('none');
        if (p < 1) {
          rolePinned.raf = requestAnimationFrame(shrink);
        } else {
          ds.pointRadius = 4;
          foundChart.update('none');
          rolePinned = { canvas: null, chart: null, pt: null, raf: null };
        }
      }
      rolePinned.raf = requestAnimationFrame(shrink);
    }, 3000);
  }, 720);
}

window.addEventListener('scroll', function() {
  var rp = document.getElementById('rolePopup');
  if (!rolePinned.canvas || !rp.classList.contains('visible')) return;
  var rect    = rolePinned.canvas.getBoundingClientRect();
  var dotX    = rect.left + rolePinned.chart.scales.x.getPixelForValue(rolePinned.pt.x);
  var dotY    = rect.top  + rolePinned.chart.scales.y.getPixelForValue(rolePinned.pt.y);
  var onRight = dotX > window.innerWidth / 2;
  rp.style.left = (onRight ? dotX - rp.offsetWidth - 16 : dotX + 16) + 'px';
  rp.style.top  = (dotY - rp.offsetHeight / 2) + 'px';
}, { passive: true });


const modelParams = {{ model_json | safe }};
(function() {
  if (!modelParams || !Object.keys(modelParams).length) return;
  var roleOrder = ['Initiator','Controller','Duelist','Sentinel','Flex'];
  var lineColors = { Initiator:'#6aafd4', Controller:'#5ab88a', Duelist:'#d47080', Sentinel:'#9060c4', Flex:'#d49040' };

  var mxDev = 0;
  Object.values(allRolesData).forEach(function(pts) {
    pts.forEach(function(p) { mxDev = Math.max(mxDev, Math.abs(p.x - 0.5)); });
  });
  var mxPad = Math.ceil((mxDev + 0.02) * 100) / 100;
  var mxMin = parseFloat((0.5 - mxPad).toFixed(2));
  var mxMax = parseFloat((0.5 + mxPad).toFixed(2));

  var tbody = document.getElementById('modelTableBody');
  roleOrder.forEach(function(role) {
    var m = modelParams[role];
    if (!m) return;
    var tr = document.createElement('tr');
    tr.innerHTML = '<td>' + role + '</td><td>' + m.intercept.toFixed(4) + '</td><td>' + m.slope.toFixed(4) + '</td>';
    tbody.appendChild(tr);
  });

  var activeRole = 'Initiator';
  var showDots = false;

  var btnsEl = document.getElementById('modelRoleBtns');
  roleOrder.forEach(function(role) {
    var btn = document.createElement('button');
    btn.className = 'role-btn' + (role === activeRole ? ' active' : '');
    btn.textContent = role;
    btn.addEventListener('click', function() {
      activeRole = role;
      btnsEl.querySelectorAll('.role-btn').forEach(function(b) { b.classList.remove('active'); });
      btn.classList.add('active');
      updateModel();
    });
    btnsEl.appendChild(btn);
  });

  document.getElementById('showDotsCheck').addEventListener('change', function() {
    showDots = this.checked;
    updateModel();
  });

  var modelChart = new Chart(document.getElementById('modelChart').getContext('2d'), {
    type: 'scatter',
    data: { datasets: [] },
    options: {
      maintainAspectRatio: false,
      events: [],
      plugins: { legend: { display: false }, tooltip: { enabled: false } },
      scales: {
        x: {
          type: 'linear', min: mxMin, max: mxMax,
          title: { display: true, text: 'Team Round Win %', font: { family: 'DM Sans' }, color: '#7a6e7e' },
          ticks: { font: { family: 'DM Sans' }, color: '#7a6e7e', callback: function(v) { return (v*100).toFixed(0)+'%'; } },
          grid: { color: '#f0eaf4' }
        },
        y: {
          min: 0.5, max: 1.5,
          title: { display: true, text: 'VLR Rating', font: { family: 'DM Sans' }, color: '#7a6e7e' },
          ticks: { font: { family: 'DM Sans' }, color: '#7a6e7e', callback: function(v) { return v.toFixed(2); } },
          grid: { color: '#f0eaf4' }
        }
      }
    }
  });

  function updateModel() {
    var m = modelParams[activeRole];
    var slopeEffect = (m.slope / 10).toFixed(3);
    document.getElementById('modelSlopeLabel').textContent = 'β = ' + m.slope.toFixed(4) + '  —  each 10% in round win % adds ' + slopeEffect + ' in predicted rating';
    var datasets = [];
    if (showDots) {
      var pts = allRolesData[activeRole] || [];
      ['Americas','EMEA','Pacific'].forEach(function(region) {
        var rPts = pts.filter(function(p) { return p.region === region; });
        if (!rPts.length) return;
        datasets.push({ type: 'scatter', data: rPts, backgroundColor: scatterColors[region] + 'aa', borderColor: scatterColors[region], borderWidth: 1, pointRadius: 4 });
      });
    }
    datasets.push({
      type: 'line',
      data: [{ x: mxMin, y: m.intercept + m.slope * mxMin }, { x: mxMax, y: m.intercept + m.slope * mxMax }],
      borderColor: lineColors[activeRole], borderWidth: 2.5, pointRadius: 0, tension: 0, fill: false,
    });
    modelChart.data.datasets = datasets;
    modelChart.update();
  }

  updateModel();

  var modelCanvas = document.getElementById('modelChart');
  var modelRolePopup = document.getElementById('rolePopup');
  modelCanvas.addEventListener('mousemove', function(e) {
    if (!showDots) { if (!rolePinned.canvas) modelRolePopup.classList.remove('visible'); return; }
    var rect = modelCanvas.getBoundingClientRect();
    var mouseX = e.clientX - rect.left, mouseY = e.clientY - rect.top;
    var ca = modelChart.chartArea;
    if (mouseX < ca.left || mouseX > ca.right || mouseY < ca.top || mouseY > ca.bottom) {
      if (!rolePinned.canvas) modelRolePopup.classList.remove('visible'); return;
    }
    var HIT = 8, nearest = null, minDist = Infinity;
    modelChart.data.datasets.forEach(function(ds) {
      if (ds.type !== 'scatter') return;
      ds.data.forEach(function(pt) {
        var px = modelChart.scales.x.getPixelForValue(pt.x);
        var py = modelChart.scales.y.getPixelForValue(pt.y);
        var d = Math.hypot(mouseX - px, mouseY - py);
        if (d < HIT && d < minDist) { minDist = d; nearest = pt; }
      });
    });
    if (!nearest) { if (!rolePinned.canvas) modelRolePopup.classList.remove('visible'); return; }
    if (rolePinned.canvas) return;
    var hs = headshots[nearest.profile] || '';
    document.getElementById('rPopupImg').src = hs;
    document.getElementById('rPopupImg').style.display = hs ? 'block' : 'none';
    document.getElementById('rPopupPtdRow').style.display = 'none';
    document.getElementById('rPopupName').textContent = nearest.player;
    document.getElementById('rPopupOrg').textContent = nearest.org;
    document.getElementById('rPopupEvent').textContent = nearest.event;
    if (nearest.roles) {
      document.getElementById('rPopupAgentLabel').textContent = 'Roles';
      var rs = Object.entries(nearest.roles).sort(function(a,b){ return b[1]-a[1]; });
      document.getElementById('rPopupAgent').textContent = rs.map(function(e){ return e[0]+' '+e[1]+'%'; }).join(' / ');
    } else {
      document.getElementById('rPopupAgentLabel').textContent = 'Most Played Agent';
      var ag = nearest.agent || '';
      document.getElementById('rPopupAgent').textContent = ag.charAt(0).toUpperCase() + ag.slice(1);
    }
    document.getElementById('rPopupRating').textContent = nearest.y.toFixed(2);
    document.getElementById('rPopupWin').textContent = (nearest.x * 100).toFixed(1) + '%';
    var onRight = e.clientX > window.innerWidth / 2;
    modelRolePopup.style.left = (onRight ? e.clientX - modelRolePopup.offsetWidth - 16 : e.clientX + 16) + 'px';
    modelRolePopup.style.top = (e.clientY - modelRolePopup.offsetHeight / 2) + 'px';
    modelRolePopup.classList.add('visible');
  });
  modelCanvas.addEventListener('mouseleave', function() {
    if (!rolePinned.canvas) modelRolePopup.classList.remove('visible');
  });
})();

const residualsData = {{ residuals_json | safe }};
(function() {
  if (!residualsData) return;
  function buildResTable(bodyId, rows) {
    var tbody = document.getElementById(bodyId);
    rows.forEach(function(row) {
      var hs = headshots[row.profile] || '';
      var imgHtml = hs ? '<img src="' + hs + '" alt="">' : '';
      var resSign  = row.residual >= 0 ? '+' : '';
      var zSign    = row.z >= 0 ? '+' : '';
      var resClass = row.residual >= 0 ? 'td-pos' : 'td-neg';
      var zClass   = row.z >= 0 ? 'td-pos' : 'td-neg';
      var tr = document.createElement('tr');
      tr.innerHTML =
        '<td class="td-img">' + imgHtml + '</td>' +
        '<td>' + row.player + '<br><span style="color:var(--soft);font-size:.75rem;font-weight:300;">' + row.org + ' &middot; ' + row.event + '</span></td>' +
        '<td>' + row.rating.toFixed(2) + '</td>' +
        '<td>' + row.expected.toFixed(2) + '</td>' +
        '<td class="' + resClass + '">' + resSign + row.residual.toFixed(3) + '</td>' +
        '<td class="' + zClass + '">' + zSign + row.z.toFixed(2) + '&sigma;</td>';
      tbody.appendChild(tr);
    });
  }
  buildResTable('resBody_alltime_best',  residualsData.alltime_best);
  buildResTable('resBody_alltime_worst', residualsData.alltime_worst);
  buildResTable('resBody_kickoff_best',  residualsData.kickoff_best);
  buildResTable('resBody_kickoff_worst', residualsData.kickoff_worst);
})();

const alltimeNoBaiters = {{ alltime_nobaiters_json | safe }};
(function() {
  if (!alltimeNoBaiters) return;
  var tbody = document.getElementById('resBody_kickoff_nobaiters');
  alltimeNoBaiters.top10.forEach(function(row) {
    var hs = headshots[row.profile] || '';
    var imgHtml = hs ? '<img src="' + hs + '" alt="">' : '';
    var resSign  = row.residual >= 0 ? '+' : '';
    var zSign    = row.z >= 0 ? '+' : '';
    var resClass = row.residual >= 0 ? 'td-pos' : 'td-neg';
    var zClass   = row.z >= 0 ? 'td-pos' : 'td-neg';
    var tr = document.createElement('tr');
    tr.innerHTML =
      '<td class="td-img">' + imgHtml + '</td>' +
      '<td>' + row.player + '<br><span style="color:var(--soft);font-size:.75rem;font-weight:300;">' + row.org + ' &middot; ' + row.event + '</span></td>' +
      '<td>' + row.rating.toFixed(2) + '</td>' +
      '<td>' + row.expected.toFixed(2) + '</td>' +
      '<td class="' + resClass + '">' + resSign + row.residual.toFixed(3) + '</td>' +
      '<td class="' + zClass + '">' + zSign + row.z.toFixed(2) + '&sigma;</td>';
    tbody.appendChild(tr);
  });




})();

const ptdKickoffData = {{ ptd_kickoff_json | safe }};
const ptdModelData   = {{ ptd_model_json   | safe }};
const ptdLookup      = {{ ptd_lookup_json  | safe }};
const oxyPtdChart    = {{ oxy_ptd_chart_json  | safe }};
(function() {
  if (!ptdKickoffData || !ptdModelData) return;

  // Top/bottom 10 PTD tables
  function buildPtdTable(bodyId, rows, thirdCol) {
    var tbody = document.getElementById(bodyId);
    if (!tbody) return;
    rows.forEach(function(row) {
      var hs = headshots[row.profile] || '';
      var imgHtml = hs ? '<img src="' + hs + '" alt="">' : '';
      var tr = document.createElement('tr');
      tr.innerHTML =
        '<td class="td-img">' + imgHtml + '</td>' +
        '<td>' + row.player + '</td>' +
        '<td>' + row.org + '</td>' +
        '<td>' + (row[thirdCol] || '') + '</td>' +
        '<td>' + row.rating.toFixed(2) + '</td>' +
        '<td>' + (row.ptd * 100).toFixed(1) + '%</td>';
      tbody.appendChild(tr);
    });
  }
  // 100T players
  var t100Body = document.getElementById('ptd100TBody');
  if (t100Body) {
    var allScatterPts = [].concat(
      ptdKickoffData.scatter.Americas || [],
      ptdKickoffData.scatter.EMEA     || [],
      ptdKickoffData.scatter.Pacific  || []
    ).filter(function(p) { return p.org === '100T'; })
     .sort(function(a, b) { return b.x - a.x; });
    allScatterPts.forEach(function(p) {
      var hs = headshots[p.profile] || '';
      var imgHtml = hs ? '<img src="' + hs + '" alt="">' : '';
      var tr = document.createElement('tr');
      tr.innerHTML =
        '<td class="td-img">' + imgHtml + '</td>' +
        '<td>' + p.player + '</td>' +
        '<td>' + p.y.toFixed(2) + '</td>' +
        '<td>' + (p.x * 100).toFixed(1) + '%</td>';
      t100Body.appendChild(tr);
    });
  }

  buildPtdTable('ptdTop10Body',    ptdKickoffData.top10,    'region');
  buildPtdTable('ptdBottom10Body', ptdKickoffData.bottom10, 'event');

  // PTD scatter chart
  var ptdScatterCtx = document.getElementById('ptdScatterChart');
  if (ptdScatterCtx) {
    var ptdColors = { Americas: '#f9a855', EMEA: '#82d49a', Pacific: '#6aaee8' };
    var ptdScatterChart = new Chart(ptdScatterCtx.getContext('2d'), {
      type: 'scatter',
      data: {
        datasets: ['Americas', 'EMEA', 'Pacific'].map(function(region) {
          return {
            label: region,
            data: ptdKickoffData.scatter[region] || [],
            backgroundColor: ptdColors[region] + 'bb',
            borderColor: ptdColors[region],
            borderWidth: 1,
            pointRadius: 6,
            pointHoverRadius: 13,
            hitRadius: 12,
          };
        })
      },
      options: {
        maintainAspectRatio: false,
        events: [],
        plugins: {
          legend: { labels: { font: { family: 'DM Sans', size: 12 }, color: '#2a1f2d' } },
          tooltip: { enabled: false }
        },
        scales: {
          x: {
            title: { display: true, text: 'PTD (Proportion of Team Deaths)', font: { family: 'DM Sans' }, color: '#7a6e7e' },
            ticks: { font: { family: 'DM Sans' }, color: '#7a6e7e', callback: function(v) { return (v * 100).toFixed(0) + '%'; } },
            grid: { color: '#f0eaf4' }
          },
          y: {
            min: 0.6, max: 1.4,
            title: { display: true, text: 'VLR Rating', font: { family: 'DM Sans' }, color: '#7a6e7e' },
            ticks: { font: { family: 'DM Sans' }, color: '#7a6e7e', callback: function(v) { return v.toFixed(2); } },
            grid: { color: '#f0eaf4' }
          }
        }
      }
    });

    // Hover popup (reuse rolePopup)
    var ptdPopup = document.getElementById('playerPopup');
    ptdScatterCtx.addEventListener('mousemove', function(e) {
      var rect = ptdScatterCtx.getBoundingClientRect();
      var mouseX = e.clientX - rect.left, mouseY = e.clientY - rect.top;
      var ca = ptdScatterChart.chartArea;
      if (mouseX < ca.left || mouseX > ca.right || mouseY < ca.top || mouseY > ca.bottom) {
        ptdPopup.classList.remove('visible'); return;
      }
      var HIT = 10, nearest = null, minDist = Infinity;
      ptdScatterChart.data.datasets.forEach(function(ds) {
        ds.data.forEach(function(pt) {
          var px = ptdScatterChart.scales.x.getPixelForValue(pt.x);
          var py = ptdScatterChart.scales.y.getPixelForValue(pt.y);
          var d = Math.hypot(mouseX - px, mouseY - py);
          if (d < HIT && d < minDist) { minDist = d; nearest = pt; }
        });
      });
      if (!nearest) { ptdPopup.classList.remove('visible'); return; }
      var hs = headshots[nearest.profile] || '';
      document.getElementById('popupImg').src = hs;
      document.getElementById('popupImg').style.display = hs ? 'block' : 'none';
      document.getElementById('popupName').textContent   = nearest.player;
      document.getElementById('popupOrg').textContent    = nearest.org;
      document.getElementById('popupRating').textContent = nearest.y.toFixed(2);
      document.getElementById('popupWin').textContent    = 'PTD ' + (nearest.x * 100).toFixed(1) + '%';
      var onRight = e.clientX > window.innerWidth / 2;
      ptdPopup.style.left = (onRight ? e.clientX - ptdPopup.offsetWidth - 16 : e.clientX + 16) + 'px';
      ptdPopup.style.top  = (e.clientY - ptdPopup.offsetHeight / 2) + 'px';
      ptdPopup.classList.add('visible');
    });
    ptdScatterCtx.addEventListener('mouseleave', function() { ptdPopup.classList.remove('visible'); });
  }

  // Equation display
  var eqBox = document.getElementById('ptdEqBox');
  if (eqBox && ptdModelData.role_eqs) {
    var anyRole = ptdModelData.role_eqs['Initiator'] || Object.values(ptdModelData.role_eqs)[0];
    if (anyRole) {

    }
  }

  // PTD model residuals tables
  function buildPtdResTable(bodyId, rows) {
    var tbody = document.getElementById(bodyId);
    if (!tbody || !rows) return;
    rows.forEach(function(row) {
      var hs = headshots[row.profile] || '';
      var imgHtml = hs ? '<img src="' + hs + '" alt="">' : '';
      var resSign  = row.residual >= 0 ? '+' : '';
      var zSign    = row.z >= 0 ? '+' : '';
      var resClass = row.residual >= 0 ? 'td-pos' : 'td-neg';
      var zClass   = row.z >= 0 ? 'td-pos' : 'td-neg';
      var ptdVal   = ptdLookup[row.profile + '|' + row.event];
      var ptdHtml  = ptdVal != null ? (ptdVal * 100).toFixed(1) + '%' : '—';
      var tr = document.createElement('tr');
      tr.innerHTML =
        '<td class="td-img">' + imgHtml + '</td>' +
        '<td>' + row.player + '<br><span style="color:var(--soft);font-size:.75rem;font-weight:300;">' + row.org + ' &middot; ' + row.event + '</span></td>' +
        '<td>' + row.rating.toFixed(2) + '</td>' +
        '<td>' + row.expected.toFixed(2) + '</td>' +
        '<td class="' + resClass + '">' + resSign + row.residual.toFixed(3) + '</td>' +
        '<td class="' + zClass   + '">' + zSign   + row.z.toFixed(2) + '&sigma;</td>' +
        '<td>' + ptdHtml + '</td>';
      tbody.appendChild(tr);
    });
  }
  buildPtdResTable('ptdAlltimeBestBody',  ptdModelData.alltime_best);
  buildPtdResTable('ptdAlltimeWorstBody', ptdModelData.alltime_worst);
  buildPtdResTable('ptdKickoffBestBody',  ptdModelData.kickoff_best);
  buildPtdResTable('ptdKickoffWorstBody', ptdModelData.kickoff_worst);

  // Patmen conclusion table
  (function() {
    var tbody = document.getElementById('patmenBody');
    if (!tbody || !ptdModelData.patmen) return;
    var p = ptdModelData.patmen;
    var hs = headshots[p.profile] || '';
    var imgHtml = hs ? '<img src="' + hs + '" alt="">' : '';
    var resSign  = p.residual >= 0 ? '+' : '';
    var zSign    = p.z >= 0 ? '+' : '';
    var resClass = p.residual >= 0 ? 'td-pos' : 'td-neg';
    var zClass   = p.z >= 0 ? 'td-pos' : 'td-neg';
    var ptdVal   = ptdLookup[p.profile + '|' + p.event];
    var ptdHtml  = ptdVal != null ? (ptdVal * 100).toFixed(1) + '%' : '—';
    var tr = document.createElement('tr');
    tr.innerHTML =
      '<td class="td-img">' + imgHtml + '</td>' +
      '<td>' + p.player + '<br><span style="color:var(--soft);font-size:.75rem;font-weight:300;">' + p.org + ' &middot; ' + p.event + '</span></td>' +
      '<td>' + p.rating.toFixed(2) + '</td>' +
      '<td>' + p.expected.toFixed(2) + '</td>' +
      '<td class="' + resClass + '">' + resSign + p.residual.toFixed(3) + '</td>' +
      '<td class="' + zClass   + '">' + zSign   + p.z.toFixed(2) + '&sigma;</td>' +
      '<td>' + ptdHtml + '</td>';
    tbody.appendChild(tr);
  })();

  // Oxy PTD normal distribution chart
  (function() {
    var ctx = document.getElementById('oxyPtdNormalChart');
    if (!ctx || !oxyPtdChart || !oxyPtdChart.mean) return;
    var mean = oxyPtdChart.mean, std = oxyPtdChart.std, oxyVal = oxyPtdChart.ptd;
    var pts = 200, lo = mean - 4 * std, hi = mean + 4 * std;
    var labels = [], normalData = [], oxyData = [];
    for (var i = 0; i <= pts; i++) {
      var x = lo + (hi - lo) * i / pts;
      var y = (1 / (std * Math.sqrt(2 * Math.PI))) * Math.exp(-0.5 * Math.pow((x - mean) / std, 2));
      labels.push(x);
      normalData.push({x: x, y: y});
      oxyData.push({x: x, y: Math.abs(x - oxyVal) < (hi - lo) / pts * 1.5 ? y : null});
    }
    new Chart(ctx.getContext('2d'), {
      type: 'line',
      data: {
        datasets: [
          {
            label: 'PTD Distribution',
            data: normalData,
            borderColor: '#9060c4',
            backgroundColor: 'rgba(144,96,196,0.15)',
            borderWidth: 2,
            pointRadius: 0,
            fill: true,
            tension: 0.4,
          },
          {
            label: 'Oxy — 2024 Stage 1 (' + (oxyVal * 100).toFixed(1) + '%)',
            data: [{x: oxyVal, y: 0}, {x: oxyVal, y: (1 / (std * Math.sqrt(2 * Math.PI)))}],
            borderColor: '#f9a855',
            borderWidth: 2,
            borderDash: [6, 3],
            pointRadius: 0,
            fill: false,
          }
        ]
      },
      options: {
        maintainAspectRatio: false,
        parsing: false,
        plugins: {
          legend: { labels: { font: { family: 'DM Sans', size: 12 }, color: '#2a1f2d' }, onClick: null },
          tooltip: { enabled: false }
        },
        scales: {
          x: {
            type: 'linear',
            title: { display: true, text: 'PTD', font: { family: 'DM Sans', size: 11 }, color: '#2a1f2d' },
            ticks: {
              callback: function(v) { return (v * 100).toFixed(0) + '%'; },
              font: { family: 'DM Sans', size: 11 }, color: '#2a1f2d'
            },
            grid: { color: 'rgba(0,0,0,0.06)' }
          },
          y: { display: false }
        }
      }
    });
  })();

  // PTD model line-of-best-fit chart
  if (ptdModelData.role_eqs) {
    var ptdLineColors = { Initiator:'#6aafd4', Controller:'#5ab88a', Duelist:'#d47080', Sentinel:'#9060c4', Flex:'#d49040' };
    var ptdRoleOrder  = ['Initiator','Controller','Duelist','Sentinel','Flex'];
    var ptdSlider      = document.getElementById('ptdSlider');
    var ptdSliderLabel = document.getElementById('ptdSliderLabel');
    function getMeanPtd() { return parseInt(ptdSlider.value, 10) / 100; }

    ptdSlider.addEventListener('input', function() {
      ptdSliderLabel.textContent = ptdSlider.value + '%';
      updatePtdLine();
    });

    var ptdMxDev = 0;
    Object.values(allRolesData).forEach(function(pts) {
      pts.forEach(function(p) { ptdMxDev = Math.max(ptdMxDev, Math.abs(p.x - 0.5)); });
    });
    var ptdMxPad = Math.ceil((ptdMxDev + 0.02) * 100) / 100;
    var ptdMxMin = parseFloat((0.5 - ptdMxPad).toFixed(2));
    var ptdMxMax = parseFloat((0.5 + ptdMxPad).toFixed(2));

    var ptdActiveRole = 'Initiator';
    var ptdShowDots   = false;

    var ptdModelBtnsEl = document.getElementById('ptdModelRoleBtns');
    ptdRoleOrder.forEach(function(role) {
      if (!ptdModelData.role_eqs[role]) return;
      var btn = document.createElement('button');
      btn.className = 'role-btn' + (role === ptdActiveRole ? ' active' : '');
      btn.textContent = role;
      btn.addEventListener('click', function() {
        ptdActiveRole = role;
        ptdModelBtnsEl.querySelectorAll('.role-btn').forEach(function(b) { b.classList.remove('active'); });
        btn.classList.add('active');
        updatePtdModelChart();
      });
      ptdModelBtnsEl.appendChild(btn);
    });

    document.getElementById('ptdShowDotsCheck').addEventListener('change', function() {
      ptdShowDots = this.checked;
      updatePtdModelChart();
    });

    var ptdModelChart = new Chart(document.getElementById('ptdModelChart').getContext('2d'), {
      type: 'scatter',
      data: { datasets: [] },
      options: {
        maintainAspectRatio: false,
        events: [],
        plugins: { legend: { display: false }, tooltip: { enabled: false } },
        scales: {
          x: {
            type: 'linear', min: ptdMxMin, max: ptdMxMax,
            title: { display: true, text: 'Team Round Win %', font: { family: 'DM Sans' }, color: '#7a6e7e' },
            ticks: { font: { family: 'DM Sans' }, color: '#7a6e7e', callback: function(v) { return (v*100).toFixed(0)+'%'; } },
            grid: { color: '#f0eaf4' }
          },
          y: {
            min: 0.5, max: 1.5,
            title: { display: true, text: 'VLR Rating', font: { family: 'DM Sans' }, color: '#7a6e7e' },
            ticks: { font: { family: 'DM Sans' }, color: '#7a6e7e', callback: function(v) { return v.toFixed(2); } },
            grid: { color: '#f0eaf4' }
          }
        }
      }
    });

    function updatePtdModelChart() {
      var m = ptdModelData.role_eqs[ptdActiveRole];
      var adjIntercept = m.intercept + m.ptd * getMeanPtd();
      var slopeEffect  = (m.slope / 10).toFixed(3);
      document.getElementById('ptdModelSlopeLabel').innerHTML =
        'β<sub>win%</sub> = ' + m.slope.toFixed(4) +
        '  —  each 10% in round win % adds ' + slopeEffect + ' in predicted rating';
      document.getElementById('ptdBetaLabel').innerHTML =
        'β<sub>PTD</sub> = ' + m.ptd.toFixed(4);
      var datasets = [];
      if (ptdShowDots) {
        var pts = allRolesData[ptdActiveRole] || [];
        ['Americas','EMEA','Pacific'].forEach(function(region) {
          var rPts = pts.filter(function(p) { return p.region === region; });
          if (!rPts.length) return;
          datasets.push({ type: 'scatter', data: rPts, backgroundColor: scatterColors[region] + 'aa', borderColor: scatterColors[region], borderWidth: 1, pointRadius: 4 });
        });
      }
      datasets.push({
        type: 'line',
        data: [{ x: ptdMxMin, y: adjIntercept + m.slope * ptdMxMin }, { x: ptdMxMax, y: adjIntercept + m.slope * ptdMxMax }],
        borderColor: ptdLineColors[ptdActiveRole] || '#888', borderWidth: 2.5, pointRadius: 0, tension: 0, fill: false,
      });
      ptdModelChart.data.datasets = datasets;
      ptdModelChart.update();
    }

    function updatePtdLine() {
      var m = ptdModelData.role_eqs[ptdActiveRole];
      var adjIntercept = m.intercept + m.ptd * getMeanPtd();
      var lineDs = ptdModelChart.data.datasets.find(function(ds) { return ds.type === 'line'; });
      if (!lineDs) return;
      lineDs.data = [{ x: ptdMxMin, y: adjIntercept + m.slope * ptdMxMin }, { x: ptdMxMax, y: adjIntercept + m.slope * ptdMxMax }];
      ptdModelChart.update('none');
    }

    updatePtdModelChart();

    var ptdModelCanvas    = document.getElementById('ptdModelChart');
    var ptdModelRolePopup = document.getElementById('rolePopup');
    ptdModelCanvas.addEventListener('mousemove', function(e) {
      if (!ptdShowDots) { if (!rolePinned.canvas) ptdModelRolePopup.classList.remove('visible'); return; }
      var rect = ptdModelCanvas.getBoundingClientRect();
      var mouseX = e.clientX - rect.left, mouseY = e.clientY - rect.top;
      var ca = ptdModelChart.chartArea;
      if (mouseX < ca.left || mouseX > ca.right || mouseY < ca.top || mouseY > ca.bottom) {
        if (!rolePinned.canvas) ptdModelRolePopup.classList.remove('visible'); return;
      }
      var HIT = 8, nearest = null, minDist = Infinity;
      ptdModelChart.data.datasets.forEach(function(ds) {
        if (ds.type !== 'scatter') return;
        ds.data.forEach(function(pt) {
          var px = ptdModelChart.scales.x.getPixelForValue(pt.x);
          var py = ptdModelChart.scales.y.getPixelForValue(pt.y);
          var d  = Math.hypot(mouseX - px, mouseY - py);
          if (d < HIT && d < minDist) { minDist = d; nearest = pt; }
        });
      });
      if (!nearest) { if (!rolePinned.canvas) ptdModelRolePopup.classList.remove('visible'); return; }
      if (rolePinned.canvas) return;
      var hs = headshots[nearest.profile] || '';
      document.getElementById('rPopupImg').src = hs;
      document.getElementById('rPopupImg').style.display = hs ? 'block' : 'none';
      document.getElementById('rPopupName').textContent  = nearest.player;
      document.getElementById('rPopupOrg').textContent   = nearest.org;
      document.getElementById('rPopupEvent').textContent = nearest.event;
      if (nearest.roles) {
        document.getElementById('rPopupAgentLabel').textContent = 'Roles';
        var rs = Object.entries(nearest.roles).sort(function(a,b){ return b[1]-a[1]; });
        document.getElementById('rPopupAgent').textContent = rs.map(function(e){ return e[0]+' '+e[1]+'%'; }).join(' / ');
      } else {
        document.getElementById('rPopupAgentLabel').textContent = 'Most Played Agent';
        var ag = nearest.agent || '';
        document.getElementById('rPopupAgent').textContent = ag.charAt(0).toUpperCase() + ag.slice(1);
      }
      document.getElementById('rPopupRating').textContent = nearest.y.toFixed(2);
      document.getElementById('rPopupWin').textContent    = (nearest.x * 100).toFixed(1) + '%';
      var ptdVal = ptdLookup[nearest.profile + '|' + nearest.event];
      var ptdRow = document.getElementById('rPopupPtdRow');
      if (ptdVal != null) {
        document.getElementById('rPopupPtd').textContent = (ptdVal * 100).toFixed(1) + '%';
        ptdRow.style.display = '';
      } else {
        ptdRow.style.display = 'none';
      }
      var onRight = e.clientX > window.innerWidth / 2;
      ptdModelRolePopup.style.left = (onRight ? e.clientX - ptdModelRolePopup.offsetWidth - 16 : e.clientX + 16) + 'px';
      ptdModelRolePopup.style.top  = (e.clientY - ptdModelRolePopup.offsetHeight / 2) + 'px';
      ptdModelRolePopup.classList.add('visible');
    });
    ptdModelCanvas.addEventListener('mouseleave', function() {
      if (!rolePinned.canvas) ptdModelRolePopup.classList.remove('visible');
    });
  }
})();

(function() {
  var winSlider  = document.getElementById('interactWinSlider');
  var ptdSlider  = document.getElementById('interactPtdSlider');
  var winVal     = document.getElementById('interactWinVal');
  var ptdVal     = document.getElementById('interactPtdVal');
  var expectedEl = document.getElementById('interactExpected');
  var matchBody  = document.getElementById('interactMatchBody');
  if (!winSlider || !ptdSlider) return;

  var selectedRole = 'Initiator';
  document.getElementById('interactRoleButtons').addEventListener('click', function(e) {
    var btn = e.target.closest('.interact-role');
    if (!btn) return;
    document.querySelectorAll('.interact-role').forEach(function(b) { b.classList.remove('active'); });
    btn.classList.add('active');
    selectedRole = btn.dataset.role;
    update();
  });

  function update() {
    var win = parseFloat(winSlider.value);
    var ptd = parseFloat(ptdSlider.value);
    winVal.textContent = win.toFixed(0);
    ptdVal.textContent = ptd.toFixed(1);

    var eq = ptdModelData.role_eqs && ptdModelData.role_eqs[selectedRole];
    if (!eq) return;
    var expected = eq.intercept + eq.slope * (win / 100) + eq.ptd * (ptd / 100);
    expectedEl.textContent = expected.toFixed(3);

    // Find closest match in allRolesData for selected role
    var pts = allRolesData[selectedRole] || [];
    var best = null, bestDist = Infinity;
    pts.forEach(function(p) {
      var pPtd = ptdLookup[p.profile + '|' + p.event];
      if (pPtd == null) return;
      var dx = (p.x - win / 100) / 0.20;
      var dy = (pPtd - ptd / 100) / 0.05;
      var dist = Math.sqrt(dx * dx + dy * dy);
      if (dist < bestDist) { bestDist = dist; best = { pt: p, ptd: pPtd }; }
    });

    matchBody.innerHTML = '';
    if (best) {
      var p = best.pt;
      var hs = headshots[p.profile] || '';
      var imgHtml = hs ? '<img src="' + hs + '" alt="">' : '';
      var tr = document.createElement('tr');
      tr.innerHTML =
        '<td class="td-img">' + imgHtml + p.player + '</td>' +
        '<td>' + p.org + '</td>' +
        '<td>' + p.event + '</td>' +
        '<td>' + (p.x * 100).toFixed(1) + '%</td>' +
        '<td>' + (best.ptd * 100).toFixed(1) + '%</td>' +
        '<td>' + p.y.toFixed(2) + '</td>';
      matchBody.appendChild(tr);
    }
  }

  winSlider.addEventListener('input', update);
  ptdSlider.addEventListener('input', update);
  update();
})();

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
<script src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
<script>
  var ptdEl = document.getElementById('ptdFormula');
  if (ptdEl) katex.render({{ ptd_formula | safe }}, ptdEl, {displayMode: true, throwOnError: false});
  var eqBoxEl = document.getElementById('ptdEqBox');
  if (eqBoxEl) katex.render({{ ptd_eq_latex | safe }}, eqBoxEl, {displayMode: true, throwOnError: false});
</script>
</body>
</html>
"""

ROLE_ORDER = ["Initiator", "Controller", "Duelist", "Sentinel"]

@article_overunder_bp.route("/")
def index():
    role_data = get_role_chart_data()
    idx = {r: i for i, r in enumerate(role_data["roles"])}
    order = [idx[r] for r in ROLE_ORDER]
    ordered = {
        "roles": ROLE_ORDER,
        "regions": role_data["regions"],
        "data": {region: [role_data["data"][region][i] for i in order] for region in role_data["regions"]},
    }
    ordered["overall"] = get_overall_by_role(ordered)
    scatter = get_scatter_data()
    headshots_path = os.path.join(DATA_DIR, "headshots.json")
    with open(headshots_path) as f:
        headshots = json.load(f)
    all_roles = get_all_roles_scatter_data()
    model_params      = get_model_coefficients(all_roles)
    residuals         = get_residuals_tables(all_roles, model_params)
    alltime_nobaiters = get_alltime_no_baiters(all_roles, model_params)
    ptd_vals          = get_ptd_values()
    ptd_kickoff       = get_ptd_kickoff_analysis(all_roles, ptd_vals)
    oxy_ptd_chart     = get_oxy_ptd_chart_data(ptd_vals)
    ptd_model         = get_model_with_ptd(all_roles, model_params, ptd_vals)
    ptd_lookup        = {f"{k[0]}|{k[1]}": v for k, v in ptd_vals.items()}
    ptd_all_scatter   = get_ptd_all_scatter(all_roles, ptd_vals)
    ptd_formula = json.dumps(r'\mathrm{PTD}_i = \frac{D_i / \mathrm{Rnd}_i}{\displaystyle\sum_{j=1}^{\underbrace{n}_{\text{qualifying players on team}}}\left(D_j / \mathrm{Rnd}_j\right)} \times \frac{\underbrace{n}_{\text{qualifying players on team}}}{5}')
    _ptd_coeff = ptd_model.get("role_eqs", {}).get("Initiator", {}).get("ptd", 0)
    _ptd_sign  = "+" if _ptd_coeff >= 0 else "-"
    ptd_eq_latex = json.dumps(
        r'\widehat{\text{Rating}} = \alpha_{\text{role}} + (\beta_{\text{role}} \times \text{Round Win\%}) '
        + _ptd_sign + r' (' + f'{abs(_ptd_coeff):.4f}' + r' \cdot \text{PTD})'
    )
    return render_template_string(PAGE_HTML,
        chart_json=json.dumps(ordered),
        scatter_json=json.dumps(scatter),
        headshots_json=json.dumps(headshots),
        all_roles_json=json.dumps(all_roles),
        model_json=json.dumps(model_params),
        residuals_json=json.dumps(residuals),
        alltime_nobaiters_json=json.dumps(alltime_nobaiters),
        ptd_kickoff_json=json.dumps(ptd_kickoff),
        oxy_ptd_chart_json=json.dumps(oxy_ptd_chart),
        ptd_model_json=json.dumps(ptd_model),
        ptd_lookup_json=json.dumps(ptd_lookup),
        ptd_all_scatter_json=json.dumps(ptd_all_scatter),
        ptd_formula=ptd_formula,
        ptd_eq_latex=ptd_eq_latex,
    )
