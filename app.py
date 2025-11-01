from flask import Flask, jsonify, render_template, request
import os, time, math, datetime as dt
from typing import Dict, List, Optional
import requests
import pandas as pd
import numpy as np
from scipy.stats import norm

app = Flask(__name__)

# ----------------- CONFIG -----------------
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "YOUR_THE_ODDS_API_KEY")
BL_BASE = "https://api.balldontlie.io/v1"
BL_HEADERS = {}  # If you have a key: {"Authorization": "Bearer ..."}
SAVE_DIR = "data"
os.makedirs(SAVE_DIR, exist_ok=True)

def today_str():
    # naive ET date string (good enough for a daily slate)
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=-4))).strftime("%Y-%m-%d")

# ----------------- ESPN INJURIES -----------------
def espn_injuries() -> pd.DataFrame:
    url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
    try:
        j = requests.get(url, timeout=20).json()
    except Exception:
        return pd.DataFrame(columns=["team","player","status","comment","date"])
    rows = []
    for team in j.get("injuries", []):
        tabbr = team["team"]["abbreviation"]
        for p in team.get("injuries", []):
            rows.append({
                "team": tabbr,
                "player": p["athlete"]["displayName"],
                "status": p.get("status"),
                "comment": p.get("details") or p.get("comment") or "",
                "date": p.get("date") or ""
            })
    return pd.DataFrame(rows)

# ----------------- ODDS (DK/FD via The Odds API) -----------------
def odds_events() -> List[dict]:
    url = "https://api.the-odds-api.com/v4/sports/basketball_nba/events"
    return requests.get(url, params={"apiKey": ODDS_API_KEY}, timeout=25).json()

def event_props(event_id: str, markets: List[str]) -> dict:
    url = f"https://api.the-odds-api.com/v4/sports/basketball_nba/events/{event_id}/odds"
    params = {"apiKey": ODDS_API_KEY, "regions": "us", "markets": ",".join(markets), "oddsFormat": "american"}
    return requests.get(url, params=params, timeout=25).json()

def pull_dk_fd_player_props(markets=None) -> pd.DataFrame:
    markets = markets or ["player_points","player_rebounds","player_assists","player_steals","player_blocks"]
    events = odds_events()
    props_rows = []
    for ev in events:
        ev_id = ev["id"]; home = ev["home_team"]; away = ev["away_team"]
        odds = event_props(ev_id, markets)
        for bm in odds.get("bookmakers", []):
            if bm.get("key") not in ("draftkings", "fanduel"):
                continue
            for m in bm.get("markets", []):
                mkey = m["key"]
                for out in m.get("outcomes", []):
                    props_rows.append({
                        "event_id": ev_id, "home": home, "away": away,
                        "book": bm["key"],
                        "market": mkey,
                        "player": out.get("description") or out.get("name"),
                        "line": out.get("point"),
                        "price": out.get("price"),
                    })
        time.sleep(0.05)
    return pd.DataFrame(props_rows)

# ----------------- BALLDONTLIE: L10 + H2H -----------------
def bl_search_player(name: str) -> Optional[dict]:
    j = requests.get(f"{BL_BASE}/players", params={"search": name, "per_page": 25}, headers=BL_HEADERS, timeout=20).json()
    data = j.get("data", [])
    return data[0] if data else None

def bl_player_last_n_stats(player_id: int, n: int=10) -> pd.DataFrame:
    stats = []
    page = 1
    while len(stats) < n and page <= 10:
        j = requests.get(f"{BL_BASE}/stats", params={"player_ids[]": player_id, "per_page": 100, "page": page, "postseason": "false"}, headers=BL_HEADERS, timeout=20).json()
        data = j.get("data", [])
        if not data: break
        stats.extend(data); page += 1
        time.sleep(0.05)
    df = pd.DataFrame([{
        "date": d["game"]["date"][:10],
        "opponent": d["game"]["home_team"]["abbreviation"] if d["game"]["home_team"]["id"] != d["team"]["id"] else d["game"]["visitor_team"]["abbreviation"],
        "PTS": d["pts"], "REB": d["reb"], "AST": d["ast"], "STL": d["stl"], "BLK": d["blk"]
    } for d in stats])
    if df.empty: return df
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date", ascending=False).head(n).reset_index(drop=True)

def bl_team_id_map() -> pd.DataFrame:
    j = requests.get(f"{BL_BASE}/teams", headers=BL_HEADERS, timeout=20).json()
    return pd.DataFrame(j["data"])[["id","abbreviation","full_name"]]

def bl_h2h_last5(home_abbr: str, away_abbr: str) -> pd.DataFrame:
    teams = bl_team_id_map()
    def tid(abbr):
        s = teams[teams["abbreviation"]==abbr]
        return int(s.iloc[0]["id"]) if not s.empty else None
    home_id, away_id = tid(home_abbr), tid(away_abbr)
    if not home_id or not away_id: return pd.DataFrame()

    def pull_team_games(team_id: int, seasons: List[int]) -> List[dict]:
        games = []
        for season in seasons:
            page = 1
            while page <= 5:
                j = requests.get(f"{BL_BASE}/games", params={"team_ids[]": team_id, "seasons[]": season, "per_page": 100, "page": page}, headers=BL_HEADERS, timeout=20).json()
                data = j.get("data", [])
                if not data: break
                games.extend(data); page += 1
                time.sleep(0.05)
        return games

    year = dt.datetime.now().year
    seasons = [year, year-1, year-2]
    g = pull_team_games(home_id, seasons) + pull_team_games(away_id, seasons)

    def is_pair(x):
        a = x["home_team"]["id"]; b = x["visitor_team"]["id"]
        return {a,b} == {home_id, away_id}
    h2h = [x for x in g if is_pair(x)]
    uniq = {x["id"]: x for x in h2h}
    rows = []
    for x in sorted(uniq.values(), key=lambda y: y["date"], reverse=True)[:5]:
        rows.append({
            "date": x["date"][:10],
            "home": x["home_team"]["abbreviation"],
            "away": x["visitor_team"]["abbreviation"],
            "home_score": x["home_team_score"],
            "away_score": x["visitor_team_score"],
            "margin_home": x["home_team_score"] - x["visitor_team_score"]
        })
    return pd.DataFrame(rows)

# ----------------- MODEL -----------------
STAT_COL_FOR_MARKET = {
    "player_points": "PTS",
    "player_rebounds": "REB",
    "player_assists": "AST",
    "player_steals": "STL",
    "player_blocks": "BLK"
}

def safe_mean(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna()
    return float(x.mean()) if len(x) else np.nan

def safe_std(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna()
    return float(x.std(ddof=1)) if len(x) > 1 else np.nan

def predict_and_score(row: pd.Series, l10_df: pd.DataFrame) -> Dict:
    stat = STAT_COL_FOR_MARKET.get(row["market"])
    if not stat or l10_df.empty:
        return {"pred_mean": np.nan, "sd": np.nan, "hit_prob": np.nan, "ev_per_$": np.nan, "score": np.nan}

    base = safe_mean(l10_df[stat])
    sd = safe_std(l10_df[stat])
    if np.isnan(sd):
        sd = max(1.5, math.sqrt(abs(base))) if not np.isnan(base) else 2.0

    line = float(row["line"]) if row["line"] is not None else np.nan
    price = float(row["price"]) if row["price"] is not None else np.nan
    if np.isnan(base) or np.isnan(line) or np.isnan(price):
        return {"pred_mean": base, "sd": sd, "hit_prob": np.nan, "ev_per_$": np.nan, "score": np.nan}

    # Over prob (Normal assumption). Upgrade later with opponent/pace + injury usage shifts.
    z = (base - line) / sd
    hit_prob = 1 - norm.cdf(0 - z)

    payout = (price/100.0) if price>0 else (100.0/abs(price))
    ev = hit_prob * payout - (1 - hit_prob) * 1.0
    score = 0.7*hit_prob + 0.3*ev

    return {"pred_mean": float(base), "sd": float(sd), "hit_prob": float(hit_prob), "ev_per_$": float(ev), "score": float(score)}

# ----------------- ROUTES (UI + APIs) -----------------
@app.route("/")
def home():
    # redirect traffic to the props page (simple UI)
    return render_template("props.html")

@app.route("/api/injuries")
def api_injuries():
    df = espn_injuries()
    return jsonify(df.to_dict(orient="records"))

@app.route("/api/games")
def api_games():
    props = pull_dk_fd_player_props(markets=["player_points"])  # light call to get events
    games = props[["event_id","home","away"]].drop_duplicates() if not props.empty else pd.DataFrame(columns=["event_id","home","away"])
    out = []
    for _, g in games.iterrows():
        h = bl_h2h_last5(g["home"], g["away"])
        if not h.empty:
            out.append({
                "event_id": g["event_id"],
                "home": g["home"],
                "away": g["away"],
                "last5": h.to_dict(orient="records"),
                "avg_home_pts": float(h["home_score"].mean()),
                "avg_away_pts": float(h["away_score"].mean()),
                "avg_margin_home": float(h["margin_home"].mean())
            })
        time.sleep(0.05)
    return jsonify(out)

@app.route("/api/props")
def api_props():
    date_str = request.args.get("date") or today_str()
    markets = request.args.get("markets", "player_points,player_rebounds,player_assists,player_steals,player_blocks").split(",")
    inj = espn_injuries()
    props = pull_dk_fd_player_props(markets=markets)
    if props.empty:
        return jsonify([])

    # last-10 per player appearing in props
    players = sorted(set(props["player"].dropna().tolist()))
    l10_map: Dict[str, pd.DataFrame] = {}
    for p in players:
        info = bl_search_player(p)
        l10_map[p] = bl_player_last_n_stats(info["id"], n=10) if info else pd.DataFrame()
        time.sleep(0.05)

    # score each prop
    scored = []
    for _, r in props.iterrows():
        l10 = l10_map.get(r["player"], pd.DataFrame())
        sc = predict_and_score(r, l10)
        row = r.to_dict()
        row.update(sc)
        # attach player injury flag
        pinj = inj[inj["player"]==r["player"]]
        if not pinj.empty:
            row["injury_status"] = pinj.iloc[0]["status"]
            row["injury_comment"] = pinj.iloc[0]["comment"]
        else:
            row["injury_status"] = None
            row["injury_comment"] = None
        scored.append(row)

    ranked = pd.DataFrame(scored).sort_values(["score","ev_per_$","hit_prob"], ascending=False)
    ranked.to_csv(os.path.join(SAVE_DIR, f"ranked_props_{date_str}.csv"), index=False)

    topk = int(request.args.get("topk", 100))
    return jsonify(ranked.head(topk).to_dict(orient="records"))

# healthcheck
@app.route("/healthz")
def healthz():
    return "ok", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
