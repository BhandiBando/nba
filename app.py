# app.py — complete file

from flask import Flask, render_template, jsonify, request
from dataclasses import dataclass
from datetime import datetime
import os, io, math, time, json, requests, pdfplumber

app = Flask(__name__)

# ---------------- App/game config ----------------
SEASON = 2025  # 2025-26
GAME = {
    "key": "LAL@MEM_2025-10-31",
    "date": "2025-10-31",
    "tipoff_et": "9:30 PM ET",
    "away_team": "Los Angeles Lakers",
    "home_team": "Memphis Grizzlies",
    "arena": "FedExForum, Memphis, TN",
}
TEAM_ABBR = {"Los Angeles Lakers": "LAL", "Memphis Grizzlies": "MEM"}
TEAM_ID_BDL = {
    "ATL": 1, "BOS": 2, "BKN": 3, "CHA": 4, "CHI": 5, "CLE": 6, "DAL": 7, "DEN": 8, "DET": 9,
    "GSW": 10, "HOU": 11, "IND": 12, "LAC": 13, "LAL": 14, "MEM": 15, "MIA": 16, "MIL": 17,
    "MIN": 18, "NOP": 19, "NYK": 20, "OKC": 21, "ORL": 22, "PHI": 23, "PHX": 24, "POR": 25,
    "SAC": 26, "SAS": 27, "TOR": 28, "UTA": 29, "WAS": 30
}

# ---------------- Official NBA injury report (PDF) ----------------
NBA_INJURY_PDF_PATTERNS = [
    "https://ak-static.cms.nba.com/referee/injury/Injury-Report_{date}_12PM.pdf",
    "https://ak-static.cms.nba.com/referee/injury/Injury-Report_{date}_03PM.pdf",
    "https://ak-static.cms.nba.com/referee/injury/Injury-Report_{date}_05PM.pdf",
    "https://ak-static.cms.nba.com/referee/injury/Injury-Report_{date}_07PM.pdf",
]

def fetch_injury_pdf_bytes(date_str: str) -> bytes | None:
    for pattern in NBA_INJURY_PDF_PATTERNS:
        url = pattern.format(date=date_str)
        try:
            r = requests.get(url, timeout=12)
            if r.status_code == 200 and r.content:
                return r.content
        except Exception:
            pass
    return None

def parse_injuries_from_pdf(pdf_bytes: bytes) -> list[dict]:
    injuries = []
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                for raw in text.splitlines():
                    line = " ".join(raw.split())
                    parts = line.split()
                    if len(parts) < 4:
                        continue
                    team = parts[0]
                    if team not in TEAM_ID_BDL:
                        continue
                    if "," not in line:
                        continue
                    status_tokens = ("OUT","DOUBTFUL","QUESTIONABLE","PROBABLE")
                    status = None
                    for tok in status_tokens:
                        if f" {tok} " in f" {line} ":
                            status = tok; break
                    if not status:
                        continue
                    after_team = line.split(team, 1)[1].strip()
                    try:
                        player_part = after_team.split(status, 1)[0].strip()
                        notes = after_team.split(status, 1)[1].strip()
                    except Exception:
                        continue
                    injuries.append({"team": team, "player": player_part, "status": status, "notes": notes})
    except Exception:
        pass
    return injuries

def get_injuries_for_game(date_str: str, away_team: str, home_team: str) -> dict:
    pdf_bytes = fetch_injury_pdf_bytes(date_str)
    rows = parse_injuries_from_pdf(pdf_bytes) if pdf_bytes else []
    away = TEAM_ABBR[away_team]; home = TEAM_ABBR[home_team]
    return {
        "date": date_str,
        away: [r for r in rows if r["team"] == away],
        home: [r for r in rows if r["team"] == home],
        "raw_count": len(rows)
    }

# ---------------- Injury-driven projection core ----------------
@dataclass
class PlayerBaseline:
    player: str
    team: str
    mpg: float
    per36_pts: float
    per36_reb: float
    per36_ast: float
    role: str  # "primary_ballhandler", "wing_shooter", "big", etc.

# Minimal sample baselines for Lakers/Grizzlies (replace with live pulls later)
BASELINES = [
    PlayerBaseline("LeBron James", "LAL", 34.0, 25.5, 7.8, 7.5, "primary_ballhandler"),
    PlayerBaseline("Anthony Davis", "LAL", 35.0, 27.2, 13.0, 2.8, "big"),
    PlayerBaseline("Austin Reaves", "LAL", 30.0, 17.0, 4.6, 5.1, "secondary_ballhandler"),
    PlayerBaseline("D'Angelo Russell", "LAL", 30.0, 20.3, 3.6, 6.3, "secondary_ballhandler"),
    PlayerBaseline("Desmond Bane", "MEM", 34.0, 25.8, 5.3, 4.8, "wing_shooter"),
    PlayerBaseline("Jaren Jackson Jr.", "MEM", 31.0, 23.6, 7.1, 1.5, "big"),
    PlayerBaseline("Ja Morant", "MEM", 34.0, 27.0, 5.5, 8.0, "primary_ballhandler"),
]

def roster(team): 
    return [p for p in BASELINES if p.team == team]

def redistribute_minutes(team_roster, injured_names):
    exp = {p.player: p.mpg for p in team_roster}
    for inj in injured_names:
        injp = next((p for p in team_roster if inj.lower() in p.player.lower()), None)
        if not injp: 
            continue
        freed = injp.mpg
        same_role = [p for p in team_roster if p.player != injp.player and p.role == injp.role]
        others = [p for p in team_roster if p.player != injp.player and p.role != injp.role]
        if same_role:
            primary = same_role[:2]
            weights = {}
            if len(primary) == 1:
                weights[primary[0].player] = 0.55
            else:
                weights[primary[0].player] = 0.45
                weights[primary[1].player] = 0.30
            remain = 1.0 - sum(weights.values())
            for p in others:
                weights[p.player] = weights.get(p.player, 0) + remain / max(len(others), 1)
        else:
            weights = {p.player: 1.0 / (len(team_roster)-1) for p in team_roster if p.player != injp.player}
        for name, w in weights.items():
            exp[name] = exp.get(name, 0) + freed * w
        exp[injp.player] = 0.0
    for k, v in exp.items():
        exp[k] = max(12.0, min(40.0, v))
    return exp

def usage_adjust(role_gained):
    if role_gained == "primary_ballhandler": return (1.08, 0.85, 1.00)
    if role_gained == "big":                  return (1.02, 1.00, 1.08)
    if role_gained == "wing_shooter":         return (1.05, 1.00, 1.00)
    if role_gained == "secondary_ballhandler":return (1.06, 0.92, 1.00)
    return (1.00, 1.00, 1.00)

def reproject_stats(team_roster, expected_minutes, injured_names):
    missing_roles = [p.role for p in team_roster for name in injured_names if name.lower() in p.player.lower()]
    pts_mult = ast_mult = reb_mult = 1.0
    for r in missing_roles:
        pm, am, rm = usage_adjust(r)
        pts_mult *= pm; ast_mult *= am; reb_mult *= rm
    proj = {}
    for p in team_roster:
        m = expected_minutes.get(p.player, p.mpg)
        pts = p.per36_pts * (m/36.0) * (pts_mult if p.player not in injured_names else 0)
        reb = p.per36_reb * (m/36.0) * (reb_mult if p.player not in injured_names else 0)
        ast = p.per36_ast * (m/36.0) * (ast_mult if p.player not in injured_names else 0)
        proj[p.player] = {"pts": round(pts,2), "reb": round(reb,2), "ast": round(ast,2), "mpg": round(m,1)}
    return proj

def prob_hit_normal(proj_mean, line, sigma=3.2):
    z = (proj_mean - line) / max(1e-6, sigma)
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2)))

def score_edge(proj, line, sigma=3.2, confidence=1.0):
    edge = proj - line
    prob = prob_hit_normal(proj, line, sigma)
    return {"edge": round(edge,2), "prob_over": round(prob,3), "score": round(edge * prob * confidence, 3)}

# ---------------- On/Off via balldontlie (presence proxy) ----------------
def bdl_find_player_id(name):
    first, *rest = name.split()
    q = first
    r = requests.get(f"https://www.balldontlie.io/api/v1/players?search={q}", timeout=12).json()
    best = None
    for p in r.get("data", []):
        full = f"{p.get('first_name','')} {p.get('last_name','')}".strip().lower()
        if name.lower() in full:
            best = p; break
    if not best and r.get("data"):
        best = r["data"][0]
    return best["id"] if best else None

def bdl_team_games(team_abbr, season):
    team_id = TEAM_ID_BDL[team_abbr]
    page = 1; out = []
    while True:
        url = f"https://www.balldontlie.io/api/v1/games?seasons[]={season}&team_ids[]={team_id}&per_page=100&page={page}"
        data = requests.get(url, timeout=12).json()
        games = data.get("data", [])
        for g in games:
            is_home = (g["home_team"]["id"] == team_id)
            team_pts = g["home_team_score"] if is_home else g["visitor_team_score"]
            out.append({"game_id": g["id"], "date": g["date"][:10], "home": is_home, "team_pts": team_pts})
        if not games or len(games) < 100: break
        page += 1
    return out

def bdl_star_played(game_id, player_id):
    url = f"https://www.balldontlie.io/api/v1/stats?game_ids[]={game_id}&player_ids[]={player_id}&per_page=1"
    data = requests.get(url, timeout=12).json()
    return len(data.get("data", [])) > 0

def compute_onoff_points(team_abbr, star_name, season):
    pid = bdl_find_player_id(star_name)
    if not pid:
        return {"team": team_abbr, "star": star_name, "avg_in": None, "avg_out": None, "n_in": 0, "n_out": 0}
    gl = bdl_team_games(team_abbr, season)
    pts_in, pts_out = [], []
    for g in gl:
        played = bdl_star_played(g["game_id"], pid)
        (pts_in if played else pts_out).append(g["team_pts"])
    avg_in  = round(sum(pts_in)/len(pts_in), 1) if pts_in else None
    avg_out = round(sum(pts_out)/len(pts_out), 1) if pts_out else None
    return {"team": team_abbr, "star": star_name, "avg_in": avg_in, "avg_out": avg_out, "n_in": len(pts_in), "n_out": len(pts_out)}

# ---------------- Odds placeholder ----------------
def get_odds_stub():
    return {"moneyline": {"LAL": None, "MEM": None},
            "spread": {"fav": None, "line": None},
            "total": None}

# ---------------- Routes ----------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/game/lakers-grizzlies")
def game_lal_mem():
    date_str = GAME["date"]
    injuries = get_injuries_for_game(date_str, GAME["away_team"], GAME["home_team"])
    odds = get_odds_stub()

    lal_inj = [r["player"] for r in injuries.get("LAL", []) if r["status"] in ("OUT","DOUBTFUL")]
    mem_inj = [r["player"] for r in injuries.get("MEM", []) if r["status"] in ("OUT","DOUBTFUL")]

    lal_proj = reproject_stats(roster("LAL"), redistribute_minutes(roster("LAL"), lal_inj), lal_inj)
    mem_proj = reproject_stats(roster("MEM"), redistribute_minutes(roster("MEM"), mem_inj), mem_inj)

    lines = {
        "Anthony Davis": {"reb": 12.5, "pts": 25.5},
        "LeBron James": {"pts": 24.5, "ast": 6.5},
        "Desmond Bane": {"pts": 24.5},
        "Jaren Jackson Jr.": {"pts": 21.5, "reb": 7.5},
        "Ja Morant": {"pts": 25.5, "ast": 7.5}
    }

    candidates = []
    def add(name, stat, sigma):
        proj = (lal_proj.get(name) or mem_proj.get(name))
        if not proj or stat not in lines.get(name, {}): return
        s = score_edge(proj[stat], lines[name][stat], sigma=sigma, confidence=0.9)
        candidates.append({"player": name, "stat": stat, "proj": proj[stat],
                           "line": lines[name][stat], "prob_over": s["prob_over"], "edge": s["edge"]})
    add("Anthony Davis","reb", 2.8)
    add("LeBron James","ast", 2.5)
    add("Desmond Bane","pts", 4.0)
    add("Jaren Jackson Jr.","pts", 4.0)
    add("Ja Morant","pts", 4.5)
    candidates.sort(key=lambda x: x["edge"], reverse=True)

    vm = {
        "meta": GAME,
        "odds": odds,
        "injuries": injuries,
        "lal_proj": lal_proj,
        "mem_proj": mem_proj,
        "picks": candidates[:6]
    }
    return render_template("game.html", vm=vm)

@app.route("/analysis/lakers-grizzlies")
def analysis_lal_mem():
    date_str = GAME["date"]
    injuries = get_injuries_for_game(date_str, GAME["away_team"], GAME["home_team"])
    # Reuse picks from /game route logic (simplified here)
    lal_inj = [r["player"] for r in injuries.get("LAL", []) if r["status"] in ("OUT","DOUBTFUL")]
    mem_inj = [r["player"] for r in injuries.get("MEM", []) if r["status"] in ("OUT","DOUBTFUL")]
    lal_proj = reproject_stats(roster("LAL"), redistribute_minutes(roster("LAL"), lal_inj), lal_inj)
    mem_proj = reproject_stats(roster("MEM"), redistribute_minutes(roster("MEM"), mem_inj), mem_inj)
    lines = {
        "Anthony Davis": {"reb": 12.5, "pts": 25.5},
        "LeBron James": {"pts": 24.5, "ast": 6.5},
        "Desmond Bane": {"pts": 24.5},
        "Jaren Jackson Jr.": {"pts": 21.5, "reb": 7.5},
        "Ja Morant": {"pts": 25.5, "ast": 7.5}
    }
    candidates = []
    def add(name, stat, sigma):
        proj = (lal_proj.get(name) or mem_proj.get(name))
        if not proj or stat not in lines.get(name, {}): return
        s = score_edge(proj[stat], lines[name][stat], sigma=sigma, confidence=0.9)
        candidates.append({"player": name, "stat": stat, "proj": proj[stat],
                           "line": lines[name][stat], "prob_over": s["prob_over"], "edge": s["edge"]})
    add("Anthony Davis","reb", 2.8)
    add("LeBron James","ast", 2.5)
    add("Desmond Bane","pts", 4.0)
    add("Jaren Jackson Jr.","pts", 4.0)
    add("Ja Morant","pts", 4.5)
    candidates.sort(key=lambda x: x["edge"], reverse=True)

    onoff = [
        compute_onoff_points("LAL", "LeBron James", SEASON),
        compute_onoff_points("MEM", "Ja Morant", SEASON),
    ]
    vm = {"meta": GAME, "injuries": injuries, "picks": candidates[:6], "onoff": onoff}
    return render_template("analysis.html", vm=vm)

# ---- APIs
@app.route("/api/onoff")
def api_onoff():
    team = request.args.get("team", "LAL")
    star = request.args.get("star", "LeBron James")
    res = compute_onoff_points(team, star, SEASON)
    return jsonify(res)

@app.route("/api/game/lakers-grizzlies")
def api_game_lal_mem():
    date_str = GAME["date"]
    injuries = get_injuries_for_game(date_str, GAME["away_team"], GAME["home_team"])
    return jsonify({"meta": GAME, "injuries": injuries})

@app.route("/api")
def api_status():
    return jsonify({"message": "NBA Betting Model API is live!", "status": "success"})

@app.route("/healthz")
def health():
    return "ok", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)


