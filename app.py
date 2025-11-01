from flask import Flask, render_template, jsonify, request
import requests, math
from dataclasses import dataclass

app = Flask(__name__)

# ---------------- BASIC GAME META ----------------
SEASON = 2025
GAME = {
    "date": "2025-10-31",
    "home_team": "MEM",
    "away_team": "LAL",
    "tipoff": "9:30 PM EST"
}

TEAM_ID_BDL = {
    "LAL": 14,  # Lakers
    "MEM": 15,  # Grizzlies
    "DAL": 7    # Mavericks
}

# ---------------- PLAYER BASELINES ----------------
@dataclass
class PlayerBaseline:
    player: str
    team: str
    mpg: float
    per36_pts: float
    per36_reb: float
    per36_ast: float
    role: str

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

# ---------------- INJURY ADJUSTMENT LOGIC ----------------
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
            weights = {}
            primary = same_role[:2]
            if len(primary) == 1:
                weights[primary[0].player] = 0.55
            else:
                weights[primary[0].player] = 0.45
                weights[primary[1].player] = 0.30
            remain = 1.0 - sum(weights.values())
            for p in others:
                weights[p.player] = weights.get(p.player, 0) + remain / max(len(others), 1)
        else:
            weights = {p.player: 1.0 / (len(team_roster) - 1) for p in team_roster if p.player != injp.player}
        for name, w in weights.items():
            exp[name] = exp.get(name, 0) + freed * w
        exp[injp.player] = 0.0
    for k, v in exp.items():
        exp[k] = max(12.0, min(40.0, v))
    return exp

def usage_adjust(role):
    if role == "primary_ballhandler":
        return (1.08, 0.85, 1.00)
    if role == "big":
        return (1.02, 1.00, 1.08)
    if role == "wing_shooter":
        return (1.05, 1.00, 1.00)
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
        pts = p.per36_pts * (m / 36.0) * (pts_mult if p.player not in injured_names else 0)
        reb = p.per36_reb * (m / 36.0) * (reb_mult if p.player not in injured_names else 0)
        ast = p.per36_ast * (m / 36.0) * (ast_mult if p.player not in injured_names else 0)
        proj[p.player] = {"pts": round(pts, 2), "reb": round(reb, 2), "ast": round(ast, 2), "mpg": round(m, 1)}
    return proj

# ---------------- BALDONTLIE HELPERS ----------------
def _json_or_none(resp):
    try:
        if resp is None or not resp.ok:
            return None
        ctype = resp.headers.get("Content-Type", "")
        if "json" not in ctype:
            return None
        return resp.json()
    except Exception:
        return None

def bdl_find_player_id(name):
    try:
        first = name.split()[0]
        url = f"https://www.balldontlie.io/api/v1/players?search={first}"
        r = requests.get(url, timeout=12, headers={"User-Agent": "nba-props/1.0"})
        data = _json_or_none(r) or {}
        for p in data.get("data", []):
            full = f"{p.get('first_name','')} {p.get('last_name','')}".strip().lower()
            if name.lower() in full:
                return p.get("id")
        return (data.get("data") or [{}])[0].get("id")
    except Exception:
        return None

def bdl_team_games(team_abbr, season):
    try:
        team_id = TEAM_ID_BDL[team_abbr]
        page, out = 1, []
        while True:
            url = f"https://www.balldontlie.io/api/v1/games?seasons[]={season}&team_ids[]={team_id}&per_page=100&page={page}"
            r = requests.get(url, timeout=12, headers={"User-Agent": "nba-props/1.0"})
            data = _json_or_none(r) or {}
            games = data.get("data", [])
            for g in games:
                is_home = g.get("home_team", {}).get("id") == team_id
                team_pts = g.get("home_team_score") if is_home else g.get("visitor_team_score")
                out.append({
                    "game_id": g.get("id"), "date": (g.get("date", "")[:10]),
                    "home": is_home, "team_pts": team_pts
                })
            if not games or len(games) < 100: break
            page += 1
        return out
    except Exception:
        return []

def bdl_star_played(game_id, player_id):
    try:
        if not game_id or not player_id: return False
        url = f"https://www.balldontlie.io/api/v1/stats?game_ids[]={game_id}&player_ids[]={player_id}&per_page=1"
        r = requests.get(url, timeout=12, headers={"User-Agent": "nba-props/1.0"})
        data = _json_or_none(r) or {}
        return len(data.get("data", [])) > 0
    except Exception:
        return False

def compute_onoff_points(team_abbr, star_name, season):
    pid = bdl_find_player_id(star_name)
    if not pid:
        return {"team": team_abbr, "star": star_name, "avg_in": None, "avg_out": None, "n_in": 0, "n_out": 0}
    gl = bdl_team_games(team_abbr, season)
    if not gl:
        return {"team": team_abbr, "star": star_name, "avg_in": None, "avg_out": None, "n_in": 0, "n_out": 0}
    pts_in, pts_out = [], []
    for g in gl:
        played = bdl_star_played(g["game_id"], pid)
        (pts_in if played else pts_out).append(g["team_pts"])
    avg_in  = round(sum(pts_in)/len(pts_in), 1) if pts_in else None
    avg_out = round(sum(pts_out)/len(pts_out), 1) if pts_out else None
    return {"team": team_abbr, "star": star_name, "avg_in": avg_in, "avg_out": avg_out, "n_in": len(pts_in), "n_out": len(pts_out)}

def safe_compute_onoff(team, star, season):
    try:
        return compute_onoff_points(team, star, season)
    except Exception:
        return {"team": team, "star": star, "avg_in": None, "avg_out": None, "n_in": 0, "n_out": 0}

# ---------------- ROUTES ----------------
@app.route('/')
def home():
    return render_template("index.html")

@app.route('/api/onoff')
def api_onoff():
    team = request.args.get("team", "LAL")
    star = request.args.get("star", "LeBron James")
    res = safe_compute_onoff(team, star, SEASON)
    return jsonify(res)

@app.route('/analysis/lakers-grizzlies')
def analysis_lal_mem():
    lal_inj = ["Luka Doncic", "Marcus Smart", "Gabe Vincent", "Maxi Kleber"]
    mem_inj = ["Brandon Clarke", "Zach Edey", "Ty Jerome"]

    lal_roster = roster("LAL")
    mem_roster = roster("MEM")

    lal_proj = reproject_stats(lal_roster, redistribute_minutes(lal_roster, lal_inj), lal_inj)
    mem_proj = reproject_stats(mem_roster, redistribute_minutes(mem_roster, mem_inj), mem_inj)

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
        line = lines[name][stat]
        edge = round(proj[stat] - line, 2)
        prob = 0.5 * (1 + math.erf((proj[stat]-line)/(3.2*math.sqrt(2))))
        candidates.append({"player": name, "stat": stat, "proj": proj[stat], "line": line, "prob_over": round(prob,3), "edge": edge})
    add("Anthony Davis","reb", 2.8)
    add("LeBron James","ast", 2.5)
    add("Desmond Bane","pts", 4.0)
    add("Jaren Jackson Jr.","pts", 4.0)
    add("Ja Morant","pts", 4.5)
    candidates.sort(key=lambda x: x["edge"], reverse=True)

    onoff = [
        safe_compute_onoff("LAL", "LeBron James", SEASON),
        safe_compute_onoff("MEM", "Ja Morant", SEASON),
    ]
    vm = {"meta": GAME, "picks": candidates[:6], "onoff": onoff}
    return render_template("analysis.html", vm=vm)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)


