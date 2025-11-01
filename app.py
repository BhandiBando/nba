# app.py
from flask import Flask, render_template, jsonify, request
import requests, math
from dataclasses import dataclass

app = Flask(__name__)

# ---------------- Game meta ----------------
SEASON = 2025  # 2025-26 season
GAME = {
    "date": "2025-10-31",
    "tipoff": "9:30 PM ET",
    "away_team": "Los Angeles Lakers",
    "home_team": "Memphis Grizzlies",
    "arena": "FedExForum, Memphis, TN",
}

# balldontlie team IDs (subset we need)
TEAM_ID_BDL = {
    "LAL": 14,
    "MEM": 15,
    "DAL": 7,
}

# ---------------- Player baselines (expand as you go) ----------------
@dataclass
class PlayerBaseline:
    player: str
    team: str  # team abbr
    mpg: float
    per36_pts: float
    per36_reb: float
    per36_ast: float
    role: str  # primary_ballhandler | secondary_ballhandler | wing_shooter | big

BASELINES = [
    # Lakers
    PlayerBaseline("LeBron James", "LAL", 34.0, 25.5, 7.8, 7.5, "primary_ballhandler"),
    PlayerBaseline("Anthony Davis", "LAL", 35.0, 27.2, 13.0, 2.8, "big"),
    PlayerBaseline("Austin Reaves", "LAL", 30.0, 17.0, 4.6, 5.1, "secondary_ballhandler"),
    PlayerBaseline("D'Angelo Russell", "LAL", 30.0, 20.3, 3.6, 6.3, "secondary_ballhandler"),
    # Added LAL to cover tonight
    PlayerBaseline("Rui Hachimura", "LAL", 29.0, 17.5, 6.2, 1.8, "wing_shooter"),
    PlayerBaseline("Jaxson Hayes", "LAL", 18.0, 11.5, 10.4, 1.0, "big"),

    # Grizzlies
    PlayerBaseline("Desmond Bane", "MEM", 34.0, 25.8, 5.3, 4.8, "wing_shooter"),
    PlayerBaseline("Jaren Jackson Jr.", "MEM", 31.0, 23.6, 7.1, 1.5, "big"),
    PlayerBaseline("Ja Morant", "MEM", 34.0, 27.0, 5.5, 8.0, "primary_ballhandler"),
    # Added MEM to cover tonight
    PlayerBaseline("Santi Aldama", "MEM", 28.0, 16.2, 7.6, 2.6, "wing_shooter"),
    PlayerBaseline("Kentavious Caldwell-Pope", "MEM", 32.0, 13.5, 3.6, 2.4, "wing_shooter"),
]

def roster(team_abbr):
    return [p for p in BASELINES if p.team == team_abbr]

# ---------------- Minutes + usage adjustments ----------------
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
    # clamp
    for k, v in exp.items():
        exp[k] = max(12.0, min(40.0, v))
    return exp

def usage_adjust(role):
    if role == "primary_ballhandler":       return (1.08, 0.85, 1.00)
    if role == "secondary_ballhandler":     return (1.06, 0.92, 1.00)
    if role == "big":                       return (1.02, 1.00, 1.08)
    if role == "wing_shooter":              return (1.05, 1.00, 1.00)
    return (1.00, 1.00, 1.00)

def reproject_stats(team_roster, expected_minutes, injured_names):
    missing_roles = [p.role for p in team_roster for name in injured_names if name.lower() in p.player.lower()]
    pts_mult = ast_mult = reb_mult = 1.0
    for r in missing_roles:
        pm, am, rm = usage_adjust(r)
        pts_mult *= pm; ast_mult *= am; reb_mult *= rm

    out = {}
    for p in team_roster:
        m = expected_minutes.get(p.player, p.mpg)
        pts = p.per36_pts * (m / 36.0) * (pts_mult if p.player not in injured_names else 0)
        reb = p.per36_reb * (m / 36.0) * (reb_mult if p.player not in injured_names else 0)
        ast = p.per36_ast * (m / 36.0) * (ast_mult if p.player not in injured_names else 0)
        out[p.player] = {"pts": round(pts, 2), "reb": round(reb, 2), "ast": round(ast, 2), "mpg": round(m, 1)}
    return out

def prob_over_normal(mean, line, sigma=3.2):
    z = (mean - line) / max(1e-6, sigma)
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2)))

# ---------------- balldontlie on/off (robust) ----------------
def _json_or_none(resp):
    try:
        if not resp or not resp.ok:
            return None
        if "json" not in (resp.headers.get("Content-Type", "") or ""):
            return None
        return resp.json()
    except Exception:
        return None

def bdl_find_player_id(name):
    try:
        q = name.split()[0]
        r = requests.get(
            f"https://www.balldontlie.io/api/v1/players?search={q}",
            timeout=12, headers={"User-Agent": "nba-props/1.0"}
        )
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
        out, page = [], 1
        while True:
            url = f"https://www.balldontlie.io/api/v1/games?seasons[]={season}&team_ids[]={team_id}&per_page=100&page={page}"
            r = requests.get(url, timeout=12, headers={"User-Agent": "nba-props/1.0"})
            data = _json_or_none(r) or {}
            games = data.get("data", [])
            for g in games:
                is_home = (g.get("home_team", {}).get("id") == team_id)
                team_pts = g.get("home_team_score") if is_home else g.get("visitor_team_score")
                out.append({"game_id": g.get("id"), "team_pts": team_pts})
            if not games or len(games) < 100:
                break
            page += 1
        return out
    except Exception:
        return []

def bdl_star_played(game_id, player_id):
    try:
        if not game_id or not player_id:
            return False
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
        (pts_in if bdl_star_played(g["game_id"], pid) else pts_out).append(g["team_pts"])
    avg_in  = round(sum(pts_in)/len(pts_in), 1) if pts_in else None
    avg_out = round(sum(pts_out)/len(pts_out), 1) if pts_out else None
    return {"team": team_abbr, "star": star_name, "avg_in": avg_in, "avg_out": avg_out, "n_in": len(pts_in), "n_out": len(pts_out)}

def safe_compute_onoff(team, star, season):
    try:
        return compute_onoff_points(team, star, season)
    except Exception:
        return {"team": team, "star": star, "avg_in": None, "avg_out": None, "n_in": 0, "n_out": 0}

# ---------------- Routes ----------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/onoff")
def api_onoff():
    team = request.args.get("team", "LAL")
    star = request.args.get("star", "LeBron James")
    return jsonify(safe_compute_onoff(team, star, SEASON))

@app.route("/analysis/lakers-grizzlies")
def analysis_lal_mem():
    # Your verified OUT lists for Oct 31, 2025
    lal_inj = ["Adou Thiero", "Gabe Vincent", "Maxi Kleber", "LeBron James"]
    mem_inj = ["Ty Jerome", "Scotty Pippen Jr.", "Brandon Clarke", "Zach Edey"]

    lal = roster("LAL"); mem = roster("MEM")
    lal_proj = reproject_stats(lal, redistribute_minutes(lal, lal_inj), lal_inj)
    mem_proj = reproject_stats(mem, redistribute_minutes(mem, mem_inj), mem_inj)

    # Book lines we want to evaluate (edit freely)
    lines = {
        # Lakers
        "Anthony Davis": {"pts": 25.5, "reb": 12.5, "ast": 3.5},
        "Austin Reaves": {"pts": 16.5, "ast": 4.5, "reb": 3.5},
        "Rui Hachimura": {"pts": 14.5, "reb": 5.5},
        # Grizzlies
        "Desmond Bane": {"pts": 24.5, "ast": 4.5},
        "Jaren Jackson Jr.": {"pts": 21.5, "reb": 7.5},
        "Ja Morant": {"pts": 25.5, "ast": 7.5},
        "Santi Aldama": {"pts": 13.5, "reb": 6.5},
    }

    candidates = []
    def add(name, stat, sigma):
        proj = (lal_proj.get(name) or mem_proj.get(name))
        if not proj: return
        if stat not in lines.get(name, {}): return
        line = lines[name][stat]
        mean = proj[stat]
        prob = prob_over_normal(mean, line, sigma=sigma)
        edge = round(mean - line, 2)
        candidates.append({
            "player": name, "stat": stat, "proj": round(mean,2),
            "line": line, "prob_over": round(prob,3), "edge": edge
        })

    # tune sigmas by stat/role
    add("Anthony Davis","reb", 2.8)
    add("Anthony Davis","pts", 4.5)
    add("Austin Reaves","ast", 2.6)
    add("Rui Hachimura","pts", 3.6)

    add("Desmond Bane","pts", 4.0)
    add("Jaren Jackson Jr.","pts", 4.0)
    add("Ja Morant","pts", 4.5)
    add("Santi Aldama","reb", 2.8)

    candidates.sort(key=lambda x: x["edge"], reverse=True)

    vm = {
        "meta": {"date": GAME["date"], "tipoff": GAME["tipoff"], "away_team": "LAL", "home_team": "MEM"},
        "picks": candidates[:10],
        "onoff": [
            safe_compute_onoff("LAL", "LeBron James", SEASON),
            safe_compute_onoff("MEM", "Ja Morant", SEASON),
        ],
        # Optional: if you wire the PDF injury parser later, pass injuries here
        "injuries": {"LAL": [{"player": n, "status": "OUT", "notes": ""} for n in lal_inj],
                     "MEM": [{"player": n, "status": "OUT", "notes": ""} for n in mem_inj]},
    }
    return render_template("analysis.html", vm=vm)

# healthcheck
@app.route("/healthz")
def healthz(): return "ok", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)


