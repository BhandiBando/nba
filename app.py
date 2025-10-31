from flask import Flask, render_template, jsonify
from datetime import datetime

app = Flask(__name__)

# ---------- sample data for ONE game (static for now) ----------
GAME = {
    "date": "2025-10-31",
    "tipoff_et": "9:30 PM ET",
    "away_team": "Los Angeles Lakers",
    "home_team": "Memphis Grizzlies",
    "arena": "FedExForum, Memphis, TN",
    # add odds later (moneyline/spread/total) once you hook up an API
    "odds": {
        "moneyline": {"LAL": None, "MEM": None},
        "spread": {"line": None, "fav": None},
        "total": None
    },
    # example props you can fill later
    "props_example": [
        {"player": "LeBron James", "market": "Points", "line": None},
        {"player": "Anthony Davis", "market": "Rebounds", "line": None},
        {"player": "Ja Morant", "market": "Points", "line": None}
    ]
}

# ----------------- pages -----------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/game/lakers-grizzlies")
def game_lal_mem():
    """HTML page for the specific game."""
    return render_template("game.html", g=GAME)

# ----------------- APIs (JSON) -----------------
@app.route("/api")
def api_status():
    return jsonify({"message": "NBA Betting Model API is live!", "status": "success"})

@app.route("/api/game/lakers-grizzlies")
def api_game_lal_mem():
    return jsonify(GAME)

@app.route("/healthz")
def health():
    return "ok", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
