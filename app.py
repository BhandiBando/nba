from flask import Flask, render_template, jsonify

app = Flask(__name__)

# HTML homepage
@app.route("/")
def home():
    return render_template("index.html")

# JSON status endpoint (handy for checks)
@app.route("/api")
def api_status():
    return jsonify({"message": "NBA Betting Model API is live!", "status": "success"})

# Health check
@app.route("/healthz")
def health():
    return "ok", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
