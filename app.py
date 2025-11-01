from flask import Flask, render_template, jsonify, request, make_response
import os, requests, math
from dataclasses import dataclass

app = Flask(__name__, template_folder="templates", static_folder="static")

# ---------------- Routes ----------------
@app.route("/")
def home():
    html = render_template("props.html")     # make sure this file is in /templates/
    resp = make_response(html, 200)
    resp.headers["Content-Type"] = "text/html; charset=utf-8"
    return resp

@app.route("/healthz")
def healthz():
    return "ok", 200

# You can keep any of your other API routes here — e.g. /api/onoff or /api/props

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

