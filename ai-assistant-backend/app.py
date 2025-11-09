from flask import Flask, request, jsonify
from flask_cors import CORS
import threading
import time
import base64
import os
import io
from PIL import Image
import random
import requests

app = Flask(__name__)
CORS(app)

# Simple in-memory state to simulate live metrics
state = {
    "running": False,
    "metrics": None,
    "assessment": None,
}

# Helper to simulate metrics
def generate_metrics():
    # Random-ish values to show in the UI
    blink_count = random.randint(0, 10)
    ear = round(random.uniform(0.18, 0.32), 3)
    mar = round(random.uniform(0.3, 0.8), 3)
    perclos = round(random.uniform(5.0, 40.0), 1)
    head_rotation = round(random.uniform(-20, 20), 0)

    lane_deviation_score = round(random.uniform(0, 100), 0)
    micro_correction_rate = random.randint(0, 20)
    steering_smoothness = round(random.uniform(30, 100), 0)

    # small logic to set flags
    is_drowsy = perclos > 25 or ear < 0.20
    is_yawning = mar > 0.6

    steering_alert_level = "LOW"
    if lane_deviation_score > 75:
        steering_alert_level = "CRITICAL"
    elif lane_deviation_score > 50:
        steering_alert_level = "WARNING"
    elif lane_deviation_score > 25:
        steering_alert_level = "CAUTION"

    metrics = {
        "camera_metrics": {
            "camera_status": "active",
            "blink_count": blink_count,
            "ear": ear,
            "mar": mar,
            "perclos": perclos,
            "head_rotation": head_rotation,
            "is_drowsy": is_drowsy,
            "is_yawning": is_yawning,
        },
        "steering_metrics": {
            "lane_deviation_score": lane_deviation_score,
            "micro_correction_rate": micro_correction_rate,
            "steering_smoothness": steering_smoothness,
            "steering_alert_level": steering_alert_level,
        },
        "performance": {
            "fps": random.randint(8, 30)
        }
    }

    # derive assessment
    risk_score = int((perclos / 40.0) * 100 * 0.6 + (lane_deviation_score / 100.0) * 100 * 0.4)
    risk_level = "LOW"
    if risk_score > 75:
        risk_level = "CRITICAL"
    elif risk_score > 55:
        risk_level = "HIGH"
    elif risk_score > 30:
        risk_level = "MODERATE"

    risk_factors = []
    if is_drowsy:
        risk_factors.append("drowsiness")
    if is_yawning:
        risk_factors.append("yawning")
    if lane_deviation_score > 50:
        risk_factors.append("lane deviation")

    assessment = {
        "risk_level": risk_level,
        "risk_score": risk_score,
        "action_needed": (
            "Pull over and rest immediately" if risk_level in ["CRITICAL", "HIGH"] else "Take a short break soon"
        ),
        "risk_factors": risk_factors,
    }

    return metrics, assessment

# Background thread that updates metrics while running
def background_loop():
    while state["running"]:
        metrics, assessment = generate_metrics()
        state["metrics"] = metrics
        state["assessment"] = assessment
        time.sleep(1)  # update every second

@app.route("/api/start", methods=["POST"])
def api_start():
    if state["running"]:
        return jsonify({"status": "info", "message": "Already running"})
    state["running"] = True
    # start background thread
    t = threading.Thread(target=background_loop, daemon=True)
    t.start()
    return jsonify({"status": "success", "message": "Monitoring started"})

@app.route("/api/stop", methods=["POST"])
def api_stop():
    state["running"] = False
    # clear metrics
    state["metrics"] = None
    state["assessment"] = None
    return jsonify({"status": "success", "message": "Monitoring stopped"})

@app.route("/api/reset", methods=["POST"])
def api_reset():
    # reset counters (for demo just clear metrics)
    state["metrics"] = None
    state["assessment"] = None
    return jsonify({"status": "success", "message": "Counters reset"})

@app.route("/api/process_frame", methods=["POST"])
def api_process_frame():
    # Accept a JSON payload { frame: "<base64>" }
    data = request.get_json() or {}
    frame_b64 = data.get("frame")
    if not frame_b64:
        return jsonify({"status": "error", "message": "No frame provided"}), 400

    try:
        image_bytes = base64.b64decode(frame_b64)
        # Optional: store a small preview locally for debugging
        try:
            img = Image.open(io.BytesIO(image_bytes))
            os.makedirs("ai-assistant-backend/tmp", exist_ok=True)
            img.save("ai-assistant-backend/tmp/last_frame.jpg", format="JPEG", quality=60)
        except Exception:
            pass

        # For demo, we don't run heavy processing here; processing is simulated in background
        return jsonify({"status": "success", "message": "Frame received"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/generate_alert", methods=["POST"])
def api_generate_alert():
    """Proxy endpoint to generate a short alert message using Gemini.

    Expects JSON body with keys: metrics (object), assessment (object), alert_type (str, optional)
    Reads GEMINI_API_KEY from environment. Returns { status: 'success', text: '...' }
    """
    data = request.get_json() or {}
    metrics = data.get("metrics") or {}
    assessment = data.get("assessment") or {}
    alert_type = data.get("alert_type", "alert")

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return jsonify({"status": "error", "message": "Gemini API key not configured on server (GEMINI_API_KEY)"}), 500

    # Build prompt from provided metrics/assessment (kept short)
    prompt_lines = [
        f"You are AURA, a caring AI driving companion. Generate a short (1-2 sentence) spoken alert for: {alert_type}.",
    ]

    if assessment:
        prompt_lines.append(f"Risk Level: {assessment.get('risk_level')}.")
        prompt_lines.append(f"Risk Score: {assessment.get('risk_score')}/100.")
        if assessment.get('action_needed'):
            prompt_lines.append(f"Action: {assessment.get('action_needed')}.")

    # Add a few camera/steering details if present
    cam = metrics.get("camera_metrics") if isinstance(metrics, dict) else None
    steer = metrics.get("steering_metrics") if isinstance(metrics, dict) else None
    if cam:
        prompt_lines.append(
            f"Eye metrics - PERCLOS: {cam.get('perclos', 'N/A')}%, EAR: {cam.get('ear', 'N/A')}, blinks: {cam.get('blink_count', 'N/A')}.")
    if steer:
        prompt_lines.append(
            f"Steering - lane deviation: {steer.get('lane_deviation_score', 'N/A')}, alert level: {steer.get('steering_alert_level', 'N/A')}.")

    prompt_lines.append("Keep tone calm, concise, and provide one clear action the driver can take.")
    prompt = "\n".join(prompt_lines)

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={api_key}"
    body = {
        "contents": [
            {"parts": [{"text": prompt}]}
        ]
    }

    try:
        resp = requests.post(url, json=body, timeout=10)
        resp.raise_for_status()
        text_data = resp.json()
        # Try to extract generated text safely
        candidates = text_data.get("candidates") or []
        if candidates and isinstance(candidates, list):
            first = candidates[0]
            parts = first.get("content", {}).get("parts") if isinstance(first.get("content"), dict) else None
            if parts and isinstance(parts, list) and len(parts) > 0:
                text = parts[0].get("text", "")
            else:
                text = first.get("output", "") or ""
        else:
            text = text_data.get("output", "") or ""

        if not text:
            # fallback message
            text = "I'm detecting reduced attention. Please take a short break or pull over safely if you feel drowsy."

        return jsonify({"status": "success", "text": text})
    except Exception as e:
        # Return fallback on any error
        fallback = "I'm detecting reduced attention. Please take a short break or pull over safely if you feel drowsy."
        return jsonify({"status": "error", "message": str(e), "text": fallback}), 500

@app.route("/api/metrics", methods=["GET"])
def api_metrics():
    if state["metrics"]:
        return jsonify({"status": "success", "data": state["metrics"]})
    else:
        # Return 200 with an 'info' status so frontend fetch doesn't treat this as an HTTP error.
        return jsonify({"status": "info", "message": "No metrics available", "data": None}), 200

@app.route("/api/assessment", methods=["GET"])
def api_assessment():
    if state["assessment"]:
        return jsonify({"status": "success", "data": state["assessment"]})
    else:
        # Return 200/info instead of 404 to simplify client handling
        return jsonify({"status": "info", "message": "No assessment available", "data": None}), 200


@app.route("/api/health", methods=["GET"])
def api_health():
    return jsonify({"status": "ok", "message": "server running"}), 200

if __name__ == "__main__":
    # Run on 0.0.0.0 so Render/local Docker can access it; debug off for production
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
