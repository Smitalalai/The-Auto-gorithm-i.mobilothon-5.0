AI Assistant 

This is a Flask backend that simulates the AI assistant endpoints expected by the frontend.
It provides the following endpoints:

- POST /api/start  -> starts background simulation of metrics
- POST /api/stop   -> stops simulation
- POST /api/reset  -> resets counters
- POST /api/process_frame -> accepts JSON { frame: "<base64 jpeg>" } and saves a preview
- GET  /api/metrics -> returns latest simulated metrics
- GET  /api/assessment -> returns latest simulated risk assessment

Run locally

1. Create a Python virtual environment (recommended):

```powershell
# AURA — AI Assistant (Flask demo backend)

Lightweight Flask backend to simulate the AURA AI assistant for local development and integration tests.

It exposes the same endpoints the frontend expects so you can develop the React UI without needing full ML inference running locally.

---

## Why use this

- Rapid front-end iteration: simulate metrics, risk assessment and voice-generation behavior.
- Safe generative integration: proxy Gemini calls server-side so API keys stay secret.

---

## Key features

- Simulated real-time metrics and risk assessment (background thread).
- Endpoints to accept frames, start/stop monitoring and request server-generated alerts.
- Server-side Gemini proxy endpoint (`/api/generate_alert`) — reads `GEMINI_API_KEY` from environment.
- Health endpoint for simple probes.

---

## Endpoints

| Method | Path | Description |
|---|---|---|
| POST | /api/start | Start simulated monitoring (background metrics generator) |
| POST | /api/stop | Stop monitoring and clear metrics |
| POST | /api/reset | Reset counters / clear metrics |
| POST | /api/process_frame | Accepts JSON { frame: "<base64 jpeg>" } (saves preview to tmp/) |
| GET  | /api/metrics | Return latest metrics (200 + status: info when no data yet) |
| GET  | /api/assessment | Return latest assessment (200 + status: info when no data yet) |
| POST | /api/generate_alert | Proxy to Gemini to generate a short alert. Body: { metrics, assessment, alert_type? } |
| GET  | /api/health | Quick health check (200 OK) |

---

## Quickstart — run locally (PowerShell)

1) Create and activate a virtual environment

```powershell
python -m venv .venv
. \.venv\Scripts\Activate.ps1
```

2) Install dependencies

```powershell
pip install -r requirements.txt
```

3) (optional) Set your Gemini API key for server-side generation

> Do NOT commit the key. Use an environment variable.

```powershell
$env:GEMINI_API_KEY = 'YOUR_GEMINI_API_KEY_HERE'
```

4) Start the server

```powershell
# defaults to port 5000
python app.py
```

The server will bind to `0.0.0.0:5000` by default (or the port in the `PORT` env var).

---