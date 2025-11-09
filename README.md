# I.Mobilothon-5.0

A driver-monitoring demo project with two Flask backends and a React + Vite frontend.

This repository contains:

- `Frontend/` — React + Vite UI (Tailwind, Mediapipe web camera utils). Default dev server: 5173.
- `ai-assistant-backend/` — Lightweight Flask simulator (AURA). Fast to run locally for frontend development. Default port: 5000.
- `Backend/` — Full frame-based Driver Monitoring System using MediaPipe and OpenCV. Heavier, intended for ML inference. Default port: 5001.

## Quick overview

- The frontend captures webcam frames and posts base64 frames to a backend endpoint (`/api/process_frame`).
- `ai-assistant-backend` simulates metrics and risk assessment with a background generator and offers a Gemini proxy endpoint (`/api/generate_alert`) (requires `GEMINI_API_KEY`).
- `Backend` provides a frame-based detector implementation using MediaPipe/OpenCV and produces more realistic metrics but requires heavier dependencies.

## Quickstart (Windows / PowerShell)

Below are minimal steps to run the frontend and the lightweight backend for development.

### 1) Frontend (dev)

Open a terminal in the `Frontend/` folder and run:

```powershell
cd Frontend
npm install
npm run dev
```

Vite will start (dev server usually on http://localhost:5173) and hot-reload as you edit.

### 2) Lightweight backend (ai-assistant-backend)

This backend is the fastest way to iterate the UI without installing heavy ML dependencies.

```powershell
cd ai-assistant-backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# Optional: set Gemini API key (server-side generation)
$env:GEMINI_API_KEY = 'YOUR_GEMINI_API_KEY'
python app.py
```

By default the service listens on port `5000`. You can override the port with the `PORT` environment variable.

Endpoints: `/api/start`, `/api/stop`, `/api/reset`, `/api/process_frame`, `/api/metrics`, `/api/assessment`, `/api/generate_alert`, `/api/health`.

### 3) Full ML backend (Backend)

This service depends on `mediapipe`, `opencv-python-headless`, `scipy`, and other native packages. Installing these on Windows can be error-prone — using Conda, WSL, or Docker is recommended.

Example using Conda (recommended on Windows):

```powershell
conda create -n dms python=3.11 -y
conda activate dms
cd Backend
pip install -r requirements.txt
python app.py
```

If `pip install` fails for `mediapipe` or `opencv`, consider:
- Using `conda` packages or `conda-forge` where available
- Running the service inside WSL (Ubuntu) where prebuilt wheels are easier to install
- Building/running a Linux Docker container (I can add a Dockerfile if you want)

This service prints usage info and binds to port `5001` by default.

## Health check (quick)

From PowerShell you can probe a running backend like:

```powershell
# ai-assistant-backend (port 5000)
Invoke-RestMethod -Uri http://localhost:5000/api/health

# Backend (port 5001)
Invoke-RestMethod -Uri http://localhost:5001/api/health
```

## Environment variables

- `GEMINI_API_KEY` — optional, required if you want `ai-assistant-backend` to proxy to Google Generative Language (Gemini) for alert generation.
- `PORT` — override the Flask app port (used by `ai-assistant-backend`) if needed.

## Notes & recommendations

- Use `ai-assistant-backend` for fast front-end development. Switch to `Backend` when you want real frame-based metrics.
- Add an `.env.example` to document required environment variables and any default ports.
- If you want, I can add Dockerfiles (for `Backend` especially) and a small pytest smoke-test for each service (`/api/health`).

## Contributing

Feel free to open issues or PRs. If you want me to add tests, Dockerfiles, or a consolidated dev script that launches frontend + lightweight backend concurrently, tell me which you'd prefer and I can implement it.

---
Generated README — tailored to this repository's structure and local development workflow.
