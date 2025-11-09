# 🏆 I.Mobilothon-5.0: AI Driver Wellness Monitoring System (DMS)

**Team Name:** Team Auto-gorithm

**A high-fidelity, real-time driver-monitoring demo project addressing the problem of fatigue and stress. It features two Flask backends for simulating/detecting driver impairment and a modern React + Vite frontend for visualization and AI-enhanced intervention.**

## 🚀 Live Demo & Presentation

**Experience the core UI and detection logic instantly—no local setup required:**

👉 **[Launch Demo: I.Mobilothon-5.0 DMS](https://i-mobilothon-5-0-git-main-yashini-s-projects.vercel.app?_vercel_share=wWUTP27sgsAEm4oO08Nz4iBfANs4YfKb)**

***

## ✨ Features & Solution Highlights

This solution directly addresses the **AI-Enhanced Driver Wellness Monitoring** problem statement by focusing on predictive, privacy-preserving monitoring and subtle, non-distracting interventions.

* **Multi-Modal Monitoring:** The system is designed to integrate data from multiple sources (simulated in this demo): Behavioral (facial cues like PERCLOS, Gaze), Physiological (HRV, ECG, GSR), and Vehicular (steering entropy and micro-corrections).
* **Predictive AI:** Aims to monitor the driver's physiological state to **predict impairment 5–10 minutes before visible signs**.
* **Driver Wellness Index (DWI):** Calculates a unified score using a multi-layer fusion model (**CNN + LSTM/BiVIT Transformer**) to classify fatigue or stress.
* **AI-Enhanced Interventions:** Uses the Gemini proxy (simulating an **OpenAI/HMI** integration) to generate **safe, non-distracting** text-based interventions (e.g., subtle voice prompts, ambient lights, breathing exercises, scheduled breaks).
* **Dual-Backend Flexibility:** Offers a **Lightweight Backend (`ai-assistant-backend`)** for rapid UI development and an **Full ML Backend (`Backend/`)** for realistic, frame-based Driver Monitoring using **MediaPipe** and **OpenCV**.

***

## 🖼️ Demo Screenshots

The core functionality includes real-time facial landmark detection, a visual road simulator, and multi-stage alerts ranging from low-risk monitoring to critical intervention.

* **Yawning Detection Alert:** Low-risk intervention suggesting a break, triggered by mouth landmarks.
    `![](./images/yawning_detection.png)`
* **Drowsiness Alert:** Critical intervention ("PULL OVER IMMEDIATELY") triggered by sustained eye closure (PERCLOS).
    `![](./images/drowsiness_alert.png)`
* **Metrics & Risk Display:** Overview showing live FPS, DWI metrics, and continuous risk monitoring.
    `![](./images/critical_risk.png)`

*(Note: Please ensure these images are uploaded to the specified path and the placeholder links are updated.)*

***

## ⚡ Quickstart: Run the Demo Locally

Below are the minimal steps to run the **Frontend** and the **Lightweight Backend** concurrently for the fastest local development and testing.

### Prerequisites

* **Node.js** (for frontend)
* **Python 3.8+** (for backend)

### 1. Launch the Frontend

Open a terminal in the `Frontend/` folder:

```bash
cd Frontend
npm install
npm run dev
````

### 2\. Launch the Lightweight AI Backend

Open a **second terminal** in the `ai-assistant-backend/` folder:

```bash
cd ai-assistant-backend
# Set up environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # (Use 'source .venv/bin/activate' on Linux/macOS)
pip install -r requirements.txt

# OPTIONAL: Enable Gemini Alert Generation
$env:GEMINI_API_KEY = 'YOUR_GEMINI_API_KEY'

# Run the service (default port: 5000)
python app.py
```

-----

## 🛠 Project Structure & Advanced Setup

  * `Frontend/` — React + Vite UI. Default dev server: 5173.
  * `ai-assistant-backend/` — Lightweight Flask simulator and Gemini proxy. Default port: 5000.
  * `Backend/` — Full frame-based DMS using MediaPipe and OpenCV. Default port: 5001.

### Full ML Backend Setup (Advanced)

Example using Conda:

```bash
conda create -n dms python=3.11 -y
conda activate dms
cd Backend
pip install -r requirements.txt
python app.py # Default port: 5001
```

## 👥 Contributors

A big thank you to **Team Auto-gorithm**!

  * **Smital Alai** ((https://github.com/Smitalalai)) 
  * **Pranav Kumbhojkar** ((https://github.com/PRANAVK3004))
  * **Vaishnavi Thorat** ((https://www.google.com/search?q=https://github.com/contributor-3-github))
  * **Yashini Pardeshi** ((https://github.com/Yashini13))

<!-- end list -->

```
```

