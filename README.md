# 🏆 I.Mobilothon-5.0: AI Driver Wellness Monitoring System (DMS)

**A high-fidelity, real-time driver-monitoring demo project addressing the problem of fatigue and stress. It features two Flask backends for simulating/detecting driver impairment and a modern React + Vite frontend for visualization and AI-enhanced intervention.**

## 🚀 Live Demo & Presentation

**Experience the core UI and detection logic instantly—no local setup required:**

👉 **[Launch Demo: I.Mobilothon-5.0 DMS](https://i-mobilothon-5-0-git-main-yashini-s-projects.vercel.app?_vercel_share=wWUTP27sgsAEm4oO08Nz4iBfANs4YfKb)**

***

## ✨ Features & Solution Highlights

This solution directly addresses the **AI-Enhanced Driver Wellness Monitoring** problem statement [cite: 8, 9] by focusing on predictive, privacy-preserving monitoring and subtle, non-distracting interventions[cite: 39, 42].

* [cite_start]**Multi-Modal Monitoring:** The system is designed to integrate data from multiple sources (simulated in this demo): Behavioral (facial cues), Physiological (HRV/ECG), and Vehicular (steering)[cite: 39, 47].
* [cite_start]**Predictive AI:** Aims to monitor the driver's physiological state to **predict impairment 5–10 minutes before visible signs**[cite: 71].
* **Driver Wellness Index (DWI):** Calculates a unified score using a multi-layer fusion model (CNN + LSTM/BiVIT Transformer) to classify fatigue or stress[cite: 26, 204].
* [cite_start]**AI-Enhanced Interventions:** Uses the Gemini proxy (simulating an **OpenAI** integration) to generate **safe, non-distracting** text-based interventions (e.g., subtle voice prompts, ambient lights, breathing exercises)[cite: 39, 138, 184].
* [cite_start]**Dual-Backend Flexibility:** Offers a **Lightweight Backend (`ai-assistant-backend`)** for rapid UI development and an **Full ML Backend (`Backend/`)** for realistic, frame-based Driver Monitoring using **MediaPipe** and **OpenCV**[cite: 71, 231].

***

## 🖼️ Demo Screenshots

The core functionality includes real-time facial landmark detection, a visual road simulator, and multi-stage alerts ranging from low-risk monitoring to critical intervention[cite: 170, 174, 164].

| Alert Type | Description | Image |
| :--- | :--- | :--- |
| **Yawning Detection** | Low-risk intervention suggesting a break, triggered by mouth landmarks. | `![](./images/yawning_detection.png)` |
| **Drowsiness Alert** | Critical intervention ("PULL OVER IMMEDIATELY") triggered by sustained eye closure (PERCLOS). | `![](./images/drowsiness_alert.png)` |
| **Metrics & Risk** | Overview showing live FPS, DWI metrics, and continuous risk monitoring. | `![](./images/critical_risk.png)` |

*(Note: Please ensure the images are uploaded to the specified path and the placeholder links are correct.)*

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
