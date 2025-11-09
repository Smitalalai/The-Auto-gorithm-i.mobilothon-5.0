# 🏆 I.Mobilothon-5.0: AI Driver Wellness Monitoring System (DMS)

[cite_start]**A high-fidelity, real-time driver-monitoring demo project addressing the problem of fatigue and stress[cite: 8]. [cite_start]It features two Flask backends for simulating/detecting driver impairment and a modern React + Vite frontend for visualization and AI-enhanced intervention[cite: 9].**

## 🚀 Live Demo & Presentation

**Experience the core UI and detection logic instantly—no local setup required:**

👉 **[Launch Demo: I.Mobilothon-5.0 DMS](https://i-mobilothon-5-0-git-main-yashini-s-projects.vercel.app?_vercel_share=wWUTP27sgsAEm4oO08Nz4iBfANs4YfKb)**

***

## ✨ Features & Solution Highlights

[cite_start]This solution directly addresses the **AI-Enhanced Driver Wellness Monitoring** problem statement [cite: 8] [cite_start]by focusing on predictive, privacy-preserving monitoring and subtle, non-distracting interventions[cite: 9].

* [cite_start]**Multi-Modal Monitoring:** The system is designed to integrate data from multiple sources (simulated in this demo): Behavioral (facial cues), Physiological (HRV/ECG), and Vehicular (steering)[cite: 9].
* **Predictive AI:** Aims to monitor the driver's physiological state to predict impairment **5–10 minutes before visible signs**[cite: 71].
* [cite_start]**Driver Wellness Index (DWI):** Calculates a unified score using a multi-layer fusion model (CNN + LSTM/BiVIT Transformer) to classify fatigue or stress[cite: 26, 71, 123].
* [cite_start]**AI-Enhanced Interventions:** Uses the Gemini proxy (simulating an OpenAI integration) to generate **safe, non-distracting** text-based interventions (e.g., subtle voice prompts, ambient lights, breathing exercises)[cite: 32, 184, 218, 220].
* **Dual-Backend Flexibility:** Offers a **Lightweight Backend (`ai-assistant-backend`)** for rapid UI development and an **Full ML Backend (`Backend/`)** for realistic, frame-based Driver Monitoring using **MediaPipe** and **OpenCV**.

***

## 🖼️ Demo Screenshots

The core functionality includes real-time facial landmark detection, a visual road simulator, and multi-stage alerts ranging from low-risk monitoring to critical intervention.

| Alert Type | Description | Image |
| :--- | :--- | :--- |
| **Yawning Detection** | Low-risk intervention suggesting a break, triggered by mouth landmarks. | `![](./images/yawning_detection.png)` |
| **Drowsiness Alert** | Critical intervention ("PULL OVER IMMEDIATELY") triggered by sustained eye closure (PERCLOS). | `![](./images/drowsiness_alert.png)` |
| **Metrics & Risk** | Overview showing live FPS, DWI metrics, and continuous risk monitoring. | `![](./images/critical_risk.png)` |

*(Note: You must save and upload the screenshots to your repository (e.g., in an `images/` folder) and replace the placeholder paths with the final image links.)*

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
