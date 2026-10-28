# SmartFlow AI - Adaptive Traffic Signal Controller (Hybrid)

> **Theme 5**: Smart Cities and Connected Communities  
> **Challenge**: SC-01 Adaptive Traffic Signal Controller (Hybrid) &bull; SDG 11: Sustainable Cities  
> **Replicated From**: [Figma Design File](https://www.figma.com/design/ckZmea5E34CM9ICl6viuH3/Untitled?node-id=0-1)

---

## 🚦 System Architecture & Figma Conversion

This repository contains the complete frontend and backend implementation matching the Figma designs down to every component, interaction, layout, and control action:

1. **Screen 1: Dashboard (`#screen-dashboard`)**
   - **4-Way Junction Visualizer**: Dynamic animated intersection with lane flow, crosswalks, traffic lights, and active countdown timer.
   - **Live Camera Stream (YOLO AI)**: Camera feed overlay with vehicle detection tags (Cars, Buses, Bikes, Trucks) and real-time congestion density.
   - **Signal Phase & Timing**: Current active green progress bar, queue clearance percentage, and next phase countdown.
   - **Approach Queue Matrix**: Real-time vehicle queue counts, assigned signal status, and allocated green duration.
   - **Waiting Time Comparison**: Adaptive Signal Plan vs. Fixed-Time Baseline Plan (-47.2% delay reduction) with sparkline trends.
   - **Live Incident & Event Log**: Real-time audit feed of system decisions and phase adjustments.

2. **Screen 2: Live Junction Simulator (`#screen-live-junction`)**
   - Interactive 2D multi-agent vehicle flow simulator with collision avoidance and lane queue buildup.
   - Manual phase override controls: `[Force N-S]`, `[Force E-W]`, `[Pedestrian Crosswalk]`.
   - Emergency Vehicle Preemption: `[🚨 Spawn Ambulance]` triggers flashing sirens and grants an immediate green corridor.
   - Live signal timing constraint sliders (Min Green, Max Green, Yellow clearance).
   - Cumulative metrics: Total vehicles cleared (3,422), average flow rate (42 veh/min), fuel saved (42.5 L).

3. **Screen 3: Analytics & Reports (`#screen-analytics`)**
   - 4 Top KPI Cards: Average Waiting Time, Total Throughput, Average Queue Length, Fuel & Emissions Saved.
   - Dual-line 24-hour comparative chart (Chart.js): Adaptive Signal Plan (Blue) vs Fixed-Time Baseline (Dashed Gray).
   - Tabs to switch metrics: Waiting Time (sec), Queue Length (vehicles), Throughput (veh/hr).
   - Peak hours breakdown: Morning Rush (08:00 - 10:00), Midday (12:00 - 14:00), Evening Peak (17:00 - 19:30).
   - Approach-wise delay evaluation table with CSV export.

4. **Screen 4: System Config (`#screen-system-config`)**
   - System Connected status badge.
   - 4 Technical Architecture cards:
     1. YOLO Computer Vision (YOLOv8n, Confidence threshold, FPS)
     2. Optimization Engine (Deep Q-Network + Webster Equilibrium Equation)
     3. Signal Controller Hardware Interface (NEMA TS2 / GPIO Relay / COM3)
     4. Micro-Simulator (Stochastic Poisson Arrival / Virtual loop sensors)
   - Parameter input form with live backend sync: Min Green, Max Green, Yellow duration, All-Red clearance, YOLO threshold, RL queue weight.
   - Self-test diagnostics and simulated traffic spike testing.

---

## 🛠️ Backend Architecture & REST APIs

Built with **Python**, **Starlette**, **Uvicorn**, and **SQLite**:

- `GET /api/status`: Current junction state, active phase, countdown, approach metrics, and recent logs.
- `GET /api/analytics`: 24-hour historical comparative data and summary KPI metrics.
- `GET /api/config`: Current timing constraints and model parameters.
- `POST /api/config`: Save and apply updated signal timing constraints.
- `POST /api/junction/mode`: Switch between Autonomous Hybrid, Fixed-Time, and Manual.
- `POST /api/junction/phase-override`: Manually force a specific phase.
- `POST /api/junction/emergency`: Activate emergency preemption green corridor.
- `GET /api/logs`: Audit trail of all traffic decisions and incidents.

---

## 🚀 How to Run

### 1. Install Dependencies
Ensure dependencies are installed into your Python environment:

```powershell
python -m pip install -r requirements.txt
```

> **Note on Multiple Python Environments**: If you have multiple Python installations on Windows, ensure you run `python -m pip install -r requirements.txt` with the exact same interpreter that executes `python run.py`.

### 2. YOLO Model Weights
The default lightweight weights (`yolov8n.pt`) are included in the project root. If you want to use custom weights, specify their path:
```powershell
$env:YOLO_MODEL_PATH = "C:\path\to\custom_weights.pt"
```

### 3. Start the Server
From the project directory:

```powershell
python run.py
```

The application will start on `http://127.0.0.1:8000` and automatically open in your default browser.
