# Adaptive Traffic Signal Analytics — Multi-Camera CV Module
**INFINIX'26 Hackathon** | **Member 3 — CSBS: Computer Vision**

This module provides real-time vehicle detection, tracking, line-crossing counting, and signal controller interfacing for an intelligent adaptive traffic signal system. It supports **four independent traffic camera feeds** (one for each junction approach: **North, South, East, and West**).

The module utilizes **Ultralytics YOLOv8** and **ByteTrack** with isolated tracking ID namespaces per camera, eliminates duplicate counts, provides vehicle class breakdowns (**cars, motorcycles, buses, trucks**), features a modern smart-city Streamlit dashboard, and feeds live counts to the adaptive signal controller.

---

## 📋 Table of Contents
1. [Architecture & Project Structure](#architecture--project-structure)
2. [Prerequisites & Installation](#prerequisites--installation)
3. [Quick Start & Launching the Dashboard](#quick-start--launching-the-dashboard)
4. [How to Upload & Process All Four Videos](#how-to-upload--process-all-four-videos)
5. [Vehicle Counting Accuracy & De-Duplication](#vehicle-counting-accuracy--de-duplication)
6. [Counting Line Configuration Guide](#counting-line-configuration-guide)
7. [Known Limitations & Real-World Challenges](#known-limitations--real-world-challenges)
8. [Connecting to the Main Traffic Signal Controller](#connecting-to-the-main-traffic-signal-controller)
9. [Fallback Simulation Mode](#fallback-simulation-mode)
10. [Running Automated Tests](#running-automated-tests)
11. [Files to Share with Your Controller Teammate](#files-to-share-with-your-controller-teammate)
12. [Testing Status Report](#testing-status-report)

---

## 🏛️ Architecture & Project Structure

```
computer_vision/
│
├── app.py                   # Redesigned Streamlit dashboard (dark smart-city theme, cached YOLO, 4 video panels, analytics)
├── detector.py              # YOLOv8 object detector & ByteTrack tracker with bottom-center reference point & tracker reset
├── vehicle_counter.py       # Core counting engine with hysteresis line-crossing, isolated ID spaces & category breakdown
├── simulator.py             # Standalone fallback simulator (runs independently of YOLO/OpenCV)
├── controller_interface.py  # JSON export, schema validation, and lightweight REST API for the signal controller
├── sample_generator.py      # Generates synthetic traffic footage for instant offline testing
├── test_module.py           # Comprehensive unit and integration test suite (11 tests covering all 8 accuracy requirements)
├── requirements.txt         # Project dependencies
├── README.md                # This comprehensive documentation guide
└── output/                  # Generated videos and live vehicle_counts.json
    ├── sample_traffic.mp4   # Demo synthetic intersection clip
    ├── traffic.mp4          # Real-world traffic footage
    └── vehicle_counts.json  # Real-time counts consumed by signal controller
```

---

## ⚙️ Prerequisites & Installation

### 1. Requirements
- Python 3.9+ (Tested on Python 3.10 – 3.14 on Windows)
- OpenCV, PyTorch, Ultralytics YOLOv8, Streamlit

### 2. Install Dependencies
Open PowerShell in the project directory:

```powershell
pip install -r computer_vision/requirements.txt
```
*(Or with Python launcher: `py -3 -m pip install -r computer_vision/requirements.txt`)*

---

## 🚀 Quick Start & Launching the Dashboard

To launch the web dashboard:

```powershell
streamlit run computer_vision/app.py
```
*(Or `py -3 -m streamlit run computer_vision/app.py`)*

The dashboard will open automatically in your browser at `http://localhost:8501`.

---

## 📹 How to Upload & Process All Four Videos

1. **Independent Upload Quadrants**:
   - The dashboard displays 4 designated approach quadrants:
     - ⬇️ **North Approach**
     - ⬆️ **South Approach**
     - ⬅️ **East Approach**
     - ➡️ **West Approach**
   - Drag and drop any `.mp4`, `.avi`, or `.mov` video into each respective quadrant.
   - For instant evaluation without external files, click **"Load Demo Traffic Clip"** (`traffic.mp4`) or **"Load Synthetic Clip"** (`sample_traffic.mp4`) on any quadrant.
   - Uploading a video for one approach is completely independent and will not overwrite another.

2. **Batch & Individual Processing**:
   - **Process All Feeds**: Click the primary blue **"🚀 Process All Videos"** button at the top to batch-process all uploaded videos in sequence.
   - **Process Single Feed**: Click **"▶️ Process [Approach] Only"** on any specific card to process only that direction.
   - **Stop / Pause**: Click **"⏹️ Stop Processing"** to halt processing at any frame.

3. **View Previews & Breakdown**:
   - Watch the live video preview with bounding boxes, tracking IDs, bottom-center reference points, and counting lines.
   - Each card displays:
     - **Unique Vehicles Count**
     - **Currently in View** (instantaneous count in current frame)
     - 🚗 **Cars** | 🏍️ **Motorcycles** | 🚌 **Buses** | 🚚 **Trucks**
     - Processing status badge (**READY**, **PROCESSING**, **COMPLETED**, **ERROR**).

---

## 🎯 Vehicle Counting Accuracy & De-Duplication

### 1. Bottom-Center Reference Point
Standard bounding-box centroids `(cx, cy)` sit in the air above the road surface, which introduces perspective distortion. This module uses the **bottom-centre reference point** `(cx, y2)`—the exact contact patch of the tires on the roadway—for reliable line-crossing detection.

### 2. Isolated Tracking ID Namespaces
Ultralytics ByteTrack state is reset before processing each video feed (`detector.reset_tracker()`). This guarantees that Vehicle `#1` in North will **never** collide with or falsely de-duplicate Vehicle `#1` in South.

### 3. Hysteresis Line Crossing & Stationary Jitter Protection
- A vehicle must cross from one side of the line to the other with a minimum displacement vector.
- Micro-jitter from stationary vehicles idling at a red light will **never** trigger multiple counts.
- Once a vehicle's tracking ID is counted, it is added to `self.counted_ids` and is never re-counted.

### 4. Mathematical Identity: Total == Sum of Classes
Every counted tracking ID is mapped to its vehicle class in `self.id_to_class`. Therefore:
$$\text{Total Unique Vehicles} = \text{Cars} + \text{Motorcycles} + \text{Buses} + \text{Trucks}$$
This identity is mathematically guaranteed with zero discrepancy.

---

## 📊 Two Separated Metrics: Queue vs Throughput

The dashboard strictly decouples two distinct traffic metrics to avoid confusion:
1. **Unique Vehicles Counted (Crossed Line)**: Cumulative count of unique tracked vehicles that physically crossed the configured virtual counting line. This represents intersection throughput.
2. **Vehicles Currently Detected in Frame**: Instantaneous count of valid detected vehicles currently visible on screen. This represents real-time queue length / traffic density.

Both metrics are overlaid directly on the video HUD and prominently displayed on the dashboard summary cards and approach panels.

---

## 📐 Counting Line Calibration & Live Diagnostics

### Why Counting Line Position Matters
Cameras pointing at roadways compress perspective. In highway CCTV footage (`traffic.mp4`), vehicles drive across the lower portion of the frame ($y \in [300, 540]$). If a counting line is placed at $y=0.50$ ($y=270$), vehicles never touch or cross it, causing low counts.

### Interactive Calibration Editor
Directly underneath each camera feed, open **"📐 Calibrate [Approach] Counting Line"**:
1. **Quick Presets**:
   - `Highway Roadway (70% - Recommended)`: Optimal for roadside CCTV & highway clips.
   - `Mid-Screen Horizon (50%)`: For elevated overhead views or synthetic 4-way intersections.
   - `Upper Roadway (35%)`: For downward top-down pole cameras.
   - `Lower Roadway (80%)`: For steep-angle or low-mounted cameras.
   - `Custom`: Interactive slider ($0.10 \to 0.90$) with live pixel height calculation.
2. **Allowed Direction Selector**:
   - `any`: Bidirectional counting (counts vehicles moving in either direction).
   - `down`: Strictly counts vehicles moving downwards ($dy > 0$, inbound towards camera).
   - `up`: Strictly counts vehicles moving upwards ($dy < 0$, outbound).
   - `left` / `right`: For horizontal flow cross-streets.

### 🔍 Real-Time Live Diagnostics Inspector
Open **"🔍 [Approach] Live Tracking & Line-Crossing Diagnostics"** to view an interactive table of all active vehicles:
- **Track ID**: ByteTrack unique persistent ID.
- **Class**: Vehicle classification (`CAR`, `MOTORCYCLE`, `BUS`, `TRUCK`).
- **Current Ref Pt**: Bottom-center contact patch `(x, y)` in pixels.
- **Previous Ref Pt**: Position in the previous frame.
- **Distance to Line**: Perpendicular distance in pixels.
- **Line Crossed**: `✅ YES` or `⏳ NO`.
- **Decision / Reason**: Real-time status (`CROSSED_VALID`, `ALREADY_COUNTED`, `APPROACHING`, `STATIONARY_JITTER`, `DIRECTION_REJECTED`, `INITIAL_FRAME`).

---

## ⚠️ Known Limitations & Real-World Challenges

1. **Occlusion in Heavy Traffic**:
   - In congested bumper-to-bumper traffic, a large bus or truck may partially occlude smaller cars or motorcycles behind it.
2. **Extreme Camera Angles & Glare**:
   - Cameras mounted low to the ground suffer more perspective distortion than overhead traffic poles. Night footage with headlight glare may require lowering the confidence threshold to `0.20` – `0.25`.
3. **Tracking ID Switches on Severe Occlusion**:
   - If a vehicle is completely obscured for more than 30 frames, ByteTrack may assign a new tracking ID upon re-emergence. The hysteresis buffer minimizes this effect, but extreme occlusions remain an open challenge in computer vision.
4. **Real Video Verification**:
   - Verified on real highway traffic footage (`traffic.mp4`) and synthetic intersection clips (`sample_traffic.mp4`). For specific municipal CCTV feeds, counting lines should be calibrated to match the roadway geometry.

---

## 🔌 Connecting to the Main Traffic Signal Controller

Your teammate building the traffic signal timing algorithm (Member 1 / Member 2) can connect using any of these **3 options**:

### Option 1: Direct Python Import (Recommended)
```python
from computer_vision.vehicle_counter import get_vehicle_counts

# Inside your controller loop:
counts = get_vehicle_counts()
print(counts)
# Output:
# {
#     "North": 12,
#     "South": 5,
#     "East": 8,
#     "West": 2
# }

# Example adaptive green-time allocation:
green_time_north_south = max(10, (counts["North"] + counts["South"]) * 3)
green_time_east_west   = max(10, (counts["East"] + counts["West"]) * 3)
```

### Option 2: Shared JSON File (`output/vehicle_counts.json`)
The file updates continuously as videos are processed. Root-level keys make access trivial:

```python
import json, time

while True:
    with open("computer_vision/output/vehicle_counts.json", "r") as f:
        data = json.load(f)

    # Simple root-level access:
    north = data["North"]
    south = data["South"]
    east  = data["East"]
    west  = data["West"]

    # Access per-direction status & vehicle breakdowns:
    north_status = data["approaches"]["North"]["status"]
    north_cars   = data["approaches"]["North"]["cars"]
    north_buses  = data["approaches"]["North"]["buses"]

    print(f"Counts: N={north}, S={south}, E={east}, W={west} | Status: {data['status']}")
    time.sleep(1)
```

### Option 3: HTTP REST API Server
Run the API server in a separate terminal:
```powershell
py -3 computer_vision/controller_interface.py --serve --port 5000
```
Then fetch via HTTP GET `http://localhost:5000/api/counts`.

---

## 🧪 Fallback Simulation Mode

If YOLO weights are unavailable, video feeds fail, or your teammate needs to test edge-case traffic spikes:
1. In the sidebar, select **"🧪 Simulation Fallback (Manual / Mock)"**.
2. Notice the prominent alert banner: **`SIMULATED DATA FALLBACK MODE ACTIVE`**.
3. Adjust vehicle counts for North, South, East, and West via number inputs.
4. Or click **"🎲 Randomize Traffic Spikes"** to simulate queue buildup and discharge.
5. All exported JSON payloads are tagged with `"status": "SIMULATED DATA"` and `"is_simulated": true` so simulated data is never mistaken for live video detections.

---

## 🧪 Running Automated Tests

Run the comprehensive unit test suite on your laptop:

```powershell
py -3 computer_vision/test_module.py
```

### All 11 Tests Passing:
1. `test_1_vehicle_detected_across_many_frames_counts_only_once`: PASS
2. `test_2_tracked_vehicle_crossing_the_line_counts_once`: PASS
3. `test_3_stationary_vehicle_must_not_repeatedly_increment_count`: PASS
4. `test_4_reverse_direction_follows_policy`: PASS
5. `test_5_two_different_tracking_ids_count_as_two_vehicles`: PASS
6. `test_6_counts_from_different_directions_must_not_mix`: PASS
7. `test_7_restarting_video_resets_tracking_and_counting_state`: PASS
8. `test_8_failed_video_processing_shows_error_instead_of_zero_count`: PASS
9. `test_total_equals_sum_of_classes`: PASS
10. `test_simulated_payload_labeling`: PASS
11. `test_simulation_format_and_overrides`: PASS

---

## 📦 Files to Share with Your Controller Teammate

Share these 4 files with Member 1 / Member 2:
1. [`vehicle_counter.py`](file:///c:/Users/param/OneDrive/Desktop/ngp%20hack/computer_vision/vehicle_counter.py)
2. [`controller_interface.py`](file:///c:/Users/param/OneDrive/Desktop/ngp%20hack/computer_vision/controller_interface.py)
3. [`output/vehicle_counts.json`](file:///c:/Users/param/OneDrive/Desktop/ngp%20hack/computer_vision/output/vehicle_counts.json)
4. [`README.md`](file:///c:/Users/param/OneDrive/Desktop/ngp%20hack/computer_vision/README.md)

*(Your teammate does not need PyTorch or OpenCV installed if they read `vehicle_counts.json` or call the REST API!)*

---

## 🔍 Testing Status Report

| Feature | Status | Notes |
| :--- | :---: | :--- |
| **YOLOv8 Object Detection** | ✅ Tested | Detects cars, motorcycles, buses, and trucks; ignores people and background. |
| **ByteTrack Tracking** | ✅ Tested | Persistent IDs with tracker reset per video stream. |
| **De-duplication per Approach** | ✅ Tested | Stationary jitter and multi-frame detections count strictly once. |
| **Bottom-Center Reference Point** | ✅ Tested | Tires-to-road contact patch used for crossing math. |
| **Class Breakdown Consistency** | ✅ Tested | Total unique count mathematically equals sum of class breakdowns. |
| **Four Independent Feeds** | ✅ Tested | North, South, East, West processed independently without ID collisions. |
| **Fault Isolation** | ✅ Tested | Corrupted/missing video displays error status while others continue. |
| **Streamlit Smart-City UI** | ✅ Tested | Professional dark navy theme, summary cards, video panels, analytics charts. |
| **Controller JSON Schema** | ✅ Tested | Root-level direction keys preserved with enriched approach metadata. |
| **Simulation Fallback Mode** | ✅ Tested | Fully editable counts watermarked as `SIMULATED DATA`. |
| **Real Traffic Footage** | ✅ Tested | Verified on real traffic footage (`traffic.mp4`), detecting up to 15 vehicles simultaneously and tracking line crossings. |
