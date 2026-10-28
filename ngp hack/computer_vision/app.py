"""
Adaptive Traffic Signal Analytics — Computer Vision Module
INFINIX'26 Hackathon | Member 3 — CSBS: Computer Vision

Professional, smart-city traffic analytics dashboard supporting four independent
approach videos (North, South, East, West) with accurate YOLOv8 + ByteTrack
vehicle counting, de-duplication, category breakdowns, and controller integration.
"""

import os
import sys
import tempfile
import time
import json
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from typing import Optional, Tuple, List, Dict, Any

# Ensure project directory is in path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from detector import VehicleDetector, check_detector_health
from vehicle_counter import VehicleCounter, CountingLine
from simulator import TrafficSimulator
from controller_interface import (
    export_counts_to_json,
    read_counts_from_json,
    DEFAULT_JSON_PATH,
)
from sample_generator import generate_synthetic_intersection_video

# ==============================================================================
# 1. Page Configuration & Professional Smart-City Theme
# ==============================================================================
st.set_page_config(
    page_title="Adaptive Traffic Signal Analytics | INFINIX'26",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');

    /* Complete Removal of Default Streamlit White Header Strip & Toolbar */
    header[data-testid="stHeader"],
    [data-testid="stHeader"],
    .stApp > header {
        background: transparent !important;
        background-color: transparent !important;
        border-bottom: none !important;
        color: #f8fafc !important;
        z-index: 99 !important;
    }
    [data-testid="stToolbar"] {
        right: 1.5rem !important;
        top: 0.5rem !important;
        color: #f8fafc !important;
    }
    [data-testid="stToolbar"] button {
        color: #f8fafc !important;
    }
    [data-testid="stDecoration"] {
        display: none !important;
    }
    .main .block-container {
        padding-top: 1.8rem !important;
        padding-bottom: 3rem !important;
        max-width: 100% !important;
    }

    /* Global Ambient Glassmorphic Canvas */
    .stApp {
        background-color: #07090e;
        background-image: 
            radial-gradient(circle at 12% 10%, rgba(14, 165, 233, 0.14) 0%, transparent 40%),
            radial-gradient(circle at 88% 12%, rgba(139, 92, 246, 0.15) 0%, transparent 45%),
            radial-gradient(circle at 50% 60%, rgba(16, 185, 129, 0.08) 0%, transparent 50%),
            radial-gradient(circle at 82% 88%, rgba(245, 158, 11, 0.08) 0%, transparent 40%);
        background-attachment: fixed;
        color: #f8fafc;
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    /* Headings & Text Color Standards */
    h1, h2, h3, h4, h5, h6 {
        color: #f8fafc !important;
        font-weight: 700 !important;
        letter-spacing: -0.01em;
    }
    p, span, label {
        color: #f8fafc;
    }

    /* Glassmorphic Header Bar */
    .header-container-glass {
        background: rgba(13, 20, 36, 0.65);
        backdrop-filter: blur(20px) saturate(180%);
        -webkit-backdrop-filter: blur(20px) saturate(180%);
        border: 1px solid rgba(255, 255, 255, 0.09);
        border-top: 1px solid rgba(255, 255, 255, 0.22);
        border-radius: 16px;
        padding: 20px 28px;
        margin-bottom: 22px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 12px 35px -8px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.08);
    }
    .header-badge-tag {
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 1px;
        color: #38bdf8;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    .header-title-gradient {
        font-size: 1.85rem;
        font-weight: 800;
        background: linear-gradient(135deg, #ffffff 0%, #38bdf8 55%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        line-height: 1.2;
    }
    .header-subtitle-text {
        font-size: 0.88rem;
        color: #cbd5e1;
        font-weight: 500;
        margin-top: 4px;
    }
    .header-status-area {
        display: flex;
        gap: 12px;
        align-items: center;
    }
    .status-glass-pill {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 6px 14px;
        border-radius: 20px;
        border: 1px solid;
        backdrop-filter: blur(10px);
    }
    .status-pulse-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        display: inline-block;
        box-shadow: 0 0 8px currentColor;
    }
    .engine-badge-pill {
        display: flex;
        align-items: center;
        gap: 6px;
        padding: 6px 12px;
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.09);
        border-radius: 20px;
    }

    /* Metric Summary Cards */
    .summary-card {
        background: rgba(13, 20, 36, 0.72);
        backdrop-filter: blur(16px) saturate(180%);
        -webkit-backdrop-filter: blur(16px) saturate(180%);
        border: 1px solid rgba(255, 255, 255, 0.09);
        border-top: 1px solid rgba(255, 255, 255, 0.22);
        border-radius: 14px;
        padding: 16px 14px;
        text-align: center;
        box-shadow: 0 10px 28px -5px rgba(0, 0, 0, 0.45), inset 0 1px 0 rgba(255, 255, 255, 0.06);
        transition: transform 0.2s cubic-bezier(0.4, 0, 0.2, 1), box-shadow 0.2s ease;
        min-height: 146px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .summary-card:hover {
        transform: translateY(-2px);
    }
    .card-hero-unique {
        border-top: 2px solid #00e5ff !important;
        box-shadow: 0 10px 28px -5px rgba(0, 229, 255, 0.18), inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
    }
    .card-hero-inframe {
        border-top: 2px solid #f59e0b !important;
        box-shadow: 0 10px 28px -5px rgba(245, 158, 11, 0.18), inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
    }
    .card-north { border-top: 2px solid #10b981 !important; }
    .card-south { border-top: 2px solid #0ea5e9 !important; }
    .card-east  { border-top: 2px solid #f59e0b !important; }
    .card-west  { border-top: 2px solid #a855f7 !important; }

    .summary-header {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 6px;
        margin-bottom: 2px;
    }
    .summary-icon {
        font-size: 1.1rem;
    }
    .summary-label {
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        color: #f8fafc;
        margin: 0;
    }
    .summary-val {
        font-size: 2.15rem;
        font-weight: 800;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: -0.02em;
        margin: 2px 0 4px 0;
        line-height: 1.1;
    }
    .summary-footer {
        display: flex;
        flex-direction: column;
        gap: 3px;
        font-size: 0.78rem;
        font-weight: 600;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
        padding-top: 6px;
        color: #cbd5e1;
    }
    .summary-footer-hero {
        font-size: 0.78rem;
        font-weight: 700;
        color: #cbd5e1;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
        padding-top: 6px;
        text-align: center;
    }

    /* Approach Video Panels */
    .panel-container {
        background: rgba(13, 20, 36, 0.65);
        backdrop-filter: blur(16px) saturate(180%);
        -webkit-backdrop-filter: blur(16px) saturate(180%);
        border: 1px solid rgba(255, 255, 255, 0.09);
        border-top: 1px solid rgba(255, 255, 255, 0.20);
        border-radius: 16px;
        padding: 18px;
        margin-bottom: 22px;
        box-shadow: 0 12px 35px -8px rgba(0, 0, 0, 0.45), inset 0 1px 0 rgba(255, 255, 255, 0.06);
    }
    .panel-header-glass {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 6px 12px;
        border-radius: 10px;
        background: rgba(255, 255, 255, 0.03);
        margin-bottom: 12px;
    }

    /* Status Badges */
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.74rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    .badge-idle       { background: rgba(148, 163, 184, 0.12); color: #94a3b8; border: 1px solid rgba(148, 163, 184, 0.3); }
    .badge-ready      { background: rgba(245, 158, 11, 0.14); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.4); }
    .badge-processing { background: rgba(14, 165, 233, 0.16); color: #38bdf8; border: 1px solid rgba(14, 165, 233, 0.5); }
    .badge-completed  { background: rgba(16, 185, 129, 0.16); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.5); }
    .badge-error      { background: rgba(239, 68, 68, 0.16); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.5); }

    /* Metadata Badge */
    .meta-glass-badge {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 6px 12px;
        font-size: 0.80rem;
        display: flex;
        justify-content: space-between;
        margin: 6px 0 10px 0;
        color: #cbd5e1;
    }

    /* Dual Metric Cards (Unique vs In-Frame) */
    .glass-metrics-wrapper {
        background: rgba(13, 20, 36, 0.60);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 12px;
        margin-top: 10px;
    }
    .metric-badge-container {
        display: flex;
        gap: 10px;
        margin-bottom: 10px;
    }
    .metric-box-unique {
        flex: 1;
        background: rgba(16, 185, 129, 0.09);
        border: 1px solid rgba(16, 185, 129, 0.40);
        border-top: 1px solid rgba(16, 185, 129, 0.75);
        border-radius: 10px;
        padding: 10px 12px;
        text-align: center;
    }
    .metric-box-inframe {
        flex: 1;
        background: rgba(245, 158, 11, 0.09);
        border: 1px solid rgba(245, 158, 11, 0.40);
        border-top: 1px solid rgba(245, 158, 11, 0.75);
        border-radius: 10px;
        padding: 10px 12px;
        text-align: center;
    }
    .metric-title-unique {
        font-size: 0.78rem;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 0.5px;
        color: #34d399;
    }
    .metric-title-inframe {
        font-size: 0.78rem;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 0.5px;
        color: #fbbf24;
    }
    .metric-number-unique {
        font-size: 2.1rem;
        font-weight: 800;
        font-family: 'JetBrains Mono', monospace;
        color: #10b981;
        margin: 2px 0;
        line-height: 1.1;
    }
    .metric-number-inframe {
        font-size: 2.1rem;
        font-weight: 800;
        font-family: 'JetBrains Mono', monospace;
        color: #f59e0b;
        margin: 2px 0;
        line-height: 1.1;
    }
    .metric-caption-sub {
        font-size: 0.76rem;
        color: #cbd5e1;
    }

    /* Vehicle Class Breakdown Grid */
    .breakdown-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 8px;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
        padding-top: 10px;
        margin-top: 6px;
    }
    .breakdown-pill {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 8px;
        padding: 6px 4px;
        text-align: center;
    }
    .pill-icon {
        font-size: 0.90rem;
        display: block;
        margin-bottom: 2px;
    }
    .pill-val {
        font-size: 1.15rem;
        font-weight: 800;
        font-family: 'JetBrains Mono', monospace;
        display: block;
        line-height: 1.1;
    }
    .pill-lbl {
        font-size: 0.74rem;
        font-weight: 600;
        color: #cbd5e1;
        text-transform: uppercase;
        letter-spacing: 0.3px;
    }

    /* Alert Banners */
    .sim-banner {
        background: rgba(245, 158, 11, 0.12);
        border: 1px solid rgba(245, 158, 11, 0.40);
        border-left: 5px solid #f59e0b;
        color: #fef3c7;
        padding: 14px 18px;
        border-radius: 12px;
        margin-bottom: 22px;
        backdrop-filter: blur(12px);
    }

    /* Streamlit Native Widget Injections */
    div[data-testid="stButton"] button[kind="primary"] {
        background: linear-gradient(135deg, #0284c7 0%, #06b6d4 100%) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 10px !important;
        box-shadow: 0 4px 18px rgba(6, 182, 212, 0.35) !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    div[data-testid="stButton"] button[kind="primary"]:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 24px rgba(6, 182, 212, 0.50) !important;
    }
    div[data-testid="stButton"] button[kind="secondary"], div[data-testid="stButton"] button {
        background: rgba(255, 255, 255, 0.05) !important;
        color: #f8fafc !important;
        font-weight: 600 !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 10px !important;
        backdrop-filter: blur(8px) !important;
        transition: all 0.2s ease !important;
    }
    div[data-testid="stButton"] button:hover {
        background: rgba(255, 255, 255, 0.09) !important;
        border-color: rgba(255, 255, 255, 0.22) !important;
        transform: translateY(-1px) !important;
    }

    div[data-testid="stFileUploader"] {
        background: rgba(13, 20, 36, 0.40) !important;
        border: 1px dashed rgba(56, 189, 248, 0.35) !important;
        border-radius: 10px !important;
        padding: 6px !important;
    }
    div[data-testid="stFileUploader"] section {
        background: transparent !important;
    }
    div[data-testid="stFileUploader"] label {
        color: #cbd5e1 !important;
        font-weight: 600 !important;
        font-size: 0.84rem !important;
    }

    div[data-testid="stExpander"] {
        background: rgba(13, 20, 36, 0.60) !important;
        border: 1px solid rgba(255, 255, 255, 0.09) !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25) !important;
        margin-top: 8px !important;
    }
    div[data-testid="stExpander"] details summary {
        color: #f8fafc !important;
        font-weight: 600 !important;
    }
    div[data-testid="stExpander"] details summary:hover {
        color: #38bdf8 !important;
    }

    section[data-testid="stSidebar"] {
        background: rgba(8, 12, 23, 0.90) !important;
        backdrop-filter: blur(20px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    }
    section[data-testid="stSidebar"] h3 {
        color: #38bdf8 !important;
        font-weight: 800 !important;
    }

    div[data-testid="stSlider"] label {
        color: #cbd5e1 !important;
        font-weight: 600 !important;
    }
    div[data-testid="stSelectbox"] label {
        color: #cbd5e1 !important;
        font-weight: 600 !important;
    }

    div[data-testid="stDataFrame"] {
        border-radius: 10px !important;
        overflow: hidden !important;
        border: 1px solid rgba(255, 255, 255, 0.10) !important;
        background: rgba(13, 20, 36, 0.60) !important;
    }

    /* Glassmorphic Streamlit Border Containers (st.container(border=True)) */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(13, 20, 36, 0.65) !important;
        backdrop-filter: blur(16px) saturate(180%) !important;
        -webkit-backdrop-filter: blur(16px) saturate(180%) !important;
        border: 1px solid rgba(255, 255, 255, 0.09) !important;
        border-top: 1px solid rgba(255, 255, 255, 0.20) !important;
        border-radius: 16px !important;
        box-shadow: 0 12px 35px -8px rgba(0, 0, 0, 0.45), inset 0 1px 0 rgba(255, 255, 255, 0.06) !important;
        padding: 16px !important;
    }

    div[data-testid="stCodeBlock"] {
        border-radius: 10px !important;
        border: 1px solid rgba(255, 255, 255, 0.10) !important;
        background: rgba(8, 12, 23, 0.85) !important;
    }

    /* Dynamic Timing Card */
    .timing-card-glass {
        background: rgba(13, 20, 36, 0.50);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px;
    }
    .timing-phase-row {
        display: flex;
        gap: 16px;
        margin-bottom: 14px;
    }
    .timing-phase-box {
        flex: 1;
        border-radius: 12px;
        padding: 14px;
        text-align: center;
    }
    .phase-ns {
        background: rgba(16, 185, 129, 0.08);
        border: 1px solid rgba(16, 185, 129, 0.35);
        border-top: 2px solid #10b981;
    }
    .phase-ew {
        background: rgba(245, 158, 11, 0.08);
        border: 1px solid rgba(245, 158, 11, 0.35);
        border-top: 2px solid #f59e0b;
    }
    .timing-phase-label {
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #f8fafc;
        margin-bottom: 4px;
    }
    .timing-phase-seconds {
        font-size: 2.4rem;
        font-weight: 800;
        font-family: 'JetBrains Mono', monospace;
        line-height: 1.1;
        margin: 2px 0 6px 0;
    }
    .phase-ns .timing-phase-seconds {
        color: #34d399;
    }
    .phase-ew .timing-phase-seconds {
        color: #fbbf24;
    }
    .timing-phase-meta {
        font-size: 0.78rem;
        color: #cbd5e1;
    }
    .timing-sync-info {
        border-top: 1px solid rgba(255, 255, 255, 0.08);
        padding-top: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

APPROACH_NAMES = ["North", "South", "East", "West"]

# ==============================================================================
# 2. Cached Model Loading (Prevents Repeated Model Reloads & Freezes)
# ==============================================================================
@st.cache_resource(show_spinner="Initializing YOLOv8 Neural Network...")
def get_cached_detector(conf: float = 0.35) -> VehicleDetector:
    """Loads YOLOv8 model once into memory and reuses it across reruns."""
    return VehicleDetector(conf_threshold=conf)


def get_video_metadata(video_path: Optional[str]) -> Tuple[str, str, int]:
    """Extract filename, duration string, and frame count from video file."""
    if not video_path or not os.path.exists(video_path):
        return "No file selected", "0s", 0
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return os.path.basename(video_path), "Unknown", 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    duration_sec = frames / fps if fps > 0 else 0
    mins = int(duration_sec // 60)
    secs = int(duration_sec % 60)
    duration_str = f"{mins}m {secs}s" if mins > 0 else f"{secs}s"
    return os.path.basename(video_path), duration_str, frames


# ==============================================================================
# 3. Session State Initialization
# ==============================================================================
if "counter" not in st.session_state:
    st.session_state.counter = VehicleCounter()
if "simulator" not in st.session_state:
    st.session_state.simulator = TrafficSimulator()
if "video_sources" not in st.session_state:
    st.session_state.video_sources = {app: None for app in APPROACH_NAMES}
if "video_display_names" not in st.session_state:
    st.session_state.video_display_names = {app: None for app in APPROACH_NAMES}
if "processed_previews" not in st.session_state:
    st.session_state.processed_previews = {app: None for app in APPROACH_NAMES}
if "stop_processing" not in st.session_state:
    st.session_state.stop_processing = False

# ==============================================================================
# 4. Sidebar Configuration
# ==============================================================================
with st.sidebar:
    st.markdown("### 🚦 Signal CV Controller")
    st.caption("INFINIX'26 | Member 3 — CSBS: Computer Vision")

    mode = st.radio(
        "Operating Mode",
        ["📹 Four-Video Processing (YOLO)", "🧪 Simulation Fallback (Manual / Mock)"],
        index=0,
    )
    is_simulation = "Simulation" in mode

    st.markdown("---")

    if not is_simulation:
        st.subheader("⚙️ Detection Settings")
        conf_slider = st.slider("Detection Confidence", 0.15, 0.85, 0.35, 0.05,
                                help="Minimum confidence threshold for vehicle detection.")
        max_frames_slider = st.slider("Max Frames per Video", 50, 1500, 300, 50,
                                      help="Frame processing limit for demonstrations.")

        show_boxes = st.checkbox("Show Bounding Boxes & IDs", value=True)
        show_lines = st.checkbox("Show Counting Line", value=True)
        show_trails = st.checkbox("Show Motion Trails", value=True)

        with st.expander("💡 Counting Line Calibration Guide"):
            st.markdown(
                """
                **Why Counting Line Position Matters:**
                - Cameras point downwards at roadways with perspective compression.
                - The counting algorithm uses the vehicle's **tire contact patch** (bottom-center of bounding box).
                - **Default: 70% height** cuts across the active roadway.
                - If placed at 50% (mid-screen), vehicles driving in the lower half never reach or cross the line!
                - Fine-tuning sliders and presets are available directly under each camera feed.
                """
            )
            if st.button("Apply Highway Preset (70%) to All", use_container_width=True):
                for a in APPROACH_NAMES:
                    st.session_state.counter.configure_approach_line(a, (0.05, 0.70), (0.95, 0.70), direction="any")
                st.success("Set all approach counting lines to 70% (roadway height).")
                st.rerun()

        auto_export = st.checkbox("Auto-sync to output/vehicle_counts.json", value=True)

        if st.button("🔄 Reset All Counters", use_container_width=True):
            st.session_state.counter.reset()
            for app in APPROACH_NAMES:
                if st.session_state.video_sources[app]:
                    st.session_state.counter.set_approach_status(app, "READY")
                else:
                    st.session_state.counter.set_approach_status(app, "IDLE")
            export_counts_to_json(
                st.session_state.counter.get_counts(),
                is_simulated=False,
                approach_details=st.session_state.counter.get_all_approach_details(),
            )
            st.success("All approach counters reset to zero.")
            st.rerun()
    else:
        st.subheader("🧪 Manual Approach Overrides")
        st.caption("Directly adjust counts to test signal timing algorithms:")
        s_north = st.number_input("North Approach Count", min_value=0, max_value=300, value=12, step=1)
        s_south = st.number_input("South Approach Count", min_value=0, max_value=300, value=5, step=1)
        s_east  = st.number_input("East Approach Count", min_value=0, max_value=300, value=8, step=1)
        s_west  = st.number_input("West Approach Count", min_value=0, max_value=300, value=2, step=1)

        st.session_state.simulator.set_counts(s_north, s_south, s_east, s_west)
        sim_counts = st.session_state.simulator.get_vehicle_counts()
        export_counts_to_json(sim_counts, is_simulated=True)

        if st.button("🎲 Randomize Traffic Spikes", use_container_width=True):
            st.session_state.simulator.simulate_step(arrival_rate=0.7)
            export_counts_to_json(st.session_state.simulator.get_vehicle_counts(), is_simulated=True)
            st.rerun()

    st.markdown("---")
    st.caption(f"Sync file: `{DEFAULT_JSON_PATH}`")

# ==============================================================================
# 5. Header Bar & Mode Alerts
# ==============================================================================
overall_status = "SIMULATED DATA" if is_simulation else "LIVE CAMERA DETECTION"
status_color = "#f59e0b" if is_simulation else "#10b981"
status_bg = "rgba(245, 158, 11, 0.12)" if is_simulation else "rgba(16, 185, 129, 0.12)"
status_border = "rgba(245, 158, 11, 0.40)" if is_simulation else "rgba(16, 185, 129, 0.40)"

st.markdown(
    f"""
    <div class="header-container-glass">
        <div>
            <div class="header-badge-tag">🚦 INFINIX'26 HACKATHON &bull; CSBS MEMBER 3</div>
            <h1 class="header-title-gradient">Adaptive Traffic Signal Analytics</h1>
            <div class="header-subtitle-text">Autonomous Multi-Camera Edge Vision &bull; Real-Time Vehicle Counting &bull; Dynamic Signal Phase Optimization</div>
        </div>
        <div class="header-status-area">
            <div class="status-glass-pill" style="border-color: {status_border}; background: {status_bg};">
                <span class="status-pulse-dot" style="background-color: {status_color};"></span>
                <span style="font-weight: 700; color: {status_color}; font-size: 0.80rem; letter-spacing: 0.5px;">{overall_status}</span>
            </div>
            <div class="engine-badge-pill">
                <span style="color: #00e5ff;">⚡</span>
                <span style="color: #cbd5e1; font-size: 0.76rem; font-weight: 600;">YOLOv8 + ByteTrack 25+ FPS</span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if is_simulation:
    st.markdown(
        """
        <div class="sim-banner">
            ⚠️ <strong>SIMULATED DATA FALLBACK MODE ACTIVE</strong><br>
            Using manually configurable vehicle counts for controller integration testing. Values are mock inputs and not from live camera footage.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ==============================================================================
# 6. Combined Junction Summary Cards (6 Distinct Cards)
# ==============================================================================
if is_simulation:
    disp_counts = st.session_state.simulator.get_vehicle_counts()
    disp_in_frame = {app: 0 for app in APPROACH_NAMES}
else:
    disp_counts = st.session_state.counter.get_counts()
    disp_in_frame = {app: st.session_state.counter.approaches[app].currently_in_frame for app in APPROACH_NAMES}

tot_unique = sum(disp_counts.values())
tot_in_frame = sum(disp_in_frame.values())

sum_c1, sum_c2, sum_c3, sum_c4, sum_c5, sum_c6 = st.columns(6)

with sum_c1:
    st.markdown(
        f"""
        <div class="summary-card card-hero-unique">
            <div class="summary-header">
                <span class="summary-icon">🚦</span>
                <span class="summary-label">Total Unique</span>
            </div>
            <div class="summary-val" style="color: #00e5ff;">{tot_unique}</div>
            <div class="summary-footer-hero">
                <span style="color: #38bdf8;">✓ Junction Flow</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with sum_c2:
    st.markdown(
        f"""
        <div class="summary-card card-hero-inframe">
            <div class="summary-header">
                <span class="summary-icon">👁️</span>
                <span class="summary-label">In View Now</span>
            </div>
            <div class="summary-val" style="color: #f59e0b;">{tot_in_frame}</div>
            <div class="summary-footer-hero">
                <span style="color: #fbbf24;">⚡ Active Queue</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with sum_c3:
    n_cross = disp_counts.get("North", 0)
    n_view = disp_in_frame.get("North", 0)
    st.markdown(
        f"""
        <div class="summary-card card-north">
            <div class="summary-header">
                <span class="summary-icon">⬇️</span>
                <span class="summary-label">North</span>
            </div>
            <div class="summary-val" style="color: #34d399;">{n_cross}</div>
            <div class="summary-footer">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="color:#94a3b8; font-size:0.75rem;">Crossed:</span>
                    <strong style="color:#34d399; font-size:0.84rem; font-family:'JetBrains Mono',monospace;">{n_cross}</strong>
                </div>
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="color:#94a3b8; font-size:0.75rem;">In View:</span>
                    <strong style="color:#fbbf24; font-size:0.84rem; font-family:'JetBrains Mono',monospace;">{n_view}</strong>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with sum_c4:
    s_cross = disp_counts.get("South", 0)
    s_view = disp_in_frame.get("South", 0)
    st.markdown(
        f"""
        <div class="summary-card card-south">
            <div class="summary-header">
                <span class="summary-icon">⬆️</span>
                <span class="summary-label">South</span>
            </div>
            <div class="summary-val" style="color: #38bdf8;">{s_cross}</div>
            <div class="summary-footer">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="color:#94a3b8; font-size:0.75rem;">Crossed:</span>
                    <strong style="color:#38bdf8; font-size:0.84rem; font-family:'JetBrains Mono',monospace;">{s_cross}</strong>
                </div>
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="color:#94a3b8; font-size:0.75rem;">In View:</span>
                    <strong style="color:#fbbf24; font-size:0.84rem; font-family:'JetBrains Mono',monospace;">{s_view}</strong>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with sum_c5:
    e_cross = disp_counts.get("East", 0)
    e_view = disp_in_frame.get("East", 0)
    st.markdown(
        f"""
        <div class="summary-card card-east">
            <div class="summary-header">
                <span class="summary-icon">⬅️</span>
                <span class="summary-label">East</span>
            </div>
            <div class="summary-val" style="color: #fbbf24;">{e_cross}</div>
            <div class="summary-footer">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="color:#94a3b8; font-size:0.75rem;">Crossed:</span>
                    <strong style="color:#fbbf24; font-size:0.84rem; font-family:'JetBrains Mono',monospace;">{e_cross}</strong>
                </div>
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="color:#94a3b8; font-size:0.75rem;">In View:</span>
                    <strong style="color:#f59e0b; font-size:0.84rem; font-family:'JetBrains Mono',monospace;">{e_view}</strong>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with sum_c6:
    w_cross = disp_counts.get("West", 0)
    w_view = disp_in_frame.get("West", 0)
    st.markdown(
        f"""
        <div class="summary-card card-west">
            <div class="summary-header">
                <span class="summary-icon">➡️</span>
                <span class="summary-label">West</span>
            </div>
            <div class="summary-val" style="color: #c084fc;">{w_cross}</div>
            <div class="summary-footer">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="color:#94a3b8; font-size:0.75rem;">Crossed:</span>
                    <strong style="color:#c084fc; font-size:0.84rem; font-family:'JetBrains Mono',monospace;">{w_cross}</strong>
                </div>
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="color:#94a3b8; font-size:0.75rem;">In View:</span>
                    <strong style="color:#fbbf24; font-size:0.84rem; font-family:'JetBrains Mono',monospace;">{w_view}</strong>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")


def render_approach_metrics(app_name: str) -> str:
    """Render prominent glassmorphism dual metrics (Unique Counted vs Currently In View) and category breakdown."""
    app_det = st.session_state.counter.get_approach_details(app_name)
    cnt = app_det.get("count", 0)
    in_frame = app_det.get("currently_in_frame", 0)
    cars = app_det.get("cars", 0)
    bikes = app_det.get("motorcycles", 0)
    buses = app_det.get("buses", 0)
    trucks = app_det.get("trucks", 0)

    return f"""
    <div class="glass-metrics-wrapper">
        <div class="metric-badge-container">
            <div class="metric-box-unique">
                <div class="metric-title-unique">✓ Unique Counted</div>
                <div class="metric-number-unique">{cnt}</div>
                <div class="metric-caption-sub">Crossed Counting Line</div>
            </div>
            <div class="metric-box-inframe">
                <div class="metric-title-inframe">👁️ Currently in View</div>
                <div class="metric-number-inframe">{in_frame}</div>
                <div class="metric-caption-sub">Instantaneous Detections</div>
            </div>
        </div>
        <div class="breakdown-grid">
            <div class="breakdown-pill">
                <span class="pill-icon">🚗</span>
                <span class="pill-val" style="color: #38bdf8;">{cars}</span>
                <span class="pill-lbl">Cars</span>
            </div>
            <div class="breakdown-pill">
                <span class="pill-icon">🏍️</span>
                <span class="pill-val" style="color: #c084fc;">{bikes}</span>
                <span class="pill-lbl">Bikes</span>
            </div>
            <div class="breakdown-pill">
                <span class="pill-icon">🚌</span>
                <span class="pill-val" style="color: #fbbf24;">{buses}</span>
                <span class="pill-lbl">Buses</span>
            </div>
            <div class="breakdown-pill">
                <span class="pill-icon">🚚</span>
                <span class="pill-val" style="color: #f472b6;">{trucks}</span>
                <span class="pill-lbl">Trucks</span>
            </div>
        </div>
    </div>
    """

# ==============================================================================
# 7. Helper: Process Single Approach Video Stream
# ==============================================================================
def process_approach_video(
    app_name: str,
    video_path: str,
    detector: VehicleDetector,
    vid_placeholder,
    prog_placeholder,
    stat_placeholder,
    metric_placeholder=None,
    max_frames: int = 300,
) -> bool:
    """
    Process one approach video independently with isolated ByteTrack tracking state.
    """
    if not video_path or not os.path.exists(video_path):
        st.session_state.counter.set_approach_status(app_name, "ERROR", "Video file not found")
        stat_placeholder.error(f"[{app_name}] Video file missing.")
        return False

    # CRITICAL: Reset tracker before processing this video stream so IDs don't cross-contaminate!
    detector.reset_tracker()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.session_state.counter.set_approach_status(app_name, "ERROR", "Unsupported format or codec error")
        stat_placeholder.error(f"[{app_name}] Unable to decode video file.")
        return False

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    limit_frames = min(total_frames, max_frames) if total_frames > 0 else max_frames

    st.session_state.counter.set_approach_status(app_name, "PROCESSING")
    stat_placeholder.info(f"[{app_name}] Processing {limit_frames} frames with YOLOv8 & ByteTrack...")

    frame_idx = 0
    last_frame = None

    try:
        while cap.isOpened() and frame_idx < limit_frames:
            if st.session_state.stop_processing:
                st.session_state.counter.set_approach_status(app_name, "PAUSED")
                stat_placeholder.warning(f"[{app_name}] Processing paused by user.")
                break

            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1

            # Proportional downscale for 4K / 1080p footage to maintain 25+ FPS real-time speed
            if frame.shape[1] > 960:
                scale = 960.0 / frame.shape[1]
                frame = cv2.resize(frame, (960, int(frame.shape[0] * scale)), interpolation=cv2.INTER_AREA)

            # YOLO ByteTrack detection
            detections = detector.track(frame, persist=True)

            # Update counting logic using bottom-center reference point
            st.session_state.counter.update_approach(app_name, detections, frame.shape, use_bottom_center=True)

            # Draw visual overlay
            annotated = st.session_state.counter.draw_approach_overlay(
                app_name,
                frame,
                detections,
                show_line=show_lines,
                show_trails=show_trails,
                show_boxes=show_boxes,
            )
            last_frame = annotated

            # Responsive live preview and metrics every 3 frames
            if frame_idx % 3 == 0 or frame_idx == limit_frames:
                rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                vid_placeholder.image(
                    rgb,
                    caption=f"{app_name} Feed | Frame {frame_idx}/{limit_frames}",
                    use_container_width=True,
                )
                if limit_frames > 0:
                    prog_placeholder.progress(min(1.0, frame_idx / limit_frames))
                if metric_placeholder is not None:
                    metric_placeholder.markdown(render_approach_metrics(app_name), unsafe_allow_html=True)

        st.session_state.counter.set_approach_status(app_name, "COMPLETED")
        app_summary = st.session_state.counter.approaches[app_name]
        stat_placeholder.success(
            f"[{app_name}] Completed: {app_summary.total_unique_count} unique vehicles counted (in frame: {app_summary.currently_in_frame})"
        )
        if metric_placeholder is not None:
            metric_placeholder.markdown(render_approach_metrics(app_name), unsafe_allow_html=True)
        if last_frame is not None:
            st.session_state.processed_previews[app_name] = last_frame
        return True

    except Exception as exc:
        st.session_state.counter.set_approach_status(app_name, "ERROR", str(exc))
        stat_placeholder.error(f"[{app_name}] Processing Error: {exc}")
        return False
    finally:
        cap.release()


# ==============================================================================
# 8. Four Independent Video Panels (2x2 Grid)
# ==============================================================================
if not is_simulation:
    st.subheader("📹 Approach Camera Feeds")

    # Global Processing Controls Bar
    with st.container(border=True):
        btn_c1, btn_c2, btn_c3, btn_c4 = st.columns([2.5, 1.8, 1.8, 2.0])
        with btn_c1:
            run_all_btn = st.button("🚀 Process All Videos", type="primary", use_container_width=True)
        with btn_c2:
            stop_btn = st.button("⏹️ Stop Processing", use_container_width=True)
        with btn_c3:
            export_btn = st.button("💾 Export JSON", use_container_width=True)
        with btn_c4:
            rst_btn = st.button("🔄 Reset Counters", use_container_width=True)

    if stop_btn:
        st.session_state.stop_processing = True
    else:
        st.session_state.stop_processing = False

    if rst_btn:
        st.session_state.counter.reset()
        for app in APPROACH_NAMES:
            if st.session_state.video_sources[app]:
                st.session_state.counter.set_approach_status(app, "READY")
            else:
                st.session_state.counter.set_approach_status(app, "IDLE")
        export_counts_to_json(
            st.session_state.counter.get_counts(),
            is_simulated=False,
            approach_details=st.session_state.counter.get_all_approach_details(),
        )
        st.success("All approach counters reset to zero.")
        st.rerun()

    st.write("")

    grid_r1_c1, grid_r1_c2 = st.columns(2)
    grid_r2_c1, grid_r2_c2 = st.columns(2)

    ui_mapping = {
        "North": {"col": grid_r1_c1, "icon": "⬇️", "color": "#10b981"},
        "South": {"col": grid_r1_c2, "icon": "⬆️", "color": "#0ea5e9"},
        "East":  {"col": grid_r2_c1, "icon": "⬅️", "color": "#f59e0b"},
        "West":  {"col": grid_r2_c2, "icon": "➡️", "color": "#a855f7"},
    }

    placeholders = {}

    for app in APPROACH_NAMES:
        uic = ui_mapping[app]
        with uic["col"]:
            with st.container(border=True):
                app_status = st.session_state.counter.approaches[app].status
                st.markdown(
                    f"""
                    <div class="panel-header-glass" style="border-left: 4px solid {uic['color']};">
                        <div style="display: flex; align-items: center; gap: 10px;">
                            <span style="font-size: 1.35rem;">{uic['icon']}</span>
                            <div>
                                <div style="font-size: 1.05rem; font-weight: 800; color: #f8fafc; letter-spacing: 0.4px;">{app.upper()} APPROACH FEED</div>
                                <div style="font-size: 0.72rem; color: #94a3b8; text-transform: uppercase;">Real-Time Approach Ingestion</div>
                            </div>
                        </div>
                        <span class="badge badge-{app_status.lower()}">
                            ● {app_status}
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # File Uploader with format validation (.mp4, .avi, .mov)
                up_vid = st.file_uploader(
                    f"Upload {app} Video (.mp4, .avi, .mov)",
                    type=["mp4", "avi", "mov"],
                    key=f"file_up_{app}",
                )

                if up_vid is not None:
                    if st.session_state.video_display_names[app] != up_vid.name:
                        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                        tfile.write(up_vid.read())
                        st.session_state.video_sources[app] = tfile.name
                        st.session_state.video_display_names[app] = up_vid.name
                        st.session_state.counter.set_approach_status(app, "READY")

                # Quick Demo Buttons
                d_c1, d_c2 = st.columns(2)
                with d_c1:
                    if st.button("Load Demo Traffic Clip", key=f"demo_r_{app}", use_container_width=True):
                        r_path = os.path.join(CURRENT_DIR, "output", "traffic.mp4")
                        if os.path.exists(r_path):
                            st.session_state.video_sources[app] = r_path
                            st.session_state.video_display_names[app] = "traffic.mp4 (Highway)"
                            st.session_state.counter.set_approach_status(app, "READY")
                            st.session_state.counter.configure_approach_line(app, (0.05, 0.70), (0.95, 0.70), direction="any")
                with d_c2:
                    if st.button("Load Synthetic Clip", key=f"demo_s_{app}", use_container_width=True):
                        s_path = os.path.join(CURRENT_DIR, "output", "sample_traffic.mp4")
                        if not os.path.exists(s_path):
                            generate_synthetic_intersection_video(s_path, duration_sec=10)
                        st.session_state.video_sources[app] = s_path
                        st.session_state.video_display_names[app] = "sample_traffic.mp4 (Intersection)"
                        st.session_state.counter.set_approach_status(app, "READY")
                        st.session_state.counter.configure_approach_line(app, (0.05, 0.50), (0.95, 0.50), direction="any")

                # Video metadata indicator
                fname, duration_str, frames_cnt = get_video_metadata(st.session_state.video_sources[app])
                st.markdown(
                    f"""
                    <div class="meta-glass-badge">
                        <span>📁 <strong style="color: #f8fafc;">{fname}</strong></span>
                        <span>⏱️ <strong style="color: #38bdf8;">{duration_str}</strong> <small style="color: #94a3b8;">({frames_cnt} frames)</small></span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Individual Process Button
                single_proc_btn = st.button(f"▶️ Process {app} Only", key=f"run_btn_{app}", use_container_width=True)

                # Viewport & Progress Placeholders
                v_ph = st.empty()
                p_ph = st.empty()
                s_ph = st.empty()
                m_ph = st.empty()

                placeholders[app] = {
                    "vid": v_ph,
                    "prog": p_ph,
                    "stat": s_ph,
                    "metric": m_ph,
                    "single_btn": single_proc_btn,
                }

                # Display saved preview if available
                if st.session_state.processed_previews[app] is not None:
                    rgb_prev = cv2.cvtColor(st.session_state.processed_previews[app], cv2.COLOR_BGR2RGB)
                    v_ph.image(rgb_prev, caption=f"Last Processed: {app}", use_container_width=True)

                # Display prominent dual metrics card
                m_ph.markdown(render_approach_metrics(app), unsafe_allow_html=True)

                # Interactive Counting Line Calibration Expander
                with st.expander(f"📐 Calibrate {app} Counting Line", expanded=False):
                    st.markdown(
                        f"""
                        <div style="font-size:0.8rem; color:#cbd5e1; margin-bottom:8px;">
                            Adjust virtual line position for this camera angle. Roadway traffic is typically in the lower portion (<strong>65% - 75%</strong> of frame height).
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    curr_st = st.session_state.counter.approaches[app]
                    curr_h_val = float(curr_st.line.start_ratio[1])
                    curr_d_val = curr_st.line.allowed_direction

                    cal_c1, cal_c2 = st.columns([1.4, 1.0])
                    with cal_c1:
                        preset_opts = [
                            "Highway Roadway (70% - Recommended)",
                            "Mid-Screen Horizon (50%)",
                            "Upper Roadway (35%)",
                            "Lower Roadway (80%)",
                            "Custom",
                        ]
                        if abs(curr_h_val - 0.70) < 0.02:
                            d_idx = 0
                        elif abs(curr_h_val - 0.50) < 0.02:
                            d_idx = 1
                        elif abs(curr_h_val - 0.35) < 0.02:
                            d_idx = 2
                        elif abs(curr_h_val - 0.80) < 0.02:
                            d_idx = 3
                        else:
                            d_idx = 4

                        preset_val = st.selectbox(
                            "Angle Preset",
                            preset_opts,
                            index=d_idx,
                            key=f"preset_sel_{app}",
                        )
                    with cal_c2:
                        dir_opts = ["any", "down", "up", "left", "right"]
                        dir_val = st.selectbox(
                            "Allowed Direction",
                            dir_opts,
                            index=dir_opts.index(curr_d_val) if curr_d_val in dir_opts else 0,
                            key=f"dir_sel_{app}",
                            help="'any' allows counting in either direction. 'down' counts vehicles moving downward toward camera.",
                        )

                    if "Highway Roadway" in preset_val:
                        target_val = 0.70
                    elif "Mid-Screen" in preset_val:
                        target_val = 0.50
                    elif "Upper Roadway" in preset_val:
                        target_val = 0.35
                    elif "Lower Roadway" in preset_val:
                        target_val = 0.80
                    else:
                        target_val = curr_h_val

                    line_slider = st.slider(
                        f"Line Height ({int(target_val * 100)}%)",
                        min_value=0.10,
                        max_value=0.90,
                        value=float(target_val),
                        step=0.05,
                        key=f"slider_h_{app}",
                        help=f"Virtual line crosses roadway from x=5% to x=95% at y={int(target_val*100)}% height.",
                    )

                    st.session_state.counter.configure_approach_line(
                        app,
                        (0.05, line_slider),
                        (0.95, line_slider),
                        direction=dir_val,
                    )
                    st.caption(
                        f"📍 Line: y = `{line_slider:.2f}` (approx. `{int(line_slider * 540)}px` @ 540p / `{int(line_slider * 720)}px` @ 720p) | Direction: `{dir_val}`"
                    )

                # Live Diagnostics Inspector Expander
                with st.expander(f"🔍 {app} Live Tracking & Line-Crossing Diagnostics", expanded=False):
                    diags = st.session_state.counter.get_approach_diagnostics(app)
                    app_st = st.session_state.counter.approaches[app]

                    st.markdown(
                        f"""
                        <div style="font-size: 0.82rem; margin-bottom: 8px; color: #cbd5e1; background: rgba(13, 20, 36, 0.60); padding: 10px; border-radius: 8px; border: 1px solid rgba(255, 255, 255, 0.08);">
                            <strong>Virtual Line:</strong> ({int(app_st.line.start_ratio[0]*100)}%, {int(app_st.line.start_ratio[1]*100)}%) → ({int(app_st.line.end_ratio[0]*100)}%, {int(app_st.line.end_ratio[1]*100)}%) | 
                            <strong>Allowed Direction:</strong> <code>{app_st.line.allowed_direction}</code><br>
                            <strong>Vehicles Currently in View:</strong> <span style="color:#fbbf24; font-weight:bold;">{app_st.currently_in_frame}</span> | 
                            <strong>Unique Vehicles Counted:</strong> <span style="color:#34d399; font-weight:bold;">{app_st.total_unique_count}</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    if diags:
                        diag_rows = []
                        for d in diags:
                            p_prev_str = f"({d['p_prev'][0]}, {d['p_prev'][1]})" if d.get("p_prev") else "—"
                            p_curr_str = f"({d['p_curr'][0]}, {d['p_curr'][1]})" if d.get("p_curr") else "—"
                            diag_rows.append({
                                "Track ID": f"#{d.get('track_id')}",
                                "Class": d.get("class_name", "car").upper(),
                                "Current Ref Pt": p_curr_str,
                                "Previous Ref Pt": p_prev_str,
                                "Dist to Line": f"{float(d.get('distance_to_line', 0)):.1f} px",
                                "Line Crossed": "✅ YES" if d.get("crossed") else "⏳ NO",
                                "Decision / Reason": str(d.get("decision", "N/A")),
                            })
                        st.dataframe(pd.DataFrame(diag_rows), use_container_width=True, hide_index=True)
                    else:
                        st.info(f"No active vehicle tracks recorded for {app} approach yet. Click '▶️ Process {app} Only' or '🚀 Process All Videos' to view live diagnostics.")

    # --------------------------------------------------------------------------
    # Video Processing Dispatcher
    # --------------------------------------------------------------------------
    run_list = []
    if run_all_btn:
        st.session_state.stop_processing = False
        run_list = [app for app in APPROACH_NAMES if st.session_state.video_sources[app] is not None]
        if not run_list:
            st.warning("Please upload or select at least one approach video first.")

    for app in APPROACH_NAMES:
        if placeholders[app]["single_btn"]:
            st.session_state.stop_processing = False
            if st.session_state.video_sources[app] is not None:
                run_list = [app]
            else:
                st.warning(f"Please upload a video for {app} approach first.")

    if run_list:
        detector = get_cached_detector(conf=conf_slider)
        if not detector.is_available:
            st.error(f"Detector failed to load: {detector.load_error}")
        else:
            with st.spinner(f"Running detection on approaches: {', '.join(run_list)}..."):
                for app in run_list:
                    if st.session_state.stop_processing:
                        break

                    v_path = st.session_state.video_sources[app]
                    # Fault-tolerant execution per approach video
                    try:
                        process_approach_video(
                            app_name=app,
                            video_path=v_path,
                            detector=detector,
                            vid_placeholder=placeholders[app]["vid"],
                            prog_placeholder=placeholders[app]["prog"],
                            stat_placeholder=placeholders[app]["stat"],
                            metric_placeholder=placeholders[app]["metric"],
                            max_frames=max_frames_slider,
                        )
                    except Exception as err:
                        placeholders[app]["stat"].error(f"Error processing {app}: {err}")
                        st.session_state.counter.set_approach_status(app, "ERROR", str(err))

                    # Update metric box
                    placeholders[app]["metric"].markdown(render_approach_metrics(app), unsafe_allow_html=True)

                    # Export to JSON
                    if auto_export:
                        export_counts_to_json(
                            st.session_state.counter.get_counts(),
                            is_simulated=False,
                            approach_details=st.session_state.counter.get_all_approach_details(),
                        )

            st.rerun()

    if export_btn:
        saved_file = export_counts_to_json(
            st.session_state.counter.get_counts(),
            is_simulated=False,
            approach_details=st.session_state.counter.get_all_approach_details(),
        )
        st.success(f"Exported to {saved_file}")

# ==============================================================================
# 9. Analytics & Traffic Distribution Charts
# ==============================================================================
st.markdown("---")
st.subheader("📊 Traffic Analytics & Junction Distribution")

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    with st.container(border=True):
        st.markdown(
            """
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px;">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="font-size: 1.25rem;">📈</span>
                    <div>
                        <div style="font-size: 0.98rem; font-weight: 700; color: #f8fafc;">Approach Traffic Volume</div>
                        <div style="font-size: 0.72rem; color: #94a3b8;">Crossed (Unique) vs Currently In View</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        app_df = pd.DataFrame({
            "Approach": APPROACH_NAMES,
            "Unique Counted (Crossed)": [disp_counts.get(app, 0) for app in APPROACH_NAMES],
            "Currently in View": [disp_in_frame.get(app, 0) for app in APPROACH_NAMES],
        })
        if app_df["Unique Counted (Crossed)"].sum() > 0 or app_df["Currently in View"].sum() > 0:
            st.bar_chart(app_df.set_index("Approach"), color=["#10b981", "#f59e0b"])
        else:
            st.info("No vehicles counted yet. Upload videos and run detection to view traffic distribution.")

with chart_col2:
    with st.container(border=True):
        st.markdown(
            """
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px;">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="font-size: 1.25rem;">🚗</span>
                    <div>
                        <div style="font-size: 0.98rem; font-weight: 700; color: #f8fafc;">Vehicle Category Breakdown</div>
                        <div style="font-size: 0.72rem; color: #94a3b8;">Junction-Wide Aggregate Detections</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if is_simulation:
            cat_data = {"Cars": tot_unique, "Motorcycles": 0, "Buses": 0, "Trucks": 0}
        else:
            cat_data = {"Cars": 0, "Motorcycles": 0, "Buses": 0, "Trucks": 0}
            for app in APPROACH_NAMES:
                det = st.session_state.counter.get_approach_details(app)
                cat_data["Cars"] += det.get("cars", 0)
                cat_data["Motorcycles"] += det.get("motorcycles", 0)
                cat_data["Buses"] += det.get("buses", 0)
                cat_data["Trucks"] += det.get("trucks", 0)

        cat_df = pd.DataFrame(list(cat_data.items()), columns=["Category", "Count"])
        if cat_df["Count"].sum() > 0:
            st.bar_chart(cat_df.set_index("Category"), color="#06b6d4")
        else:
            st.info("No category data available yet.")

# Main Dashboard: Prominent Dynamic Signal Timing Card
if is_simulation:
    active_counts = st.session_state.simulator.get_vehicle_counts()
    active_details = {
        app: {
            "total_unique": active_counts[app],
            "currently_in_frame": 0,
            "status": "SIMULATED",
            "cars": active_counts[app],
            "motorcycles": 0,
            "buses": 0,
            "trucks": 0,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        for app in APPROACH_NAMES
    }
else:
    active_counts = st.session_state.counter.get_counts()
    active_details = st.session_state.counter.get_all_approach_details()

ns_vol = active_counts["North"] + active_counts["South"]
ew_vol = active_counts["East"] + active_counts["West"]
tot_vol = max(1, ns_vol + ew_vol)
base_cycle = 60
ns_green = max(10, min(50, int((ns_vol / tot_vol) * base_cycle)))
ew_green = max(10, min(50, int((ew_vol / tot_vol) * base_cycle)))

st.write("")
with st.container(border=True):
    st.markdown(
        f"""
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 14px;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="font-size: 1.35rem;">⏱️</span>
                <div>
                    <div style="font-size: 1.05rem; font-weight: 800; color: #f8fafc; letter-spacing: 0.3px;">ADAPTIVE GREEN TIME ALLOCATION PREVIEW</div>
                    <div style="font-size: 0.78rem; color: #cbd5e1;">Real-time Webster proportional signal phase optimization based on cumulative approach volume</div>
                </div>
            </div>
            <span class="badge badge-completed">● LIVE OPTIMIZATION</span>
        </div>
        <div class="timing-card-glass">
            <div class="timing-phase-row">
                <div class="timing-phase-box phase-ns">
                    <div class="timing-phase-label">↕️ North-South Phase Green Time</div>
                    <div class="timing-phase-seconds">{ns_green}s</div>
                    <div class="timing-phase-meta">Traffic Volume: <strong style="color: #34d399;">{ns_vol}</strong> vehicles &bull; Cycle Share: <strong style="color: #f8fafc;">{int((ns_vol/tot_vol)*100)}%</strong></div>
                </div>
                <div class="timing-phase-box phase-ew">
                    <div class="timing-phase-label">↔️ East-West Phase Green Time</div>
                    <div class="timing-phase-seconds">{ew_green}s</div>
                    <div class="timing-phase-meta">Traffic Volume: <strong style="color: #fbbf24;">{ew_vol}</strong> vehicles &bull; Cycle Share: <strong style="color: #f8fafc;">{int((ew_vol/tot_vol)*100)}%</strong></div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ==============================================================================
# 10. Developer Tools — Controller Integration (Collapsible)
# ==============================================================================
st.markdown("---")
with st.expander("🛠️ Developer Tools — Controller Integration", expanded=False):
    # Construct complete payload dictionary
    payload_json = {
        "North": active_counts["North"],
        "South": active_counts["South"],
        "East": active_counts["East"],
        "West": active_counts["West"],
        "total_vehicles": sum(active_counts.values()),
        "status": "SIMULATED DATA" if is_simulation else "LIVE CAMERA DETECTION",
        "is_simulated": is_simulation,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "counts": active_counts,
        "approaches": active_details,
    }

    # Strict JSON serialization with standard library json.dumps
    json_str = json.dumps(payload_json, indent=2)

    # Validate with json.loads to guarantee 100% valid JSON syntax
    try:
        parsed_test = json.loads(json_str)
        is_valid_json = True
        validation_status = "✅ Valid JSON confirmed (json.loads verified)"
    except Exception as val_err:
        is_valid_json = False
        validation_status = f"❌ JSON Validation Error: {val_err}"

    dev_top_c1, dev_top_c2 = st.columns([1.6, 1.0])
    with dev_top_c1:
        st.markdown(
            f"""
            <div style="font-size: 0.85rem; color: #cbd5e1; line-height: 1.6; margin-bottom: 8px;">
                <strong>Sync Target File:</strong> <code style="color: #38bdf8;">{DEFAULT_JSON_PATH}</code><br>
                <strong>Engine Mode:</strong> <span style="color: #38bdf8; font-weight: 700;">{payload_json['status']}</span> &bull; 
                <span style="color: #34d399; font-weight: 600;">{validation_status}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with dev_top_c2:
        btn_exp_c1, btn_exp_c2 = st.columns(2)
        with btn_exp_c1:
            if st.button("💾 Export JSON Now", use_container_width=True):
                saved_file = export_counts_to_json(
                    active_counts,
                    is_simulated=is_simulation,
                    approach_details=active_details,
                )
                st.success(f"JSON verified & written to {os.path.basename(saved_file)}")
        with btn_exp_c2:
            st.download_button(
                label="⬇️ Download JSON",
                data=json_str,
                file_name="vehicle_counts.json",
                mime="application/json",
                use_container_width=True,
            )

    st.markdown("##### Formatted Controller JSON Payload Preview")
    st.caption("Standard JSON serialized with commas and formatting. Click the copy icon in the top-right corner to copy:")
    st.code(json_str, language="json")

    st.markdown("##### Python Integration for Traffic Signal Controller Teammate")
    st.code(
        f"""from vehicle_counter import get_vehicle_counts

# Poll real-time counts from computer vision module:
counts = get_vehicle_counts()
# Returns: {active_counts}

# Calculate green time allocations:
ns_green = max(10, counts['North'] + counts['South'])
ew_green = max(10, counts['East'] + counts['West'])
""",
        language="python",
    )
