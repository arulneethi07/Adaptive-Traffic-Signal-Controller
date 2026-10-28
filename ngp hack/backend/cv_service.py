"""
Computer Vision Service for SmartFlow Backend.
Integrates YOLOv8 + ByteTrack object tracking and vehicle counting
with Starlette / FastAPI backend and Adaptive Traffic Signal Controller.
Supports:
- Laptop webcam capture with shared hardware lock.
- Multi-threaded concurrent processing for video/webcam across approaches.
- Approach selection & independent source options (Webcam, Upload, Demo clips).
"""

import os
import sys
import time
import threading
from typing import Dict, Any, Optional, Tuple, List
import cv2
import numpy as np

# Ensure computer_vision directory is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CV_DIR = os.path.join(BASE_DIR, "computer_vision")
if CV_DIR not in sys.path:
    sys.path.insert(0, CV_DIR)

from detector import VehicleDetector
from vehicle_counter import VehicleCounter
from simulator import TrafficSimulator
from sample_generator import generate_synthetic_intersection_video

APPROACH_NAMES = ["North", "South", "East", "West"]


class CVService:
    """
    Manages multi-approach video & webcam ingestion, YOLOv8 vehicle detection,
    ByteTrack tracking, live frame streaming, and controller synchronization.
    Enforces a shared webcam lock so only one approach can capture from the laptop webcam at a time.
    """

    def __init__(self, controller=None, model_name: Optional[str] = None):
        self.controller = controller
        self.model_name = model_name
        # Dedicated detector instance per approach for 100% thread safety & ByteTrack isolation
        self.detectors: Dict[str, VehicleDetector] = {
            app: VehicleDetector(model_name=self.model_name, conf_threshold=0.35) for app in APPROACH_NAMES
        }
        self.counter = VehicleCounter()
        self.simulator = TrafficSimulator()

        # Approach Configuration
        # Defaults:
        # - North: Selected, Laptop Webcam
        # - South: Selected, Laptop Webcam
        # - East: Unselected, Upload Traffic Video (preloaded demo)
        # - West: Unselected, Simulated Data
        self.approach_sources: Dict[str, str] = {
            "North": "webcam",
            "South": "webcam",
            "East": "upload",
            "West": "simulated",
        }
        self.approach_selected: Dict[str, bool] = {
            "North": True,
            "South": True,
            "East": False,
            "West": False,
        }
        self.approach_ip_urls: Dict[str, str] = {app: "" for app in APPROACH_NAMES}

        # Shared Laptop Webcam Lock
        self.webcam_holder: Optional[str] = None
        self.webcam_lock = threading.Lock()

        self.video_paths: Dict[str, Optional[str]] = {app: None for app in APPROACH_NAMES}
        self.video_names: Dict[str, Optional[str]] = {
            "North": "Laptop Webcam (Live)",
            "South": "Laptop Webcam (Live)",
            "East": "traffic.mp4 (Highway CCTV)",
            "West": "Simulated Traffic Telemetry",
        }
        self.latest_frames: Dict[str, Optional[bytes]] = {app: None for app in APPROACH_NAMES}
        self.progress: Dict[str, float] = {app: 0.0 for app in APPROACH_NAMES}
        self.current_frame_idx: Dict[str, int] = {app: 0 for app in APPROACH_NAMES}
        self.total_frames: Dict[str, int] = {app: 0 for app in APPROACH_NAMES}

        # Multi-threading coordination
        self.worker_threads: Dict[str, Optional[threading.Thread]] = {app: None for app in APPROACH_NAMES}
        self.stop_flags: Dict[str, bool] = {app: False for app in APPROACH_NAMES}
        self.stop_requested = False
        self.is_simulation = False
        self._lock = threading.Lock()

        # Output and Upload folders
        self.upload_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
        os.makedirs(self.upload_dir, exist_ok=True)

        # Set initial status for North & South as READY
        self.counter.set_approach_status("North", "READY")
        self.counter.set_approach_status("South", "READY")
        self.counter.set_approach_status("East", "IDLE")
        self.counter.set_approach_status("West", "IDLE")

        # Pre-generate or prepare demo clips for East and West
        try:
            self._prepare_initial_demos()
        except Exception as e:
            print(f"[CVService] Notice on initial demo setup: {e}")

    def _prepare_initial_demos(self) -> None:
        """Pre-stage demo paths for East and West while leaving them unselected."""
        out_dir = os.path.join(CV_DIR, "output")
        os.makedirs(out_dir, exist_ok=True)
        h_path = os.path.join(out_dir, "traffic.mp4")
        if os.path.exists(h_path):
            self.video_paths["East"] = h_path
            self.video_names["East"] = "traffic.mp4 (Highway CCTV)"
            self.counter.configure_approach_line("East", (0.05, 0.70), (0.95, 0.70), direction="any")

        s_path = os.path.join(out_dir, "sample_traffic.mp4")
        if not os.path.exists(s_path):
            try:
                generate_synthetic_intersection_video(s_path, duration_sec=5)
            except Exception:
                pass
        if os.path.exists(s_path):
            self.video_paths["West"] = s_path
            self.video_names["West"] = "sample_traffic.mp4 (Synthetic 4-Way)"
            self.counter.configure_approach_line("West", (0.05, 0.50), (0.95, 0.50), direction="any")

    @property
    def detector(self) -> VehicleDetector:
        """Backward-compatible accessor for primary detector."""
        return self.detectors.get("North") or next(iter(self.detectors.values()))

    def check_is_processing(self) -> bool:
        """Returns True if any approach worker thread is currently alive."""
        with self._lock:
            return any(t is not None and t.is_alive() for t in self.worker_threads.values())

    @property
    def is_processing(self) -> bool:
        """Property returning True if any worker thread is actively running."""
        return self.check_is_processing()

    @property
    def active_approach(self) -> Optional[str]:
        """Returns currently active approach name, 'all', or None."""
        with self._lock:
            running = [app for app, t in self.worker_threads.items() if t and t.is_alive()]
            if len(running) > 1:
                return "all"
            elif len(running) == 1:
                return running[0]
            return None

    # ==================== SHARED WEBCAM LOCK MANAGEMENT ====================

    def acquire_webcam(self, app_name: str) -> Tuple[bool, Optional[str]]:
        """
        Attempt to acquire the shared laptop webcam lock for an approach.
        Returns (success, error_message).
        """
        with self.webcam_lock:
            if self.webcam_holder is not None and self.webcam_holder != app_name:
                return False, f"Webcam already in use by {self.webcam_holder}"
            self.webcam_holder = app_name
            return True, None

    def release_webcam(self, app_name: str) -> None:
        """Release the shared laptop webcam lock if held by app_name."""
        with self.webcam_lock:
            if self.webcam_holder == app_name:
                self.webcam_holder = None

    # ==================== APPROACH CONFIGURATION ====================

    def configure_approach(
        self,
        app_name: str,
        source_type: Optional[str] = None,
        selected: Optional[bool] = None,
        ip_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Update source type, selected state, and IP camera URL for an approach."""
        if app_name not in APPROACH_NAMES:
            raise ValueError(f"Invalid approach: {app_name}")

        with self._lock:
            if selected is not None:
                self.approach_selected[app_name] = bool(selected)
            if ip_url is not None:
                self.approach_ip_urls[app_name] = str(ip_url).strip()
            if source_type is not None:
                self.approach_sources[app_name] = str(source_type)
                if source_type == "webcam":
                    self.video_names[app_name] = "Laptop Webcam (Live)"
                    self.counter.set_approach_status(app_name, "READY")
                elif source_type == "ip_camera":
                    clean_url = self.approach_ip_urls.get(app_name, "")
                    self.video_names[app_name] = f"IP Camera ({clean_url})" if clean_url else "IP Camera (No URL set)"
                    self.counter.set_approach_status(app_name, "READY" if clean_url else "IDLE")
                elif source_type == "simulated":
                    self.video_names[app_name] = "Simulated Traffic Telemetry"
                    self.counter.set_approach_status(app_name, "READY")
                elif source_type == "demo_highway":
                    self.load_demo(app_name, "highway")
                elif source_type == "demo_synthetic":
                    self.load_demo(app_name, "synthetic")
                elif source_type == "upload":
                    if not self.video_paths.get(app_name):
                        self.video_names[app_name] = "No video loaded"
                        self.counter.set_approach_status(app_name, "IDLE")
                    else:
                        self.counter.set_approach_status(app_name, "READY")

        return self.get_approach_details(app_name)

    def set_video(self, app_name: str, file_path: str, display_name: Optional[str] = None) -> None:
        """Register an uploaded video file for an approach."""
        if app_name not in APPROACH_NAMES:
            raise ValueError(f"Invalid approach: {app_name}")
        self.video_paths[app_name] = file_path
        self.video_names[app_name] = display_name or os.path.basename(file_path)
        self.approach_sources[app_name] = "upload"
        self.counter.set_approach_status(app_name, "READY")
        self.progress[app_name] = 0.0

        # Try to extract the first frame for initial preview
        try:
            cap = cv2.VideoCapture(file_path)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    if frame.shape[1] > 960:
                        scale = 960.0 / frame.shape[1]
                        frame = cv2.resize(frame, (960, int(frame.shape[0] * scale)), interpolation=cv2.INTER_AREA)
                    ret_enc, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                    if ret_enc:
                        self.latest_frames[app_name] = buf.tobytes()
            cap.release()
        except Exception as e:
            print(f"Preview extraction error for {app_name}: {e}")

    def load_demo(self, app_name: str, clip_type: str = "highway") -> Dict[str, Any]:
        """Load a built-in demo traffic clip."""
        if app_name not in APPROACH_NAMES:
            raise ValueError(f"Invalid approach: {app_name}")

        out_dir = os.path.join(CV_DIR, "output")
        os.makedirs(out_dir, exist_ok=True)

        if clip_type == "highway":
            path = os.path.join(out_dir, "traffic.mp4")
            display_name = "traffic.mp4 (Highway CCTV)"
            line_pos = 0.70
            self.approach_sources[app_name] = "demo_highway"
        else:
            path = os.path.join(out_dir, "sample_traffic.mp4")
            if not os.path.exists(path):
                generate_synthetic_intersection_video(path, duration_sec=10)
            display_name = "sample_traffic.mp4 (Synthetic 4-Way)"
            line_pos = 0.50
            self.approach_sources[app_name] = "demo_synthetic"

        if not os.path.exists(path):
            raise FileNotFoundError(f"Demo clip not found at {path}")

        self.video_paths[app_name] = path
        self.video_names[app_name] = display_name
        self.counter.set_approach_status(app_name, "READY")
        self.progress[app_name] = 0.0
        self.counter.configure_approach_line(app_name, (0.05, line_pos), (0.95, line_pos), direction="any")

        # Extract initial frame
        try:
            cap = cv2.VideoCapture(path)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    if frame.shape[1] > 960:
                        scale = 960.0 / frame.shape[1]
                        frame = cv2.resize(frame, (960, int(frame.shape[0] * scale)), interpolation=cv2.INTER_AREA)
                    ret_enc, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                    if ret_enc:
                        self.latest_frames[app_name] = buf.tobytes()
            cap.release()
        except Exception:
            pass

        return {
            "success": True,
            "approach": app_name,
            "filename": display_name,
            "line_pos": line_pos,
        }

    def calibrate_line(self, app_name: str, line_height: float, direction: str = "any") -> None:
        """Calibrate counting line position and direction for an approach."""
        if app_name not in APPROACH_NAMES:
            raise ValueError(f"Invalid approach: {app_name}")
        clamped_h = max(0.10, min(0.90, float(line_height)))
        self.counter.configure_approach_line(app_name, (0.05, clamped_h), (0.95, clamped_h), direction=direction)

    def reset_counts(self) -> None:
        """Reset all approach counters, tracking histories, and worker threads."""
        self.stop_processing()
        time.sleep(0.1)

        with self._lock:
            self.stop_requested = False
            self.counter.reset()
            for app in APPROACH_NAMES:
                self.stop_flags[app] = False
                self.detectors[app].reset_tracker()
                self.release_webcam(app)
                src = self.approach_sources.get(app, "upload")
                if src == "webcam" or self.video_paths[app]:
                    self.counter.set_approach_status(app, "READY")
                else:
                    self.counter.set_approach_status(app, "IDLE")
                self.progress[app] = 0.0
                self.current_frame_idx[app] = 0

            if self.controller:
                self.controller.update_cv_counts(self.get_all_approach_details(), source="RESET")

    def set_simulation_mode(self, enabled: bool) -> None:
        """Toggle simulation fallback mode."""
        self.is_simulation = bool(enabled)
        if self.controller:
            self.controller.update_cv_counts(
                self.get_all_approach_details(),
                source="SIMULATION MODE" if self.is_simulation else "LIVE VIDEO DETECTION"
            )

    def set_simulated_counts(self, north: int, south: int, east: int, west: int) -> None:
        """Set manual simulated counts."""
        self.simulator.set_counts(north, south, east, west)
        if self.is_simulation and self.controller:
            self.controller.update_cv_counts(self.get_all_approach_details(), source="SIMULATION MODE")

    def stop_processing(self, target_approach: Optional[str] = None) -> None:
        """Signal running video/webcam worker(s) to stop gracefully."""
        with self._lock:
            if target_approach and target_approach in self.stop_flags:
                self.stop_flags[target_approach] = True
                self.release_webcam(target_approach)
            else:
                self.stop_requested = True
                for app in APPROACH_NAMES:
                    self.stop_flags[app] = True
                    self.release_webcam(app)

    # ==================== PROCESSING ORCHESTRATION ====================

    def start_processing(self, target_approach: str = "all", max_frames: int = 600) -> bool:
        """
        Start concurrent background video/webcam processing thread(s).
        - If target_approach is 'all': processes ONLY selected approaches (approach_selected[app] == True).
        - If target_approach is a single approach: processes that approach.
        - Routes webcam sources to _process_webcam_stream and video sources to _process_single_video.
        - Respects shared webcam lock: if another approach holds the webcam, marks approach as FAILED with
          'Webcam already in use by {holder}'.
        """
        with self._lock:
            if target_approach == "all":
                # Only include approaches that are SELECTED
                run_list = [app for app in APPROACH_NAMES if self.approach_selected.get(app, False)]
                if not run_list:
                    return False
                # If all requested are already running, return False
                already_running = [app for app in run_list if self.worker_threads.get(app) and self.worker_threads[app].is_alive()]
                if len(already_running) == len(run_list):
                    return False
            elif target_approach in APPROACH_NAMES:
                if self.worker_threads.get(target_approach) and self.worker_threads[target_approach].is_alive():
                    return False
                run_list = [target_approach]
            else:
                return False

            self.stop_requested = False

            # Launch independent worker threads
            for app in run_list:
                self.stop_flags[app] = False
                src_type = self.approach_sources.get(app, "upload")

                # If source is webcam, check lock upfront so UI reflects conflict immediately
                if src_type == "webcam":
                    if self.webcam_holder is not None and self.webcam_holder != app:
                        # Another approach is actively holding the webcam!
                        self.counter.set_approach_status(app, "FAILED", f"Webcam already in use by {self.webcam_holder}")
                        continue

                self.counter.set_approach_status(app, "PROCESSING")
                th = threading.Thread(
                    target=self._approach_worker_loop,
                    args=(app, max_frames, src_type),
                    daemon=True,
                    name=f"CVWorker-{app}"
                )
                self.worker_threads[app] = th
                th.start()

            return True

    def _approach_worker_loop(self, app_name: str, max_frames: int, source_type: str = "upload") -> None:
        """Worker loop executed inside a dedicated approach thread."""
        try:
            if source_type == "webcam":
                self._process_webcam_stream(app_name)
            elif source_type == "ip_camera":
                ip_url = self.approach_ip_urls.get(app_name, "")
                self._process_ip_camera_stream(app_name, ip_url, max_frames)
            elif source_type == "simulated":
                self._process_simulated_stream(app_name, max_frames)
            else:
                self._process_single_video(app_name, max_frames)
        except Exception as exc:
            print(f"[CVService] Exception in worker for {app_name}: {exc}")
            self.counter.set_approach_status(app_name, "FAILED", str(exc))
        finally:
            if self.controller:
                try:
                    self.controller.update_cv_counts(
                        self.get_all_approach_details(),
                        source="LIVE VIDEO DETECTION" if not self.is_simulation else "SIMULATION MODE"
                    )
                except Exception as e:
                    print(f"[CVService] Controller sync error on {app_name} exit: {e}")

    # ==================== WEBCAM PROCESSING STREAM ====================

    def _process_webcam_stream(self, app_name: str) -> bool:
        """
        Process live video from laptop webcam for a specific approach.
        Enforces shared webcam lock, handles permission denial, camera errors, and clean shutdown.
        """
        acquired, err = self.acquire_webcam(app_name)
        if not acquired:
            self.counter.set_approach_status(app_name, "FAILED", err)
            return False

        detector = self.detectors[app_name]
        if not detector.is_available:
            detector.load_model()

        if not detector.is_available:
            self.release_webcam(app_name)
            self.counter.set_approach_status(
                app_name, "FAILED", f"YOLO detector unavailable: {detector.load_error}"
            )
            return False

        detector.reset_tracker()

        # Try DirectShow first on Windows, fallback to default backend
        cap = None
        try:
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap.release()
                cap = cv2.VideoCapture(0)
        except Exception:
            try:
                cap = cv2.VideoCapture(0)
            except Exception as e:
                self.release_webcam(app_name)
                self.counter.set_approach_status(app_name, "FAILED", f"Webcam error: {e}")
                return False

        if not cap or not cap.isOpened():
            self.release_webcam(app_name)
            self.counter.set_approach_status(
                app_name,
                "FAILED",
                "Laptop webcam unavailable (permission denied or no camera device found)"
            )
            return False

        self.counter.set_approach_status(app_name, "PROCESSING")
        self.progress[app_name] = 0.5  # Live streaming indicator
        frame_idx = 0

        try:
            while cap.isOpened():
                if self.stop_requested or self.stop_flags.get(app_name, False):
                    self.counter.set_approach_status(app_name, "PAUSED")
                    break

                ret, frame = cap.read()
                if not ret or frame is None:
                    time.sleep(0.03)
                    continue

                frame_idx += 1
                self.current_frame_idx[app_name] = frame_idx

                # Downscale for smooth real-time 25+ FPS inference
                if frame.shape[1] > 960:
                    scale = 960.0 / frame.shape[1]
                    frame = cv2.resize(frame, (960, int(frame.shape[0] * scale)), interpolation=cv2.INTER_AREA)

                # YOLO + ByteTrack inference on approach's dedicated detector
                detections = detector.track(frame, persist=True)

                # Update counting logic with tire-road bottom center contact point
                self.counter.update_approach(app_name, detections, frame.shape, use_bottom_center=True)

                # Draw visual overlay on video
                annotated = self.counter.draw_approach_overlay(
                    app_name,
                    frame,
                    detections,
                    show_line=True,
                    show_trails=True,
                    show_boxes=True,
                )

                # Encode JPEG snapshot every 2 frames for smooth web view
                if frame_idx % 2 == 0:
                    ret_enc, buf = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                    if ret_enc:
                        self.latest_frames[app_name] = buf.tobytes()

                # Periodically sync counts with controller every 15 frames
                if frame_idx % 15 == 0 and self.controller:
                    self.controller.update_cv_counts(self.get_all_approach_details(), source="LIVE VIDEO DETECTION")

            if not (self.stop_requested or self.stop_flags.get(app_name, False)):
                self.counter.set_approach_status(app_name, "COMPLETED")
            return True

        except Exception as exc:
            self.counter.set_approach_status(app_name, "FAILED", f"Webcam runtime error: {exc}")
            return False
        finally:
            if cap:
                cap.release()
            self.release_webcam(app_name)

    # ==================== IP CAMERA PROCESSING STREAM ====================

    def _process_ip_camera_stream(self, app_name: str, ip_url: str, max_frames: int = 600) -> bool:
        """
        Process live video stream from an IP Camera (RTSP / HTTP stream) for a specific approach.
        Runs dedicated YOLOv8 + ByteTrack object tracking and vehicle counting.
        """
        if not ip_url or not ip_url.strip():
            self.counter.set_approach_status(app_name, "FAILED", "Please provide a valid IP Camera RTSP/HTTP URL")
            return False

        detector = self.detectors[app_name]
        if not detector.is_available:
            detector.load_model()

        if not detector.is_available:
            self.counter.set_approach_status(
                app_name, "FAILED", f"YOLO detector unavailable: {detector.load_error}"
            )
            return False

        clean_url = ip_url.strip()
        detector.reset_tracker()

        cap = None
        try:
            cap = cv2.VideoCapture(clean_url)
        except Exception as e:
            self.counter.set_approach_status(app_name, "FAILED", f"Cannot connect to IP Camera at {clean_url}: {e}")
            return False

        if not cap or not cap.isOpened():
            if cap:
                cap.release()
            self.counter.set_approach_status(
                app_name,
                "FAILED",
                f"Failed to connect to IP Camera at {clean_url} (stream unreachable or timed out)"
            )
            return False

        self.counter.set_approach_status(app_name, "PROCESSING")
        self.progress[app_name] = 0.0
        frame_idx = 0
        fail_count = 0

        try:
            while not (self.stop_requested or self.stop_flags.get(app_name, False)):
                ret, frame = cap.read()
                if not ret or frame is None:
                    fail_count += 1
                    if fail_count > 30:  # ~1 second of missing frames
                        self.counter.set_approach_status(
                            app_name, "FAILED", f"Lost connection to IP Camera stream at {clean_url}"
                        )
                        return False
                    time.sleep(0.03)
                    continue

                fail_count = 0
                frame_idx += 1
                self.current_frame_idx[app_name] = frame_idx

                # Resize if too wide
                h, w = frame.shape[:2]
                if w > 960:
                    scale = 960.0 / w
                    frame = cv2.resize(frame, (960, int(h * scale)), interpolation=cv2.INTER_AREA)

                detections = detector.track(frame)
                self.counter.update_approach(app_name, detections, frame.shape)

                annotated = self.counter.draw_overlay(
                    app_name,
                    frame,
                    detections,
                    show_line=True,
                    show_trails=True,
                    show_boxes=True,
                )

                if frame_idx % 2 == 0:
                    ret_enc, buf = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                    if ret_enc:
                        self.latest_frames[app_name] = buf.tobytes()

                if frame_idx % 15 == 0 and self.controller:
                    self.controller.update_cv_counts(self.get_all_approach_details(), source="LIVE VIDEO DETECTION")

                if max_frames and frame_idx >= max_frames:
                    break

            if not (self.stop_requested or self.stop_flags.get(app_name, False)):
                self.counter.set_approach_status(app_name, "COMPLETED")
                self.progress[app_name] = 1.0
            return True
        except Exception as exc:
            self.counter.set_approach_status(app_name, "FAILED", f"IP Camera stream error: {exc}")
            return False
        finally:
            if cap:
                cap.release()

    # ==================== SIMULATED DATA PROCESSING STREAM ====================

    def _process_simulated_stream(self, app_name: str, max_frames: int = 300) -> bool:
        """
        Generates realistic simulated vehicle crossings and live visual feedback for an approach.
        Renders synthetic road frames with lane markings, crossing line, and vehicle objects,
        updating unique crossings and vehicle classes (cars, bikes, buses, trucks).
        """
        self.stop_requested = False
        self.stop_flags[app_name] = False
        self.counter.set_approach_status(app_name, "PROCESSING")
        self.progress[app_name] = 0.0

        s_path = os.path.join(CV_DIR, "output", "sample_traffic.mp4")
        cap = None
        if os.path.exists(s_path):
            try:
                cap = cv2.VideoCapture(s_path)
            except Exception:
                cap = None

        classes_pool = ["car", "car", "car", "motorcycle", "bus", "truck", "car", "car"]
        step = 0

        try:
            while not (self.stop_requested or self.stop_flags.get(app_name, False)):
                step += 1
                self.current_frame_idx[app_name] = step

                frame = None
                if cap and cap.isOpened():
                    ret, cap_frame = cap.read()
                    if ret and cap_frame is not None:
                        frame = cap_frame
                    else:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret2, cap_frame2 = cap.read()
                        if ret2 and cap_frame2 is not None:
                            frame = cap_frame2

                if frame is None:
                    w, h = 640, 360
                    frame = np.full((h, w, 3), (35, 38, 42), dtype=np.uint8)
                    cv2.line(frame, (60, 0), (60, h), (255, 255, 255), 3)
                    cv2.line(frame, (w - 60, 0), (w - 60, h), (255, 255, 255), 3)
                    for y_dash in range(0, h, 40):
                        cv2.line(frame, (w // 2, y_dash), (w // 2, y_dash + 20), (255, 255, 255), 2)

                    y_pos1 = (step * 8) % (h + 100) - 50
                    y_pos2 = ((step + 25) * 6) % (h + 100) - 50
                    cv2.rectangle(frame, (w // 2 - 120, int(y_pos1)), (w // 2 - 40, int(y_pos1 + 65)), (0, 200, 255), -1)
                    cv2.putText(frame, "SIM CAR", (w // 2 - 115, int(y_pos1 + 40)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                    cv2.rectangle(frame, (w // 2 + 30, int(y_pos2)), (w // 2 + 130, int(y_pos2 + 90)), (255, 120, 0), -1)
                    cv2.putText(frame, "SIM TRUCK", (w // 2 + 35, int(y_pos2 + 50)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                if step % 4 == 0 or step == 1:
                    import random
                    chosen_cls = random.choice(classes_pool)
                    in_view_now = random.randint(1, 4)
                    self.counter.record_simulated_step(app_name, vehicle_class=chosen_cls, in_view=in_view_now)

                line = self.counter.approaches[app_name].line
                p1, p2 = line.get_points(frame.shape[1], frame.shape[0])
                cv2.line(frame, p1, p2, (0, 255, 0), 2)
                cv2.putText(frame, f"SIMULATED STREAM - {app_name.upper()}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 230, 255), 2)

                if step % 2 == 0:
                    ret_enc, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                    if ret_enc:
                        self.latest_frames[app_name] = buf.tobytes()

                if step % 15 == 0 and self.controller:
                    self.controller.update_cv_counts(self.get_all_approach_details(), source="SIMULATION MODE")

                self.progress[app_name] = min(0.99, step / float(max_frames)) if max_frames else 0.5
                time.sleep(0.04)

                if max_frames and step >= max_frames:
                    break

            if not (self.stop_requested or self.stop_flags.get(app_name, False)):
                self.counter.set_approach_status(app_name, "COMPLETED")
                self.progress[app_name] = 1.0
            return True
        except Exception as exc:
            self.counter.set_approach_status(app_name, "FAILED", f"Simulation error: {exc}")
            return False
        finally:
            if cap:
                cap.release()

    # ==================== VIDEO FILE PROCESSING ====================

    def _process_single_video(self, app_name: str, max_frames: int) -> bool:
        """
        Process one approach video stream with dedicated detector & ByteTrack isolation.
        Validates video readability before and during processing.
        """
        video_path = self.video_paths.get(app_name)
        if not video_path or not os.path.exists(video_path):
            self.counter.set_approach_status(app_name, "FAILED", "Video file not found")
            return False

        detector = self.detectors[app_name]
        if not detector.is_available:
            detector.load_model()

        if not detector.is_available:
            self.counter.set_approach_status(
                app_name, "FAILED", f"YOLO detector unavailable: {detector.load_error}"
            )
            return False

        detector.reset_tracker()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.counter.set_approach_status(app_name, "FAILED", "Failed to open video codec or file is unreadable")
            return False

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        limit_frames = min(total_frames, max_frames) if total_frames > 0 else max_frames
        self.total_frames[app_name] = limit_frames

        self.counter.set_approach_status(app_name, "PROCESSING")
        frame_idx = 0

        try:
            while cap.isOpened() and frame_idx < limit_frames:
                if self.stop_requested or self.stop_flags.get(app_name, False):
                    self.counter.set_approach_status(app_name, "PAUSED")
                    break

                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                self.current_frame_idx[app_name] = frame_idx
                self.progress[app_name] = round(frame_idx / limit_frames, 3) if limit_frames > 0 else 0.0

                if frame.shape[1] > 960:
                    scale = 960.0 / frame.shape[1]
                    frame = cv2.resize(frame, (960, int(frame.shape[0] * scale)), interpolation=cv2.INTER_AREA)

                detections = detector.track(frame, persist=True)
                self.counter.update_approach(app_name, detections, frame.shape, use_bottom_center=True)

                annotated = self.counter.draw_approach_overlay(
                    app_name,
                    frame,
                    detections,
                    show_line=True,
                    show_trails=True,
                    show_boxes=True,
                )

                if frame_idx % 2 == 0 or frame_idx == limit_frames:
                    ret_enc, buf = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                    if ret_enc:
                        self.latest_frames[app_name] = buf.tobytes()

                if frame_idx % 15 == 0 and self.controller:
                    self.controller.update_cv_counts(self.get_all_approach_details(), source="LIVE VIDEO DETECTION")

            if not (self.stop_requested or self.stop_flags.get(app_name, False)):
                self.counter.set_approach_status(app_name, "COMPLETED")
                self.progress[app_name] = 1.0
            return True

        except Exception as exc:
            self.counter.set_approach_status(app_name, "FAILED", str(exc))
            return False
        finally:
            cap.release()

    # ==================== METRICS & DIAGNOSTICS ====================

    def get_approach_details(self, app_name: str) -> Dict[str, Any]:
        """Return real-time metrics for a specific approach including source status and selection."""
        source_type = self.approach_sources.get(app_name, "upload")
        is_selected = self.approach_selected.get(app_name, False)
        ip_url = self.approach_ip_urls.get(app_name, "")

        if source_type == "webcam":
            src_label = "LIVE CAMERA"
        elif source_type == "ip_camera":
            src_label = "IP CAMERA"
        elif source_type == "simulated" or self.is_simulation:
            src_label = "SIMULATED DATA"
        else:
            src_label = "UPLOADED VIDEO"

        has_media = (
            (source_type in ["webcam", "simulated"])
            or (source_type == "ip_camera" and bool(ip_url))
            or (self.video_paths.get(app_name) is not None)
        )
        has_frame = self.latest_frames.get(app_name) is not None
        detector = self.detectors.get(app_name)
        det_avail = detector.is_available if detector else False
        det_err = detector.load_error if detector else "Detector not initialized"
        model_p = detector.resolved_model_path if detector else None

        if self.is_simulation:
            sim_counts = self.simulator.get_vehicle_counts()
            val = sim_counts.get(app_name, 0)
            return {
                "name": app_name,
                "count": val,
                "total_unique": val,
                "currently_in_frame": 0,
                "status": "SIMULATED",
                "error_message": None,
                "cars": val,
                "motorcycles": 0,
                "buses": 0,
                "trucks": 0,
                "breakdown": {"car": val, "motorcycle": 0, "bus": 0, "truck": 0},
                "line_height_ratio": 0.70,
                "line_direction": "any",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "has_video": has_media,
                "has_frame": has_frame,
                "video_name": self.video_names[app_name],
                "progress": self.progress[app_name],
                "source_type": source_type,
                "source_status": src_label,
                "selected": is_selected,
                "ip_url": ip_url,
                "webcam_holder": self.webcam_holder,
                "detector_available": det_avail,
                "detector_error": det_err,
                "model_path": model_p,
            }
        else:
            det = self.counter.get_approach_details(app_name)
            det["has_video"] = has_media
            det["has_frame"] = has_frame
            det["video_name"] = self.video_names[app_name]
            det["progress"] = self.progress[app_name]
            det["source_type"] = source_type
            det["source_status"] = src_label
            det["selected"] = is_selected
            det["ip_url"] = ip_url
            det["webcam_holder"] = self.webcam_holder
            det["detector_available"] = det_avail
            det["detector_error"] = det_err
            det["model_path"] = model_p
            return det

    def get_all_approach_details(self) -> Dict[str, Any]:
        """Return dictionary of metrics across all 4 approaches."""
        return {app: self.get_approach_details(app) for app in APPROACH_NAMES}

    def get_status(self) -> Dict[str, Any]:
        """Return system-wide CV status object."""
        approaches = self.get_all_approach_details()
        tot_unique = sum(app["total_unique"] for app in approaches.values())
        tot_in_view = sum(app["currently_in_frame"] for app in approaches.values())
        is_proc = self.check_is_processing()
        act_app = self.active_approach

        if self.is_simulation:
            mode_label = "SIMULATION MODE"
        elif any(app["status"] == "COMPLETED" for app in approaches.values()):
            mode_label = "PROCESSED VIDEO"
        elif is_proc or any(app["status"] == "PROCESSING" for app in approaches.values()):
            mode_label = "LIVE VIDEO DETECTION"
        elif any(app["has_video"] for app in approaches.values()):
            mode_label = "READY TO PROCESS"
        else:
            mode_label = "WAITING FOR VIDEO DATA"

        # Detector health
        any_available = any(d.is_available for d in self.detectors.values())
        first_err = next((d.load_error for d in self.detectors.values() if d.load_error), None)
        resolved_path = next((d.resolved_model_path for d in self.detectors.values() if d.resolved_model_path), None)

        return {
            "mode": mode_label,
            "is_simulation": self.is_simulation,
            "is_processing": is_proc,
            "active_approach": act_app,
            "total_unique_counted": tot_unique,
            "total_currently_in_view": tot_in_view,
            "approaches": approaches,
            "detector_available": any_available,
            "detector_error": first_err,
            "model_path": resolved_path,
            "webcam_holder": self.webcam_holder,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

    def get_frame(self, app_name: str) -> Optional[bytes]:
        """Return the latest JPEG frame buffer for an approach."""
        return self.latest_frames.get(app_name)
