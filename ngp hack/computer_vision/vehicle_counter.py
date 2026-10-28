"""
Vehicle Counter Module for Adaptive Traffic Signal Controller.
INFINIX'26 Hackathon - Member 3 (CSBS: Computer Vision)

Provides accurate, direction-specific vehicle counting for four independent
junction approaches (North, South, East, West) with:
- Strict separation between 'Vehicles Currently Detected' and 'Unique Vehicles Counted'.
- Robust line-crossing detection using signed distance and segment intersection.
- Bounding-box bottom-center reference point (tire-road contact patch).
- Comprehensive crossing diagnostics (decision reason, distance to line, positions).
- ByteTrack ID de-duplication (stationary jitter & repeated detections never double-count).
- Configurable counting lines with exact pixel-to-dimension alignment.
"""

from collections import deque
import time
from typing import Dict, List, Set, Tuple, Optional, Any
import cv2
import numpy as np


def ccw(A: Tuple[float, float], B: Tuple[float, float], C: Tuple[float, float]) -> bool:
    """Check if three points are listed in counter-clockwise order."""
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def segments_intersect(
    A: Tuple[float, float],
    B: Tuple[float, float],
    C: Tuple[float, float],
    D: Tuple[float, float],
) -> bool:
    """
    Return True if line segment AB intersects line segment CD.
    Handles collinear and touching boundary cases cleanly.
    """
    # Quick bounding-box rejection test
    if (
        max(A[0], B[0]) < min(C[0], D[0])
        or min(A[0], B[0]) > max(C[0], D[0])
        or max(A[1], B[1]) < min(C[1], D[1])
        or min(A[1], B[1]) > max(C[1], D[1])
    ):
        return False

    def orientation(p, q, r):
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if abs(val) < 1e-5:
            return 0  # collinear
        return 1 if val > 0 else 2  # clock or counterclock

    o1 = orientation(A, B, C)
    o2 = orientation(A, B, D)
    o3 = orientation(C, D, A)
    o4 = orientation(C, D, B)

    # General case
    if o1 != o2 and o3 != o4:
        return True

    # Special Cases (point touches segment)
    def on_segment(p, q, r):
        return (
            min(p[0], r[0]) <= q[0] <= max(p[0], r[0])
            and min(p[1], r[1]) <= q[1] <= max(p[1], r[1])
        )

    if o1 == 0 and on_segment(A, C, B):
        return True
    if o2 == 0 and on_segment(A, D, B):
        return True
    if o3 == 0 and on_segment(C, A, D):
        return True
    if o4 == 0 and on_segment(C, B, D):
        return True

    return False


def signed_distance_to_line(
    P: Tuple[float, float],
    L1: Tuple[float, float],
    L2: Tuple[float, float],
) -> float:
    """
    Compute signed perpendicular distance of point P relative to line L1->L2.
    Positive indicates one side; negative indicates the opposite side.
    """
    line_len = np.hypot(L2[0] - L1[0], L2[1] - L1[1])
    if line_len < 1e-5:
        return 0.0
    cross = (L2[0] - L1[0]) * (P[1] - L1[1]) - (L2[1] - L1[1]) * (P[0] - L1[0])
    return float(cross / line_len)


class CountingLine:
    """Represents a virtual counting line across a roadway."""

    def __init__(
        self,
        name: str,
        start_ratio: Tuple[float, float] = (0.05, 0.50),
        end_ratio: Tuple[float, float] = (0.95, 0.50),
        allowed_direction: str = "any",  # "down", "up", "right", "left", or "any"
        color: Tuple[int, int, int] = (0, 230, 118),
    ):
        """
        Args:
            name: Approach name ('North', 'South', 'East', 'West')
            start_ratio: (x_ratio, y_ratio) relative coordinate [0.0 - 1.0]
            end_ratio: (x_ratio, y_ratio) relative coordinate [0.0 - 1.0]
            allowed_direction: Required movement direction ('down', 'up', 'left', 'right', 'any')
            color: BGR line color
        """
        self.name = name
        self.start_ratio = start_ratio
        self.end_ratio = end_ratio
        self.allowed_direction = allowed_direction
        self.color = color

    def get_points(self, width: int, height: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Convert relative ratio endpoints to absolute pixel coordinates for the current frame dimensions."""
        p1 = (int(self.start_ratio[0] * width), int(self.start_ratio[1] * height))
        p2 = (int(self.end_ratio[0] * width), int(self.end_ratio[1] * height))
        return p1, p2

    def is_direction_allowed(self, p_prev: Tuple[int, int], p_curr: Tuple[int, int]) -> bool:
        """Check if movement from p_prev to p_curr matches the allowed direction policy."""
        dx = p_curr[0] - p_prev[0]
        dy = p_curr[1] - p_prev[1]

        if self.allowed_direction == "down":
            return dy > 0
        elif self.allowed_direction == "up":
            return dy < 0
        elif self.allowed_direction == "right":
            return dx > 0
        elif self.allowed_direction == "left":
            return dx < 0
        return True  # "any" allows both directions


class ApproachState:
    """
    Manages detection, tracking, counting, diagnostics, and status for one road approach.
    Maintains complete isolation from other approaches.
    """

    def __init__(self, name: str, default_line: CountingLine, history_len: int = 30):
        self.name = name
        self.line = default_line
        self.history_len = history_len

        # Unique vehicle tracking state (cumulative crossing count)
        self.counted_ids: Set[int] = set()
        self.id_to_class: Dict[int, str] = {}  # track_id -> normalized class ('car', 'motorcycle', 'bus', 'truck')

        # Track history isolated to this video: track_id -> deque of points [(x, y), ...]
        self.track_history: Dict[int, deque] = {}

        # Instantaneous metrics for current frame
        self.currently_in_frame: int = 0
        self.current_frame_classes: Dict[str, int] = {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0}

        # Real-time per-vehicle diagnostics for debug inspector
        self.latest_diagnostics: List[Dict[str, Any]] = []

        # Status: IDLE, READY, PROCESSING, COMPLETED, ERROR
        self.status: str = "IDLE"
        self.error_message: Optional[str] = None
        self.last_updated: float = time.time()

    def reset(self) -> None:
        """Reset counts, sets, and tracking trajectories for this approach."""
        self.counted_ids.clear()
        self.id_to_class.clear()
        self.track_history.clear()
        self.currently_in_frame = 0
        self.current_frame_classes = {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0}
        self.latest_diagnostics.clear()
        self.status = "IDLE"
        self.error_message = None
        self.last_updated = time.time()

    @property
    def total_unique_count(self) -> int:
        """Strictly equals the number of unique vehicles that crossed the line."""
        return len(self.counted_ids)

    def get_class_breakdown(self) -> Dict[str, int]:
        """
        Calculate breakdown directly from counted vehicles.
        Mathematically guarantees: total_unique == sum(breakdown.values()).
        """
        breakdown = {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0}
        for cls in self.id_to_class.values():
            if cls in breakdown:
                breakdown[cls] += 1
            else:
                breakdown["car"] += 1
        return breakdown

    def update(
        self,
        detections: List[Dict[str, Any]],
        frame_shape: Tuple[int, int, int],
        use_bottom_center: bool = True,
    ) -> int:
        """
        Process detections from this approach's video frame.

        Args:
            detections: List of detection records from VehicleDetector.track()
            frame_shape: (height, width, channels)
            use_bottom_center: Use bottom-center for line crossing (recommended)

        Returns:
            Current total unique vehicle count.
        """
        h, w = frame_shape[:2]
        current_active_ids = set()
        l1, l2 = self.line.get_points(w, h)

        # Update currently detected in this frame
        self.currently_in_frame = len(detections)
        frame_classes = {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0}
        current_frame_diagnostics = []

        for det in detections:
            track_id = det.get("track_id")
            cls_name = det.get("class_name", "car").lower()

            # Normalize class
            if "motorcycle" in cls_name or "bike" in cls_name:
                norm_cls = "motorcycle"
            elif "bus" in cls_name:
                norm_cls = "bus"
            elif "truck" in cls_name:
                norm_cls = "truck"
            else:
                norm_cls = "car"

            frame_classes[norm_cls] = frame_classes.get(norm_cls, 0) + 1

            # Require valid tracking ID
            if track_id is None:
                continue

            current_active_ids.add(track_id)

            # Choose reference point: bottom_center (tire contact patch) or centroid
            if use_bottom_center and "bottom_center" in det:
                ref_pt = det["bottom_center"]
            else:
                ref_pt = det["centroid"]

            if track_id not in self.track_history:
                self.track_history[track_id] = deque(maxlen=self.history_len)

            history = self.track_history[track_id]
            p_curr = ref_pt

            # Diagnostic information record
            dist = signed_distance_to_line(p_curr, l1, l2)
            diag_entry = {
                "track_id": track_id,
                "class_name": norm_cls,
                "p_curr": p_curr,
                "p_prev": None,
                "distance_to_line": round(abs(dist), 1),
                "line_coords": f"({l1[0]},{l1[1]})->({l2[0]},{l2[1]})",
                "crossed": False,
                "decision": "INITIAL_FRAME",
            }

            if len(history) > 0:
                p_prev = history[-1]
                diag_entry["p_prev"] = p_prev

                # Case 1: Already counted previously
                if track_id in self.counted_ids:
                    diag_entry["crossed"] = True
                    diag_entry["decision"] = "ALREADY_COUNTED"
                else:
                    # Minimum motion displacement check
                    displacement = np.hypot(p_curr[0] - p_prev[0], p_curr[1] - p_prev[1])

                    if displacement < 1.0:
                        diag_entry["decision"] = "STATIONARY_JITTER"
                    else:
                        # Test if trajectory intersects counting line
                        intersects = segments_intersect(p_prev, p_curr, l1, l2)

                        if intersects:
                            if self.line.is_direction_allowed(p_prev, p_curr):
                                # Valid line crossing!
                                self.counted_ids.add(track_id)
                                self.id_to_class[track_id] = norm_cls
                                self.last_updated = time.time()
                                diag_entry["crossed"] = True
                                diag_entry["decision"] = "CROSSED_VALID"
                            else:
                                diag_entry["decision"] = "DIRECTION_REJECTED"
                        else:
                            diag_entry["decision"] = f"APPROACHING ({round(abs(dist), 1)}px to line)"
            else:
                if track_id in self.counted_ids:
                    diag_entry["crossed"] = True
                    diag_entry["decision"] = "ALREADY_COUNTED"
                else:
                    diag_entry["decision"] = f"IN_VIEW ({round(abs(dist), 1)}px to line)"

            history.append(p_curr)
            current_frame_diagnostics.append(diag_entry)

        self.current_frame_classes = frame_classes
        self.latest_diagnostics = current_frame_diagnostics

        # Clean stale tracks from dictionary
        if len(self.track_history) > 300:
            stale_keys = [k for k in self.track_history.keys() if k not in current_active_ids]
            for k in stale_keys[: len(stale_keys) // 2]:
                del self.track_history[k]

        return self.total_unique_count

    def get_details(self) -> Dict[str, Any]:
        """Return comprehensive status and breakdown for this approach."""
        breakdown = self.get_class_breakdown()
        return {
            "name": self.name,
            "count": self.total_unique_count,
            "total_unique": self.total_unique_count,
            "currently_in_frame": self.currently_in_frame,
            "status": self.status,
            "error_message": self.error_message,
            "cars": breakdown["car"],
            "motorcycles": breakdown["motorcycle"],
            "buses": breakdown["bus"],
            "trucks": breakdown["truck"],
            "breakdown": breakdown,
            "line_height_ratio": self.line.start_ratio[1],
            "line_direction": self.line.allowed_direction,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.last_updated)),
        }

    def record_simulated_step(self, vehicle_class: str = "car", in_view: int = 1) -> None:
        """Record simulated vehicle crossing and active in-view count."""
        sim_id = 300000 + len(self.counted_ids) + 1
        self.counted_ids.add(sim_id)
        self.id_to_class[sim_id] = vehicle_class
        self.currently_in_frame = max(0, in_view)
        self.current_frame_classes = {vehicle_class: in_view}
        self.last_updated = time.time()


class VehicleCounter:
    """
    Central vehicle counter managing all 4 intersection approaches (North, South, East, West).
    """

    DEFAULT_APPROACHES = ["North", "South", "East", "West"]

    def __init__(self, history_len: int = 30):
        self.history_len = history_len

        self.colors = {
            "North": (0, 230, 118),  # Vibrant Green
            "South": (41, 182, 246),  # Sky Blue
            "East": (255, 167, 38),   # Amber / Orange
            "West": (171, 71, 188),   # Purple / Magenta
        }

        # Initialize independent approach states with default mid-screen line at 50%
        self.approaches: Dict[str, ApproachState] = {}
        for app in self.DEFAULT_APPROACHES:
            line = CountingLine(
                name=app,
                start_ratio=(0.05, 0.50),
                end_ratio=(0.95, 0.50),
                allowed_direction="any",
                color=self.colors.get(app, (0, 230, 118)),
            )
            self.approaches[app] = ApproachState(app, line, history_len=self.history_len)

        # Legacy 4-way single-camera configuration
        self.lines: Dict[str, CountingLine] = {
            "North": CountingLine("North", (0.20, 0.35), (0.80, 0.35), "down", (0, 230, 118)),
            "South": CountingLine("South", (0.20, 0.65), (0.80, 0.65), "up", (41, 182, 246)),
            "West": CountingLine("West", (0.30, 0.20), (0.30, 0.80), "right", (171, 71, 188)),
            "East": CountingLine("East", (0.70, 0.20), (0.70, 0.80), "left", (255, 167, 38)),
        }

    def configure_approach_line(
        self,
        approach: str,
        start_ratio: Tuple[float, float],
        end_ratio: Tuple[float, float],
        direction: str = "any",
    ) -> None:
        """Configure counting line coordinates and direction policy for an approach."""
        if approach in self.approaches:
            self.approaches[approach].line.start_ratio = start_ratio
            self.approaches[approach].line.end_ratio = end_ratio
            self.approaches[approach].line.allowed_direction = direction

    def configure_line(
        self,
        approach: str,
        start_ratio: Tuple[float, float],
        end_ratio: Tuple[float, float],
        direction: str = "any",
    ) -> None:
        """Legacy helper for configure_line."""
        self.configure_approach_line(approach, start_ratio, end_ratio, direction)
        if approach in self.lines:
            self.lines[approach].start_ratio = start_ratio
            self.lines[approach].end_ratio = end_ratio
            self.lines[approach].allowed_direction = direction

    def set_approach_status(
        self, approach: str, status: str, error_message: Optional[str] = None
    ) -> None:
        """Update status for an approach."""
        if approach in self.approaches:
            self.approaches[approach].status = status
            self.approaches[approach].error_message = error_message
            self.approaches[approach].last_updated = time.time()

    def update_approach(
        self,
        approach: str,
        detections: List[Dict[str, Any]],
        frame_shape: Tuple[int, int, int],
        use_bottom_center: bool = True,
    ) -> int:
        """
        Update counting logic for a specific approach video.
        Ensures isolated ByteTrack IDs and de-duplication.
        """
        if approach in self.approaches:
            return self.approaches[approach].update(detections, frame_shape, use_bottom_center)
        return 0

    def update(
        self,
        detections: List[Dict[str, Any]],
        frame_shape: Tuple[int, int, int],
    ) -> Dict[str, int]:
        """Legacy single-video 4-way intersection update method."""
        h, w = frame_shape[:2]
        for app_name, line in self.lines.items():
            l1, l2 = line.get_points(w, h)
            app_state = self.approaches[app_name]

            for det in detections:
                track_id = det.get("track_id")
                cx, cy = det["centroid"]
                cls_name = det.get("class_name", "car").lower()
                norm_cls = "car"
                if "motorcycle" in cls_name or "bike" in cls_name:
                    norm_cls = "motorcycle"
                elif "bus" in cls_name:
                    norm_cls = "bus"
                elif "truck" in cls_name:
                    norm_cls = "truck"

                if track_id is None:
                    continue

                if track_id not in app_state.track_history:
                    app_state.track_history[track_id] = deque(maxlen=self.history_len)

                hist = app_state.track_history[track_id]
                p_curr = (cx, cy)

                if len(hist) > 0:
                    p_prev = hist[-1]
                    if track_id not in app_state.counted_ids:
                        if segments_intersect(p_prev, p_curr, l1, l2):
                            if line.is_direction_allowed(p_prev, p_curr):
                                app_state.counted_ids.add(track_id)
                                app_state.id_to_class[track_id] = norm_cls
                                app_state.last_updated = time.time()
                hist.append(p_curr)

        return self.get_counts()

    def get_counts(self) -> Dict[str, int]:
        """
        Standard controller interface:
        Returns vehicle counts formatted as:
        {
            "North": int,
            "South": int,
            "East": int,
            "West": int
        }
        """
        return {app: self.approaches[app].total_unique_count for app in self.DEFAULT_APPROACHES}

    @property
    def counts(self) -> Dict[str, int]:
        """Property for backward compatibility."""
        return self.get_counts()

    @counts.setter
    def counts(self, value: Dict[str, int]) -> None:
        """Allow manual override of counts."""
        for app, val in value.items():
            if app in self.approaches:
                diff = int(val) - self.approaches[app].total_unique_count
                if diff > 0:
                    for i in range(diff):
                        fake_id = 100000 + len(self.approaches[app].counted_ids)
                        self.approaches[app].counted_ids.add(fake_id)
                        self.approaches[app].id_to_class[fake_id] = "car"
                elif diff < 0:
                    self.approaches[app].reset()
                    for i in range(int(val)):
                        fake_id = 100000 + i
                        self.approaches[app].counted_ids.add(fake_id)
                        self.approaches[app].id_to_class[fake_id] = "car"

    def get_total_count(self) -> int:
        """Return total unique vehicle count across all four directions."""
        return sum(self.approaches[app].total_unique_count for app in self.DEFAULT_APPROACHES)

    def get_approach_details(self, approach: str) -> Dict[str, Any]:
        """Return comprehensive status and breakdown for one approach."""
        if approach in self.approaches:
            return self.approaches[approach].get_details()
        return {}

    def get_all_approach_details(self) -> Dict[str, Dict[str, Any]]:
        """Return details for all 4 approaches."""
        return {app: self.approaches[app].get_details() for app in self.DEFAULT_APPROACHES}

    def get_approach_diagnostics(self, approach: str) -> List[Dict[str, Any]]:
        """Return per-vehicle line-crossing diagnostics for debug inspection."""
        if approach in self.approaches:
            return self.approaches[approach].latest_diagnostics
        return []

    def get_class_breakdown(self) -> Dict[str, Dict[str, int]]:
        """Return class breakdown per approach."""
        return {app: self.approaches[app].get_class_breakdown() for app in self.DEFAULT_APPROACHES}

    def reset(self, approach: Optional[str] = None) -> None:
        """Reset counters. If approach is None, resets all approaches."""
        if approach and approach in self.approaches:
            self.approaches[approach].reset()
        else:
            for app in self.approaches.values():
                app.reset()

    def record_simulated_step(self, approach: str, vehicle_class: str = "car", in_view: int = 1) -> None:
        """Record simulated vehicle step on an approach."""
        if approach in self.approaches:
            self.approaches[approach].record_simulated_step(vehicle_class, in_view)

    def draw_approach_overlay(
        self,
        approach: str,
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
        show_line: bool = True,
        show_trails: bool = True,
        show_boxes: bool = True,
    ) -> np.ndarray:
        """
        Draw visual overlay specifically for a single approach video:
        renders counting line, tracking trails, bounding boxes, reference points,
        and prominently overlays BOTH:
        1. Vehicles Currently Detected in Frame
        2. Unique Vehicles Counted Crossing Line
        """
        if approach not in self.approaches:
            return frame

        app_state = self.approaches[approach]
        overlay = frame.copy()
        h, w = frame.shape[:2]

        # 1. Draw Counting Line
        if show_line:
            p1, p2 = app_state.line.get_points(w, h)
            # Outer dark border for high contrast
            cv2.line(overlay, p1, p2, (15, 20, 25), 6, cv2.LINE_AA)
            cv2.line(overlay, p1, p2, app_state.line.color, 3, cv2.LINE_AA)

            mid_x = (p1[0] + p2[0]) // 2
            mid_y = (p1[1] + p2[1]) // 2
            label = f"{approach} Line (y={p1[1]}px) | Crossed: {app_state.total_unique_count}"

            (txt_w, txt_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(
                overlay,
                (mid_x - 6, mid_y - txt_h - 6),
                (mid_x + txt_w + 6, mid_y + 6),
                (15, 20, 30),
                -1,
            )
            cv2.rectangle(
                overlay,
                (mid_x - 6, mid_y - txt_h - 6),
                (mid_x + txt_w + 6, mid_y + 6),
                app_state.line.color,
                1,
            )
            cv2.putText(
                overlay,
                label,
                (mid_x, mid_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                app_state.line.color,
                1,
                cv2.LINE_AA,
            )

        # 2. Draw Motion Trails
        if show_trails:
            for track_id, history in app_state.track_history.items():
                if len(history) > 1:
                    pts = np.array(history, np.int32).reshape((-1, 1, 2))
                    cv2.polylines(
                        overlay,
                        [pts],
                        isClosed=False,
                        color=(0, 220, 255),
                        thickness=2,
                        lineType=cv2.LINE_AA,
                    )

        # 3. Draw Bounding Boxes, IDs, and Reference Points
        if show_boxes:
            for det in detections:
                x1, y1, x2, y2 = det["box"]
                track_id = det.get("track_id")
                cls_name = det.get("class_name", "veh")
                conf = det.get("confidence", 0.0)
                cx = (x1 + x2) // 2

                # If vehicle already crossed, label as COUNTED in green; otherwise IN VIEW in amber
                is_counted = track_id in app_state.counted_ids if track_id else False
                box_color = (0, 230, 118) if is_counted else (255, 170, 0)

                cv2.rectangle(overlay, (x1, y1), (x2, y2), box_color, 2)

                # Draw bottom-center reference point (tire-road contact patch)
                cv2.circle(overlay, (cx, y2), 5, (0, 0, 255), -1)
                cv2.circle(overlay, (cx, y2), 2, (255, 255, 255), -1)

                id_str = f"#{track_id} " if track_id is not None else ""
                status_str = " [CROSSED]" if is_counted else " [IN VIEW]"
                caption = f"{id_str}{cls_name} {conf:.2f}{status_str}"
                (cw, ch), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
                cv2.rectangle(
                    overlay,
                    (x1, max(0, y1 - ch - 6)),
                    (x1 + cw + 4, max(ch + 6, y1)),
                    box_color,
                    -1,
                )
                cv2.putText(
                    overlay,
                    caption,
                    (x1 + 2, max(0, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (10, 15, 20),
                    1,
                    cv2.LINE_AA,
                )

        # 4. Top HUD overlay displaying BOTH metrics prominently
        hud_h = 48
        cv2.rectangle(overlay, (0, 0), (w, hud_h), (10, 15, 25), -1)
        cv2.line(overlay, (0, hud_h), (w, hud_h), (35, 45, 60), 1)

        # Left Metric: Unique Counted
        count_text = f"[{approach.upper()}] UNIQUE COUNTED: {app_state.total_unique_count}"
        cv2.putText(
            overlay,
            count_text,
            (14, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (0, 230, 118),
            1,
            cv2.LINE_AA,
        )

        # In-Frame Metric: Currently in View
        in_frame_text = f"CURRENTLY DETECTED IN FRAME: {app_state.currently_in_frame}"
        cv2.putText(
            overlay,
            in_frame_text,
            (14, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 170, 0),
            1,
            cv2.LINE_AA,
        )

        # Right side: Class breakdown summary
        breakdown = app_state.get_class_breakdown()
        cls_summary = f"Cars: {breakdown['car']} | Bikes: {breakdown['motorcycle']} | Buses: {breakdown['bus']} | Trucks: {breakdown['truck']}"
        (tw, _), _ = cv2.getTextSize(cls_summary, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        cv2.putText(
            overlay,
            cls_summary,
            (max(20, w - tw - 15), 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (200, 210, 220),
            1,
            cv2.LINE_AA,
        )

        return overlay

    def draw_overlay(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
        show_lines: bool = True,
        show_trails: bool = True,
        show_boxes: bool = True,
    ) -> np.ndarray:
        """Legacy overlay for single 4-way intersection video."""
        return self.draw_approach_overlay(
            "North", frame, detections, show_lines, show_trails, show_boxes
        )


# ==============================================================================
# Global Module Instance & Export Functions
# ==============================================================================

_global_counter = VehicleCounter()


def get_default_counter() -> VehicleCounter:
    """Return the global VehicleCounter singleton."""
    return _global_counter


def get_vehicle_counts() -> Dict[str, int]:
    """
    Standard interface function required by Project Specifications:

    Returns vehicle counts formatted as:
    {
        "North": 12,
        "South": 5,
        "East": 8,
        "West": 2
    }
    """
    return _global_counter.get_counts()


def get_total_count() -> int:
    """Return total unique vehicle count across all four directions."""
    return _global_counter.get_total_count()


def reset_counts(approach: Optional[str] = None) -> None:
    """Reset vehicle counters."""
    _global_counter.reset(approach)
