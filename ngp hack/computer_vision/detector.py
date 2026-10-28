"""
Computer Vision Vehicle Detector for Adaptive Traffic Signal Controller.
INFINIX'26 Hackathon - Member 3 (CSBS: Computer Vision)

Wraps Ultralytics YOLOv8 with ByteTrack persistent multi-object tracking.
Filters strictly for vehicles (car, motorcycle, bus, truck) and extracts
both bounding box bottom-centre and centroid reference points.
"""

import os
from typing import List, Dict, Any, Optional, Tuple
import cv2
import numpy as np


CV_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CV_DIR)
DEFAULT_MODEL_NAME = "yolov8n.pt"


def resolve_model_path(model_name_or_path: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
    """
    Safely resolves the YOLO model file path without triggering automatic network downloads.

    Search order:
    1. Explicit model_name_or_path argument (if provided, only this target is searched)
    2. YOLO_MODEL_PATH environment variable (if set, only this target is searched)
    3. Standard default model locations (yolov8n.pt in project root, CV dir, or cwd)

    Returns:
        (resolved_absolute_path, error_message)
    """
    # 1. Explicit argument
    if model_name_or_path:
        cand = str(model_name_or_path).strip()
        candidates = [cand]
        if not os.path.isabs(cand):
            candidates.extend([
                os.path.join(PROJECT_ROOT, cand),
                os.path.join(CV_DIR, cand),
                os.path.join(os.getcwd(), cand),
            ])
        for c in candidates:
            if c and os.path.isfile(c):
                return os.path.abspath(c), None
        return None, (
            f"Specified model weights file '{cand}' not found. "
            f"Please verify the file exists or check the specified path."
        )

    # 2. Environment variable
    env_path = os.environ.get("YOLO_MODEL_PATH")
    if env_path:
        cand = env_path.strip()
        candidates = [cand]
        if not os.path.isabs(cand):
            candidates.extend([
                os.path.join(PROJECT_ROOT, cand),
                os.path.join(CV_DIR, cand),
                os.path.join(os.getcwd(), cand),
            ])
        for c in candidates:
            if c and os.path.isfile(c):
                return os.path.abspath(c), None
        return None, (
            f"Model weights file '{cand}' specified in YOLO_MODEL_PATH was not found."
        )

    # 3. Default model locations
    candidates = [
        os.path.join(PROJECT_ROOT, DEFAULT_MODEL_NAME),
        os.path.join(CV_DIR, DEFAULT_MODEL_NAME),
        os.path.join(os.getcwd(), DEFAULT_MODEL_NAME),
    ]

    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return os.path.abspath(candidate), None

    err = (
        f"YOLOv8 weights file '{DEFAULT_MODEL_NAME}' not found. "
        f"Please place 'yolov8n.pt' in '{PROJECT_ROOT}' or set the "
        "YOLO_MODEL_PATH environment variable."
    )
    return None, err


class VehicleDetector:
    """
    YOLOv8-based vehicle detector and tracker.
    Strictly filters for target vehicle classes:
    COCO 2: car, 3: motorcycle, 5: bus, 7: truck.
    Ignores pedestrians, bicycles, animals, and non-vehicle objects.
    """

    # COCO dataset vehicle classes
    DEFAULT_VEHICLE_CLASSES = {
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck",
    }

    def __init__(
        self,
        model_name: Optional[str] = None,
        conf_threshold: float = 0.35,
        target_classes: Optional[Dict[int, str]] = None,
    ):
        """
        Initialize YOLOv8 vehicle detector.

        Args:
            model_name: Optional path to model weights or model name ('yolov8n.pt')
            conf_threshold: Minimum confidence score to accept detection
            target_classes: Dictionary mapping COCO class ID to string label
        """
        self.model_name = model_name
        self.conf_threshold = conf_threshold
        self.target_classes = target_classes or self.DEFAULT_VEHICLE_CLASSES
        self.model = None
        self._is_loaded = False
        self._load_error: Optional[str] = None
        self.resolved_model_path: Optional[str] = None

        self._load_model()

    def load_model(self, force: bool = False) -> bool:
        """
        Dynamically attempt to (re)load the YOLO model.
        Allows the app to recover dynamically if dependencies or weights were added.
        """
        if self._is_loaded and self.model is not None and not force:
            return True
        self._load_model()
        return self.is_available

    def _load_model(self) -> None:
        """Attempt to load the YOLO model safely without triggering auto-download."""
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            self.model = None
            self._is_loaded = False
            self._load_error = (
                f"Missing required dependency 'ultralytics': {exc}. "
                f"To resolve, run: \"{sys.executable}\" -m pip install -r requirements.txt"
            )
            return

        resolved_path, err = resolve_model_path(self.model_name)
        if not resolved_path:
            self.model = None
            self._is_loaded = False
            self._load_error = err
            return

        self.resolved_model_path = resolved_path

        try:
            self.model = YOLO(resolved_path)
            self._is_loaded = True
            self._load_error = None
        except Exception as e:
            self.model = None
            self._is_loaded = False
            self._load_error = f"Failed to initialize YOLO model from '{resolved_path}': {e}"

    @property
    def is_available(self) -> bool:
        """Returns True if the YOLO model was successfully loaded."""
        return self._is_loaded and self.model is not None

    @property
    def load_error(self) -> Optional[str]:
        """Returns error message if model failed to load."""
        return self._load_error

    def reset_tracker(self) -> None:
        """
        Reset the Ultralytics ByteTrack tracker state.
        CRITICAL: Must be called when starting a new video stream so tracking IDs
        from previous videos do not contaminate new video streams!
        """
        if self.model is None:
            return
        try:
            # Clear internal tracker state in Ultralytics predictor if present
            if hasattr(self.model, "predictor") and self.model.predictor is not None:
                if hasattr(self.model.predictor, "trackers"):
                    self.model.predictor.trackers.clear()
        except Exception:
            pass

    def track(
        self, frame: np.ndarray, persist: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Detect and track vehicles in a single frame.

        Args:
            frame: BGR image from OpenCV
            persist: Whether to maintain ByteTrack history across consecutive frames

        Returns:
            List of detected vehicle dictionaries containing:
                - 'track_id': Unique integer tracking ID (or None if untracked)
                - 'box': (x1, y1, x2, y2) bounding box coordinates
                - 'centroid': (cx, cy) center coordinate of bounding box
                - 'bottom_center': (cx, y2) contact point with the roadway (recommended for counting)
                - 'class_id': COCO class ID (strictly 2, 3, 5, or 7)
                - 'class_name': Readable class label ('car', 'motorcycle', 'bus', 'truck')
                - 'confidence': Detection confidence score [0.0 - 1.0]
        """
        if not self.is_available:
            self.load_model()

        if not self.is_available:
            raise RuntimeError(
                f"YOLO detector is not available. {self._load_error}"
            )

        if frame is None or frame.size == 0:
            return []

        class_indices = list(self.target_classes.keys())

        # Run ByteTrack tracking via Ultralytics
        results = self.model.track(
            source=frame,
            persist=persist,
            classes=class_indices,
            conf=self.conf_threshold,
            tracker="bytetrack.yaml",
            verbose=False,
        )

        detections: List[Dict[str, Any]] = []

        if not results or len(results) == 0:
            return detections

        result = results[0]
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            return detections

        for box in boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())

            # Strictly enforce target vehicle classes (ignore humans, bikes, dogs, etc.)
            if cls_id not in self.target_classes:
                continue

            if conf < self.conf_threshold:
                continue

            # Bounding box coordinates [x1, y1, x2, y2]
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, xyxy)

            # Horizontal midpoint
            cx = int((x1 + x2) / 2)
            # Centroid (midpoint of box)
            cy = int((y1 + y2) / 2)
            # Bottom-centre: Contact patch of tires with road surface
            bottom_center = (cx, y2)

            # Persistent tracking ID from ByteTrack
            track_id = None
            if box.id is not None:
                track_id = int(box.id[0].item())

            class_name = self.target_classes.get(cls_id, f"class_{cls_id}")

            detections.append(
                {
                    "track_id": track_id,
                    "box": (x1, y1, x2, y2),
                    "centroid": (cx, cy),
                    "bottom_center": bottom_center,
                    "class_id": cls_id,
                    "class_name": class_name,
                    "confidence": conf,
                }
            )

        return detections


def check_detector_health(model_name: Optional[str] = None) -> Tuple[bool, str]:
    """Utility to test if YOLO can be instantiated."""
    try:
        detector = VehicleDetector(model_name=model_name)
        if detector.is_available:
            return True, f"YOLO detector initialized successfully from '{detector.resolved_model_path}'."
        return False, f"YOLO load failure: {detector.load_error}"
    except Exception as exc:
        return False, f"Exception during YOLO initialization: {exc}"

