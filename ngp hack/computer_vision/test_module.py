"""
Comprehensive Test Suite for Vehicle Counting Accuracy & Fault Tolerance.
INFINIX'26 Hackathon - Member 3 (CSBS: Computer Vision)

Directly tests all 8 requirements specified in Project Priority 1.E:
1. A vehicle detected across many frames must count only once.
2. A tracked vehicle crossing the line must count once.
3. A vehicle remaining stationary must not repeatedly increment the count.
4. A vehicle crossing in the reverse direction must follow the configured counting policy.
5. Two different tracking IDs crossing the line must count as two vehicles.
6. Counts from different directions must not mix.
7. Restarting a video must reset its previous tracking and counting state.
8. Unsupported or failed video processing must show an error instead of a misleading zero count.

Additional tests:
- Total unique vehicles strictly equals the sum of class-wise unique counts.
- Controller JSON schema verification with top-level keys.
- Fallback simulation mode output.
"""

import os
import sys
import tempfile
import unittest

# Ensure module path is accessible
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from vehicle_counter import (
    VehicleCounter,
    CountingLine,
    segments_intersect,
    get_vehicle_counts,
    get_total_count,
    reset_counts,
)
from simulator import TrafficSimulator, get_simulated_payload
from controller_interface import (
    export_counts_to_json,
    read_counts_from_json,
)
from sample_generator import generate_synthetic_intersection_video


class TestVehicleCountingAccuracy(unittest.TestCase):
    """Rigorous unit tests for vehicle counting accuracy, line crossing, and state isolation."""

    def setUp(self):
        self.counter = VehicleCounter()
        self.frame_shape = (480, 640, 3)  # Height=480, Width=640

    def test_1_vehicle_detected_across_many_frames_counts_only_once(self):
        """
        Requirement 1: A vehicle detected across many frames must count only once.
        Simulates vehicle #42 detected over 50 consecutive frames as it approaches,
        crosses the line, and drives away.
        """
        # North counting line is at y = 0.50 * 480 = 240
        for frame_num in range(50):
            # y moves from 100 to 350
            y_pos = int(100 + (frame_num / 50.0) * 250)
            det = [{
                "track_id": 42,
                "box": (300, y_pos - 40, 340, y_pos),
                "centroid": (320, y_pos - 20),
                "bottom_center": (320, y_pos),
                "class_name": "car",
            }]
            self.counter.update_approach("North", det, self.frame_shape)

        # Count must be strictly 1, never 50
        counts = self.counter.get_counts()
        self.assertEqual(counts["North"], 1)
        self.assertEqual(self.counter.get_total_count(), 1)

    def test_2_tracked_vehicle_crossing_the_line_counts_once(self):
        """
        Requirement 2: A tracked vehicle crossing the line must count once.
        """
        # Pre-crossing frame (above line at y=240)
        self.counter.update_approach(
            "South",
            [{
                "track_id": 10,
                "box": (300, 180, 340, 220),
                "centroid": (320, 200),
                "bottom_center": (320, 220),
                "class_name": "truck",
            }],
            self.frame_shape,
        )
        self.assertEqual(self.counter.get_counts()["South"], 0)

        # Crossing frame (moves across line to y=260)
        self.counter.update_approach(
            "South",
            [{
                "track_id": 10,
                "box": (300, 220, 340, 260),
                "centroid": (320, 240),
                "bottom_center": (320, 260),
                "class_name": "truck",
            }],
            self.frame_shape,
        )
        self.assertEqual(self.counter.get_counts()["South"], 1)

        # Post-crossing frame (continues moving)
        self.counter.update_approach(
            "South",
            [{
                "track_id": 10,
                "box": (300, 260, 340, 300),
                "centroid": (320, 280),
                "bottom_center": (320, 300),
                "class_name": "truck",
            }],
            self.frame_shape,
        )
        self.assertEqual(self.counter.get_counts()["South"], 1)

    def test_3_stationary_vehicle_must_not_repeatedly_increment_count(self):
        """
        Requirement 3: A vehicle remaining stationary must not repeatedly increment the count.
        Simulates a stopped car whose bounding box slightly fluctuates near the line.
        """
        # Step 1: Car crosses line once
        self.counter.update_approach(
            "East",
            [{"track_id": 99, "box": (200, 210, 240, 230), "bottom_center": (220, 230), "class_name": "car"}],
            self.frame_shape,
        )
        self.counter.update_approach(
            "East",
            [{"track_id": 99, "box": (200, 230, 240, 250), "bottom_center": (220, 250), "class_name": "car"}],
            self.frame_shape,
        )
        self.assertEqual(self.counter.get_counts()["East"], 1)

        # Step 2: Car halts at the line and micro-jitters for 30 frames
        for i in range(30):
            jitter = (i % 2) * 1  # 0 or 1 px
            self.counter.update_approach(
                "East",
                [{"track_id": 99, "box": (200, 230 + jitter, 240, 250 + jitter), "bottom_center": (220, 250 + jitter), "class_name": "car"}],
                self.frame_shape,
            )
            # Count must NEVER increase beyond 1
            self.assertEqual(self.counter.get_counts()["East"], 1)

    def test_4_reverse_direction_follows_policy(self):
        """
        Requirement 4: A vehicle crossing in the reverse direction must follow configured counting policy.
        Configure line to ONLY accept 'down' (entering inbound from top).
        A vehicle moving 'up' (exiting or reverse) must NOT be counted.
        """
        # Configure North approach counting line to strictly accept 'down'
        self.counter.configure_approach_line("North", (0.05, 0.50), (0.95, 0.50), direction="down")

        # Vehicle moving upwards (reverse: y from 260 to 220)
        self.counter.update_approach(
            "North",
            [{"track_id": 7, "box": (300, 240, 340, 260), "bottom_center": (320, 260), "class_name": "car"}],
            self.frame_shape,
        )
        self.counter.update_approach(
            "North",
            [{"track_id": 7, "box": (300, 200, 340, 220), "bottom_center": (320, 220), "class_name": "car"}],
            self.frame_shape,
        )

        # Rejected by direction policy!
        self.assertEqual(self.counter.get_counts()["North"], 0)

        # Now vehicle moving downwards (valid: y from 220 to 260)
        self.counter.update_approach(
            "North",
            [{"track_id": 8, "box": (300, 200, 340, 220), "bottom_center": (320, 220), "class_name": "car"}],
            self.frame_shape,
        )
        self.counter.update_approach(
            "North",
            [{"track_id": 8, "box": (300, 240, 340, 260), "bottom_center": (320, 260), "class_name": "car"}],
            self.frame_shape,
        )
        self.assertEqual(self.counter.get_counts()["North"], 1)

    def test_5_two_different_tracking_ids_count_as_two_vehicles(self):
        """
        Requirement 5: Two different tracking IDs crossing the line must count as two vehicles.
        """
        # Vehicle #1 crosses
        self.counter.update_approach(
            "West",
            [{"track_id": 1, "box": (100, 220, 140, 230), "bottom_center": (120, 230), "class_name": "motorcycle"}],
            self.frame_shape,
        )
        self.counter.update_approach(
            "West",
            [{"track_id": 1, "box": (100, 240, 140, 260), "bottom_center": (120, 260), "class_name": "motorcycle"}],
            self.frame_shape,
        )

        # Vehicle #2 crosses
        self.counter.update_approach(
            "West",
            [{"track_id": 2, "box": (200, 220, 240, 230), "bottom_center": (220, 230), "class_name": "bus"}],
            self.frame_shape,
        )
        self.counter.update_approach(
            "West",
            [{"track_id": 2, "box": (200, 240, 240, 260), "bottom_center": (220, 260), "class_name": "bus"}],
            self.frame_shape,
        )

        counts = self.counter.get_counts()
        self.assertEqual(counts["West"], 2)

        # Verify class breakdown
        details = self.counter.get_approach_details("West")
        self.assertEqual(details["motorcycles"], 1)
        self.assertEqual(details["buses"], 1)
        self.assertEqual(details["cars"], 0)
        self.assertEqual(details["trucks"], 0)
        self.assertEqual(details["count"], 2)

    def test_6_counts_from_different_directions_must_not_mix(self):
        """
        Requirement 6: Counts from different directions must not mix.
        """
        # North video gets vehicle #1
        self.counter.update_approach(
            "North",
            [{"track_id": 1, "box": (100, 220, 140, 230), "bottom_center": (120, 230), "class_name": "car"}],
            self.frame_shape,
        )
        self.counter.update_approach(
            "North",
            [{"track_id": 1, "box": (100, 240, 140, 260), "bottom_center": (120, 260), "class_name": "car"}],
            self.frame_shape,
        )

        # South video gets its OWN vehicle #1 (same ID number, completely different video stream)
        self.counter.update_approach(
            "South",
            [{"track_id": 1, "box": (200, 220, 240, 230), "bottom_center": (220, 230), "class_name": "bus"}],
            self.frame_shape,
        )
        self.counter.update_approach(
            "South",
            [{"track_id": 1, "box": (200, 240, 240, 260), "bottom_center": (220, 260), "class_name": "bus"}],
            self.frame_shape,
        )

        # East and West have 0
        counts = self.counter.get_counts()
        self.assertEqual(counts["North"], 1)
        self.assertEqual(counts["South"], 1)
        self.assertEqual(counts["East"], 0)
        self.assertEqual(counts["West"], 0)
        self.assertEqual(self.counter.get_total_count(), 2)

    def test_7_restarting_video_resets_tracking_and_counting_state(self):
        """
        Requirement 7: Restarting a video must reset its previous tracking and counting state.
        """
        # Accumulate 3 counts on North (x=320 is well within counting line x-range [32, 608])
        for i in [101, 102, 103]:
            self.counter.update_approach(
                "North",
                [{"track_id": i, "box": (300, 220, 340, 230), "bottom_center": (320, 230), "class_name": "car"}],
                self.frame_shape,
            )
            self.counter.update_approach(
                "North",
                [{"track_id": i, "box": (300, 240, 340, 260), "bottom_center": (320, 260), "class_name": "car"}],
                self.frame_shape,
            )
        self.assertEqual(self.counter.get_counts()["North"], 3)

        # Reset only North
        self.counter.reset("North")
        self.assertEqual(self.counter.get_counts()["North"], 0)
        details = self.counter.get_approach_details("North")
        self.assertEqual(details["count"], 0)
        self.assertEqual(details["cars"], 0)
        self.assertEqual(len(self.counter.approaches["North"].counted_ids), 0)
        self.assertEqual(len(self.counter.approaches["North"].track_history), 0)

    def test_8_failed_video_processing_shows_error_instead_of_zero_count(self):
        """
        Requirement 8: Unsupported or failed video processing must show an error instead of a misleading zero count.
        """
        self.counter.set_approach_status("East", "ERROR", "Unsupported video codec / file corrupted")
        details = self.counter.get_approach_details("East")
        self.assertEqual(details["status"], "ERROR")
        self.assertEqual(details["error_message"], "Unsupported video codec / file corrupted")

        # Confirm export captures this status
        test_counts = self.counter.get_counts()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
            temp_path = tf.name
        try:
            export_counts_to_json(
                test_counts,
                filepath=temp_path,
                is_simulated=False,
                approach_details=self.counter.get_all_approach_details(),
            )
            data = read_counts_from_json(temp_path)
            self.assertEqual(data["approaches"]["East"]["status"], "ERROR")
            self.assertEqual(data["approaches"]["East"]["error"], "Unsupported video codec / file corrupted")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_total_equals_sum_of_classes(self):
        """
        Verify that total_unique strictly equals the sum of class-wise unique counts.
        """
        # Add 2 cars, 1 bus, 1 truck, 1 motorcycle
        vehs = [
            (1, "car"),
            (2, "car"),
            (3, "bus"),
            (4, "truck"),
            (5, "motorcycle"),
        ]
        for vid, vcls in vehs:
            self.counter.update_approach(
                "South",
                [{"track_id": vid, "box": (300, 220, 340, 230), "bottom_center": (320, 230), "class_name": vcls}],
                self.frame_shape,
            )
            self.counter.update_approach(
                "South",
                [{"track_id": vid, "box": (300, 240, 340, 260), "bottom_center": (320, 260), "class_name": vcls}],
                self.frame_shape,
            )

        det = self.counter.get_approach_details("South")
        total = det["count"]
        class_sum = det["cars"] + det["motorcycles"] + det["buses"] + det["trucks"]
        self.assertEqual(total, 5)
        self.assertEqual(total, class_sum)

    def test_two_distinct_metrics_separation(self):
        """
        Verify that 'currently_in_frame' and 'total_unique' remain strictly separated.
        If 4 vehicles are present in the frame but only 1 crossed the line,
        currently_in_frame must be 4, and total_unique must be 1.
        """
        # Vehicle 1 crossed the line (y: 220 -> 260)
        self.counter.update_approach(
            "North",
            [{"track_id": 1, "box": (100, 200, 140, 220), "bottom_center": (120, 220), "class_name": "car"}],
            self.frame_shape,
        )
        # In next frame, Vehicle 1 moves forward, and 3 new vehicles appear before the line
        dets = [
            {"track_id": 1, "box": (100, 240, 140, 260), "bottom_center": (120, 260), "class_name": "car"},
            {"track_id": 2, "box": (200, 100, 240, 120), "bottom_center": (220, 120), "class_name": "car"},
            {"track_id": 3, "box": (300, 130, 340, 150), "bottom_center": (320, 150), "class_name": "car"},
            {"track_id": 4, "box": (400, 160, 440, 180), "bottom_center": (420, 180), "class_name": "truck"},
        ]
        self.counter.update_approach("North", dets, self.frame_shape)
        det = self.counter.get_approach_details("North")

        self.assertEqual(det["currently_in_frame"], 4)
        self.assertEqual(det["total_unique"], 1)
        self.assertNotEqual(det["currently_in_frame"], det["total_unique"])

    def test_configurable_counting_line_and_diagnostics(self):
        """
        Verify that line height ratio can be calibrated and diagnostics record real-time decisions.
        """
        # Calibrate line to 75% height
        self.counter.configure_approach_line("West", (0.05, 0.75), (0.95, 0.75), direction="any")
        line_pts = self.counter.approaches["West"].line.get_points(640, 480)
        self.assertEqual(line_pts[0][1], int(0.75 * 480))

        # Vehicle moves from 340 to 380 (crossing 360)
        self.counter.update_approach(
            "West",
            [{"track_id": 55, "box": (100, 320, 140, 340), "bottom_center": (120, 340), "class_name": "car"}],
            self.frame_shape,
        )
        self.counter.update_approach(
            "West",
            [{"track_id": 55, "box": (100, 360, 140, 380), "bottom_center": (120, 380), "class_name": "car"}],
            self.frame_shape,
        )

        self.assertEqual(self.counter.get_counts()["West"], 1)
        diags = self.counter.get_approach_diagnostics("West")
        self.assertEqual(len(diags), 1)
        self.assertTrue(diags[0]["crossed"])
        self.assertEqual(diags[0]["decision"], "CROSSED_VALID")


class TestTrafficSimulator(unittest.TestCase):
    """Test suite for fallback simulation engine."""

    def setUp(self):
        self.sim = TrafficSimulator()

    def test_simulation_format_and_overrides(self):
        self.sim.set_counts(north=15, south=9, east=12, west=4)
        counts = self.sim.get_vehicle_counts()
        self.assertEqual(counts["North"], 15)
        self.assertEqual(counts["South"], 9)
        self.assertEqual(counts["East"], 12)
        self.assertEqual(counts["West"], 4)
        self.assertEqual(self.sim.get_total_count(), 40)

    def test_simulated_payload_labeling(self):
        payload = self.sim.get_payload()
        self.assertEqual(payload["status"], "SIMULATED DATA")
        self.assertTrue(payload["is_simulated"])
        self.assertIn("counts", payload)


if __name__ == "__main__":
    unittest.main(verbosity=2)
