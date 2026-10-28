"""
Verification script for vehicle counting accuracy on real-world footage (traffic.mp4).
"""
import os
import cv2
from detector import VehicleDetector
from vehicle_counter import VehicleCounter
from controller_interface import export_counts_to_json

def run_verification():
    video_path = os.path.join(os.path.dirname(__file__), "output", "traffic.mp4")
    if not os.path.exists(video_path):
        print(f"Video {video_path} not found.")
        return

    detector = VehicleDetector(conf_threshold=0.35)
    counter = VehicleCounter()

    # Configure counting line across roadway at 70% height
    counter.configure_approach_line("North", (0.05, 0.70), (0.95, 0.70), direction="any")

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Opened video: {os.path.basename(video_path)} with {total_frames} total frames.")

    frame_idx = 0
    max_test_frames = 100

    while cap.isOpened() and frame_idx < max_test_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        if frame.shape[1] > 960:
            scale = 960.0 / frame.shape[1]
            frame = cv2.resize(frame, (960, int(frame.shape[0] * scale)), interpolation=cv2.INTER_AREA)

        detections = detector.track(frame, persist=True)
        counter.update_approach("North", detections, frame.shape, use_bottom_center=True)

    cap.release()

    details = counter.get_approach_details("North")
    print("\n" + "=" * 60)
    print("COUNTING ACCURACY VERIFICATION REPORT (traffic.mp4)")
    print("=" * 60)
    print(f"Frames Processed               : {frame_idx}")
    print(f"Vehicles Currently in View     : {details['currently_in_frame']}")
    print(f"Unique Vehicles Counted        : {details['total_unique']}")
    print(f"Class Breakdown                : Cars={details['cars']}, Motorcycles={details['motorcycles']}, Buses={details['buses']}, Trucks={details['trucks']}")
    print(f"Line Height Position           : {details['line_height_ratio']} (Roadway at ~378px on 540p)")
    print(f"Line Allowed Direction         : {details['line_direction']}")
    print("-" * 60)

    diagnostics = counter.get_approach_diagnostics("North")
    print(f"Active Vehicles in Frame {frame_idx} Diagnostics Sample ({len(diagnostics)} total):")
    for d in diagnostics[:5]:
        p_prev = d.get('p_prev') or 'Initial'
        print(f"  Track #{d['track_id']:<3} | Class: {d['class_name']:<10} | Pt: {d['p_curr']} (prev: {p_prev}) | Dist: {d['distance_to_line']:>5.1f}px | Crossed: {str(d['crossed']):<5} | Status: {d['decision']}")
    print("=" * 60)

    # Save to JSON
    saved_json = export_counts_to_json(
        counter.get_counts(),
        is_simulated=False,
        approach_details=counter.get_all_approach_details(),
    )
    print(f"Successfully exported controller JSON payload to: {saved_json}")

if __name__ == "__main__":
    run_verification()
