"""
Sample Traffic Video Generator.
INFINIX'26 Hackathon - Member 3 (CSBS)

Generates a synthetic 4-way traffic intersection video using OpenCV.
Features vehicles (cars, buses, trucks) approaching from North, South, East,
and West with realistic directional motion for immediate testing.
"""

import os
import cv2
import numpy as np
from typing import Optional


def generate_synthetic_intersection_video(
    output_path: str,
    duration_sec: int = 15,
    fps: int = 25,
    width: int = 640,
    height: int = 480,
) -> str:
    """
    Generate a 4-way intersection video with vehicles traveling across
    North, South, East, and West approaches.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    total_frames = duration_sec * fps

    # Vehicle definitions: (approach, start_x, start_y, speed_x, speed_y, width, height, color, label)
    # We will spawn vehicles over time
    class Vehicle:
        def __init__(self, approach, x, y, vx, vy, w, h, color, name):
            self.approach = approach
            self.x = float(x)
            self.y = float(y)
            self.vx = float(vx)
            self.vy = float(vy)
            self.w = w
            self.h = h
            self.color = color
            self.name = name

        def step(self):
            self.x += self.vx
            self.y += self.vy

        def draw(self, frame):
            x1 = int(self.x - self.w / 2)
            y1 = int(self.y - self.h / 2)
            x2 = int(self.x + self.w / 2)
            y2 = int(self.y + self.h / 2)

            # Draw car body
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.color, -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

            # Draw windshield / roof
            inner_pad = 6
            cv2.rectangle(
                frame,
                (x1 + inner_pad, y1 + inner_pad),
                (x2 - inner_pad, y2 - inner_pad),
                (40, 40, 40),
                -1,
            )

            # Draw label
            cv2.putText(
                frame,
                self.name,
                (x1, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    vehicles = []

    # Road coordinates (4-way intersection)
    road_color = (60, 60, 65)
    lane_color = (240, 240, 240)
    grass_color = (35, 90, 40)

    road_w = 140
    cx = width // 2
    cy = height // 2

    for frame_idx in range(total_frames):
        # Background
        frame = np.full((height, width, 3), grass_color, dtype=np.uint8)

        # Draw Roads
        # North-South Road
        cv2.rectangle(
            frame, (cx - road_w // 2, 0), (cx + road_w // 2, height), road_color, -1
        )
        # East-West Road
        cv2.rectangle(
            frame, (0, cy - road_w // 2), (width, cy + road_w // 2), road_color, -1
        )

        # Center intersection box
        cv2.rectangle(
            frame,
            (cx - road_w // 2, cy - road_w // 2),
            (cx + road_w // 2, cy + road_w // 2),
            (50, 50, 55),
            -1,
        )

        # Draw lane dividing dashes
        for y in range(0, height, 30):
            if abs(y - cy) > road_w // 2:
                cv2.line(frame, (cx, y), (cx, y + 15), lane_color, 2)
        for x in range(0, width, 30):
            if abs(x - cx) > road_w // 2:
                cv2.line(frame, (x, cy), (x + 15, cy), lane_color, 2)

        # Road boundaries
        cv2.line(frame, (cx - road_w // 2, 0), (cx - road_w // 2, cy - road_w // 2), (200, 200, 200), 2)
        cv2.line(frame, (cx + road_w // 2, 0), (cx + road_w // 2, cy - road_w // 2), (200, 200, 200), 2)
        cv2.line(frame, (cx - road_w // 2, cy + road_w // 2), (cx - road_w // 2, height), (200, 200, 200), 2)
        cv2.line(frame, (cx + road_w // 2, cy + road_w // 2), (cx + road_w // 2, height), (200, 200, 200), 2)

        # Spawn vehicles periodically
        # North inbound (lane x = cx - road_w // 4, moving down)
        if frame_idx % 45 == 0 and frame_idx < total_frames - 50:
            vehicles.append(
                Vehicle("North", cx - road_w // 4, 10, 0, 3.5, 36, 52, (30, 70, 210), "CAR")
            )

        # South inbound (lane x = cx + road_w // 4, moving up)
        if (frame_idx + 20) % 55 == 0 and frame_idx < total_frames - 50:
            vehicles.append(
                Vehicle("South", cx + road_w // 4, height - 10, 0, -3.2, 40, 70, (200, 100, 30), "BUS")
            )

        # West inbound (lane y = cy + road_w // 4, moving right)
        if (frame_idx + 10) % 50 == 0 and frame_idx < total_frames - 50:
            vehicles.append(
                Vehicle("West", 10, cy + road_w // 4, 3.4, 0, 54, 34, (50, 180, 50), "TRUCK")
            )

        # East inbound (lane y = cy - road_w // 4, moving left)
        if (frame_idx + 35) % 60 == 0 and frame_idx < total_frames - 50:
            vehicles.append(
                Vehicle("East", width - 10, cy - road_w // 4, -3.8, 0, 48, 30, (180, 50, 180), "CAR")
            )

        # Update and draw active vehicles
        alive_vehicles = []
        for veh in vehicles:
            veh.step()
            veh.draw(frame)
            # Retain if within bounds
            if -80 < veh.x < width + 80 and -80 < veh.y < height + 80:
                alive_vehicles.append(veh)
        vehicles = alive_vehicles

        # Write frame to video
        out.write(frame)

    out.release()
    return output_path


if __name__ == "__main__":
    target_out = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "output", "sample_traffic.mp4"
    )
    print(f"Generating synthetic traffic intersection video to {target_out}...")
    generate_synthetic_intersection_video(target_out, duration_sec=10)
    print("Done! Video successfully created.")
