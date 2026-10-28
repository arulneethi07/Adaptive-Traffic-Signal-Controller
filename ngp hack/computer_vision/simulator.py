"""
Traffic Simulation Fallback Engine.
INFINIX'26 Hackathon - Member 3 (CSBS)

Operates completely independently of OpenCV and PyTorch.
Generates simulated vehicle counts for North, South, East, and West approaches
when YOLO is unavailable, video feeds fail, or controller testing requires mock data.
"""

import time
import random
from typing import Dict, Any, Optional


class TrafficSimulator:
    """
    Independent vehicle count simulator.
    Provides manual count setting and realistic dynamic traffic queue simulation.
    """

    DEFAULT_APPROACHES = ["North", "South", "East", "West"]

    def __init__(
        self,
        initial_counts: Optional[Dict[str, int]] = None,
    ):
        """
        Initialize simulation engine with default or provided counts.
        """
        self.counts: Dict[str, int] = initial_counts or {
            "North": 14,
            "South": 8,
            "East": 11,
            "West": 6,
        }
        self.last_update_time = time.time()
        self.auto_step = False

    def set_counts(
        self,
        north: Optional[int] = None,
        south: Optional[int] = None,
        east: Optional[int] = None,
        west: Optional[int] = None,
    ) -> Dict[str, int]:
        """Manually set vehicle counts for any or all approaches."""
        if north is not None:
            self.counts["North"] = max(0, int(north))
        if south is not None:
            self.counts["South"] = max(0, int(south))
        if east is not None:
            self.counts["East"] = max(0, int(east))
        if west is not None:
            self.counts["West"] = max(0, int(west))
        self.last_update_time = time.time()
        return self.get_vehicle_counts()

    def simulate_step(
        self,
        active_phase: Optional[str] = None,
        arrival_rate: float = 0.4,
        discharge_rate: float = 0.6,
    ) -> Dict[str, int]:
        """
        Simulate realistic dynamic traffic queue fluctuations.
        Approaches with green signal discharge vehicles; approaches with red build queues.
        """
        for approach in self.DEFAULT_APPROACHES:
            # Vehicles arriving randomly
            if random.random() < arrival_rate:
                self.counts[approach] += random.randint(1, 3)

            # If this approach is actively discharging (green light)
            if active_phase and active_phase.lower() in approach.lower():
                if random.random() < discharge_rate:
                    discharge = min(self.counts[approach], random.randint(1, 4))
                    self.counts[approach] -= discharge

        self.last_update_time = time.time()
        return self.get_vehicle_counts()

    def get_vehicle_counts(self) -> Dict[str, int]:
        """
        Standard interface function:
        Returns vehicle counts in exact required format:
        {
            "North": int,
            "South": int,
            "East": int,
            "West": int
        }
        """
        return {
            "North": int(self.counts.get("North", 0)),
            "South": int(self.counts.get("South", 0)),
            "East": int(self.counts.get("East", 0)),
            "West": int(self.counts.get("West", 0)),
        }

    def get_total_count(self) -> int:
        """Return total vehicle count across simulated approaches."""
        return sum(self.counts.values())

    def get_payload(self) -> Dict[str, Any]:
        """
        Full status payload clearly labeled as simulated data.
        """
        return {
            "status": "SIMULATED DATA",
            "is_simulated": True,
            "counts": self.get_vehicle_counts(),
            "total_vehicles": self.get_total_count(),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.last_update_time)),
            "note": "FALLBACK MODE: Mock counts generated independently of camera/YOLO.",
        }


# Global singleton instance for easy import
_global_simulator = TrafficSimulator()


def get_simulated_counts() -> Dict[str, int]:
    """Return current simulated vehicle counts."""
    return _global_simulator.get_vehicle_counts()


def get_simulated_payload() -> Dict[str, Any]:
    """Return full payload labeled SIMULATED DATA."""
    return _global_simulator.get_payload()


def set_simulated_counts(north: int, south: int, east: int, west: int) -> Dict[str, int]:
    """Update global simulator counts."""
    return _global_simulator.set_counts(north, south, east, west)
