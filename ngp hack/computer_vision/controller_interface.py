"""
Controller Interface & Export Module.
INFINIX'26 Hackathon - Member 3 (CSBS)

Provides JSON file persistence, format validation, and an optional
HTTP REST API server for the Main Traffic Signal Controller developer.
Enriched with per-direction status, timestamps, and vehicle class breakdowns.
"""

import json
import os
import sys
import time
from typing import Dict, Any, Optional, Tuple

# Default output file path
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
DEFAULT_JSON_PATH = os.path.join(DEFAULT_OUTPUT_DIR, "vehicle_counts.json")


def ensure_output_dir(path: Optional[str] = None) -> str:
    """Ensure directory exists for output files."""
    target_dir = path or DEFAULT_OUTPUT_DIR
    os.makedirs(target_dir, exist_ok=True)
    return target_dir


def export_counts_to_json(
    counts: Dict[str, int],
    filepath: Optional[str] = None,
    is_simulated: bool = False,
    approach_details: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Export vehicle counts to a JSON file formatted for the signal controller.
    Includes per-direction status and timestamp while keeping North, South,
    East, West keys immediately accessible at the root level.

    Args:
        counts: Dictionary with keys 'North', 'South', 'East', 'West'
        filepath: Target JSON file path (default: computer_vision/output/vehicle_counts.json)
        is_simulated: Flag indicating whether data originated from fallback simulator
        approach_details: Optional dictionary containing per-approach statuses & breakdowns
        metadata: Optional additional diagnostic info

    Returns:
        Absolute path to the saved JSON file.
    """
    target_path = filepath or DEFAULT_JSON_PATH
    ensure_output_dir(os.path.dirname(target_path))

    curr_time = time.strftime("%Y-%m-%d %H:%M:%S")

    # Standardize dictionary keys
    standardized_counts = {
        "North": int(counts.get("North", 0)),
        "South": int(counts.get("South", 0)),
        "East": int(counts.get("East", 0)),
        "West": int(counts.get("West", 0)),
    }

    total_vehicles = sum(standardized_counts.values())

    # Build per-approach metadata
    approaches_meta: Dict[str, Any] = {}
    for app in ["North", "South", "East", "West"]:
        if approach_details and app in approach_details:
            det = approach_details[app]
            approaches_meta[app] = {
                "count": standardized_counts[app],
                "total_unique": standardized_counts[app],
                "currently_in_frame": det.get("currently_in_frame", 0),
                "status": det.get("status", "COMPLETED" if not is_simulated else "SIMULATED"),
                "timestamp": det.get("timestamp", curr_time),
                "cars": det.get("cars", det.get("breakdown", {}).get("car", 0)),
                "motorcycles": det.get("motorcycles", det.get("breakdown", {}).get("motorcycle", 0)),
                "buses": det.get("buses", det.get("breakdown", {}).get("bus", 0)),
                "trucks": det.get("trucks", det.get("breakdown", {}).get("truck", 0)),
                "error": det.get("error_message"),
            }
        else:
            approaches_meta[app] = {
                "count": standardized_counts[app],
                "total_unique": standardized_counts[app],
                "currently_in_frame": 0,
                "status": "SIMULATED" if is_simulated else "IDLE",
                "timestamp": curr_time,
                "cars": standardized_counts[app],
                "motorcycles": 0,
                "buses": 0,
                "trucks": 0,
                "error": None,
            }

    payload = {
        # Top-level direction keys for direct convenience:
        "North": standardized_counts["North"],
        "South": standardized_counts["South"],
        "East": standardized_counts["East"],
        "West": standardized_counts["West"],
        "total_vehicles": total_vehicles,
        "status": "SIMULATED DATA" if is_simulated else "LIVE CAMERA DETECTION",
        "is_simulated": bool(is_simulated),
        "timestamp": curr_time,
        "counts": standardized_counts,
        "approaches": approaches_meta,
    }

    if metadata:
        payload["metadata"] = metadata

    # Write atomically
    temp_path = f"{target_path}.tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)
    os.replace(temp_path, target_path)

    # Post-write validation check to ensure 100% valid JSON on disk
    with open(target_path, "r", encoding="utf-8") as f:
        json.loads(f.read())

    return target_path


def read_counts_from_json(filepath: Optional[str] = None) -> Dict[str, Any]:
    """
    Read vehicle counts from JSON file and validate schema.

    Returns:
        Parsed dictionary with validated approach counts.
    """
    target_path = filepath or DEFAULT_JSON_PATH
    if not os.path.exists(target_path):
        raise FileNotFoundError(f"Vehicle counts file not found at: {target_path}")

    with open(target_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Validate that all 4 required approaches exist
    for key in ["North", "South", "East", "West"]:
        if key not in data and ("counts" not in data or key not in data["counts"]):
            raise KeyError(f"Missing required approach key '{key}' in JSON payload.")

    return data


def start_rest_api_server(
    host: str = "0.0.0.0",
    port: int = 5000,
    json_path: Optional[str] = None,
) -> None:
    """
    Launch a lightweight Flask REST API server exposing vehicle counts.
    Endpoints:
        GET /api/counts   - Returns current vehicle counts & per-approach details
        GET /api/health   - Returns service health status
    """
    from flask import Flask, jsonify

    target_json = json_path or DEFAULT_JSON_PATH
    app = Flask(__name__)

    @app.route("/api/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok", "service": "Adaptive Traffic CV Module (4 Approaches)"})

    @app.route("/api/counts", methods=["GET"])
    def get_counts():
        try:
            data = read_counts_from_json(target_json)
            return jsonify(data), 200
        except FileNotFoundError:
            return jsonify({
                "status": "INITIALIZING",
                "is_simulated": True,
                "North": 0,
                "South": 0,
                "East": 0,
                "West": 0,
                "total_vehicles": 0,
                "message": "Detection loop has not exported counts yet.",
            }), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    print(f"[*] Starting Traffic Signal Controller CV API on http://{host}:{port}")
    print(f"[*] Endpoint: http://localhost:{port}/api/counts")
    app.run(host=host, port=port, debug=False)


def print_teammate_instructions() -> None:
    """Print copy-pasteable integration code for the main controller developer."""
    snippet = """
===============================================================================
  HOW TO INTEGRATE VEHICLE COUNTS WITH YOUR TRAFFIC SIGNAL CONTROLLER
===============================================================================

OPTION 1: Direct Python Import (Recommended for local joint testing)
-------------------------------------------------------------------------------
from vehicle_counter import get_vehicle_counts

# Inside your controller loop:
counts = get_vehicle_counts()
print(counts)
# Output: {'North': 12, 'South': 5, 'East': 8, 'West': 2}

# Use counts to calculate green time allocations:
green_time_north_south = max(10, counts['North'] + counts['South'])
green_time_east_west   = max(10, counts['East'] + counts['West'])


OPTION 2: File-Based Polling (Zero dependencies, works in any language)
-------------------------------------------------------------------------------
import json, time

while True:
    try:
        with open("computer_vision/output/vehicle_counts.json", "r") as f:
            data = json.load(f)
            north = data["North"]
            south = data["South"]
            east  = data["East"]
            west  = data["West"]
            print(f"Received counts: N={north}, S={south}, E={east}, W={west}")
            print(f"Data Status: {data.get('status')}")
            
            # Access detailed breakdown & status per direction:
            if "approaches" in data:
                print("North Status:", data["approaches"]["North"]["status"])
                print("North Cars:", data["approaches"]["North"]["cars"])
    except Exception as e:
        print(f"Waiting for CV module... ({e})")
    time.sleep(2)


OPTION 3: HTTP REST API (Best for distributed microservices)
-------------------------------------------------------------------------------
import requests

response = requests.get("http://localhost:5000/api/counts")
if response.status_code == 200:
    data = response.json()
    counts = data["counts"]
    print("Fetched counts from CV API:", counts)
===============================================================================
"""
    print(snippet)


if __name__ == "__main__":
    if "--serve" in sys.argv:
        port = 5000
        if "--port" in sys.argv:
            try:
                p_idx = sys.argv.index("--port") + 1
                port = int(sys.argv[p_idx])
            except Exception:
                pass
        start_rest_api_server(port=port)
    else:
        print_teammate_instructions()
