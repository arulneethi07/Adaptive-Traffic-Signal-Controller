import time
import math
from datetime import datetime
from database import get_connection

class AdaptiveTrafficController:
    def __init__(self):
        self.last_update = time.time()

    def get_status(self):
        conn = get_connection()
        c = conn.cursor()
        c.execute("SELECT * FROM junction_status WHERE id = 1")
        status = dict(c.fetchone())

        c.execute("SELECT * FROM approach_metrics ORDER BY id")
        approaches = [dict(row) for row in c.fetchall()]

        c.execute("SELECT * FROM system_config WHERE id = 1")
        config = dict(c.fetchone())

        c.execute("SELECT * FROM event_logs ORDER BY id DESC LIMIT 15")
        logs = [dict(row) for row in c.fetchall()]

        conn.close()
        return {
            "status": status,
            "approaches": approaches,
            "config": config,
            "logs": logs
        }

    def set_mode(self, mode):
        # 'ADAPTIVE_HYBRID', 'FIXED_TIME', 'MANUAL_OVERRIDE'
        conn = get_connection()
        c = conn.cursor()
        c.execute("UPDATE junction_status SET controller_mode = ?, updated_at = CURRENT_TIMESTAMP WHERE id = 1", (mode,))
        ts = datetime.now().strftime("%H:%M:%S")
        c.execute("INSERT INTO event_logs (timestamp, level, source, message) VALUES (?, 'ACTION', 'Operator', ?)", 
                  (ts, f"Controller mode changed to {mode}"))
        conn.commit()
        conn.close()

    def trigger_phase_override(self, phase_name):
        conn = get_connection()
        c = conn.cursor()

        if phase_name == "PHASE_A_NS":
            c.execute("""
            UPDATE junction_status 
            SET active_phase = 'PHASE_A_NS', active_phase_name = 'Phase A: North-South Flow', active_direction = 'North-South',
                current_green_remaining = 35, current_green_total = 35, updated_at = CURRENT_TIMESTAMP
            WHERE id = 1
            """)
            c.execute("UPDATE approach_metrics SET current_signal = 'GREEN' WHERE direction IN ('North', 'South')")
            c.execute("UPDATE approach_metrics SET current_signal = 'RED' WHERE direction IN ('East', 'West')")
        elif phase_name == "PHASE_B_EW":
            c.execute("""
            UPDATE junction_status 
            SET active_phase = 'PHASE_B_EW', active_phase_name = 'Phase B: East-West Flow', active_direction = 'East-West',
                current_green_remaining = 30, current_green_total = 30, updated_at = CURRENT_TIMESTAMP
            WHERE id = 1
            """)
            c.execute("UPDATE approach_metrics SET current_signal = 'RED' WHERE direction IN ('North', 'South')")
            c.execute("UPDATE approach_metrics SET current_signal = 'GREEN' WHERE direction IN ('East', 'West')")
        elif phase_name == "PEDESTRIAN":
            c.execute("""
            UPDATE junction_status 
            SET active_phase = 'PEDESTRIAN', active_phase_name = 'Pedestrian Crosswalk Phase', active_direction = 'All Approaches',
                current_green_remaining = 15, current_green_total = 15, updated_at = CURRENT_TIMESTAMP
            WHERE id = 1
            """)
            c.execute("UPDATE approach_metrics SET current_signal = 'RED'")

        ts = datetime.now().strftime("%H:%M:%S")
        c.execute("INSERT INTO event_logs (timestamp, level, source, message) VALUES (?, 'ACTION', 'Manual Override', ?)",
                  (ts, f"Manual signal phase triggered: {phase_name}"))
        conn.commit()
        conn.close()

    def trigger_emergency(self, direction="South"):
        conn = get_connection()
        c = conn.cursor()
        c.execute("""
        UPDATE junction_status 
        SET emergency_preempt = 1, active_phase = 'EMERGENCY_PREEMPT',
            active_phase_name = 'EMERGENCY PREEMPTION: Priority Corridor', active_direction = ?,
            current_green_remaining = 25, current_green_total = 25, updated_at = CURRENT_TIMESTAMP
        WHERE id = 1
        """, (direction,))

        # Give green to the emergency direction, red to all others
        c.execute("UPDATE approach_metrics SET current_signal = 'RED'")
        c.execute("UPDATE approach_metrics SET current_signal = 'GREEN' WHERE direction = ?", (direction,))

        ts = datetime.now().strftime("%H:%M:%S")
        c.execute("INSERT INTO event_logs (timestamp, level, source, message) VALUES (?, 'ALERT', 'Emergency Preemption', ?)",
                  (ts, f"EMERGENCY VEHICLE PREEMPTION ACTIVATED: Green corridor for {direction} approach"))
        conn.commit()
        conn.close()

    def update_config(self, new_config):
        conn = get_connection()
        c = conn.cursor()
        c.execute("""
        UPDATE system_config 
        SET min_green_time = ?, max_green_time = ?, yellow_duration = ?, 
            all_red_clearance = ?, yolo_confidence = ?, rl_queue_weight = ?
        WHERE id = 1
        """, (
            new_config.get("min_green_time", 15),
            new_config.get("max_green_time", 60),
            new_config.get("yellow_duration", 4),
            new_config.get("all_red_clearance", 2),
            new_config.get("yolo_confidence", 0.5),
            new_config.get("rl_queue_weight", 0.7)
        ))
        ts = datetime.now().strftime("%H:%M:%S")
        c.execute("INSERT INTO event_logs (timestamp, level, source, message) VALUES (?, 'INFO', 'System Settings', 'System configuration parameters updated')", (ts,))
        conn.commit()
        conn.close()

    def update_cv_counts(self, cv_data: dict, source: str = "LIVE_VIDEO"):
        """
        Synchronize detected/simulated vehicle counts from Computer Vision module
        directly into the Adaptive Signal Controller and SQLite database.
        """
        conn = get_connection()
        c = conn.cursor()

        # Fetch timing constraints
        c.execute("SELECT min_green_time, max_green_time FROM system_config WHERE id = 1")
        cfg_row = c.fetchone()
        min_g = cfg_row["min_green_time"] if cfg_row else 15
        max_g = cfg_row["max_green_time"] if cfg_row else 60

        total_unique = 0
        total_in_frame = 0
        app_counts = {}

        # 1. Update approach_metrics for North, South, East, West
        for direction in ["North", "South", "East", "West"]:
            app_det = cv_data.get(direction, {})
            u_crossed = int(app_det.get("count", app_det.get("total_unique", 0)))
            in_view = int(app_det.get("currently_in_frame", 0))
            cars = int(app_det.get("cars", 0))
            bikes = int(app_det.get("motorcycles", app_det.get("bikes", 0)))
            buses = int(app_det.get("buses", 0))
            trucks = int(app_det.get("trucks", 0))

            total_unique += u_crossed
            total_in_frame += in_view
            app_counts[direction] = u_crossed

            # Congestion level heuristic
            queue_eff = max(in_view, u_crossed)
            if queue_eff <= 5:
                cong = "Light"
            elif queue_eff <= 12:
                cong = "Moderate"
            elif queue_eff <= 20:
                cong = "High"
            else:
                cong = "Critical"

            c.execute("""
            UPDATE approach_metrics
            SET queue_count = ?, unique_crossed = ?, currently_in_frame = ?,
                cars_count = ?, buses_count = ?, bikes_count = ?, trucks_count = ?,
                congestion_level = ?, data_source = ?, updated_at = CURRENT_TIMESTAMP
            WHERE direction = ?
            """, (in_view, u_crossed, in_view, cars, buses, bikes, trucks, cong, source, direction))

        # 2. Dynamic Webster-style green allocation
        ns_vol = app_counts.get("North", 0) + app_counts.get("South", 0)
        ew_vol = app_counts.get("East", 0) + app_counts.get("West", 0)
        tot_vol = max(1, ns_vol + ew_vol)
        cycle = 75

        ns_green = max(min_g, min(max_g, round((ns_vol / tot_vol) * cycle)))
        ew_green = max(min_g, min(max_g, round((ew_vol / tot_vol) * cycle)))

        c.execute("UPDATE approach_metrics SET allocated_green = ? WHERE direction IN ('North', 'South')", (ns_green,))
        c.execute("UPDATE approach_metrics SET allocated_green = ? WHERE direction IN ('East', 'West')", (ew_green,))

        # 3. Update junction_status
        c.execute("""
        UPDATE junction_status
        SET total_detected_vehicles = ?, cv_source = ?, cv_status = 'ACTIVE', updated_at = CURRENT_TIMESTAMP
        WHERE id = 1
        """, (total_unique, source))

        # 4. Insert audit log
        ts = datetime.now().strftime("%H:%M:%S")
        c.execute("""
        INSERT INTO event_logs (timestamp, level, source, message)
        VALUES (?, 'INFO', 'Computer Vision (YOLO)', ?)
        """, (ts, f"CV Synchronized ({source}): N={app_counts.get('North', 0)}, S={app_counts.get('South', 0)}, E={app_counts.get('East', 0)}, W={app_counts.get('West', 0)} | Allocated Green: N-S {ns_green}s, E-W {ew_green}s"))

        conn.commit()
        conn.close()

        # 5. Export validated JSON
        self._export_json_payload(app_counts, cv_data, source, total_unique)

    def _export_json_payload(self, app_counts: dict, cv_data: dict, source: str, total_unique: int):
        """Export validated controller JSON file."""
        import json
        import os

        payload = {
            "North": app_counts.get("North", 0),
            "South": app_counts.get("South", 0),
            "East": app_counts.get("East", 0),
            "West": app_counts.get("West", 0),
            "total_vehicles": total_unique,
            "status": source,
            "is_simulated": "SIMULATION" in source.upper(),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "counts": app_counts,
            "approaches": cv_data,
        }

        # Write to computer_vision/output/vehicle_counts.json and backend/output/vehicle_counts.json
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        targets = [
            os.path.join(base_dir, "computer_vision", "output", "vehicle_counts.json"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "vehicle_counts.json"),
        ]

        for target in targets:
            try:
                os.makedirs(os.path.dirname(target), exist_ok=True)
                tmp = f"{target}.tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=4)
                os.replace(tmp, target)
                # Validation check
                with open(target, "r", encoding="utf-8") as f:
                    json.loads(f.read())
            except Exception as e:
                print(f"Warning: Failed writing {target}: {e}")

    def step_simulation(self):
        """Simulate real-time vehicle flow, queue drainage, and adaptive adjustments"""
        conn = get_connection()
        c = conn.cursor()

        c.execute("SELECT * FROM junction_status WHERE id = 1")
        status = dict(c.fetchone())

        rem = status["current_green_remaining"] - 1
        cleared = status["total_vehicles_cleared"]
        active_phase = status["active_phase"]

        # Fetch current approach metrics to simulate dynamic traffic flow
        c.execute("SELECT * FROM approach_metrics ORDER BY id")
        approaches = [dict(r) for r in c.fetchall()]

        if rem <= 0:
            # Switch Phase using dynamic allocated_green from approach_metrics
            if active_phase == "PHASE_A_NS":
                new_phase = "PHASE_B_EW"
                new_name = "Phase B: East-West Flow"
                new_dir = "East-West"
                c.execute("SELECT allocated_green FROM approach_metrics WHERE direction = 'East'")
                row_g = c.fetchone()
                new_green = row_g["allocated_green"] if (row_g and row_g["allocated_green"]) else 28
                c.execute("UPDATE approach_metrics SET current_signal = 'RED' WHERE direction IN ('North', 'South')")
                c.execute("UPDATE approach_metrics SET current_signal = 'GREEN' WHERE direction IN ('East', 'West')")
            else:
                new_phase = "PHASE_A_NS"
                new_name = "Phase A: North-South Flow"
                new_dir = "North-South"
                c.execute("SELECT allocated_green FROM approach_metrics WHERE direction = 'North'")
                row_g = c.fetchone()
                new_green = row_g["allocated_green"] if (row_g and row_g["allocated_green"]) else 38
                c.execute("UPDATE approach_metrics SET current_signal = 'GREEN' WHERE direction IN ('North', 'South')")
                c.execute("UPDATE approach_metrics SET current_signal = 'RED' WHERE direction IN ('East', 'West')")

            c.execute("""
            UPDATE junction_status 
            SET active_phase = ?, active_phase_name = ?, active_direction = ?,
                current_green_remaining = ?, current_green_total = ?, emergency_preempt = 0,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = 1
            """, (new_phase, new_name, new_dir, new_green, new_green))

            ts = datetime.now().strftime("%H:%M:%S")
            c.execute("INSERT INTO event_logs (timestamp, level, source, message) VALUES (?, 'INFO', 'Phase Controller', ?)",
                      (ts, f"Adaptive Phase switch: {new_name} allocated {new_green}s green time"))
        else:
            # Dynamic traffic queue fluctuations based on simulation in given folder
            # Approaches with green signal discharge vehicles; approaches with red build queues
            for app in approaches:
                d = app["direction"]
                is_green = (active_phase == "PHASE_A_NS" and d in ["North", "South"]) or \
                           (active_phase == "PHASE_B_EW" and d in ["East", "West"])
                
                u_crossed = app.get("unique_crossed", 0) or 0
                in_view = app.get("currently_in_frame", app.get("queue_count", 0)) or 0

                if is_green:
                    # Vehicle crossing / discharging on green
                    if rem % 3 == 0 and in_view > 0:
                        discharge = 1
                        in_view = max(0, in_view - discharge)
                        u_crossed += discharge
                        cleared += discharge
                    elif rem % 4 == 0:
                        # Inbound vehicle passing directly through green
                        u_crossed += 1
                        cleared += 1
                else:
                    # Red signal: arrival vehicle joins queue
                    if rem % 4 == 0 and in_view < 15:
                        in_view += 1

                c.execute("""
                UPDATE approach_metrics
                SET unique_crossed = ?, currently_in_frame = ?, queue_count = ?,
                    current_signal = ?, updated_at = CURRENT_TIMESTAMP
                WHERE direction = ?
                """, (u_crossed, in_view, in_view, "GREEN" if is_green else "RED", d))

            c.execute("UPDATE junction_status SET current_green_remaining = ?, total_vehicles_cleared = ? WHERE id = 1", (rem, cleared))

        conn.commit()
        conn.close()
