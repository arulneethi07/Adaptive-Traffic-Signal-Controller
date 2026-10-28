import sqlite3
import os
import json
from datetime import datetime, timedelta

DB_PATH = os.path.join(os.path.dirname(__file__), "traffic_controller.db")

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_connection()
    c = conn.cursor()

    # Table: Junction Status
    c.execute("""
    CREATE TABLE IF NOT EXISTS junction_status (
        id INTEGER PRIMARY KEY,
        junction_name TEXT,
        controller_mode TEXT, -- 'ADAPTIVE_HYBRID', 'FIXED_TIME', 'MANUAL_OVERRIDE'
        active_phase TEXT,    -- 'PHASE_A_NS', 'PHASE_B_EW', 'PEDESTRIAN'
        active_phase_name TEXT,
        active_direction TEXT,
        current_green_remaining INTEGER,
        current_green_total INTEGER,
        cycle_time INTEGER,
        total_vehicles_cleared INTEGER,
        average_flow_rate INTEGER,
        emergency_preempt INTEGER DEFAULT 0,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # Table: Approach Metrics
    c.execute("""
    CREATE TABLE IF NOT EXISTS approach_metrics (
        id INTEGER PRIMARY KEY,
        direction TEXT UNIQUE, -- 'North', 'South', 'East', 'West'
        queue_count INTEGER,
        cars_count INTEGER,
        buses_count INTEGER,
        bikes_count INTEGER,
        trucks_count INTEGER,
        avg_waiting_time REAL,
        congestion_level TEXT, -- 'Low', 'Moderate', 'High', 'Critical'
        current_signal TEXT,   -- 'GREEN', 'YELLOW', 'RED'
        allocated_green INTEGER,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # Table: System Configuration
    c.execute("""
    CREATE TABLE IF NOT EXISTS system_config (
        id INTEGER PRIMARY KEY,
        yolo_confidence REAL,
        yolo_fps INTEGER,
        yolo_classes TEXT,
        rl_algorithm TEXT,
        rl_queue_weight REAL,
        rl_delay_weight REAL,
        hardware_protocol TEXT,
        hardware_port TEXT,
        hardware_baudrate INTEGER,
        min_green_time INTEGER,
        max_green_time INTEGER,
        yellow_duration INTEGER,
        all_red_clearance INTEGER,
        pedestrian_phase_duration INTEGER,
        emergency_preemption_enabled INTEGER,
        fallback_fixed_timer INTEGER,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # Table: Historical Metrics (24-hour sample data)
    c.execute("""
    CREATE TABLE IF NOT EXISTS historical_metrics (
        id INTEGER PRIMARY KEY,
        hour_label TEXT,
        fixed_waiting_time REAL,
        adaptive_waiting_time REAL,
        fixed_queue_length REAL,
        adaptive_queue_length REAL,
        throughput_count INTEGER,
        emissions_saved_kg REAL
    )
    """)

    # Table: Event Logs
    c.execute("""
    CREATE TABLE IF NOT EXISTS event_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        level TEXT, -- 'INFO', 'WARNING', 'ALERT', 'ACTION'
        source TEXT,
        message TEXT
    )
    """)

    # Safely add Computer Vision telemetry columns if existing database table is present
    for col, col_type, default in [
        ("cv_source", "TEXT", "'WAITING'"),
        ("total_detected_vehicles", "INTEGER", "0"),
        ("cv_status", "TEXT", "'IDLE'"),
    ]:
        try:
            c.execute(f"ALTER TABLE junction_status ADD COLUMN {col} {col_type} DEFAULT {default}")
        except Exception:
            pass

    for col, col_type, default in [
        ("unique_crossed", "INTEGER", "0"),
        ("currently_in_frame", "INTEGER", "0"),
        ("data_source", "TEXT", "'WAITING'"),
    ]:
        try:
            c.execute(f"ALTER TABLE approach_metrics ADD COLUMN {col} {col_type} DEFAULT {default}")
        except Exception:
            pass

    conn.commit()

    # Seed Initial Data if empty
    c.execute("SELECT COUNT(*) FROM junction_status")
    if c.fetchone()[0] == 0:
        seed_data(conn)

    conn.close()

def seed_data(conn):
    c = conn.cursor()

    # Initial Junction Status
    c.execute("""
    INSERT INTO junction_status 
    (id, junction_name, controller_mode, active_phase, active_phase_name, active_direction, current_green_remaining, current_green_total, cycle_time, total_vehicles_cleared, average_flow_rate, emergency_preempt)
    VALUES (1, 'Junction 01 - Main Cross', 'ADAPTIVE_HYBRID', 'PHASE_A_NS', 'Phase A: North-South Flow', 'North-South', 22, 45, 78, 3422, 42, 0)
    """)

    # Initial Approach Metrics
    approaches = [
        ('North', 4, 18, 4, 11, 2, 4, 1, 16.4, 'High', 'GREEN', 38),
        ('South', 3, 14, 3, 8, 1, 4, 1, 19.8, 'High', 'GREEN', 38),
        ('East', 5, 5, 5, 3, 0, 2, 0, 11.2, 'Light', 'RED', 22),
        ('West', 7, 7, 7, 4, 1, 2, 0, 13.5, 'Moderate', 'RED', 22),
    ]
    for app in approaches:
        c.execute("""
        INSERT INTO approach_metrics (direction, queue_count, unique_crossed, currently_in_frame, cars_count, buses_count, bikes_count, trucks_count, avg_waiting_time, congestion_level, current_signal, allocated_green)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, app)

    # Initial System Config
    c.execute("""
    INSERT INTO system_config 
    (id, yolo_confidence, yolo_fps, yolo_classes, rl_algorithm, rl_queue_weight, rl_delay_weight, hardware_protocol, hardware_port, hardware_baudrate, min_green_time, max_green_time, yellow_duration, all_red_clearance, pedestrian_phase_duration, emergency_preemption_enabled, fallback_fixed_timer)
    VALUES (1, 0.50, 15, 'Car,Bus,Truck,Motorcycle', 'Hybrid (DQN + Webster Formula)', 0.70, 0.30, 'NEMA TS2 / GPIO Relay', 'COM3', 115200, 15, 60, 4, 2, 15, 1, 1)
    """)

    # 24-Hour Historical Comparison Data
    history_records = [
        ('00:00', 18.0, 8.5, 4.2, 1.8, 420, 12.5),
        ('02:00', 16.5, 7.0, 3.5, 1.2, 280, 10.2),
        ('04:00', 17.0, 8.0, 3.8, 1.5, 350, 11.0),
        ('06:00', 28.5, 14.2, 8.2, 3.6, 1250, 28.4),
        ('08:00', 48.2, 24.5, 18.5, 7.8, 2840, 58.6),  # Morning Peak
        ('10:00', 36.4, 19.2, 12.4, 5.2, 2100, 42.1),
        ('12:00', 32.0, 17.5, 10.8, 4.5, 1950, 38.0),
        ('14:00', 30.5, 16.8, 9.6, 4.2, 1820, 36.5),
        ('16:00', 39.8, 20.4, 13.8, 5.9, 2350, 46.2),
        ('18:00', 52.0, 26.8, 21.2, 8.6, 3120, 64.8),  # Evening Peak
        ('20:00', 38.5, 19.6, 13.0, 5.4, 2200, 44.0),
        ('22:00', 24.0, 11.8, 6.5, 2.8, 980, 21.5),
    ]
    for rec in history_records:
        c.execute("""
        INSERT INTO historical_metrics 
        (hour_label, fixed_waiting_time, adaptive_waiting_time, fixed_queue_length, adaptive_queue_length, throughput_count, emissions_saved_kg)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, rec)

    # Initial Event Logs
    logs = [
        ('15:42:10', 'INFO', 'Adaptive Controller', 'Phase A green extended +8s based on North approach queue pressure (18 vehicles)'),
        ('15:39:45', 'ALERT', 'Computer Vision (YOLO)', 'Emergency Vehicle (Ambulance) detected approaching South lane - preemption standby'),
        ('15:35:12', 'INFO', 'RL Optimization', 'Updated phase allocation policy: N-S weight 0.62, E-W weight 0.38'),
        ('15:30:00', 'INFO', 'System Monitor', 'Hourly Webster equilibrium reached. Total delay reduction: -47.2% vs baseline'),
        ('15:24:18', 'WARNING', 'Queue Sensor', 'East approach density momentary surge (+4 vehicles). Priority score adjusted.'),
        ('15:15:02', 'INFO', 'Hardware Interface', 'NEMA TS2 controller heartbeat OK (latency 12ms, 0 packet loss)'),
    ]
    for log in logs:
        c.execute("INSERT INTO event_logs (timestamp, level, source, message) VALUES (?, ?, ?, ?)", log)

    conn.commit()

if __name__ == "__main__":
    init_db()
    print("Database initialized successfully at", DB_PATH)
