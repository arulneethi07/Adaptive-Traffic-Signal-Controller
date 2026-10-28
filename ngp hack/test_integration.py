"""
Automated End-to-End Integration Test for SmartFlow AI + Computer Vision Module.
"""
import os
import sys
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(BASE_DIR, "backend")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from starlette.testclient import TestClient
# Explicitly import from backend app
import app as backend_app
app = backend_app.app
cv_service = backend_app.cv_service
controller = backend_app.controller

def run_tests():
    print("=" * 60)
    print("RUNNING SMARTFLOW AI + CV MODULE INTEGRATION TESTS")
    print("=" * 60)

    client = TestClient(app)

    # 1. Test GET /api/status
    print("\n[1] Testing GET /api/status ...")
    res = client.get("/api/status")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    data = res.json()
    assert "status" in data
    assert "approaches" in data
    assert "cv_service" in data
    print(f" -> Success! Status: {data['status']['active_phase_name']}, CV Source: {data['status']['cv_source']}")

    # 2. Test POST /api/cv/demo (Load Demo Clips)
    print("\n[2] Testing POST /api/cv/demo (North - highway) ...")
    res = client.post("/api/cv/demo", json={"approach": "North", "clip": "highway"})
    assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
    demo_data = res.json()
    assert demo_data["success"] is True
    print(f" -> Success! Loaded: {demo_data['filename']} for {demo_data['approach']}")

    print("\n[2b] Testing POST /api/cv/demo (South - synthetic) ...")
    res = client.post("/api/cv/demo", json={"approach": "South", "clip": "synthetic"})
    assert res.status_code == 200
    print(" -> Success! South demo loaded.")

    # 3. Test POST /api/cv/calibrate
    print("\n[3] Testing POST /api/cv/calibrate ...")
    res = client.post("/api/cv/calibrate", json={"approach": "North", "line_height": 0.65, "direction": "any"})
    assert res.status_code == 200
    cal_data = res.json()
    assert cal_data["line_height"] == 0.65
    print(f" -> Success! Calibrated North line height: {cal_data['line_height']}")

    # 4. Test GET /api/cv/frame/North
    print("\n[4] Testing GET /api/cv/frame/North ...")
    res = client.get("/api/cv/frame/North")
    assert res.status_code in [200, 204], f"Unexpected status code: {res.status_code}"
    if res.status_code == 200:
        assert res.headers["content-type"] == "image/jpeg"
        print(f" -> Success! Received JPEG frame ({len(res.content)} bytes)")
    else:
        print(" -> Frame buffer empty (status 204)")

    # 5. Test POST /api/cv/mode (Simulation fallback toggle)
    print("\n[5] Testing POST /api/cv/mode ...")
    res = client.post("/api/cv/mode", json={"is_simulation": True})
    assert res.status_code == 200
    assert res.json()["is_simulation"] is True

    # 6. Test POST /api/cv/simulate (Set simulated counts)
    print("\n[6] Testing POST /api/cv/simulate ...")
    res = client.post("/api/cv/simulate", json={"North": 22, "South": 16, "East": 9, "West": 7})
    assert res.status_code == 200
    counts = res.json()["counts"]
    print(f" -> Success! Set simulated counts: {counts}")

    # 7. Test GET /api/cv/status
    print("\n[7] Testing GET /api/cv/status ...")
    res = client.get("/api/cv/status")
    assert res.status_code == 200
    cv_st = res.json()
    assert cv_st["is_simulation"] is True
    assert cv_st["total_unique_counted"] == 54  # 22+16+9+7
    print(f" -> Success! Mode: {cv_st['mode']}, Total Counted: {cv_st['total_unique_counted']}")

    # Verify that controller updated allocated green times dynamically
    status_now = controller.get_status()
    print(f" -> Controller Data Source: {status_now['status']['cv_source']}")
    app_metrics = {a['direction']: a['allocated_green'] for a in status_now['approaches']}
    print(f" -> Dynamic Webster Allocated Greens: {app_metrics}")
    assert app_metrics['North'] > app_metrics['West'], "North with 22 veh should get more green than West with 7 veh"

    # 8. Test GET /api/cv/export
    print("\n[8] Testing GET /api/cv/export ...")
    res = client.get("/api/cv/export")
    assert res.status_code == 200
    export_json = res.json()
    assert "approaches" in export_json or "junction_status" in export_json or "status" in export_json
    print(f" -> Success! Exported JSON payload contains {len(export_json)} keys")

    # 9. Test POST /api/cv/reset
    print("\n[9] Testing POST /api/cv/reset ...")
    res = client.post("/api/cv/reset")
    assert res.status_code == 200
    print(" -> Success! Reset counts complete.")

    # 10. Switch back to Real Video Mode
    client.post("/api/cv/mode", json={"is_simulation": False})

    # 11. Test Short Processing Run
    print("\n[11] Testing POST /api/cv/process (North - 20 frames) ...")
    res = client.post("/api/cv/process", json={"approach": "North", "max_frames": 20})
    assert res.status_code == 200
    print(f" -> Processing trigger response: {res.json()}")

    import time
    time.sleep(1.5)
    st = cv_service.get_status()
    print(f" -> Worker active: {st['is_processing']}, approach: {st['active_approach']}")

    # Stop processing
    client.post("/api/cv/stop")
    time.sleep(0.5)
    print(" -> Stopped worker successfully.")

    print("\n" + "=" * 60)
    print("ALL INTEGRATION TESTS PASSED PERFECTLY (100% SUCCESS)!")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()
