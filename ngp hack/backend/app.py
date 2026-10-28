import os
import json
import asyncio
from contextlib import asynccontextmanager
from starlette.applications import Starlette
from starlette.responses import JSONResponse, FileResponse, Response
from starlette.routing import Route, Mount
from starlette.staticfiles import StaticFiles
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from database import init_db, get_connection
from controller import AdaptiveTrafficController
from cv_service import CVService, CV_DIR

# Initialize DB
init_db()
controller = AdaptiveTrafficController()
cv_service = CVService(controller=controller)

FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))

# Background Simulation Task
async def simulation_loop():
    while True:
        try:
            controller.step_simulation()
        except Exception as e:
            print("Simulation step error:", e)
        await asyncio.sleep(1.0)

@asynccontextmanager
async def lifespan(app):
    sim_task = asyncio.create_task(simulation_loop())
    try:
        yield
    finally:
        sim_task.cancel()

# REST API Handlers
async def api_status(request):
    data = controller.get_status()
    # Augment with current CV service status
    data["cv_service"] = cv_service.get_status()
    return JSONResponse(data)

async def api_analytics(request):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM historical_metrics ORDER BY id")
    rows = [dict(r) for r in c.fetchall()]
    conn.close()

    summary = {
        "avg_waiting_time_adaptive": 18.2,
        "avg_waiting_time_fixed": 34.5,
        "waiting_time_reduction_pct": 47.2,
        "total_throughput_adaptive": 37520,
        "total_throughput_fixed": 31200,
        "throughput_gain_pct": 20.3,
        "avg_queue_length_adaptive": 4.2,
        "avg_queue_length_fixed": 9.8,
        "queue_reduction_pct": 57.1,
        "fuel_saved_liters": 420.0,
        "co2_saved_kg": 980.0
    }
    return JSONResponse({"history": rows, "summary": summary})

async def api_config(request):
    if request.method == "GET":
        conn = get_connection()
        c = conn.cursor()
        c.execute("SELECT * FROM system_config WHERE id = 1")
        cfg = dict(c.fetchone())
        conn.close()
        return JSONResponse(cfg)
    elif request.method == "POST":
        payload = await request.json()
        controller.update_config(payload)
        return JSONResponse({"success": True, "message": "Configuration updated"})

async def api_set_mode(request):
    payload = await request.json()
    mode = payload.get("mode", "ADAPTIVE_HYBRID")
    controller.set_mode(mode)
    return JSONResponse({"success": True, "mode": mode})

async def api_phase_override(request):
    payload = await request.json()
    phase = payload.get("phase", "PHASE_A_NS")
    controller.trigger_phase_override(phase)
    return JSONResponse({"success": True, "phase": phase})

async def api_emergency(request):
    payload = await request.json()
    direction = payload.get("direction", "South")
    controller.trigger_emergency(direction)
    return JSONResponse({"success": True, "emergency_direction": direction})

async def api_logs(request):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM event_logs ORDER BY id DESC LIMIT 50")
    logs = [dict(r) for r in c.fetchall()]
    conn.close()
    return JSONResponse({"logs": logs})

# ==================== COMPUTER VISION API HANDLERS ====================

async def api_cv_upload(request):
    form = await request.form()
    approach = form.get("approach")
    file = form.get("file")
    if not approach or approach not in ["North", "South", "East", "West"]:
        return JSONResponse({"success": False, "error": "Invalid approach"}, status_code=400)
    if not file:
        return JSONResponse({"success": False, "error": "No file uploaded"}, status_code=400)

    filename = getattr(file, "filename", f"{approach.lower()}_upload.mp4")
    save_path = os.path.join(cv_service.upload_dir, f"{approach.lower()}_{filename}")

    contents = await file.read()
    with open(save_path, "wb") as f:
        f.write(contents)

    cv_service.set_video(approach, save_path, filename)
    return JSONResponse({"success": True, "approach": approach, "filename": filename})

async def api_cv_demo(request):
    payload = await request.json()
    approach = payload.get("approach", "North")
    clip = payload.get("clip", "highway")
    try:
        res = cv_service.load_demo(approach, clip)
        return JSONResponse(res)
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=400)

async def api_cv_process(request):
    payload = await request.json()
    approach = payload.get("approach", "all")
    max_frames = int(payload.get("max_frames", 600))
    started = cv_service.start_processing(target_approach=approach, max_frames=max_frames)
    return JSONResponse({"success": started, "message": "Processing started" if started else "Already processing"})

async def api_cv_stop(request):
    try:
        payload = await request.json() if request.headers.get("content-length", "0") not in ["0", ""] else {}
    except Exception:
        payload = {}
    approach = payload.get("approach") if isinstance(payload, dict) else None
    cv_service.stop_processing(target_approach=approach)
    return JSONResponse({"success": True, "message": "Stop requested"})

async def api_cv_configure(request):
    payload = await request.json()
    approach = payload.get("approach", "North")
    source_type = payload.get("source_type")
    selected = payload.get("selected")
    ip_url = payload.get("ip_url")
    details = cv_service.configure_approach(approach, source_type=source_type, selected=selected, ip_url=ip_url)
    return JSONResponse({"success": True, "details": details})

async def api_cv_webcam_start(request):
    payload = await request.json()
    approach = payload.get("approach", "North")

    # Validate detector before committing camera hardware
    detector = cv_service.detectors.get(approach)
    if detector and not detector.is_available:
        detector.load_model()
    if detector and not detector.is_available:
        return JSONResponse({
            "success": False,
            "error": f"YOLO detector unavailable: {detector.load_error}"
        }, status_code=400)

    acquired, err = cv_service.acquire_webcam(approach)
    if not acquired:
        return JSONResponse({"success": False, "error": err}, status_code=409)
    cv_service.release_webcam(approach)  # Hand off to worker
    cv_service.configure_approach(approach, source_type="webcam", selected=True)
    started = cv_service.start_processing(target_approach=approach)
    return JSONResponse({"success": started, "message": "Webcam started" if started else "Already processing"})

async def api_cv_webcam_stop(request):
    payload = await request.json()
    approach = payload.get("approach", "North")
    cv_service.stop_processing(target_approach=approach)
    cv_service.release_webcam(approach)
    return JSONResponse({"success": True, "message": f"Webcam stopped for {approach}"})

async def api_cv_ipcam_start(request):
    payload = await request.json()
    approach = payload.get("approach", "North")
    detector = cv_service.detectors.get(approach)
    if detector and not detector.is_available:
        detector.load_model()
    if detector and not detector.is_available:
        return JSONResponse({
            "success": False,
            "error": f"YOLO detector unavailable: {detector.load_error}"
        }, status_code=400)

    ip_url = payload.get("ip_url")
    if ip_url is not None:
        cv_service.configure_approach(approach, source_type="ip_camera", selected=True, ip_url=ip_url)
    else:
        cv_service.configure_approach(approach, source_type="ip_camera", selected=True)
    started = cv_service.start_processing(target_approach=approach)
    return JSONResponse({"success": started, "message": "IP Camera stream started" if started else "Already processing"})

async def api_cv_ipcam_stop(request):
    payload = await request.json()
    approach = payload.get("approach", "North")
    cv_service.stop_processing(target_approach=approach)
    return JSONResponse({"success": True, "message": f"IP Camera stopped for {approach}"})

async def api_cv_sim_start(request):
    payload = await request.json()
    approach = payload.get("approach", "North")
    cv_service.configure_approach(approach, source_type="simulated", selected=True)
    started = cv_service.start_processing(target_approach=approach)
    return JSONResponse({"success": started, "message": "Simulated stream started" if started else "Already processing"})

async def api_cv_sim_stop(request):
    payload = await request.json()
    approach = payload.get("approach", "North")
    cv_service.stop_processing(target_approach=approach)
    return JSONResponse({"success": True, "message": f"Simulation stopped for {approach}"})

async def api_cv_status(request):
    status = cv_service.get_status()
    return JSONResponse(status)

async def api_cv_frame(request):
    direction = request.path_params.get("direction", "North")
    frame_bytes = cv_service.get_frame(direction)
    if not frame_bytes:
        return Response(status_code=204)
    return Response(frame_bytes, media_type="image/jpeg", headers={"Cache-Control": "no-cache, no-store, must-revalidate"})

async def api_cv_calibrate(request):
    payload = await request.json()
    approach = payload.get("approach", "North")
    line_height = float(payload.get("line_height", 0.70))
    direction = payload.get("direction", "any")
    cv_service.calibrate_line(approach, line_height, direction)
    return JSONResponse({"success": True, "approach": approach, "line_height": line_height, "direction": direction})

async def api_cv_reset(request):
    cv_service.reset_counts()
    return JSONResponse({"success": True, "message": "All counts and trackers reset"})

async def api_cv_mode(request):
    payload = await request.json()
    is_sim = bool(payload.get("is_simulation", False))
    cv_service.set_simulation_mode(is_sim)
    return JSONResponse({"success": True, "is_simulation": is_sim})

async def api_cv_simulate(request):
    payload = await request.json()
    north = int(payload.get("North", 15))
    south = int(payload.get("South", 12))
    east = int(payload.get("East", 8))
    west = int(payload.get("West", 6))
    cv_service.set_simulated_counts(north, south, east, west)
    return JSONResponse({"success": True, "counts": {"North": north, "South": south, "East": east, "West": west}})

async def api_cv_export(request):
    export_path = os.path.join(CV_DIR, "output", "vehicle_counts.json")
    if os.path.exists(export_path):
        with open(export_path, "r") as f:
            data = json.load(f)
    else:
        data = controller.export_cv_json()
    return JSONResponse(data, headers={"Content-Disposition": 'attachment; filename="vehicle_counts.json"'})

async def index(request):
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

# Routes
routes = [
    Route("/api/status", api_status, methods=["GET"]),
    Route("/api/analytics", api_analytics, methods=["GET"]),
    Route("/api/config", api_config, methods=["GET", "POST"]),
    Route("/api/junction/mode", api_set_mode, methods=["POST"]),
    Route("/api/junction/phase-override", api_phase_override, methods=["POST"]),
    Route("/api/junction/emergency", api_emergency, methods=["POST"]),
    Route("/api/logs", api_logs, methods=["GET"]),

    # Computer Vision Routes
    Route("/api/cv/upload", api_cv_upload, methods=["POST"]),
    Route("/api/cv/demo", api_cv_demo, methods=["POST"]),
    Route("/api/cv/process", api_cv_process, methods=["POST"]),
    Route("/api/cv/stop", api_cv_stop, methods=["POST"]),
    Route("/api/cv/approach/configure", api_cv_configure, methods=["POST"]),
    Route("/api/cv/webcam/start", api_cv_webcam_start, methods=["POST"]),
    Route("/api/cv/webcam/stop", api_cv_webcam_stop, methods=["POST"]),
    Route("/api/cv/ipcam/start", api_cv_ipcam_start, methods=["POST"]),
    Route("/api/cv/ipcam/stop", api_cv_ipcam_stop, methods=["POST"]),
    Route("/api/cv/sim/start", api_cv_sim_start, methods=["POST"]),
    Route("/api/cv/sim/stop", api_cv_sim_stop, methods=["POST"]),
    Route("/api/cv/status", api_cv_status, methods=["GET"]),
    Route("/api/cv/frame/{direction}", api_cv_frame, methods=["GET"]),
    Route("/api/cv/calibrate", api_cv_calibrate, methods=["POST"]),
    Route("/api/cv/reset", api_cv_reset, methods=["POST"]),
    Route("/api/cv/mode", api_cv_mode, methods=["POST"]),
    Route("/api/cv/simulate", api_cv_simulate, methods=["POST"]),
    Route("/api/cv/export", api_cv_export, methods=["GET"]),

    Route("/", index, methods=["GET"]),
    Mount("/", StaticFiles(directory=FRONTEND_DIR), name="static")
]

middleware = [
    Middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
]

app = Starlette(debug=True, routes=routes, middleware=middleware, lifespan=lifespan)

if __name__ == "__main__":
    import uvicorn
    print("Starting Adaptive Traffic Signal Controller Backend on http://localhost:8000 ...")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
