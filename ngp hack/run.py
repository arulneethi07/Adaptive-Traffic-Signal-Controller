import os
import sys
import webbrowser
import uvicorn

# Ensure backend directory is in path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

if __name__ == "__main__":
    port = 8000
    host = "127.0.0.1"
    url = f"http://{host}:{port}"
    print("=" * 60)
    print(" SMARTFLOW - ADAPTIVE TRAFFIC SIGNAL CONTROLLER")
    print(f" Server starting on: {url}")
    print("=" * 60)
    
    # Try opening browser after a brief delay
    try:
        webbrowser.open(url)
    except Exception:
        pass

    from app import app
    uvicorn.run(app, host=host, port=port, log_level="info")
