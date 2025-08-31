"""Railway start script - uses your working local setup."""

import os
import sys
from pathlib import Path
import threading
import time

# Set up paths exactly like your working local setup
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

def start_api():
    """Start the API server using your working script logic."""
    try:
        # Import your working API start logic
        from scripts.start_api import start_api_server
        start_api_server()
    except ImportError:
        # Fallback to direct uvicorn start
        import uvicorn
        from src.api.main import app
        port = int(os.environ.get("PORT", 8000))
        uvicorn.run(app, host="0.0.0.0", port=port)

def start_dashboard():
    """Start the dashboard using your working script logic."""
    try:
        # Import your working dashboard start logic  
        import subprocess
        subprocess.run([
            "streamlit", "run", 
            "src/dashboard/app.py",
            "--server.port", "8502",  # Different port to avoid conflict
            "--server.address", "0.0.0.0",
            "--server.headless", "true",
            "--server.enableCORS", "false"
        ])
    except Exception as e:
        print(f"Dashboard start failed: {e}")
        # Fallback to simple dashboard
        subprocess.run([
            "streamlit", "run", 
            "simple_dashboard.py",
            "--server.port", "8502",
            "--server.address", "0.0.0.0", 
            "--server.headless", "true"
        ])

if __name__ == "__main__":
    # Railway only supports single port, so we'll start API on main port
    # and provide API endpoints for the functionality
    
    # Check if we should start both or just API
    if os.environ.get("RAILWAY_ENVIRONMENT"):
        # On Railway, just start the API
        print("Starting API server for Railway...")
        start_api()
    else:
        # Local development - start both
        print("Starting both API and Dashboard for local development...")
        
        # Start dashboard in background thread
        dashboard_thread = threading.Thread(target=start_dashboard)
        dashboard_thread.daemon = True
        dashboard_thread.start()
        
        # Wait a moment for dashboard to start
        time.sleep(2)
        
        # Start API in main thread
        start_api()