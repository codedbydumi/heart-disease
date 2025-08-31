"""Script to start the Streamlit dashboard for Railway deployment."""

import sys
import os
from pathlib import Path
import subprocess

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from src.utils.logger import get_logger

logger = get_logger("dashboard_server")

def start_dashboard():
    """Start the Streamlit dashboard with Railway configuration."""
    # Railway provides PORT environment variable
    port = int(os.environ.get("PORT", 8501))
    
    logger.info(f"Starting Heart Disease Risk Assessment Dashboard on port {port}")
    
    try:
        # Start Streamlit app with Railway-compatible settings
        subprocess.run([
            "streamlit", "run", 
            "simple_dashboard.py",  # Use your working simple dashboard
            "--server.port", str(port),
            "--server.address", "0.0.0.0",
            "--server.headless", "true",
            "--server.enableCORS", "false",
            "--server.enableXsrfProtection", "false"
        ])
        
    except Exception as e:
        logger.error(f"Failed to start dashboard: {e}")
        raise

if __name__ == "__main__":
    start_dashboard()