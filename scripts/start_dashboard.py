"""Script to start the Streamlit dashboard."""

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
    """Start the Streamlit dashboard."""
    logger.info("Starting Heart Disease Risk Assessment Dashboard...")
    
    try:
        # Start Streamlit app
        subprocess.run([
            "streamlit", "run", 
            "src/dashboard/app.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0",
            "--theme.base", "light",
            "--theme.primaryColor", "#2E86AB",
            "--theme.backgroundColor", "#f8f9fa",
            "--theme.secondaryBackgroundColor", "#ffffff"
        ])
        
    except Exception as e:
        logger.error(f"Failed to start dashboard: {e}")
        raise


if __name__ == "__main__":
    start_dashboard()