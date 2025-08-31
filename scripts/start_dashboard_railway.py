"""Railway-compatible dashboard start script."""

import sys
import os
from pathlib import Path
import subprocess

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from src.utils.logger import get_logger

logger = get_logger("railway_dashboard_server")

def start_dashboard():
    """Start the Streamlit dashboard for Railway."""
    logger.info("Starting Heart Disease Risk Assessment Dashboard for Railway...")
    
    port = int(os.environ.get("PORT", 8501))
    
    try:
        # Use your working dashboard file but with Railway-specific settings
        subprocess.run([
            "streamlit", "run", 
            "simple_dashboard.py",  # Use the simple working version
            "--server.port", str(port),
            "--server.address", "0.0.0.0",
            "--server.headless", "true",
            "--server.enableCORS", "false",
            "--server.enableXsrfProtection", "false",
            "--theme.base", "light",
            "--theme.primaryColor", "#2E86AB"
        ])
        
    except Exception as e:
        logger.error(f"Failed to start dashboard: {e}")
        
        # Fallback to basic streamlit run
        try:
            subprocess.run([
                "streamlit", "run", 
                "simple_dashboard.py",
                "--server.port", str(port),
                "--server.address", "0.0.0.0"
            ])
        except Exception as e2:
            logger.error(f"Fallback also failed: {e2}")
            raise

if __name__ == "__main__":
    start_dashboard()