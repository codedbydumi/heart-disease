"""Railway-compatible API start script."""

import sys
import os
from pathlib import Path

# Add project root to Python path (Railway environment)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

import uvicorn
from src.utils.logger import get_logger

logger = get_logger("railway_api_server")

def start_api_server():
    """Start the FastAPI server for Railway deployment."""
    # Get port from Railway environment variable
    port = int(os.environ.get("PORT", 8000))
    host = "0.0.0.0"
    
    # Set Railway-specific environment
    os.environ.setdefault("ENVIRONMENT", "production")
    os.environ.setdefault("LOG_LEVEL", "INFO")
    
    logger.info(f"Starting Heart Disease API for Railway deployment...")
    logger.info(f"Host: {host}:{port}")
    logger.info(f"Environment: {os.environ.get('ENVIRONMENT', 'production')}")
    
    try:
        # Import your existing API
        from src.api.main import app
        
        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level="info",
            access_log=True,
            reload=False  # No reload in production
        )
    except Exception as e:
        logger.error(f"Failed to start API server: {e}")
        raise

if __name__ == "__main__":
    start_api_server()