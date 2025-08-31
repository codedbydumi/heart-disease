"""Script to start the FastAPI server for Railway deployment."""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

import uvicorn
from src.config import settings
from src.utils.logger import get_logger

logger = get_logger("api_server")

def start_api_server():
    """Start the FastAPI server with Railway configuration."""
    # Railway provides PORT environment variable
    port = int(os.environ.get("PORT", 8000))
    
    logger.info(f"Starting Heart Disease Prediction API server on port {port}")
    logger.info(f"Environment: {settings.environment}")
    
    try:
        uvicorn.run(
            "src.api.main:app",
            host="0.0.0.0",
            port=port,
            reload=False,  # Disable reload in production
            log_level=settings.log_level.lower(),
            access_log=True
        )
    except Exception as e:
        logger.error(f"Failed to start API server: {e}")
        raise

if __name__ == "__main__":
    start_api_server()