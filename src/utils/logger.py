"""Professional logging configuration for the application."""

import sys
from pathlib import Path
from typing import Optional

from loguru import logger

from ..config import settings


def setup_logging(
    log_level: str = settings.log_level,
    log_file: Optional[str] = settings.log_file
) -> None:
    """
    Configure application logging.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
    """
    # Remove default handler
    logger.remove()
    
    # Console handler with colors
    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
               "<level>{message}</level>",
        colorize=True
    )
    
    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            log_file,
            level=log_level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
            rotation="10 MB",
            retention="1 month",
            compression="zip"
        )
    
    logger.info(f"Logging configured - Level: {log_level}, File: {log_file}")


def get_logger(name: str):
    """Get a logger instance with the given name."""
    return logger.bind(name=name)


# Initialize logging when module is imported
setup_logging()