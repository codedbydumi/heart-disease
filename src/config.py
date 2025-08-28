"""Configuration management for Heart Disease Prediction System."""

import os
from pathlib import Path
from typing import Optional

try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Application
    environment: str = "development"
    log_level: str = "INFO"
    secret_key: str = "change-this-in-production"
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    # Streamlit Configuration
    streamlit_port: int = 8501
    
    # Database Configuration
    database_url: str = "sqlite:///./heart_disease.db"
    
    # Model Configuration
    model_version: str = "v1.0.0"
    model_path: str = "./models/trained_models/"
    
    # Paths
    project_root: Path = Path(__file__).parent.parent
    data_dir: Path = project_root / "data"
    models_dir: Path = project_root / "models"
    logs_dir: Path = project_root / "logs"
    
    # Logging
    log_file: str = "./logs/app.log"
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()

# Create necessary directories
settings.logs_dir.mkdir(exist_ok=True)
settings.models_dir.mkdir(exist_ok=True)
(settings.models_dir / "trained_models").mkdir(exist_ok=True)
(settings.models_dir / "scalers").mkdir(exist_ok=True)