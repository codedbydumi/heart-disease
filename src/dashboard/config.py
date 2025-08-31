"""Configuration management with Railway support."""

import os
from pathlib import Path
from typing import Optional
from pydantic import BaseSettings

class Settings(BaseSettings):
    """Application settings with Railway environment variable support."""
    
    # Application
    environment: str = os.getenv("RAILWAY_ENVIRONMENT", "production")
    log_level: str = "INFO"
    secret_key: str = os.getenv("SECRET_KEY", "change-this-in-production")
    
    # API Configuration - Railway sets PORT automatically
    api_host: str = "0.0.0.0"
    api_port: int = int(os.getenv("PORT", 8000))
    
    # Streamlit Configuration  
    streamlit_port: int = int(os.getenv("PORT", 8501))
    
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
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create necessary directories
        self.logs_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        (self.models_dir / "trained_models").mkdir(exist_ok=True)
        (self.models_dir / "scalers").mkdir(exist_ok=True)

    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        case_sensitive = False

# Global settings instance
settings = Settings()