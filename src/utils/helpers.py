"""General utility functions for the application."""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from loguru import logger


def load_json(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load JSON data from file.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Loaded JSON data as dictionary
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        logger.info(f"Successfully loaded JSON from {file_path}")
        return data
    except Exception as e:
        logger.error(f"Error loading JSON from {file_path}: {e}")
        raise


def save_json(data: Dict[str, Any], file_path: Union[str, Path]) -> None:
    """
    Save data to JSON file.
    
    Args:
        data: Data to save
        file_path: Path to save JSON file
    """
    try:
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"Successfully saved JSON to {file_path}")
    except Exception as e:
        logger.error(f"Error saving JSON to {file_path}: {e}")
        raise


def load_pickle(file_path: Union[str, Path]) -> Any:
    """
    Load pickled object from file.
    
    Args:
        file_path: Path to pickle file
        
    Returns:
        Loaded object
    """
    try:
        with open(file_path, 'rb') as f:
            obj = pickle.load(f)
        logger.info(f"Successfully loaded pickle from {file_path}")
        return obj
    except Exception as e:
        logger.error(f"Error loading pickle from {file_path}: {e}")
        raise


def save_pickle(obj: Any, file_path: Union[str, Path]) -> None:
    """
    Save object to pickle file.
    
    Args:
        obj: Object to save
        file_path: Path to save pickle file
    """
    try:
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(obj, f)
        logger.info(f"Successfully saved pickle to {file_path}")
    except Exception as e:
        logger.error(f"Error saving pickle to {file_path}: {e}")
        raise


def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """
    Validate that DataFrame has required columns.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        
    Returns:
        True if valid, raises ValueError if not
    """
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    logger.info(f"DataFrame validation passed - Shape: {df.shape}")
    return True


def get_model_info(model_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Get information about a saved model.
    
    Args:
        model_path: Path to model file
        
    Returns:
        Dictionary with model information
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    info = {
        "model_path": str(model_path),
        "file_size_mb": model_path.stat().st_size / (1024 * 1024),
        "last_modified": model_path.stat().st_mtime,
        "exists": True
    }
    
    return info