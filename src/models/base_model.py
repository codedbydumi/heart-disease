"""Base model class with common functionality for all ML models."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from loguru import logger

from ..config import settings
from ..utils.helpers import save_pickle, load_pickle, save_json


class BaseModel(ABC):
    """Abstract base class for heart disease prediction models."""
    
    def __init__(self, model_name: str):
        """Initialize base model."""
        self.model_name = model_name
        self.model = None
        self.is_trained = False
        self.training_history = {}
        self.feature_names = []
        self.model_metadata = {}
        
        logger.info(f"Initialized {model_name} model")
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Fit the model to training data."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data."""
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        pass
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance."""
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        # Get predictions
        y_pred = self.predict(X)
        y_pred_proba = self.predict_proba(X)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'f1_score': f1_score(y, y_pred),
            'roc_auc': roc_auc_score(y, y_pred_proba)
        }
        
        # Medical-specific metrics
        metrics.update(self._calculate_medical_metrics(y, y_pred, y_pred_proba))
        
        return metrics
    
    def _calculate_medical_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
        """Calculate medical-specific metrics."""
        from sklearn.metrics import confusion_matrix
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Medical metrics
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Same as recall
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Positive Predictive Value
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
        
        return {
            'sensitivity': sensitivity,
            'specificity': specificity,
            'ppv': ppv,
            'npv': npv,
            'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'false_negative_rate': fn / (tp + fn) if (tp + fn) > 0 else 0
        }
    
    def save_model(self, file_path: Optional[str] = None) -> str:
        """Save trained model to file."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        if file_path is None:
            file_path = settings.models_dir / "trained_models" / f"{self.model_name}_model.pkl"
        
        model_data = {
            'model': self.model,
            'model_name': self.model_name,
            'is_trained': self.is_trained,
            'training_history': self.training_history,
            'feature_names': self.feature_names,
            'model_metadata': self.model_metadata
        }
        
        save_pickle(model_data, file_path)
        logger.info(f"Model saved to {file_path}")
        
        return str(file_path)
    
    def load_model(self, file_path: str) -> None:
        """Load trained model from file."""
        model_data = load_pickle(file_path)
        
        self.model = model_data['model']
        self.model_name = model_data['model_name']
        self.is_trained = model_data['is_trained']
        self.training_history = model_data['training_history']
        self.feature_names = model_data['feature_names']
        self.model_metadata = model_data['model_metadata']
        
        logger.info(f"Model loaded from {file_path}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        return {
            'model_name': self.model_name,
            'is_trained': self.is_trained,
            'model_type': type(self.model).__name__ if self.model else None,
            'feature_count': len(self.feature_names),
            'training_history': self.training_history,
            'metadata': self.model_metadata
        }