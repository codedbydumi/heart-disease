"""Ensemble model combining Random Forest, XGBoost, and Logistic Regression."""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import xgboost as xgb
from loguru import logger

from .base_model import BaseModel


class HeartDiseaseEnsemble(BaseModel):
    """Ensemble model for heart disease prediction."""
    
    def __init__(self, 
                 rf_params: Optional[Dict] = None,
                 xgb_params: Optional[Dict] = None,
                 lr_params: Optional[Dict] = None,
                 ensemble_method: str = 'voting'):
        """
        Initialize ensemble model.
        
        Args:
            rf_params: Random Forest parameters
            xgb_params: XGBoost parameters  
            lr_params: Logistic Regression parameters
            ensemble_method: Method to combine predictions ('voting' or 'stacking')
        """
        super().__init__("heart_disease_ensemble")
        
        # Default parameters
        self.rf_params = rf_params or {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42
        }
        
        self.xgb_params = xgb_params or {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
        
        self.lr_params = lr_params or {
            'random_state': 42,
            'max_iter': 1000
        }
        
        self.ensemble_method = ensemble_method
        
        # Initialize individual models
        self.rf_model = RandomForestClassifier(**self.rf_params)
        self.xgb_model = xgb.XGBClassifier(**self.xgb_params)
        self.lr_model = LogisticRegression(**self.lr_params)
        
        self.models = {
            'random_forest': self.rf_model,
            'xgboost': self.xgb_model,
            'logistic_regression': self.lr_model
        }
        
        # Ensemble weights (learned during training)
        self.model_weights = {'random_forest': 1/3, 'xgboost': 1/3, 'logistic_regression': 1/3}
        
        logger.info(f"Ensemble model initialized with {len(self.models)} base models")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train all models in the ensemble."""
        logger.info("Training ensemble model...")
        
        training_results = {}
        
        # Train each model individually
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            
            # Cross-validation before final training
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
            
            # Train on full dataset
            model.fit(X, y)
            
            training_results[name] = {
                'cv_mean_accuracy': cv_scores.mean(),
                'cv_std_accuracy': cv_scores.std(),
                'cv_scores': cv_scores.tolist()
            }
            
            logger.info(f"{name} - CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Calculate ensemble weights based on performance
        self._calculate_ensemble_weights(training_results)
        
        # Store training information
        self.training_history = training_results
        self.is_trained = True
        
        # Store model metadata
        self.model_metadata = {
            'ensemble_method': self.ensemble_method,
            'model_weights': self.model_weights,
            'base_models': list(self.models.keys()),
            'rf_params': self.rf_params,
            'xgb_params': self.xgb_params,
            'lr_params': self.lr_params
        }
        
        logger.info("Ensemble training completed")
        
        return training_results
    
    def _calculate_ensemble_weights(self, training_results: Dict[str, Any]) -> None:
        """Calculate weights for ensemble based on cross-validation performance."""
        accuracies = {name: results['cv_mean_accuracy'] for name, results in training_results.items()}
        
        # Simple performance-based weighting
        total_accuracy = sum(accuracies.values())
        self.model_weights = {name: acc / total_accuracy for name, acc in accuracies.items()}
        
        logger.info(f"Ensemble weights: {self.model_weights}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make ensemble predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        predictions = {}
        
        # Get predictions from each model
        for name, model in self.models.items():
            predictions[name] = model.predict_proba(X)[:, 1]  # Probability of positive class
        
        # Combine predictions using weights
        ensemble_proba = np.zeros(X.shape[0])
        for name, proba in predictions.items():
            ensemble_proba += self.model_weights[name] * proba
        
        # Convert probabilities to class predictions
        return (ensemble_proba > 0.5).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get ensemble prediction probabilities."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        predictions = {}
        
        # Get predictions from each model
        for name, model in self.models.items():
            predictions[name] = model.predict_proba(X)
        
        # Combine predictions using weights
        ensemble_proba = np.zeros_like(predictions[list(predictions.keys())[0]])
        for name, proba in predictions.items():
            ensemble_proba += self.model_weights[name] * proba
        
        return ensemble_proba
    
    def get_individual_predictions(self, X: np.ndarray) -> Dict[str, Dict[str, np.ndarray]]:
        """Get predictions from individual models for analysis."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        results = {}
        for name, model in self.models.items():
            results[name] = {
                'predictions': model.predict(X),
                'probabilities': model.predict_proba(X)[:, 1]
            }
        
        return results
    
    def get_feature_importance(self) -> Dict[str, np.ndarray]:
        """Get feature importance from models that support it."""
        importance_dict = {}
        
        if hasattr(self.rf_model, 'feature_importances_'):
            importance_dict['random_forest'] = self.rf_model.feature_importances_
        
        if hasattr(self.xgb_model, 'feature_importances_'):
            importance_dict['xgboost'] = self.xgb_model.feature_importances_
        
        # For logistic regression, use absolute coefficients
        if hasattr(self.lr_model, 'coef_'):
            importance_dict['logistic_regression'] = np.abs(self.lr_model.coef_[0])
        
        return importance_dict
    
    def evaluate_individual_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Evaluate each model individually."""
        results = {}
        
        for name, model in self.models.items():
            y_pred = model.predict(X)
            y_pred_proba = model.predict_proba(X)[:, 1]
            
            # Create temporary model instance for evaluation
            temp_model = BaseModel(name)
            temp_model.model = model
            temp_model.is_trained = True
            
            results[name] = temp_model._calculate_medical_metrics(y, y_pred, y_pred_proba)
            
            # Add standard metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            results[name].update({
                'accuracy': accuracy_score(y, y_pred),
                'precision': precision_score(y, y_pred),
                'recall': recall_score(y, y_pred),
                'f1_score': f1_score(y, y_pred),
                'roc_auc': roc_auc_score(y, y_pred_proba)
            })
        
        return results