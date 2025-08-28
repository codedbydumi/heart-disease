"""Professional model training pipeline with hyperparameter tuning."""

from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer
import optuna
from loguru import logger

from .ensemble_model import HeartDiseaseEnsemble
from ..config import settings
from ..utils.helpers import save_json, load_json


class ModelTrainer:
    """Professional model training with hyperparameter optimization."""
    
    def __init__(self, 
                 optimization_method: str = 'optuna',
                 n_trials: int = 100,
                 cv_folds: int = 5):
        """
        Initialize model trainer.
        
        Args:
            optimization_method: 'optuna', 'grid', or 'random'
            n_trials: Number of optimization trials
            cv_folds: Cross-validation folds
        """
        self.optimization_method = optimization_method
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.best_params = {}
        self.optimization_history = {}
        
        logger.info(f"ModelTrainer initialized with {optimization_method} optimization")
    
    def train_ensemble_model(self, 
                           X_train: np.ndarray, 
                           y_train: np.ndarray,
                           X_val: Optional[np.ndarray] = None,
                           y_val: Optional[np.ndarray] = None,
                           optimize_hyperparams: bool = True) -> HeartDiseaseEnsemble:
        """
        Train ensemble model with optional hyperparameter optimization.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            optimize_hyperparams: Whether to optimize hyperparameters
            
        Returns:
            Trained ensemble model
        """
        logger.info("Starting ensemble model training...")
        
        if optimize_hyperparams:
            logger.info("Optimizing hyperparameters...")
            best_params = self._optimize_hyperparameters(X_train, y_train)
            model = HeartDiseaseEnsemble(
                rf_params=best_params.get('rf_params'),
                xgb_params=best_params.get('xgb_params'),
                lr_params=best_params.get('lr_params')
            )
        else:
            model = HeartDiseaseEnsemble()
        
        # Train the model
        training_results = model.fit(X_train, y_train)
        
        # Evaluate on validation set if provided
        if X_val is not None and y_val is not None:
            val_metrics = model.evaluate(X_val, y_val)
            training_results['validation_metrics'] = val_metrics
            logger.info(f"Validation Accuracy: {val_metrics['accuracy']:.4f}")
        
        # Save training results
        self._save_training_results(training_results, model)
        
        logger.info("Ensemble model training completed")
        
        return model
    
    def _optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Dict]:
        """Optimize hyperparameters using specified method."""
        
        if self.optimization_method == 'optuna':
            return self._optuna_optimization(X, y)
        elif self.optimization_method == 'grid':
            return self._grid_search_optimization(X, y)
        elif self.optimization_method == 'random':
            return self._random_search_optimization(X, y)
        else:
            logger.warning(f"Unknown optimization method: {self.optimization_method}")
            return {}
    
    def _optuna_optimization(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Dict]:
        """Optimize hyperparameters using Optuna."""
        
        def objective(trial):
            # Random Forest parameters
            rf_params = {
                'n_estimators': trial.suggest_int('rf_n_estimators', 50, 200),
                'max_depth': trial.suggest_int('rf_max_depth', 5, 20),
                'min_samples_split': trial.suggest_int('rf_min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('rf_min_samples_leaf', 1, 5),
                'random_state': 42
            }
            
            # XGBoost parameters
            xgb_params = {
                'n_estimators': trial.suggest_int('xgb_n_estimators', 50, 200),
                'max_depth': trial.suggest_int('xgb_max_depth', 3, 10),
                'learning_rate': trial.suggest_float('xgb_learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('xgb_subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('xgb_colsample_bytree', 0.6, 1.0),
                'random_state': 42
            }
            
            # Logistic Regression parameters
            lr_params = {
                'C': trial.suggest_float('lr_C', 0.01, 10.0, log=True),
                'random_state': 42,
                'max_iter': 1000
            }
            
            # Create and evaluate model
            model = HeartDiseaseEnsemble(rf_params=rf_params, xgb_params=xgb_params, lr_params=lr_params)
            
            from sklearn.model_selection import cross_val_score
            scores = cross_val_score(model, X, y, cv=self.cv_folds, scoring='accuracy')
            
            return scores.mean()
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials)
        
        # Extract best parameters
        best_trial = study.best_trial
        best_params = {
            'rf_params': {k.replace('rf_', ''): v for k, v in best_trial.params.items() if k.startswith('rf_')},
            'xgb_params': {k.replace('xgb_', ''): v for k, v in best_trial.params.items() if k.startswith('xgb_')},
            'lr_params': {k.replace('lr_', ''): v for k, v in best_trial.params.items() if k.startswith('lr_')}
        }
        
        # Add random state
        best_params['rf_params']['random_state'] = 42
        best_params['xgb_params']['random_state'] = 42
        best_params['lr_params']['random_state'] = 42
        best_params['lr_params']['max_iter'] = 1000
        
        self.optimization_history = {
            'method': 'optuna',
            'n_trials': self.n_trials,
            'best_score': study.best_value,
            'best_params': best_params
        }
        
        logger.info(f"Optuna optimization completed. Best score: {study.best_value:.4f}")
        
        return best_params
    
    def _grid_search_optimization(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Dict]:
        """Optimize hyperparameters using GridSearchCV."""
        from sklearn.ensemble import RandomForestClassifier
        
        param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [5, 10, 15],
            'min_samples_split': [2, 5, 10]
        }
        
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(rf, param_grid, cv=self.cv_folds, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X, y)
        
        best_params = {
            'rf_params': grid_search.best_params_,
            'xgb_params': {'random_state': 42},  # Use defaults for other models
            'lr_params': {'random_state': 42}
        }
        
        self.optimization_history = {
            'method': 'grid_search',
            'best_score': grid_search.best_score_,
            'best_params': best_params
        }
        
        return best_params
    
    def _random_search_optimization(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Dict]:
        """Optimize hyperparameters using RandomizedSearchCV."""
        from sklearn.ensemble import RandomForestClassifier
        from scipy.stats import randint
        
        param_dist = {
            'n_estimators': randint(50, 200),
            'max_depth': randint(5, 20),
            'min_samples_split': randint(2, 10),
            'min_samples_leaf': randint(1, 5)
        }
        
        rf = RandomForestClassifier(random_state=42)
        random_search = RandomizedSearchCV(rf, param_dist, n_iter=self.n_trials, 
                                         cv=self.cv_folds, scoring='accuracy', n_jobs=-1, random_state=42)
        random_search.fit(X, y)
        
        best_params = {
            'rf_params': random_search.best_params_,
            'xgb_params': {'random_state': 42},
            'lr_params': {'random_state': 42}
        }
        
        self.optimization_history = {
            'method': 'random_search',
            'best_score': random_search.best_score_,
            'best_params': best_params
        }
        
        return best_params
    
    def _save_training_results(self, training_results: Dict, model: HeartDiseaseEnsemble) -> None:
        """Save training results and model metadata."""
        metadata_dir = settings.models_dir / "metadata"
        metadata_dir.mkdir(exist_ok=True)
        
        training_metadata = {
            'model_name': model.model_name,
            'training_timestamp': pd.Timestamp.now().isoformat(),
            'optimization_method': self.optimization_method,
            'optimization_history': self.optimization_history,
            'training_results': training_results,
            'model_metadata': model.model_metadata
        }
        
        save_json(training_metadata, metadata_dir / f"{model.model_name}_training_metadata.json")
        logger.info("Training metadata saved")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of training process."""
        return {
            'optimization_method': self.optimization_method,
            'n_trials': self.n_trials,
            'cv_folds': self.cv_folds,
            'optimization_history': self.optimization_history
        }


def train_heart_disease_model(X_train: np.ndarray, 
                            y_train: np.ndarray,
                            X_val: Optional[np.ndarray] = None,
                            y_val: Optional[np.ndarray] = None,
                            optimize_hyperparams: bool = True,
                            optimization_method: str = 'optuna') -> HeartDiseaseEnsemble:
    """
    Convenience function to train heart disease prediction model.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        optimize_hyperparams: Whether to optimize hyperparameters
        optimization_method: Optimization method to use
        
    Returns:
        Trained ensemble model
    """
    trainer = ModelTrainer(optimization_method=optimization_method)
    
    model = trainer.train_ensemble_model(
        X_train, y_train, X_val, y_val, optimize_hyperparams
    )
    
    return model