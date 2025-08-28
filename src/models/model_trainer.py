"""Enhanced Intermediate Model Training with Hyperparameter Tuning."""

from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
from loguru import logger

from ..config import settings
from ..utils.helpers import save_json, save_pickle


class ModelTrainer:
    """Enhanced Intermediate Model Training with Hyperparameter Optimization."""
    
    def __init__(self, tuning_method: str = 'random'):
        """
        Initialize model trainer.
        
        Args:
            tuning_method: 'random', 'grid', or 'none'
        """
        self.tuning_method = tuning_method
        self.best_params = {}
        self.models = {}
        self.best_model = None
        
        logger.info(f"ModelTrainer initialized with {tuning_method} search")
    
    def train_ensemble(self, X_train: np.ndarray, y_train: np.ndarray,
                      X_val: Optional[np.ndarray] = None, 
                      y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Train ensemble with hyperparameter tuning."""
        
        results = {
            'individual_models': {},
            'ensemble_performance': {},
            'best_model_name': None
        }
        
        # Train individual models with tuning
        logger.info("Training individual models...")
        
        # 1. Random Forest
        rf_model, rf_results = self._train_random_forest(X_train, y_train, X_val, y_val)
        self.models['random_forest'] = rf_model
        results['individual_models']['random_forest'] = rf_results
        
        # 2. XGBoost  
        xgb_model, xgb_results = self._train_xgboost(X_train, y_train, X_val, y_val)
        self.models['xgboost'] = xgb_model
        results['individual_models']['xgboost'] = xgb_results
        
        # 3. Logistic Regression
        lr_model, lr_results = self._train_logistic_regression(X_train, y_train, X_val, y_val)
        self.models['logistic_regression'] = lr_model
        results['individual_models']['logistic_regression'] = lr_results
        
        # Select best model
        best_score = 0
        best_name = 'random_forest'
        
        for name, model_results in results['individual_models'].items():
            if model_results['val_accuracy'] > best_score:
                best_score = model_results['val_accuracy']
                best_name = name
        
        self.best_model = self.models[best_name]
        results['best_model_name'] = best_name
        results['best_score'] = best_score
        
        logger.info(f"Best model: {best_name} with accuracy: {best_score:.4f}")
        
        # Save results
        self._save_training_results(results)
        
        return results
    
    def _train_random_forest(self, X_train, y_train, X_val=None, y_val=None):
        """Train Random Forest with hyperparameter tuning."""
        
        if self.tuning_method == 'grid':
            param_grid = {
                'n_estimators': [50, 100, 150],
                'max_depth': [5, 10, 15],
                'min_samples_split': [2, 5, 10]
            }
            
            rf = RandomForestClassifier(random_state=42)
            grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            
        elif self.tuning_method == 'random':
            from scipy.stats import randint
            
            param_dist = {
                'n_estimators': randint(50, 150),
                'max_depth': [5, 10, 15, 20],
                'min_samples_split': randint(2, 11),
                'min_samples_leaf': randint(1, 5)
            }
            
            rf = RandomForestClassifier(random_state=42)
            random_search = RandomizedSearchCV(rf, param_dist, n_iter=10, cv=3, 
                                             scoring='accuracy', n_jobs=-1, random_state=42)
            random_search.fit(X_train, y_train)
            
            best_model = random_search.best_estimator_
            best_params = random_search.best_params_
            
        else:  # No tuning
            best_params = {'n_estimators': 100, 'max_depth': 10, 'random_state': 42}
            best_model = RandomForestClassifier(**best_params)
            best_model.fit(X_train, y_train)
        
        # Evaluate
        train_acc = accuracy_score(y_train, best_model.predict(X_train))
        val_acc = accuracy_score(y_val, best_model.predict(X_val)) if X_val is not None else train_acc
        
        results = {
            'best_params': best_params,
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'feature_importance': best_model.feature_importances_.tolist()
        }
        
        logger.info(f"Random Forest - Train: {train_acc:.4f}, Val: {val_acc:.4f}")
        
        return best_model, results
    
    def _train_xgboost(self, X_train, y_train, X_val=None, y_val=None):
        """Train XGBoost with hyperparameter tuning."""
        
        if self.tuning_method == 'grid':
            param_grid = {
                'n_estimators': [50, 100, 150],
                'max_depth': [3, 6, 10],
                'learning_rate': [0.01, 0.1, 0.2]
            }
            
            xgb_model = xgb.XGBClassifier(random_state=42)
            grid_search = GridSearchCV(xgb_model, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            
        elif self.tuning_method == 'random':
            from scipy.stats import uniform, randint
            
            param_dist = {
                'n_estimators': randint(50, 150),
                'max_depth': randint(3, 10),
                'learning_rate': uniform(0.05, 0.25),
                'subsample': uniform(0.7, 0.3)
            }
            
            xgb_model = xgb.XGBClassifier(random_state=42)
            random_search = RandomizedSearchCV(xgb_model, param_dist, n_iter=10, cv=3,
                                             scoring='accuracy', n_jobs=-1, random_state=42)
            random_search.fit(X_train, y_train)
            
            best_model = random_search.best_estimator_
            best_params = random_search.best_params_
            
        else:  # No tuning
            best_params = {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1, 'random_state': 42}
            best_model = xgb.XGBClassifier(**best_params)
            best_model.fit(X_train, y_train)
        
        # Evaluate
        train_acc = accuracy_score(y_train, best_model.predict(X_train))
        val_acc = accuracy_score(y_val, best_model.predict(X_val)) if X_val is not None else train_acc
        
        results = {
            'best_params': best_params,
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'feature_importance': best_model.feature_importances_.tolist()
        }
        
        logger.info(f"XGBoost - Train: {train_acc:.4f}, Val: {val_acc:.4f}")
        
        return best_model, results
    
    def _train_logistic_regression(self, X_train, y_train, X_val=None, y_val=None):
        """Train Logistic Regression with hyperparameter tuning."""
        
        if self.tuning_method in ['grid', 'random']:
            param_options = {
                'C': [0.1, 1, 10, 100],
                'solver': ['liblinear', 'lbfgs'],
                'max_iter': [1000]
            }
            
            lr = LogisticRegression(random_state=42)
            
            if self.tuning_method == 'grid':
                search = GridSearchCV(lr, param_options, cv=3, scoring='accuracy', n_jobs=-1)
            else:
                search = RandomizedSearchCV(lr, param_options, n_iter=5, cv=3, 
                                          scoring='accuracy', n_jobs=-1, random_state=42)
            
            search.fit(X_train, y_train)
            best_model = search.best_estimator_
            best_params = search.best_params_
            
        else:  # No tuning
            best_params = {'C': 1, 'solver': 'lbfgs', 'max_iter': 1000, 'random_state': 42}
            best_model = LogisticRegression(**best_params)
            best_model.fit(X_train, y_train)
        
        # Evaluate
        train_acc = accuracy_score(y_train, best_model.predict(X_train))
        val_acc = accuracy_score(y_val, best_model.predict(X_val)) if X_val is not None else train_acc
        
        results = {
            'best_params': best_params,
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'feature_importance': np.abs(best_model.coef_[0]).tolist()
        }
        
        logger.info(f"Logistic Regression - Train: {train_acc:.4f}, Val: {val_acc:.4f}")
        
        return best_model, results
    
    def predict(self, X):
        """Make predictions using best model."""
        if self.best_model is None:
            raise ValueError("Must train model first")
        return self.best_model.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities."""
        if self.best_model is None:
            raise ValueError("Must train model first")
        return self.best_model.predict_proba(X)
    
    def save_model(self, file_path: Optional[str] = None):
        """Save the best trained model."""
        if file_path is None:
            file_path = settings.models_dir / "trained_models" / "best_heart_disease_model.pkl"
        
        model_data = {
            'best_model': self.best_model,
            'all_models': self.models,
            'tuning_method': self.tuning_method,
            'best_params': self.best_params
        }
        
        save_pickle(model_data, file_path)
        logger.info(f"Models saved to {file_path}")
        
        return str(file_path)
    
    def _save_training_results(self, results):
        """Save training results."""
        metadata_dir = settings.models_dir / "metadata"
        metadata_dir.mkdir(exist_ok=True)
        
        training_metadata = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'tuning_method': self.tuning_method,
            'results': results
        }
        
        save_json(training_metadata, metadata_dir / "training_results.json")
        logger.info("Training metadata saved")