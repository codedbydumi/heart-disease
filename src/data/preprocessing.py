"""Data preprocessing module with professional pipeline."""

from typing import Dict, List, Optional, Tuple, Any, Union
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split
from loguru import logger

from ..config import settings
from ..utils.helpers import save_pickle, load_pickle
from .validation import DataValidator


class HeartDiseasePreprocessor:
    """Professional data preprocessing for heart disease prediction."""
    
    def __init__(self):
        """Initialize preprocessor with default settings."""
        self.scaler = None
        self.label_encoders = {}
        self.feature_names = []
        self.target_name = 'target'
        self.is_fitted = False
        self.preprocessing_info = {}
        
        # Initialize validator
        self.validator = DataValidator()
        
        logger.info("HeartDiseasePreprocessor initialized")
    
    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit preprocessor and transform training data.
        
        Args:
            df: Training DataFrame
            
        Returns:
            Tuple of (X_transformed, y)
        """
        logger.info("Fitting preprocessor on training data")
        
        # Validate data
        validation_results = self.validator.validate_dataframe(df)
        if not validation_results["is_valid"]:
            logger.warning(f"Data validation issues: {validation_results['errors']}")
        
        # Separate features and target
        X = df.drop(columns=[self.target_name])
        y = df[self.target_name].values
        
        # Store feature names
        self.feature_names = list(X.columns)
        
        # Handle categorical features (if any need encoding)
        X_processed = self._fit_categorical_encoding(X)
        
        # Create engineered features
        X_engineered = self._fit_feature_engineering(X_processed)
        
        # Scale features
        X_scaled = self._fit_scaling(X_engineered)
        
        # Store preprocessing information
        self._store_preprocessing_info(df)
        
        self.is_fitted = True
        logger.info(f"Preprocessor fitted - Features: {X_scaled.shape[1]}, Samples: {X_scaled.shape[0]}")
        
        return X_scaled, y
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform new data using fitted preprocessor.
        
        Args:
            df: DataFrame to transform
            
        Returns:
            Transformed feature array
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transforming data")
        
        # Ensure same columns as training
        if self.target_name in df.columns:
            X = df.drop(columns=[self.target_name])
        else:
            X = df.copy()
        
        # Validate columns
        missing_cols = set(self.feature_names) - set(X.columns)
        if missing_cols:
            logger.warning(f"Missing columns in transform data: {missing_cols}")
            # Add missing columns with default values
            for col in missing_cols:
                X[col] = 0
        
        # Ensure column order
        X = X[self.feature_names]
        
        # Apply same transformations
        X_processed = self._transform_categorical_encoding(X)
        X_engineered = self._transform_feature_engineering(X_processed)
        X_scaled = self._transform_scaling(X_engineered)
        
        return X_scaled
    
    def _fit_categorical_encoding(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform categorical features (most are already encoded)."""
        X_processed = X.copy()
        
        # Heart disease data is mostly already encoded, but we can add validations
        categorical_columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
        
        for col in categorical_columns:
            if col in X_processed.columns:
                # Ensure categorical columns are integers
                X_processed[col] = X_processed[col].astype(int)
        
        return X_processed
    
    def _transform_categorical_encoding(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform categorical features using fitted encoders."""
        X_processed = X.copy()
        
        categorical_columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
        
        for col in categorical_columns:
            if col in X_processed.columns:
                X_processed[col] = X_processed[col].astype(int)
        
        return X_processed
    
    def _fit_feature_engineering(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create and fit engineered features."""
        X_engineered = X.copy()
        
        # Age groups
        X_engineered['age_group'] = pd.cut(
            X_engineered['age'], 
            bins=[0, 40, 55, 65, 100], 
            labels=[0, 1, 2, 3]
        ).astype(int)
        
        # Blood pressure categories
        X_engineered['bp_category'] = pd.cut(
            X_engineered['trestbps'],
            bins=[0, 120, 140, 180, 300],
            labels=[0, 1, 2, 3]
        ).astype(int)
        
        # Cholesterol categories
        X_engineered['chol_category'] = pd.cut(
            X_engineered['chol'],
            bins=[0, 200, 240, 300, 600],
            labels=[0, 1, 2, 3]
        ).astype(int)
        
        # Heart rate reserve (max heart rate - age-predicted max)
        predicted_max_hr = 220 - X_engineered['age']
        X_engineered['hr_reserve'] = X_engineered['thalach'] - predicted_max_hr
        
        # Risk interaction features
        X_engineered['age_chol_interaction'] = X_engineered['age'] * X_engineered['chol'] / 1000
        X_engineered['bp_age_interaction'] = X_engineered['trestbps'] * X_engineered['age'] / 1000
        
        # Composite risk score
        X_engineered['composite_risk'] = (
            X_engineered['age'] * 0.1 +
            X_engineered['chol'] * 0.01 +
            X_engineered['trestbps'] * 0.1 +
            X_engineered['cp'] * 10 +
            X_engineered['exang'] * 20
        )
        
        logger.info(f"Feature engineering complete - Added {len(X_engineered.columns) - len(X.columns)} features")
        
        return X_engineered
    
    def _transform_feature_engineering(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform features using fitted engineering pipeline."""
        return self._fit_feature_engineering(X)  # Same logic for transform
    
    def _fit_scaling(self, X: pd.DataFrame) -> np.ndarray:
        """Fit and transform feature scaling."""
        # Use RobustScaler for better handling of outliers
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        logger.info("Feature scaling fitted and applied")
        
        return X_scaled
    
    def _transform_scaling(self, X: pd.DataFrame) -> np.ndarray:
        """Transform features using fitted scaler."""
        if self.scaler is None:
            raise ValueError("Scaler not fitted")
        
        return self.scaler.transform(X)
    
    def _store_preprocessing_info(self, df: pd.DataFrame) -> None:
        """Store information about preprocessing steps."""
        self.preprocessing_info = {
            "original_features": list(df.drop(columns=[self.target_name]).columns),
            "target_distribution": df[self.target_name].value_counts().to_dict(),
            "feature_statistics": df.describe().to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "data_types": df.dtypes.to_dict(),
            "preprocessing_steps": [
                "categorical_encoding",
                "feature_engineering", 
                "scaling"
            ],
            "scaler_type": type(self.scaler).__name__,
            "total_features_after_engineering": None  # Will be set after engineering
        }
    
    def get_feature_names(self) -> List[str]:
        """Get names of all features after preprocessing."""
        if not self.is_fitted:
            return []
        
        # Return engineered feature names
        base_features = self.feature_names.copy()
        engineered_features = [
            'age_group', 'bp_category', 'chol_category', 
            'hr_reserve', 'age_chol_interaction', 'bp_age_interaction',
            'composite_risk'
        ]
        
        return base_features + engineered_features
    
    def save_preprocessor(self, file_path: Optional[str] = None) -> str:
        """Save fitted preprocessor to file."""
        if file_path is None:
            file_path = settings.models_dir / "scalers" / "preprocessor.pkl"
        
        preprocessor_data = {
            "scaler": self.scaler,
            "label_encoders": self.label_encoders,
            "feature_names": self.feature_names,
            "target_name": self.target_name,
            "is_fitted": self.is_fitted,
            "preprocessing_info": self.preprocessing_info
        }
        
        save_pickle(preprocessor_data, file_path)
        logger.info(f"Preprocessor saved to {file_path}")
        
        return str(file_path)
    
    def load_preprocessor(self, file_path: str) -> None:
        """Load fitted preprocessor from file."""
        preprocessor_data = load_pickle(file_path)
        
        self.scaler = preprocessor_data["scaler"]
        self.label_encoders = preprocessor_data["label_encoders"]
        self.feature_names = preprocessor_data["feature_names"]
        self.target_name = preprocessor_data["target_name"]
        self.is_fitted = preprocessor_data["is_fitted"]
        self.preprocessing_info = preprocessor_data["preprocessing_info"]
        
        logger.info(f"Preprocessor loaded from {file_path}")
    
    def get_preprocessing_summary(self) -> Dict[str, Any]:
        """Get summary of preprocessing steps and transformations."""
        if not self.is_fitted:
            return {"status": "not_fitted"}
        
        return {
            "status": "fitted",
            "original_features": len(self.feature_names),
            "engineered_features": len(self.get_feature_names()),
            "scaler_type": type(self.scaler).__name__,
            "preprocessing_info": self.preprocessing_info,
            "feature_names": self.get_feature_names()
        }


def preprocess_heart_disease_data(
    train_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame] = None,
    save_preprocessor: bool = True
) -> Tuple[Tuple[np.ndarray, np.ndarray], Optional[Tuple[np.ndarray, np.ndarray]]]:
    """
    Convenience function for preprocessing heart disease data.
    
    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame (optional)
        save_preprocessor: Whether to save the fitted preprocessor
        
    Returns:
        Tuple of training data and optionally test data
    """
    preprocessor = HeartDiseasePreprocessor()
    
    # Fit and transform training data
    X_train, y_train = preprocessor.fit_transform(train_df)
    
    # Transform test data if provided
    test_data = None
    if test_df is not None:
        X_test = preprocessor.transform(test_df)
        y_test = test_df['target'].values if 'target' in test_df.columns else None
        test_data = (X_test, y_test)
    
    # Save preprocessor
    if save_preprocessor:
        preprocessor.save_preprocessor()
    
    return (X_train, y_train), test_data