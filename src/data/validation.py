"""Data validation and quality checking module."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import pandas as pd
import numpy as np
from loguru import logger

from ..config import settings
from ..utils.helpers import load_json


class DataValidator:
    """Comprehensive data validation for heart disease prediction."""
    
    def __init__(self, schema_path: Optional[str] = None):
        """
        Initialize data validator.
        
        Args:
            schema_path: Path to JSON schema file
        """
        if schema_path is None:
            schema_path = settings.project_root / "data" / "schemas" / "heart_disease_schema.json"
        
        self.schema = load_json(schema_path)
        self.features_schema = self.schema["features"]
        self.medical_mappings = self.schema["medical_mappings"]
        
        logger.info(f"DataValidator initialized with schema: {schema_path}")
    
    def validate_dataframe(self, df: pd.DataFrame, check_target: bool = True) -> Dict[str, Any]:
        """
        Comprehensive validation of DataFrame against schema.
        
        Args:
            df: DataFrame to validate
            check_target: Whether to check target column
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "summary": {},
            "data_quality": {}
        }
        
        try:
            # Check required columns
            required_columns = [
                col for col, config in self.features_schema.items()
                if config.get("required", False)
            ]
            
            if not check_target:
                required_columns = [col for col in required_columns if col != "target"]
            
            missing_columns = set(required_columns) - set(df.columns)
            if missing_columns:
                validation_results["errors"].append(f"Missing required columns: {missing_columns}")
                validation_results["is_valid"] = False
            
            # Check data types and ranges
            for column in df.columns:
                if column in self.features_schema:
                    column_results = self._validate_column(df[column], column)
                    if column_results["errors"]:
                        validation_results["errors"].extend(column_results["errors"])
                        validation_results["is_valid"] = False
                    validation_results["warnings"].extend(column_results["warnings"])
            
            # Data quality metrics
            validation_results["data_quality"] = self._calculate_data_quality(df)
            
            # Summary statistics
            validation_results["summary"] = {
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "missing_values": df.isnull().sum().sum(),
                "duplicate_rows": df.duplicated().sum()
            }
            
            logger.info(f"Validation complete - Valid: {validation_results['is_valid']}")
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            validation_results["errors"].append(f"Validation failed: {str(e)}")
            validation_results["is_valid"] = False
        
        return validation_results
    
    def _validate_column(self, series: pd.Series, column_name: str) -> Dict[str, List[str]]:
        """Validate individual column against schema."""
        results = {"errors": [], "warnings": []}
        config = self.features_schema[column_name]
        
        # Check data type
        expected_type = config["type"]
        if expected_type == "integer":
            if not pd.api.types.is_integer_dtype(series):
                # Try to convert
                try:
                    series = pd.to_numeric(series, downcast='integer')
                except:
                    results["errors"].append(f"{column_name}: Expected integer, got {series.dtype}")
        elif expected_type == "float":
            if not pd.api.types.is_numeric_dtype(series):
                results["errors"].append(f"{column_name}: Expected numeric, got {series.dtype}")
        
        # Check value ranges
        if "min_value" in config:
            min_violations = series < config["min_value"]
            if min_violations.any():
                count = min_violations.sum()
                results["errors"].append(f"{column_name}: {count} values below minimum {config['min_value']}")
        
        if "max_value" in config:
            max_violations = series > config["max_value"]
            if max_violations.any():
                count = max_violations.sum()
                results["errors"].append(f"{column_name}: {count} values above maximum {config['max_value']}")
        
        # Check allowed values
        if "allowed_values" in config:
            invalid_values = ~series.isin(config["allowed_values"])
            if invalid_values.any():
                count = invalid_values.sum()
                unique_invalid = series[invalid_values].unique()
                results["errors"].append(
                    f"{column_name}: {count} invalid values {unique_invalid}. "
                    f"Allowed: {config['allowed_values']}"
                )
        
        # Check for missing values
        missing_count = series.isnull().sum()
        if missing_count > 0:
            if config.get("required", False):
                results["errors"].append(f"{column_name}: {missing_count} missing values in required column")
            else:
                results["warnings"].append(f"{column_name}: {missing_count} missing values")
        
        return results
    
    def _calculate_data_quality(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate data quality metrics."""
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()
        
        quality_metrics = {
            "completeness": 1 - (missing_cells / total_cells) if total_cells > 0 else 0,
            "uniqueness": 1 - (df.duplicated().sum() / len(df)) if len(df) > 0 else 0,
            "missing_percentage": (missing_cells / total_cells * 100) if total_cells > 0 else 0,
            "duplicate_percentage": (df.duplicated().sum() / len(df) * 100) if len(df) > 0 else 0
        }
        
        return quality_metrics
    
    def clean_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Clean data based on validation results.
        
        Args:
            df: DataFrame to clean
            
        Returns:
            Tuple of (cleaned_dataframe, cleaning_report)
        """
        cleaning_report = {
            "original_shape": df.shape,
            "operations": [],
            "final_shape": None,
            "removed_rows": 0,
            "imputed_values": {}
        }
        
        df_cleaned = df.copy()
        
        # Remove duplicates
        initial_rows = len(df_cleaned)
        df_cleaned = df_cleaned.drop_duplicates()
        removed_duplicates = initial_rows - len(df_cleaned)
        if removed_duplicates > 0:
            cleaning_report["operations"].append(f"Removed {removed_duplicates} duplicate rows")
            cleaning_report["removed_rows"] += removed_duplicates
        
        # Handle outliers based on medical knowledge
        df_cleaned = self._handle_medical_outliers(df_cleaned, cleaning_report)
        
        # Handle missing values
        df_cleaned = self._handle_missing_values(df_cleaned, cleaning_report)
        
        cleaning_report["final_shape"] = df_cleaned.shape
        
        logger.info(f"Data cleaning complete: {cleaning_report['original_shape']} -> {cleaning_report['final_shape']}")
        
        return df_cleaned, cleaning_report
    
    def _handle_medical_outliers(self, df: pd.DataFrame, report: Dict) -> pd.DataFrame:
        """Handle outliers based on medical knowledge."""
        df_clean = df.copy()
        initial_rows = len(df_clean)
        
        # Extreme age outliers (keeping reasonable medical range)
        if 'age' in df_clean.columns:
            age_outliers = (df_clean['age'] < 18) | (df_clean['age'] > 100)
            df_clean = df_clean[~age_outliers]
        
        # Extreme blood pressure outliers
        if 'trestbps' in df_clean.columns:
            bp_outliers = (df_clean['trestbps'] < 70) | (df_clean['trestbps'] > 300)
            df_clean = df_clean[~bp_outliers]
        
        # Extreme cholesterol outliers
        if 'chol' in df_clean.columns:
            chol_outliers = (df_clean['chol'] < 50) | (df_clean['chol'] > 800)
            df_clean = df_clean[~chol_outliers]
        
        removed_outliers = initial_rows - len(df_clean)
        if removed_outliers > 0:
            report["operations"].append(f"Removed {removed_outliers} rows with medical outliers")
            report["removed_rows"] += removed_outliers
        
        return df_clean
    
    def _handle_missing_values(self, df: pd.DataFrame, report: Dict) -> pd.DataFrame:
        """Handle missing values with appropriate imputation."""
        df_clean = df.copy()
        
        for column in df_clean.columns:
            if column in self.features_schema:
                missing_count = df_clean[column].isnull().sum()
                
                if missing_count > 0:
                    config = self.features_schema[column]
                    
                    if config["type"] in ["integer", "float"]:
                        # Use median for numeric columns
                        median_value = df_clean[column].median()
                        df_clean[column].fillna(median_value, inplace=True)
                        report["imputed_values"][column] = f"Imputed {missing_count} values with median ({median_value})"
                    else:
                        # Use mode for categorical columns
                        mode_value = df_clean[column].mode().iloc[0] if not df_clean[column].mode().empty else 0
                        df_clean[column].fillna(mode_value, inplace=True)
                        report["imputed_values"][column] = f"Imputed {missing_count} values with mode ({mode_value})"
        
        return df_clean
    
    def get_medical_interpretation(self, row: Union[pd.Series, Dict]) -> Dict[str, str]:
        """
        Get medical interpretation of patient data.
        
        Args:
            row: Patient data as Series or Dictionary
            
        Returns:
            Dictionary with medical interpretations
        """
        if isinstance(row, pd.Series):
            row = row.to_dict()
        
        interpretations = {}
        
        # Map categorical values to medical terms
        for feature, mappings in self.medical_mappings.items():
            column_name = feature.replace("_mapping", "")
            if column_name in row:
                value = row[column_name]
                if str(value) in mappings:
                    interpretations[column_name] = mappings[str(value)]
        
        # Add risk indicators
        if 'age' in row:
            if row['age'] > 65:
                interpretations['age_risk'] = "Advanced age (>65) - Higher risk factor"
            elif row['age'] < 35:
                interpretations['age_risk'] = "Young age (<35) - Lower risk factor"
        
        if 'chol' in row:
            if row['chol'] > 240:
                interpretations['cholesterol_risk'] = "High cholesterol (>240) - Major risk factor"
            elif row['chol'] < 200:
                interpretations['cholesterol_risk'] = "Normal cholesterol (<200) - Good"
        
        if 'trestbps' in row:
            if row['trestbps'] > 140:
                interpretations['bp_risk'] = "High blood pressure (>140) - Major risk factor"
            elif row['trestbps'] < 120:
                interpretations['bp_risk'] = "Normal blood pressure (<120) - Good"
        
        return interpretations


def validate_heart_disease_data(
    df: pd.DataFrame, 
    schema_path: Optional[str] = None,
    clean_data: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Convenience function for validating heart disease data.
    
    Args:
        df: DataFrame to validate
        schema_path: Path to schema file
        clean_data: Whether to clean the data
        
    Returns:
        Tuple of (validated_dataframe, validation_report)
    """
    validator = DataValidator(schema_path)
    
    # Validate
    validation_results = validator.validate_dataframe(df)
    
    # Clean if requested
    if clean_data:
        df_cleaned, cleaning_report = validator.clean_data(df)
        validation_results["cleaning_report"] = cleaning_report
        return df_cleaned, validation_results
    
    return df, validation_results