"""Data loading and management module."""

import sqlite3
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union

import pandas as pd
import numpy as np
from loguru import logger

from ..config import settings
from ..utils.helpers import save_json, load_json
from .validation import DataValidator


class HeartDiseaseDataLoader:
    """Professional data loading and management for heart disease prediction."""
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize data loader.
        
        Args:
            db_path: Path to SQLite database
        """
        if db_path is None:
            db_path = settings.project_root / "heart_disease.db"
        
        self.db_path = db_path
        self.validator = DataValidator()
        
        # Initialize database
        self._init_database()
        
        logger.info(f"HeartDiseaseDataLoader initialized with database: {db_path}")
    
    def _init_database(self) -> None:
        """Initialize SQLite database with proper schema."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create main data table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS heart_disease_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        age INTEGER NOT NULL,
                        sex INTEGER NOT NULL,
                        cp INTEGER NOT NULL,
                        trestbps INTEGER NOT NULL,
                        chol INTEGER NOT NULL,
                        fbs INTEGER NOT NULL,
                        restecg INTEGER NOT NULL,
                        thalach INTEGER NOT NULL,
                        exang INTEGER NOT NULL,
                        oldpeak REAL NOT NULL,
                        slope INTEGER NOT NULL,
                        ca INTEGER NOT NULL,
                        thal INTEGER NOT NULL,
                        target INTEGER NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        data_source TEXT DEFAULT 'manual'
                    )
                """)
                
                # Create predictions table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS predictions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        patient_data_id INTEGER,
                        model_version TEXT NOT NULL,
                        prediction REAL NOT NULL,
                        prediction_class INTEGER NOT NULL,
                        confidence REAL,
                        risk_category TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (patient_data_id) REFERENCES heart_disease_data (id)
                    )
                """)
                
                # Create data quality logs table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS data_quality_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        batch_id TEXT NOT NULL,
                        validation_status TEXT NOT NULL,
                        error_count INTEGER DEFAULT 0,
                        warning_count INTEGER DEFAULT 0,
                        total_records INTEGER NOT NULL,
                        quality_score REAL,
                        report TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                conn.commit()
                logger.info("Database initialized successfully")
                
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    def load_csv_data(
        self, 
        file_path: Union[str, Path], 
        validate: bool = True,
        save_to_db: bool = True
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Load and process CSV data.
        
        Args:
            file_path: Path to CSV file
            validate: Whether to validate data
            save_to_db: Whether to save to database
            
        Returns:
            Tuple of (dataframe, loading_report)
        """
        file_path = Path(file_path)
        
        loading_report = {
            "file_path": str(file_path),
            "file_size_mb": file_path.stat().st_size / (1024 * 1024) if file_path.exists() else 0,
            "loading_time": None,
            "validation_results": None,
            "saved_to_db": False,
            "total_records": 0
        }
        
        try:
            # Load CSV
            start_time = pd.Timestamp.now()
            df = pd.read_csv(file_path)
            loading_time = (pd.Timestamp.now() - start_time).total_seconds()
            
            loading_report.update({
                "loading_time": loading_time,
                "total_records": len(df),
                "columns": list(df.columns),
                "data_types": df.dtypes.to_dict()
            })
            
            logger.info(f"Loaded CSV: {len(df)} rows, {len(df.columns)} columns from {file_path}")
            
            # Validate data
            if validate:
                df_validated, validation_results = self.validator.validate_dataframe(df, check_target=True)
                loading_report["validation_results"] = validation_results
                
                if validation_results["is_valid"]:
                    df = df_validated
                    logger.info("Data validation passed")
                else:
                    logger.warning(f"Data validation issues: {validation_results['errors']}")
            
            # Save to database
            if save_to_db and (not validate or validation_results.get("is_valid", True)):
                self.save_to_database(df, source=f"csv:{file_path.name}")
                loading_report["saved_to_db"] = True
            
        except Exception as e:
            logger.error(f"Failed to load CSV {file_path}: {e}")
            loading_report["error"] = str(e)
            raise
        
        return df, loading_report
    
    def save_to_database(
        self, 
        df: pd.DataFrame, 
        source: str = "manual",
        batch_id: Optional[str] = None
    ) -> str:
        """
        Save DataFrame to database.
        
        Args:
            df: DataFrame to save
            source: Source identifier
            batch_id: Batch identifier
            
        Returns:
            Batch ID of saved data
        """
        if batch_id is None:
            batch_id = f"batch_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Add metadata columns
                df_to_save = df.copy()
                df_to_save['data_source'] = source
                
                # Save data
                df_to_save.to_sql(
                    'heart_disease_data', 
                    conn, 
                    if_exists='append', 
                    index=False,
                    method='multi'
                )
                
                logger.info(f"Saved {len(df)} records to database with batch_id: {batch_id}")
                
        except Exception as e:
            logger.error(f"Failed to save to database: {e}")
            raise
        
        return batch_id
    
    def load_from_database(
        self, 
        limit: Optional[int] = None,
        source: Optional[str] = None,
        include_predictions: bool = False
    ) -> pd.DataFrame:
        """
        Load data from database.
        
        Args:
            limit: Maximum number of records to load
            source: Filter by data source
            include_predictions: Whether to include prediction data
            
        Returns:
            DataFrame with loaded data
        """
        try:
            query = "SELECT * FROM heart_disease_data"
            params = []
            
            if source:
                query += " WHERE data_source = ?"
                params.append(source)
            
            query += " ORDER BY created_at DESC"
            
            if limit:
                query += " LIMIT ?"
                params.append(limit)
            
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query(query, conn, params=params)
            
            logger.info(f"Loaded {len(df)} records from database")
            
            if include_predictions:
                predictions_df = pd.read_sql_query(
                    "SELECT * FROM predictions ORDER BY created_at DESC",
                    conn
                )
                return df, predictions_df
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load from database: {e}")
            raise
    
    def get_data_statistics(self) -> Dict[str, Any]:
        """Get comprehensive data statistics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Basic statistics
                stats = {
                    "total_records": pd.read_sql_query("SELECT COUNT(*) as count FROM heart_disease_data", conn).iloc[0]['count'],
                    "total_predictions": pd.read_sql_query("SELECT COUNT(*) as count FROM predictions", conn).iloc[0]['count'],
                }
                
                # Data distribution
                target_dist = pd.read_sql_query(
                    "SELECT target, COUNT(*) as count FROM heart_disease_data GROUP BY target",
                    conn
                )
                stats["target_distribution"] = target_dist.to_dict('records')
                
                # Data sources
                sources = pd.read_sql_query(
                    "SELECT data_source, COUNT(*) as count FROM heart_disease_data GROUP BY data_source",
                    conn
                )
                stats["data_sources"] = sources.to_dict('records')
                
                # Recent activity
                recent_data = pd.read_sql_query(
                    """SELECT DATE(created_at) as date, COUNT(*) as count 
                       FROM heart_disease_data 
                       WHERE created_at >= date('now', '-7 days')
                       GROUP BY DATE(created_at)
                       ORDER BY date DESC""",
                    conn
                )
                stats["recent_activity"] = recent_data.to_dict('records')
                
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}
    
    def create_train_test_split(
        self, 
        test_size: float = 0.2,
        validation_size: float = 0.1,
        random_state: int = 42,
        save_splits: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create train/test/validation splits.
        
        Args:
            test_size: Proportion for test set
            validation_size: Proportion for validation set
            random_state: Random seed
            save_splits: Whether to save splits to files
            
        Returns:
            Tuple of (train_df, test_df, val_df)
        """
        from sklearn.model_selection import train_test_split
        
        # Load all data
        df = self.load_from_database()
        
        # Remove metadata columns for ML
        feature_columns = [col for col in df.columns if col not in ['id', 'created_at', 'data_source']]
        df_ml = df[feature_columns]
        
        # First split: train+val vs test
        train_val_df, test_df = train_test_split(
            df_ml,
            test_size=test_size,
            random_state=random_state,
            stratify=df_ml['target']
        )
        
        # Second split: train vs validation
        val_ratio = validation_size / (1 - test_size)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_ratio,
            random_state=random_state,
            stratify=train_val_df['target']
        )
        
        logger.info(f"Data splits created - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        # Save splits
        if save_splits:
            data_dir = settings.project_root / "data" / "processed"
            data_dir.mkdir(exist_ok=True)
            
            train_df.to_csv(data_dir / "train_data.csv", index=False)
            val_df.to_csv(data_dir / "val_data.csv", index=False)
            test_df.to_csv(data_dir / "test_data.csv", index=False)
            
            # Save split metadata
            split_info = {
                "created_at": pd.Timestamp.now().isoformat(),
                "total_records": len(df_ml),
                "train_records": len(train_df),
                "val_records": len(val_df),
                "test_records": len(test_df),
                "test_size": test_size,
                "validation_size": validation_size,
                "random_state": random_state,
                "target_distribution": {
                "train": train_df['target'].value_counts().to_dict(),
                "val": val_df['target'].value_counts().to_dict(),
                "test": test_df['target'].value_counts().to_dict()
               }
           }
            save_json(split_info, data_dir / "split_info.json")
            logger.info("Data splits saved to processed directory")
       
        return train_df, test_df, val_df
    
    def get_sample_data(self, n_samples: int = 5) -> pd.DataFrame:
       """Get sample data for testing and demonstration."""
       sample_data = pd.DataFrame({
           'age': [63, 37, 41, 56, 57],
           'sex': [1, 1, 0, 1, 0],
           'cp': [3, 2, 1, 1, 0],
           'trestbps': [145, 130, 130, 120, 120],
           'chol': [233, 250, 204, 236, 354],
           'fbs': [1, 0, 0, 0, 0],
           'restecg': [0, 1, 0, 1, 1],
           'thalach': [150, 187, 172, 178, 163],
           'exang': [0, 0, 0, 0, 1],
           'oldpeak': [2.3, 3.5, 1.4, 0.8, 0.6],
           'slope': [0, 0, 2, 2, 2],
           'ca': [0, 0, 0, 0, 0],
           'thal': [1, 2, 2, 2, 2],
           'target': [1, 1, 1, 1, 1]
       })
       
       return sample_data.head(n_samples)


def load_heart_disease_dataset(
   file_path: Optional[Union[str, Path]] = None,
   from_database: bool = False,
   validate: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
   """
   Convenience function to load heart disease dataset.
   
   Args:
       file_path: Path to CSV file (if loading from file)
       from_database: Whether to load from database
       validate: Whether to validate data
       
   Returns:
       Tuple of (dataframe, loading_report)
   """
   loader = HeartDiseaseDataLoader()
   
   if from_database:
       df = loader.load_from_database()
       # Remove database metadata columns for ML use
       feature_columns = [col for col in df.columns if col not in ['id', 'created_at', 'data_source']]
       df = df[feature_columns]
       
       report = {
           "source": "database",
           "total_records": len(df),
           "columns": list(df.columns)
       }
       
       if validate:
           _, validation_results = loader.validator.validate_dataframe(df)
           report["validation_results"] = validation_results
       
       return df, report
   
   elif file_path:
       return loader.load_csv_data(file_path, validate=validate)
   
   else:
       # Return sample data
       df = loader.get_sample_data(n_samples=100)  # Generate more sample data
       report = {
           "source": "sample",
           "total_records": len(df),
           "columns": list(df.columns),
           "note": "Using generated sample data"
       }
       
       return df, report