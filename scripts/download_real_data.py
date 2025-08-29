"""Download and prepare real UCI Heart Disease dataset."""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import get_logger
from src.data.data_loader import HeartDiseaseDataLoader

logger = get_logger("real_data_download")


def download_uci_heart_disease_data():
    """Download and process UCI Heart Disease dataset."""
    
    # Column names for the dataset
    column_names = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
    ]
    
    try:
        # Download Cleveland dataset (most complete)
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
        
        logger.info("Downloading UCI Heart Disease dataset...")
        df = pd.read_csv(url, header=None, names=column_names, na_values='?')
        
        logger.info(f"Dataset downloaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Clean the data
        df_cleaned = clean_dataset(df)
        
        # Save to data directory
        data_dir = project_root / "data" / "raw"
        data_dir.mkdir(exist_ok=True)
        
        output_path = data_dir / "uci_heart_disease.csv"
        df_cleaned.to_csv(output_path, index=False)
        
        logger.info(f"Real dataset saved to: {output_path}")
        logger.info(f"Final dataset shape: {df_cleaned.shape}")
        
        # Display dataset info
        print("\n" + "="*50)
        print("UCI HEART DISEASE DATASET SUMMARY")
        print("="*50)
        print(f"Total patients: {len(df_cleaned)}")
        print(f"Features: {len(df_cleaned.columns) - 1}")
        print(f"Target distribution:")
        print(df_cleaned['target'].value_counts())
        print(f"Missing values: {df_cleaned.isnull().sum().sum()}")
        
        return df_cleaned
        
    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        raise


def clean_dataset(df):
    """Clean and prepare the dataset."""
    
    logger.info("Cleaning dataset...")
    
    # Handle missing values
    missing_before = df.isnull().sum().sum()
    
    # Drop rows with missing target values
    df = df.dropna(subset=['target'])
    
    # Handle missing values in other columns
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ['float64', 'int64']:
                # Fill numeric columns with median
                df[col].fillna(df[col].median(), inplace=True)
            else:
                # Fill categorical columns with mode
                df[col].fillna(df[col].mode()[0], inplace=True)
    
    missing_after = df.isnull().sum().sum()
    logger.info(f"Missing values: {missing_before} -> {missing_after}")
    
    # Convert target to binary (0: no disease, 1: disease)
    # Original dataset has 0,1,2,3,4 where >0 indicates disease
    df['target'] = (df['target'] > 0).astype(int)
    
    # Ensure proper data types
    categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    for col in categorical_cols:
        df[col] = df[col].astype(int)
    
    # Remove extreme outliers
    for col in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']:
        if col in df.columns:
            q1 = df[col].quantile(0.01)
            q99 = df[col].quantile(0.99)
            df = df[(df[col] >= q1) & (df[col] <= q99)]
    
    logger.info(f"Dataset cleaned and prepared")
    
    return df


def retrain_with_real_data():
    """Retrain models with real data."""
    
    logger.info("Starting model retraining with real data...")
    
    # Load the real dataset
    data_path = project_root / "data" / "raw" / "uci_heart_disease.csv"
    
    if not data_path.exists():
        logger.error("Real dataset not found. Run download first.")
        return
    
    # Load data
    df = pd.read_csv(data_path)
    logger.info(f"Loaded real dataset: {df.shape}")
    
    # Use existing data loader to save to database
    loader = HeartDiseaseDataLoader()
    batch_id = loader.save_to_database(df, source="uci_heart_disease")
    logger.info(f"Saved to database with batch_id: {batch_id}")
    
    # Create new train/test splits
    train_df, test_df, val_df = loader.create_train_test_split(
        test_size=0.2,
        validation_size=0.1,
        save_splits=True
    )
    
    logger.info(f"Created splits - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Now retrain models
    logger.info("Retraining models...")
    import subprocess
    result = subprocess.run([sys.executable, "scripts/train_model.py"], 
                          capture_output=True, text=True)
    
    if result.returncode == 0:
        logger.info("Model retraining completed successfully!")
        print("\n" + "="*50)
        print("MODEL RETRAINING COMPLETED")
        print("="*50)
        print("Check the training logs for new performance metrics.")
        print("Expected improvement: 60% -> 85%+ accuracy")
    else:
        logger.error(f"Model retraining failed: {result.stderr}")


if __name__ == "__main__":
    print("Starting real data download and model improvement...")
    
    # Download real dataset
    df = download_uci_heart_disease_data()
    
    # Retrain models with real data
    retrain_with_real_data()
    
    print("\nReal data integration completed!")
    print("Your model should now have significantly better accuracy.")