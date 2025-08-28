"""Test script for data pipeline functionality."""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.data_loader import HeartDiseaseDataLoader, load_heart_disease_dataset
from data.validation import DataValidator
from data.preprocessing import HeartDiseasePreprocessor
from utils.logger import get_logger

logger = get_logger("test_data_pipeline")


def test_data_validation():
    """Test data validation functionality."""
    logger.info("Testing data validation...")
    
    validator = DataValidator()
    
    # Get sample data
    loader = HeartDiseaseDataLoader()
    sample_df = loader.get_sample_data(n_samples=10)
    
    # Test validation
    validation_results = validator.validate_dataframe(sample_df)
    
    logger.info(f"Validation results: {validation_results['is_valid']}")
    if validation_results['errors']:
        logger.warning(f"Errors: {validation_results['errors']}")
    
    # Test medical interpretation
    sample_row = sample_df.iloc[0]
    interpretation = validator.get_medical_interpretation(sample_row)
    logger.info(f"Medical interpretation: {interpretation}")


def test_data_loading():
    """Test data loading functionality."""
    logger.info("Testing data loading...")
    
    loader = HeartDiseaseDataLoader()
    
    # Test sample data generation
    sample_df = loader.get_sample_data(n_samples=20)
    logger.info(f"Generated sample data: {sample_df.shape}")
    
    # Test database operations
    batch_id = loader.save_to_database(sample_df, source="test_script")
    logger.info(f"Saved to database with batch_id: {batch_id}")
    
    # Load from database
    loaded_df = loader.load_from_database(limit=10)
    logger.info(f"Loaded from database: {loaded_df.shape}")
    
    # Get statistics
    stats = loader.get_data_statistics()
    logger.info(f"Database statistics: {stats}")


def test_preprocessing():
    """Test preprocessing functionality."""
    logger.info("Testing preprocessing...")
    
    loader = HeartDiseaseDataLoader()
    sample_df = loader.get_sample_data(n_samples=100)
    
    preprocessor = HeartDiseasePreprocessor()
    
    # Test fit_transform
    X_train, y_train = preprocessor.fit_transform(sample_df)
    logger.info(f"Preprocessed training data: X{X_train.shape}, y{y_train.shape}")
    
    # Test transform
    X_test = preprocessor.transform(sample_df.drop(columns=['target']))
    logger.info(f"Preprocessed test data: {X_test.shape}")
    
    # Test feature names
    feature_names = preprocessor.get_feature_names()
    logger.info(f"Feature names ({len(feature_names)}): {feature_names}")
    
    # Test save/load
    preprocessor.save_preprocessor()
    logger.info("Preprocessor saved successfully")


def test_full_pipeline():
    """Test complete data pipeline."""
    logger.info("Testing complete data pipeline...")
    
    # 1. Load data
    df, loading_report = load_heart_disease_dataset()
    logger.info(f"Loaded data: {df.shape}, Report: {loading_report}")
    
    # 2. Create train/test split
    loader = HeartDiseaseDataLoader()
    train_df, test_df, val_df = loader.create_train_test_split(
        test_size=0.2,
        validation_size=0.1,
        save_splits=True
    )
    
    # 3. Preprocess data
    preprocessor = HeartDiseasePreprocessor()
    (X_train, y_train), (X_test, y_test) = preprocessor.fit_transform(train_df), preprocessor.transform(test_df)
    
    logger.info(f"Pipeline complete - Train: {X_train.shape}, Test: {X_test.shape}")


if __name__ == "__main__":
    logger.info("Starting data pipeline tests...")
    
    try:
        test_data_validation()
        test_data_loading() 
        test_preprocessing()
        test_full_pipeline()
        
        logger.info("All tests completed successfully! ✅")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise