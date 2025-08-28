"""Simple training script for Enhanced Intermediate level."""

import sys
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.data_loader import HeartDiseaseDataLoader
from src.data.preprocessing import HeartDiseasePreprocessor
from src.models.model_trainer import ModelTrainer
from src.utils.logger import get_logger

logger = get_logger("train_model")


def main():
    """Simple training pipeline."""
    logger.info("Starting model training...")
    
    try:
        # Step 1: Load and prepare data with MORE samples
        logger.info("Loading data...")
        loader = HeartDiseaseDataLoader()
        
        # Generate MORE sample data (need at least 100+ for proper ML)
        sample_df = loader.get_sample_data(n_samples=200)  # Increased from 5 to 200
        logger.info(f"Generated sample data: {sample_df.shape}")
        
        # Check class balance
        class_distribution = sample_df['target'].value_counts()
        logger.info(f"Class distribution: {dict(class_distribution)}")
        
        # Save sample data to database for future use
        batch_id = loader.save_to_database(sample_df, source="training_script")
        logger.info(f"Saved sample data to database with batch_id: {batch_id}")
        
        # Step 2: Split the sample data directly (not from database)
        logger.info("Splitting data...")
        from sklearn.model_selection import train_test_split
        
        # First split: train+val vs test
        train_val_df, test_df = train_test_split(
            sample_df,
            test_size=0.2,
            random_state=42,
            stratify=sample_df['target']
        )
        
        # Second split: train vs validation
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=0.125,  # 0.125 of total = 0.1 validation, 0.7 train
            random_state=42,
            stratify=train_val_df['target']
        )
        
        logger.info(f"Data splits - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        # Verify class balance in each split
        logger.info(f"Train class distribution: {dict(train_df['target'].value_counts())}")
        logger.info(f"Val class distribution: {dict(val_df['target'].value_counts())}")
        logger.info(f"Test class distribution: {dict(test_df['target'].value_counts())}")
        
        # Step 3: Preprocess
        logger.info("Preprocessing data...")
        preprocessor = HeartDiseasePreprocessor()
        X_train, y_train = preprocessor.fit_transform(train_df)
        X_val, y_val = preprocessor.transform(val_df), val_df['target'].values
        X_test, y_test = preprocessor.transform(test_df), test_df['target'].values
        logger.info(f"Features after preprocessing: {X_train.shape[1]}")
        
        # Step 4: Train with simplified approach (no hyperparameter tuning for small dataset)
        logger.info("Training models...")
        trainer = ModelTrainer(tuning_method='none')  # Use 'none' to avoid CV issues with small data
        results = trainer.train_ensemble(X_train, y_train, X_val, y_val)
        
        # Step 5: Evaluate on test set
        logger.info("Evaluating on test set...")
        test_predictions = trainer.predict(X_test)
        test_probabilities = trainer.predict_proba(X_test)
        test_accuracy = accuracy_score(y_test, test_predictions)
        
        # Step 6: Print comprehensive results
        logger.info("=" * 50)
        logger.info("🎉 TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 50)
        logger.info(f"📊 BEST MODEL: {results['best_model_name'].upper()}")
        logger.info(f"🎯 Best Validation Score: {results['best_score']:.4f}")
        logger.info(f"🏆 Final Test Accuracy: {test_accuracy:.4f}")
        logger.info("-" * 30)
        logger.info("📈 Individual Model Performance:")
        
        for model_name, model_results in results['individual_models'].items():
            logger.info(f"   {model_name}: Val = {model_results['val_accuracy']:.4f}, Train = {model_results['train_accuracy']:.4f}")
        
        # Step 7: Save model and preprocessor
        model_path = trainer.save_model()
        preprocessor_path = preprocessor.save_preprocessor()
        
        logger.info("-" * 30)
        logger.info("💾 Saved Artifacts:")
        logger.info(f"   Model: {model_path}")
        logger.info(f"   Preprocessor: {preprocessor_path}")
        
        # Step 8: Show some example predictions
        logger.info("-" * 30)
        logger.info("🔮 Example Predictions:")
        for i in range(min(5, len(X_test))):
            prob = test_probabilities[i, 1] if len(test_probabilities[i]) > 1 else test_probabilities[i]
            actual = y_test[i]
            predicted = test_predictions[i]
            status = "✅" if predicted == actual else "❌"
            logger.info(f"   Patient {i+1}: Risk={prob:.1%}, Predicted={predicted}, Actual={actual} {status}")
        
        logger.info("=" * 50)
        
        return trainer, results
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    trainer, results = main()
    print("\n🎉 Training completed successfully!")
    print(f"📊 Best Model: {results['best_model_name']}")