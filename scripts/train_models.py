"""Complete training script for heart disease prediction model."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.data_loader import HeartDiseaseDataLoader
from src.data.preprocessing import HeartDiseasePreprocessor
from src.models.model_trainer import ModelTrainer
from src.models.model_evaluator import ModelEvaluator
from src.models.model_interpreter import ModelInterpreter
from src.utils.logger import get_logger

# Set up logging
logger = get_logger("train_models")


def main():
    """Main training pipeline."""
    logger.info("Starting heart disease model training pipeline...")
    
    try:
        # Step 1: Load and prepare data
        logger.info("Step 1: Loading and preparing data...")
        
        # Load data
        loader = HeartDiseaseDataLoader()
        
        # Create sample data if no real data exists
        sample_df = loader.get_sample_data(n_samples=1000)  # Generate larger sample
        logger.info(f"Generated sample data: {sample_df.shape}")
        
        # Create train/test/validation splits
        train_df, test_df, val_df = loader.create_train_test_split(
            test_size=0.2,
            validation_size=0.1,
            save_splits=True
        )
        
        logger.info(f"Data splits - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        # Step 2: Preprocess data
        logger.info("Step 2: Preprocessing data...")
        
        preprocessor = HeartDiseasePreprocessor()
        
        # Fit on training data
        X_train, y_train = preprocessor.fit_transform(train_df)
        X_val, y_val = preprocessor.transform(val_df), val_df['target'].values
        X_test, y_test = preprocessor.transform(test_df), test_df['target'].values
        
        # Get feature names for interpretation
        feature_names = preprocessor.get_feature_names()
        
        logger.info(f"Preprocessing complete - Features: {X_train.shape[1]}")
        
        # Step 3: Train model
        logger.info("Step 3: Training ensemble model...")
        
        trainer = ModelTrainer(
            optimization_method='optuna',
            n_trials=20,  # Reduced for demo
            cv_folds=5
        )
        
        model = trainer.train_ensemble_model(
            X_train, y_train, 
            X_val, y_val,
            optimize_hyperparams=True
        )
        
        # Save trained model
        model_path = model.save_model()
        logger.info(f"Model saved to: {model_path}")
        
        # Step 4: Evaluate model
        logger.info("Step 4: Evaluating model...")
        
        evaluator = ModelEvaluator(save_plots=True)
        
        evaluation_results = evaluator.comprehensive_evaluation(
            model, X_test, y_test, X_train, y_train
        )
        
        # Print key metrics
        test_metrics = evaluation_results['test_metrics']
        logger.info("=== Model Performance ===")
        logger.info(f"Accuracy: {test_metrics['accuracy']:.4f}")
        logger.info(f"Precision: {test_metrics['precision']:.4f}")
        logger.info(f"Recall: {test_metrics['recall']:.4f}")
        logger.info(f"F1-Score: {test_metrics['f1_score']:.4f}")
        logger.info(f"ROC-AUC: {test_metrics['roc_auc']:.4f}")
        logger.info(f"Sensitivity: {test_metrics['sensitivity']:.4f}")
        logger.info(f"Specificity: {test_metrics['specificity']:.4f}")
        
        # Generate medical report
        medical_report = evaluator.generate_medical_report()
        
        # Save medical report
        report_path = project_root / "reports" / "medical_evaluation_report.md"
        report_path.parent.mkdir(exist_ok=True)
        with open(report_path, 'w') as f:
            f.write(medical_report)
        logger.info(f"Medical report saved to: {report_path}")
        
        # Step 5: Model interpretation
        logger.info("Step 5: Interpreting model...")
        
        # Use subset of training data as background for SHAP
        background_indices = np.random.choice(len(X_train), size=100, replace=False)
        X_background = X_train[background_indices]
        
        interpreter = ModelInterpreter(model, X_background, feature_names)
        interpreter.setup_explainer('kernel')  # Use kernel explainer for ensemble
        
        # Generate interpretations
        interpreter.explain_predictions(X_test[:50])  # Explain first 50 test samples
        
        # Save interpretation results
        interpretation_path = interpreter.save_interpretation_results(X_test[:50])
        logger.info(f"Interpretation results saved to: {interpretation_path}")
        
        # Step 6: Final summary
        logger.info("=== Training Pipeline Complete ===")
        logger.info(f"Model Type: {model.model_name}")
        logger.info(f"Training Samples: {len(X_train)}")
        logger.info(f"Test Accuracy: {test_metrics['accuracy']:.1%}")
        logger.info(f"Model Saved: {model_path}")
        logger.info(f"Evaluation Plots: {len(evaluation_results.get('plot_paths', {}))}")
        logger.info(f"Interpretation Complete: ✅")
        
        return {
            'model': model,
            'evaluation_results': evaluation_results,
            'interpretation_path': interpretation_path,
            'model_path': model_path
        }
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    results = main()
    print("🎉 Training completed successfully!")