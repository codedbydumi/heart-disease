"""Fixed model training script using real UCI data with proper validation."""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from src.utils.logger import get_logger
from src.models.model_trainer import ModelTrainer
from src.data.preprocessing import HeartDiseasePreprocessor

logger = get_logger("model_training")


def load_real_uci_data():
    """Load the actual UCI heart disease dataset."""
    uci_path = project_root / "data" / "raw" / "uci_heart_disease.csv"
    
    if not uci_path.exists():
        logger.error(f"UCI dataset not found at {uci_path}")
        logger.info("Please run: python scripts/download_real_data.py")
        raise FileNotFoundError("UCI dataset not available")
    
    df = pd.read_csv(uci_path)
    logger.info(f"Loaded UCI dataset: {df.shape[0]} patients, {df.shape[1]} features")
    
    # Verify data integrity
    if df.shape[0] < 250:
        logger.warning("Dataset seems too small - may not be real UCI data")
    
    return df


def create_proper_splits(df, test_size=0.2, random_state=42):
    """Create proper train/test splits with stratification."""
    # Separate features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Create stratified split to maintain class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    # Recombine for preprocessing
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    logger.info(f"Train set: {len(train_df)} patients")
    logger.info(f"Test set: {len(test_df)} patients")
    logger.info(f"Train class distribution: {y_train.value_counts().to_dict()}")
    logger.info(f"Test class distribution: {y_test.value_counts().to_dict()}")
    
    return train_df, test_df


def train_with_cross_validation(X_train, y_train):
    """Train models with cross-validation to get better performance estimates."""
    # Initialize models with regularization to prevent overfitting
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from xgboost import XGBClassifier
    
    models = {
        'random_forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=5,  # Reduced to prevent overfitting
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        ),
        'logistic_regression': LogisticRegression(
            C=1.0,  # Regularization
            max_iter=1000,
            random_state=42
        ),
        'xgboost': XGBClassifier(
            n_estimators=100,
            max_depth=4,  # Reduced to prevent overfitting
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
    }
    
    # 5-fold cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    cv_results = {}
    trained_models = {}
    
    for name, model in models.items():
        # Cross-validation scores
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
        cv_results[name] = {
            'mean_cv_accuracy': cv_scores.mean(),
            'std_cv_accuracy': cv_scores.std(),
            'cv_scores': cv_scores
        }
        
        # Train final model on full training set
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        logger.info(f"{name}: CV Accuracy = {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Select best model based on CV performance
    best_model_name = max(cv_results.keys(), key=lambda x: cv_results[x]['mean_cv_accuracy'])
    best_model = trained_models[best_model_name]
    
    logger.info(f"Best model: {best_model_name} with CV accuracy: {cv_results[best_model_name]['mean_cv_accuracy']:.4f}")
    
    return best_model, trained_models, cv_results, best_model_name


def evaluate_model_comprehensive(model, X_test, y_test, model_name):
    """Comprehensive model evaluation."""
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc_roc': auc
    }
    
    logger.info(f"=== {model_name} Test Performance ===")
    logger.info(f"Accuracy:  {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall:    {recall:.4f}")
    logger.info(f"F1-Score:  {f1:.4f}")
    logger.info(f"AUC-ROC:   {auc:.4f}")
    
    return metrics


def save_model_artifacts(best_model, preprocessor, cv_results, test_metrics, best_model_name):
    """Save model and metadata."""
    from src.utils.helpers import save_pickle, save_json
    
    # Save model
    model_data = {
        'best_model': best_model,
        'model_name': best_model_name,
        'model_info': {
            'training_date': pd.Timestamp.now().isoformat(),
            'model_type': type(best_model).__name__,
            'cross_validation_results': cv_results,
            'test_performance': test_metrics,
            'data_source': 'UCI Heart Disease Dataset',
            'total_training_samples': len(preprocessor.feature_names) if hasattr(preprocessor, 'feature_names') else 'unknown'
        }
    }
    
    model_path = project_root / "models" / "trained_models" / "best_heart_disease_model.pkl"
    save_pickle(model_data, model_path)
    logger.info(f"Model saved to: {model_path}")
    
    # Save training metadata
    metadata_path = project_root / "models" / "metadata" / "training_results.json"
    metadata = {
        'training_date': pd.Timestamp.now().isoformat(),
        'best_model': best_model_name,
        'cross_validation_results': {k: {
            'mean_accuracy': float(v['mean_cv_accuracy']),
            'std_accuracy': float(v['std_cv_accuracy'])
        } for k, v in cv_results.items()},
        'test_performance': {k: float(v) for k, v in test_metrics.items()},
        'data_info': {
            'source': 'UCI Heart Disease Dataset',
            'total_samples': 'Loaded from real dataset'
        }
    }
    save_json(metadata, metadata_path)
    logger.info(f"Metadata saved to: {metadata_path}")


def main():
    """Main training function using real UCI data."""
    logger.info("=== Starting Model Training with Real UCI Data ===")
    
    try:
        # Load real UCI dataset
        logger.info("Loading UCI Heart Disease dataset...")
        df = load_real_uci_data()
        
        # Create proper splits
        logger.info("Creating train/test splits...")
        train_df, test_df = create_proper_splits(df)
        
        # Preprocess data
        logger.info("Preprocessing data...")
        preprocessor = HeartDiseasePreprocessor()
        
        # Fit preprocessor on training data
        X_train, y_train = preprocessor.fit_transform(train_df)
        logger.info(f"Training features after preprocessing: {X_train.shape[1]}")
        
        # Transform test data
        X_test = preprocessor.transform(test_df.drop('target', axis=1))
        y_test = test_df['target'].values
        
        # Train models with cross-validation
        logger.info("Training models with cross-validation...")
        best_model, all_models, cv_results, best_model_name = train_with_cross_validation(X_train, y_train)
        
        # Comprehensive evaluation
        logger.info("Evaluating on test set...")
        test_metrics = evaluate_model_comprehensive(best_model, X_test, y_test, best_model_name)
        
        # Save preprocessor
        preprocessor_path = project_root / "models" / "scalers" / "preprocessor.pkl"
        preprocessor.save_preprocessor(str(preprocessor_path))
        
        # Save model artifacts
        save_model_artifacts(best_model, preprocessor, cv_results, test_metrics, best_model_name)
        
        # Summary
        logger.info("="*60)
        logger.info("TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
        logger.info(f"Dataset: UCI Heart Disease ({len(df)} patients)")
        logger.info(f"Best Model: {best_model_name}")
        logger.info(f"Cross-Validation Accuracy: {cv_results[best_model_name]['mean_cv_accuracy']:.4f}")
        logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
        logger.info(f"Test AUC-ROC: {test_metrics['auc_roc']:.4f}")
        logger.info("="*60)
        
        # Show feature importance if available
        if hasattr(best_model, 'feature_importances_'):
            feature_names = preprocessor.get_feature_names()
            importances = best_model.feature_importances_
            
            # Top 5 most important features
            feature_importance = list(zip(feature_names, importances))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            logger.info("Top 5 Most Important Features:")
            for i, (feature, importance) in enumerate(feature_importance[:5], 1):
                logger.info(f"  {i}. {feature}: {importance:.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
    
    print("\n" + "="*60)
    print("✅ Model training completed successfully!")
    print("📊 Check logs above for detailed performance metrics")
    print("💾 Models saved and ready for use in dashboard/API")
    print("="*60)