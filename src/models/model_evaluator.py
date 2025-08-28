"""Comprehensive model evaluation with medical-focused metrics."""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.calibration import calibration_curve
from loguru import logger

from .base_model import BaseModel
from ..config import settings
from ..utils.helpers import save_json


class ModelEvaluator:
    """Comprehensive model evaluation for heart disease prediction."""
    
    def __init__(self, save_plots: bool = True):
        """Initialize model evaluator."""
        self.save_plots = save_plots
        self.evaluation_results = {}
        
        logger.info("ModelEvaluator initialized")
    
    def comprehensive_evaluation(self, 
                               model: BaseModel,
                               X_test: np.ndarray,
                               y_test: np.ndarray,
                               X_train: Optional[np.ndarray] = None,
                               y_train: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Perform comprehensive model evaluation.
        
        Args:
            model: Trained model to evaluate
            X_test: Test features
            y_test: Test targets
            X_train: Training features (for comparison)
            y_train: Training targets (for comparison)
            
        Returns:
            Comprehensive evaluation results
        """
        logger.info("Starting comprehensive model evaluation...")
        
        results = {
            'model_name': model.model_name,
            'evaluation_timestamp': pd.Timestamp.now().isoformat(),
            'test_metrics': {},
            'train_metrics': {},
            'medical_analysis': {},
            'plot_paths': {}
        }
        
        # Test set evaluation
        results['test_metrics'] = self._evaluate_dataset(model, X_test, y_test, "test")
        
        # Training set evaluation (if provided)
        if X_train is not None and y_train is not None:
            results['train_metrics'] = self._evaluate_dataset(model, X_train, y_train, "train")
            results['overfitting_analysis'] = self._analyze_overfitting(
                results['train_metrics'], results['test_metrics']
            )
        
        # Medical analysis
        results['medical_analysis'] = self._medical_analysis(model, X_test, y_test)
        
        # Generate plots
        if self.save_plots:
            results['plot_paths'] = self._generate_evaluation_plots(model, X_test, y_test)
        
        # Risk stratification analysis
        results['risk_stratification'] = self._analyze_risk_stratification(model, X_test, y_test)
        
        # Feature importance (if available)
        if hasattr(model, 'get_feature_importance'):
            results['feature_importance'] = model.get_feature_importance()
        
        self.evaluation_results = results
        
        # Save evaluation report
        self._save_evaluation_report(results)
        
        logger.info("Comprehensive evaluation completed")
        
        return results
    
    def _evaluate_dataset(self, model: BaseModel, X: np.ndarray, y: np.ndarray, dataset_name: str) -> Dict[str, Any]:
        """Evaluate model on a specific dataset."""
        # Get predictions
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)[:, 1]
        
        # Standard metrics
        metrics = model.evaluate(X, y)
        
        # Additional metrics
        metrics.update({
            'average_precision': average_precision_score(y, y_pred_proba),
            'brier_score': np.mean((y_pred_proba - y) ** 2)  # Calibration metric
        })
        
        # Confusion matrix details
        cm = confusion_matrix(y, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Classification report
        class_report = classification_report(y, y_pred, output_dict=True)
        metrics['classification_report'] = class_report
        
        logger.info(f"{dataset_name.capitalize()} evaluation - Accuracy: {metrics['accuracy']:.4f}, AUC: {metrics['roc_auc']:.4f}")
        
        return metrics
    
    def _analyze_overfitting(self, train_metrics: Dict, test_metrics: Dict) -> Dict[str, Any]:
        """Analyze overfitting by comparing train and test performance."""
        overfitting_analysis = {}
        
        for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']:
            train_score = train_metrics.get(metric, 0)
            test_score = test_metrics.get(metric, 0)
            
            difference = train_score - test_score
            relative_difference = (difference / train_score * 100) if train_score > 0 else 0
            
            overfitting_analysis[metric] = {
                'train_score': train_score,
                'test_score': test_score,
                'difference': difference,
                'relative_difference_percent': relative_difference
            }
        
        # Overall overfitting assessment
        avg_relative_diff = np.mean([v['relative_difference_percent'] for v in overfitting_analysis.values()])
        
        if avg_relative_diff > 10:
            overfitting_status = "high_overfitting"
        elif avg_relative_diff > 5:
            overfitting_status = "moderate_overfitting"
        else:
            overfitting_status = "good_generalization"
        
        overfitting_analysis['overall_assessment'] = {
            'status': overfitting_status,
            'average_relative_difference': avg_relative_diff
        }
        
        return overfitting_analysis
    
    def _medical_analysis(self, model: BaseModel, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Perform medical-specific analysis."""
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        medical_analysis = {}
        
        # Cost analysis (medical perspective)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        # Medical costs (example values - adjust based on real medical costs)
        cost_fn = 50000  # Cost of missing a heart disease case
        cost_fp = 5000   # Cost of false alarm
        cost_tn = 0      # Cost of correct negative
        cost_tp = 1000   # Cost of correct positive (treatment cost)
        
        total_cost = (fn * cost_fn + fp * cost_fp + tn * cost_tn + tp * cost_tp)
        
        medical_analysis['cost_analysis'] = {
            'false_negative_cost': fn * cost_fn,
            'false_positive_cost': fp * cost_fp,
            'true_negative_cost': tn * cost_tn,
            'true_positive_cost': tp * cost_tp,
            'total_cost': total_cost,
            'cost_per_patient': total_cost / len(y_test)
        }
        
        # Clinical decision thresholds
        medical_analysis['clinical_thresholds'] = self._analyze_clinical_thresholds(y_test, y_pred_proba)
        
        # Risk group analysis
        medical_analysis['risk_groups'] = self._analyze_risk_groups(y_pred_proba, y_test)
        
        return medical_analysis
    
    def _analyze_clinical_thresholds(self, y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, Any]:
        """Analyze different clinical decision thresholds."""
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
        threshold_analysis = {}
        
        for threshold in thresholds:
            y_pred_threshold = (y_proba > threshold).astype(int)
            
            from sklearn.metrics import precision_score, recall_score
            
            precision = precision_score(y_true, y_pred_threshold, zero_division=0)
            recall = recall_score(y_true, y_pred_threshold, zero_division=0)
            
            # Calculate medical metrics
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred_threshold).ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            threshold_analysis[f'threshold_{threshold}'] = {
                'precision': precision,
                'recall': recall,
                'sensitivity': sensitivity,
                'specificity': specificity,
                'false_positives': int(fp),
                'false_negatives': int(fn)
            }
        
        return threshold_analysis
    
    def _analyze_risk_groups(self, y_proba: np.ndarray, y_true: np.ndarray) -> Dict[str, Any]:
        """Analyze model performance across different risk groups."""
        # Define risk groups
        low_risk = y_proba < 0.3
        medium_risk = (y_proba >= 0.3) & (y_proba < 0.7)
        high_risk = y_proba >= 0.7
        
        risk_analysis = {}
        
        for risk_name, risk_mask in [('low_risk', low_risk), ('medium_risk', medium_risk), ('high_risk', high_risk)]:
            if np.sum(risk_mask) > 0:
                group_true = y_true[risk_mask]
                group_proba = y_proba[risk_mask]
                
                risk_analysis[risk_name] = {
                    'count': int(np.sum(risk_mask)),
                    'actual_positive_rate': float(np.mean(group_true)),
                    'average_predicted_probability': float(np.mean(group_proba)),
                    'calibration_score': float(np.mean(np.abs(group_proba - group_true)))
                }
        
        return risk_analysis
    
    def _analyze_risk_stratification(self, model: BaseModel, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Analyze risk stratification performance."""
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Divide into risk quartiles
        quartiles = np.percentile(y_pred_proba, [25, 50, 75])
        
        risk_strata = {
            'Q1_low_risk': y_pred_proba <= quartiles[0],
            'Q2_low_medium': (y_pred_proba > quartiles[0]) & (y_pred_proba <= quartiles[1]),
            'Q3_medium_high': (y_pred_proba > quartiles[1]) & (y_pred_proba <= quartiles[2]),
            'Q4_high_risk': y_pred_proba > quartiles[2]
        }
        
        stratification_results = {}
        
        for stratum_name, stratum_mask in risk_strata.items():
            if np.sum(stratum_mask) > 0:
                stratum_true = y_test[stratum_mask]
                stratum_proba = y_pred_proba[stratum_mask]
                
                stratification_results[stratum_name] = {
                    'count': int(np.sum(stratum_mask)),
                    'actual_event_rate': float(np.mean(stratum_true)),
                    'predicted_event_rate': float(np.mean(stratum_proba)),
                    'risk_range': {
                        'min': float(np.min(stratum_proba)),
                        'max': float(np.max(stratum_proba)),
                        'mean': float(np.mean(stratum_proba))
                    }
                }
        
        return stratification_results
    
    def _generate_evaluation_plots(self, model: BaseModel, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, str]:
        """Generate comprehensive evaluation plots."""
        plot_paths = {}
        plots_dir = settings.project_root / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # 1. Confusion Matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Disease', 'Disease'],
                   yticklabels=['No Disease', 'Disease'])
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plot_path = plots_dir / f"{model.model_name}_confusion_matrix.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_paths['confusion_matrix'] = str(plot_path)
        
        # 2. ROC Curve
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plot_path = plots_dir / f"{model.model_name}_roc_curve.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_paths['roc_curve'] = str(plot_path)
        
        # 3. Precision-Recall Curve
        plt.figure(figsize=(8, 6))
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        avg_precision = average_precision_score(y_test, y_pred_proba)
        plt.plot(recall, precision, color='blue', lw=2, label=f'Precision-Recall Curve (AP = {avg_precision:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plot_path = plots_dir / f"{model.model_name}_precision_recall.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_paths['precision_recall'] = str(plot_path)
        
        # 4. Prediction Distribution
        plt.figure(figsize=(10, 6))
        plt.hist(y_pred_proba[y_test == 0], bins=30, alpha=0.7, label='No Disease', color='blue')
        plt.hist(y_pred_proba[y_test == 1], bins=30, alpha=0.7, label='Disease', color='red')
        plt.xlabel('Predicted Probability')
        plt.ylabel('Frequency')
        plt.title('Distribution of Predicted Probabilities')
        plt.legend()
        plot_path = plots_dir / f"{model.model_name}_prediction_distribution.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_paths['prediction_distribution'] = str(plot_path)
        
        # 5. Calibration Plot
        plt.figure(figsize=(8, 6))
        fraction_of_positives, mean_predicted_value = calibration_curve(y_test, y_pred_proba, n_bins=10)
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
        plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title('Calibration Plot')
        plt.legend()
        plot_path = plots_dir / f"{model.model_name}_calibration.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_paths['calibration'] = str(plot_path)
        
        logger.info(f"Generated {len(plot_paths)} evaluation plots")
        
        return plot_paths
    
    def _save_evaluation_report(self, results: Dict[str, Any]) -> None:
        """Save comprehensive evaluation report."""
        reports_dir = settings.project_root / "reports"
        reports_dir.mkdir(exist_ok=True)
        
        report_path = reports_dir / f"{results['model_name']}_evaluation_report.json"
        save_json(results, report_path)
        
        logger.info(f"Evaluation report saved to {report_path}")
    
    def generate_medical_report(self, results: Optional[Dict[str, Any]] = None) -> str:
        """Generate a medical-style evaluation report."""
        if results is None:
            results = self.evaluation_results
        
        if not results:
            raise ValueError("No evaluation results available")
        
        report = f"""
# Heart Disease Prediction Model - Clinical Evaluation Report

**Model:** {results['model_name']}
**Evaluation Date:** {results['evaluation_timestamp']}

## Executive Summary

The {results['model_name']} demonstrates strong predictive performance for heart disease risk assessment:

- **Accuracy:** {results['test_metrics']['accuracy']:.1%}
- **Sensitivity (Recall):** {results['test_metrics']['sensitivity']:.1%}
- **Specificity:** {results['test_metrics']['specificity']:.1%}
- **Area Under ROC Curve:** {results['test_metrics']['roc_auc']:.3f}

## Clinical Performance Metrics

### Diagnostic Accuracy
- **Positive Predictive Value:** {results['test_metrics']['ppv']:.1%}
- **Negative Predictive Value:** {results['test_metrics']['npv']:.1%}
- **False Positive Rate:** {results['test_metrics']['false_positive_rate']:.1%}
- **False Negative Rate:** {results['test_metrics']['false_negative_rate']:.1%}

### Risk Stratification Performance
"""
        
        if 'risk_stratification' in results:
            for stratum, metrics in results['risk_stratification'].items():
                report += f"- **{stratum.replace('_', ' ').title()}:** {metrics['count']} patients, {metrics['actual_event_rate']:.1%} event rate\n"
        
        report += f"""

## Medical Cost Analysis
"""
        
        if 'medical_analysis' in results and 'cost_analysis' in results['medical_analysis']:
            cost_analysis = results['medical_analysis']['cost_analysis']
            report += f"""
- **Total Medical Cost:** ${cost_analysis['total_cost']:,.0f}
- **Cost per Patient:** ${cost_analysis['cost_per_patient']:.0f}
- **False Negative Cost:** ${cost_analysis['false_negative_cost']:,.0f}
- **False Positive Cost:** ${cost_analysis['false_positive_cost']:,.0f}
"""
        
        report += """
## Clinical Recommendations

1. **High-Risk Patients (>70% probability):** Immediate cardiology referral
2. **Medium-Risk Patients (30-70% probability):** Additional testing recommended
3. **Low-Risk Patients (<30% probability):** Routine monitoring sufficient

## Model Limitations

- Model trained on specific population demographics
- Regular recalibration recommended with new data
- Clinical judgment should always complement model predictions
"""
        
        return report


def evaluate_heart_disease_model(model: BaseModel,
                               X_test: np.ndarray,
                               y_test: np.ndarray,
                               X_train: Optional[np.ndarray] = None,
                               y_train: Optional[np.ndarray] = None,
                               save_plots: bool = True) -> Dict[str, Any]:
    """
    Convenience function for comprehensive model evaluation.
    
    Args:
        model: Trained model to evaluate
        X_test: Test features
        y_test: Test targets
        X_train: Training features (optional)
        y_train: Training targets (optional)
        save_plots: Whether to save evaluation plots
        
    Returns:
        Comprehensive evaluation results
    """
    evaluator = ModelEvaluator(save_plots=save_plots)
    
    results = evaluator.comprehensive_evaluation(
        model, X_test, y_test, X_train, y_train
    )
    
    return results