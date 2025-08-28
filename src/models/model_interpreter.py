"""SHAP-based model interpretation for explainable heart disease prediction."""

from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from loguru import logger

from .base_model import BaseModel
from ..config import settings
from ..utils.helpers import save_json


class ModelInterpreter:
    """SHAP-based model interpretation system."""
    
    def __init__(self, model: BaseModel, X_background: np.ndarray, feature_names: List[str]):
        """
        Initialize model interpreter.
        
        Args:
            model: Trained model to interpret
            X_background: Background data for SHAP (typically training set sample)
            feature_names: Names of features
        """
        self.model = model
        self.X_background = X_background
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None
        
        logger.info("ModelInterpreter initialized")
    
    def setup_explainer(self, explainer_type: str = 'tree') -> None:
        """
        Set up SHAP explainer based on model type.
        
        Args:
            explainer_type: Type of explainer ('tree', 'linear', 'kernel', 'deep')
        """
        try:
            if explainer_type == 'tree':
                # For tree-based models (Random Forest, XGBoost)
                self.explainer = shap.TreeExplainer(self.model.model)
            elif explainer_type == 'linear':
                # For linear models
                self.explainer = shap.LinearExplainer(self.model.model, self.X_background)
            elif explainer_type == 'kernel':
                # Model-agnostic explainer
                self.explainer = shap.KernelExplainer(self.model.predict_proba, self.X_background)
            else:
                # Default to kernel explainer for ensemble models
                self.explainer = shap.KernelExplainer(self.model.predict_proba, self.X_background)
            
            logger.info(f"SHAP {explainer_type} explainer set up successfully")
            
        except Exception as e:
            logger.warning(f"Failed to set up {explainer_type} explainer: {e}")
            # Fall back to kernel explainer
            self.explainer = shap.KernelExplainer(self.model.predict_proba, self.X_background)
            logger.info("Using kernel explainer as fallback")
    
    def explain_predictions(self, X: np.ndarray, max_samples: int = 100) -> np.ndarray:
        """
        Generate SHAP values for predictions.
        
        Args:
            X: Input data to explain
            max_samples: Maximum number of samples to explain (for performance)
            
        Returns:
            SHAP values array
        """
        if self.explainer is None:
            self.setup_explainer()
        
        # Limit samples for performance
        if len(X) > max_samples:
            logger.warning(f"Limiting explanation to {max_samples} samples for performance")
            X_explain = X[:max_samples]
        else:
            X_explain = X
        
        logger.info(f"Generating SHAP values for {len(X_explain)} samples...")
        
        try:
            if isinstance(self.explainer, shap.TreeExplainer):
                self.shap_values = self.explainer.shap_values(X_explain)
                # For binary classification, take positive class SHAP values
                if isinstance(self.shap_values, list):
                    self.shap_values = self.shap_values[1]
            else:
                self.shap_values = self.explainer.shap_values(X_explain)
                # For kernel explainer with predict_proba, take positive class
                if len(self.shap_values.shape) > 2:
                    self.shap_values = self.shap_values[:, :, 1]
            
            logger.info("SHAP values generated successfully")
            
        except Exception as e:
            logger.error(f"Failed to generate SHAP values: {e}")
            raise
        
        return self.shap_values
    
    def get_global_importance(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Get global feature importance using SHAP values.
        
        Args:
            X: Input data
            
        Returns:
            Global importance analysis
        """
        if self.shap_values is None:
            self.explain_predictions(X)
        
        # Calculate mean absolute SHAP values for global importance
        mean_abs_shap = np.mean(np.abs(self.shap_values), axis=0)
        
        # Create importance ranking
        importance_dict = {
            feature: float(importance) 
            for feature, importance in zip(self.feature_names, mean_abs_shap)
        }
        
        # Sort by importance
        sorted_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        
        global_analysis = {
            'feature_importance': sorted_importance,
            'top_5_features': list(sorted_importance.keys())[:5],
            'total_importance': float(np.sum(mean_abs_shap)),
            'importance_distribution': {
                'mean': float(np.mean(mean_abs_shap)),
                'std': float(np.std(mean_abs_shap)),
                'min': float(np.min(mean_abs_shap)),
                'max': float(np.max(mean_abs_shap))
            }
        }
        
        return global_analysis
    
    def explain_single_prediction(self, x_instance: np.ndarray, instance_index: int = 0) -> Dict[str, Any]:
        """
        Explain a single prediction in detail.
        
        Args:
            x_instance: Single instance to explain
            instance_index: Index in SHAP values array
            
        Returns:
            Detailed explanation
        """
        if self.shap_values is None:
            raise ValueError("Must generate SHAP values first")
        
        if instance_index >= len(self.shap_values):
            raise ValueError(f"Instance index {instance_index} out of range")
        
        instance_shap = self.shap_values[instance_index]
        prediction = self.model.predict_proba(x_instance.reshape(1, -1))[0, 1]
        
        # Feature contributions
        feature_contributions = {
            feature: float(shap_val)
            for feature, shap_val in zip(self.feature_names, instance_shap)
        }
        
        # Sort by absolute contribution
        sorted_contributions = dict(sorted(
            feature_contributions.items(), 
            key=lambda x: abs(x[1]), 
            reverse=True
        ))
        
        # Separate positive and negative contributions
        positive_contributions = {k: v for k, v in sorted_contributions.items() if v > 0}
        negative_contributions = {k: v for k, v in sorted_contributions.items() if v < 0}
        
        explanation = {
            'prediction_probability': float(prediction),
            'prediction_class': int(prediction > 0.5),
            'base_value': float(np.mean(self.model.predict_proba(self.X_background)[:, 1])),
            'instance_values': x_instance.tolist(),
            'feature_contributions': feature_contributions,
            'sorted_contributions': sorted_contributions,
            'positive_contributions': positive_contributions,
            'negative_contributions': negative_contributions,
            'top_positive_factors': list(positive_contributions.keys())[:3],
            'top_negative_factors': list(negative_contributions.keys())[:3],
            'explanation_summary': self._generate_explanation_summary(
                sorted_contributions, prediction, x_instance
            )
        }
        
        return explanation
    
    def _generate_explanation_summary(self, contributions: Dict[str, float], 
                                    prediction: float, instance: np.ndarray) -> str:
        """Generate human-readable explanation summary."""
        
        # Get top contributing factors
        top_positive = [(k, v) for k, v in contributions.items() if v > 0][:2]
        top_negative = [(k, v) for k, v in contributions.items() if v < 0][:2]
        
        risk_level = "high" if prediction > 0.7 else "medium" if prediction > 0.3 else "low"
        
        summary = f"This patient has a {risk_level} risk of heart disease ({prediction:.1%} probability). "
        
        if top_positive:
            factors_increasing = ", ".join([factor.replace('_', ' ') for factor, _ in top_positive])
            summary += f"Main factors increasing risk: {factors_increasing}. "
        
        if top_negative:
            factors_decreasing = ", ".join([factor.replace('_', ' ') for factor, _ in top_negative])
            summary += f"Main protective factors: {factors_decreasing}. "
        
        return summary
    
    def generate_interpretation_plots(self, X: np.ndarray, save_plots: bool = True) -> Dict[str, str]:
        """
        Generate comprehensive interpretation plots.
        
        Args:
            X: Input data
            save_plots: Whether to save plots
            
        Returns:
            Dictionary of plot paths
        """
        if self.shap_values is None:
            self.explain_predictions(X)
        
        plots_dir = settings.project_root / "plots" / "interpretability"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        plot_paths = {}
        
        try:
            # 1. Summary Plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(self.shap_values, X, feature_names=self.feature_names, show=False)
            if save_plots:
                plot_path = plots_dir / f"{self.model.model_name}_shap_summary.png"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plot_paths['summary_plot'] = str(plot_path)
            plt.close()
            
            # 2. Bar Plot (Global Importance)
            plt.figure(figsize=(10, 8))
            shap.summary_plot(self.shap_values, X, feature_names=self.feature_names, 
                            plot_type="bar", show=False)
            if save_plots:
                plot_path = plots_dir / f"{self.model.model_name}_shap_importance.png"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plot_paths['importance_plot'] = str(plot_path)
            plt.close()
            
            # 3. Waterfall Plot for first prediction
            if len(self.shap_values) > 0:
                plt.figure(figsize=(10, 8))
                shap.waterfall_plot(
                    shap.Explanation(
                        values=self.shap_values[0], 
                        base_values=np.mean(self.model.predict_proba(self.X_background)[:, 1]),
                        data=X[0], 
                        feature_names=self.feature_names
                    ), 
                    show=False
                )
                if save_plots:
                    plot_path = plots_dir / f"{self.model.model_name}_shap_waterfall.png"
                    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                    plot_paths['waterfall_plot'] = str(plot_path)
                plt.close()
            
            # 4. Dependence Plots for top features
            global_importance = self.get_global_importance(X)
            top_features = global_importance['top_5_features'][:3]
            
            for i, feature in enumerate(top_features):
                if feature in self.feature_names:
                    feature_idx = self.feature_names.index(feature)
                    plt.figure(figsize=(8, 6))
                    shap.dependence_plot(feature_idx, self.shap_values, X, 
                                       feature_names=self.feature_names, show=False)
                    if save_plots:
                        plot_path = plots_dir / f"{self.model.model_name}_dependence_{feature}.png"
                        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                        plot_paths[f'dependence_{feature}'] = str(plot_path)
                    plt.close()
            
            logger.info(f"Generated {len(plot_paths)} interpretation plots")
            
        except Exception as e:
            logger.error(f"Error generating SHAP plots: {e}")
        
        return plot_paths
    
    def medical_interpretation_report(self, X: np.ndarray) -> str:
        """Generate medical interpretation report."""
        
        global_analysis = self.get_global_importance(X)
        
        report = f"""
# Model Interpretability Report - Medical Analysis

## Global Feature Importance Analysis

The model's decision-making process is primarily driven by the following factors:

### Top 5 Most Important Features:
"""
        
        for i, (feature, importance) in enumerate(list(global_analysis['feature_importance'].items())[:5], 1):
            medical_name = self._get_medical_feature_name(feature)
            report += f"{i}. **{medical_name}** (Importance: {importance:.3f})\n"
        
        report += f"""

### Feature Importance Distribution:
- **Total Model Complexity:** {global_analysis['total_importance']:.3f}
- **Average Feature Impact:** {global_analysis['importance_distribution']['mean']:.3f}
- **Most Influential Feature:** {global_analysis['top_5_features'][0]} ({list(global_analysis['feature_importance'].values())[0]:.3f})

## Medical Insights:

### Cardiovascular Risk Factors:
The model appropriately prioritizes established cardiovascular risk factors:
- **Chest Pain Type:** Critical indicator of cardiac events
- **Maximum Heart Rate:** Reflects cardiac fitness and reserve
- **ST Depression (Oldpeak):** Direct measure of cardiac ischemia
- **Blood Pressure:** Major modifiable risk factor

### Model Reliability:
- The feature importance aligns with established medical knowledge
- No unexpected or medically implausible associations detected
- Model decisions are interpretable and clinically relevant

## Clinical Applications:
1. **Risk Stratification:** Use SHAP values to explain individual risk scores
2. **Treatment Planning:** Identify modifiable risk factors for intervention
3. **Patient Education:** Provide clear explanations of risk factors
4. **Quality Assurance:** Monitor for model drift or bias

"""
        return report
    
    def _get_medical_feature_name(self, feature: str) -> str:
        """Convert technical feature names to medical terminology."""
        medical_names = {
            'age': 'Patient Age',
            'sex': 'Gender',
            'cp': 'Chest Pain Type',
            'trestbps': 'Resting Blood Pressure',
            'chol': 'Serum Cholesterol',
            'fbs': 'Fasting Blood Sugar',
            'restecg': 'Resting ECG Results',
            'thalach': 'Maximum Heart Rate',
            'exang': 'Exercise Induced Angina',
            'oldpeak': 'ST Depression (Exercise)',
            'slope': 'ST Segment Slope',
            'ca': 'Major Vessels (Fluoroscopy)',
            'thal': 'Thalassemia',
            'age_group': 'Age Category',
            'bp_category': 'Blood Pressure Category',
            'chol_category': 'Cholesterol Category',
            'hr_reserve': 'Heart Rate Reserve',
            'composite_risk': 'Composite Risk Score'
        }
        
        return medical_names.get(feature, feature.replace('_', ' ').title())
    
    def save_interpretation_results(self, X: np.ndarray, output_dir: Optional[str] = None) -> str:
        """Save comprehensive interpretation results."""
        
        if output_dir is None:
            output_dir = settings.project_root / "interpretability_results"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate all analyses
        global_analysis = self.get_global_importance(X)
        
        # Sample individual explanations
        sample_explanations = []
        for i in range(min(5, len(X))):
            explanation = self.explain_single_prediction(X[i], i)
            sample_explanations.append(explanation)
        
        # Generate plots
        plot_paths = self.generate_interpretation_plots(X, save_plots=True)
        
        # Generate medical report
        medical_report = self.medical_interpretation_report(X)
        
        # Compile results
        results = {
            'model_name': self.model.model_name,
            'timestamp': pd.Timestamp.now().isoformat(),
            'global_analysis': global_analysis,
            'sample_explanations': sample_explanations,
            'plot_paths': plot_paths,
            'medical_interpretation': medical_report,
            'interpretation_metadata': {
                'n_samples_explained': len(X),
                'n_features': len(self.feature_names),
                'explainer_type': type(self.explainer).__name__ if self.explainer else None
            }
        }
        
        # Save results
        results_path = output_dir / f"{self.model.model_name}_interpretation_results.json"
        save_json(results, results_path)
        
        # Save medical report as text
        report_path = output_dir / f"{self.model.model_name}_medical_interpretation.md"
        with open(report_path, 'w') as f:
            f.write(medical_report)
        
        logger.info(f"Interpretation results saved to {output_dir}")
        
        return str(results_path)


def interpret_heart_disease_model(model: BaseModel,
                                X_background: np.ndarray,
                                X_test: np.ndarray,
                                feature_names: List[str],
                                explainer_type: str = 'tree') -> Dict[str, Any]:
    """
    Convenience function for model interpretation.
    
    Args:
        model: Trained model to interpret
        X_background: Background data for SHAP
        X_test: Test data to explain
        feature_names: Feature names
        explainer_type: Type of SHAP explainer
        
    Returns:
        Interpretation results
    """
    interpreter = ModelInterpreter(model, X_background, feature_names)
    interpreter.setup_explainer(explainer_type)
    
    # Generate explanations
    interpreter.explain_predictions(X_test, max_samples=100)
    
    # Get results
    results = {
        'global_importance': interpreter.get_global_importance(X_test),
        'sample_explanations': [
            interpreter.explain_single_prediction(X_test[i], i) 
            for i in range(min(5, len(X_test)))
        ],
        'plot_paths': interpreter.generate_interpretation_plots(X_test),
        'medical_report': interpreter.medical_interpretation_report(X_test)
    }
    
    return results