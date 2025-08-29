"""Core prediction service for heart disease risk assessment."""

import os
import time
from typing import Dict, List, Any, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from ..config import settings
from ..utils.helpers import load_pickle, get_model_info
from ..data.preprocessing import HeartDiseasePreprocessor
from ..data.validation import DataValidator
from .schemas import PatientInput, PredictionResponse, RiskCategory


class HeartDiseasePredictionService:
    """Professional prediction service with medical interpretations."""
    
    def __init__(self):
        """Initialize prediction service."""
        self.model = None
        self.preprocessor = None
        self.validator = None
        self.model_info = {}
        self.service_start_time = time.time()
        
        # Load model and preprocessor
        self._load_model()
        self._load_preprocessor()
        self._load_validator()
        
        logger.info("HeartDiseasePredictionService initialized successfully")
    
    def _load_model(self) -> None:
        """Load the trained model."""
        try:
            model_path = settings.models_dir / "trained_models" / "best_heart_disease_model.pkl"
            
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            model_data = load_pickle(model_path)
            self.model = model_data['best_model']
            self.model_info = model_data.get('model_info', {})
            
            logger.info(f"Model loaded successfully: {type(self.model).__name__}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _load_preprocessor(self) -> None:
        """Load the fitted preprocessor."""
        try:
            preprocessor_path = settings.models_dir / "scalers" / "preprocessor.pkl"
            
            if not preprocessor_path.exists():
                raise FileNotFoundError(f"Preprocessor file not found: {preprocessor_path}")
            
            self.preprocessor = HeartDiseasePreprocessor()
            self.preprocessor.load_preprocessor(str(preprocessor_path))
            
            logger.info("Preprocessor loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load preprocessor: {e}")
            raise
    
    def _load_validator(self) -> None:
        """Load the data validator."""
        try:
            self.validator = DataValidator()
            logger.info("Validator loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load validator: {e}")
            raise
    
    def predict_single(self, patient_data: PatientInput) -> PredictionResponse:
        """
        Make prediction for a single patient.
        
        Args:
            patient_data: Patient input data
            
        Returns:
            Prediction response with medical interpretation
        """
        try:
            # Convert to DataFrame
            df = pd.DataFrame([patient_data.dict()])
            
            # Validate data
            validation_results = self.validator.validate_dataframe(df, check_target=False)
            if not validation_results["is_valid"]:
                logger.warning(f"Validation issues: {validation_results['errors']}")
            
            # Preprocess
            X = self.preprocessor.transform(df)
            
            # Make prediction
            prediction = self.model.predict(X)[0]
            probability = self.model.predict_proba(X)[0]
            
            # Calculate metrics
            disease_probability = probability[1]  # Probability of disease (class 1)
            risk_percentage = disease_probability * 100
            
            # Determine risk category
            risk_category = self._get_risk_category(disease_probability)
            
            # Get medical interpretation
            interpretation = self.validator.get_medical_interpretation(patient_data.dict())
            
            # Generate recommendations
            recommendations = self._generate_recommendations(patient_data, disease_probability)
            
            # Calculate confidence (distance from decision boundary)
            confidence = max(probability) * 100
            
            response = PredictionResponse(
                prediction=int(prediction),
                probability=float(disease_probability),
                risk_percentage=float(risk_percentage),
                risk_category=risk_category,
                confidence=float(confidence),
                interpretation=interpretation,
                recommendations=recommendations
            )
            
            logger.info(f"Prediction completed - Risk: {risk_percentage:.1f}%, Category: {risk_category}")
            
            return response
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
    
    def predict_batch(self, patients_data: List[PatientInput]) -> Dict[str, Any]:
        """
        Make predictions for multiple patients.
        
        Args:
            patients_data: List of patient input data
            
        Returns:
            Batch prediction results with summary
        """
        try:
            predictions = []
            risk_categories = {"low": 0, "medium": 0, "high": 0}
            total_risk = 0
            
            for i, patient_data in enumerate(patients_data):
                try:
                    prediction = self.predict_single(patient_data)
                    predictions.append(prediction)
                    
                    # Update statistics
                    risk_categories[prediction.risk_category.value] += 1
                    total_risk += prediction.risk_percentage
                    
                except Exception as e:
                    logger.error(f"Failed to predict patient {i+1}: {e}")
                    # Continue with other patients
                    continue
            
            # Calculate summary statistics
            total_patients = len(predictions)
            summary = {
                "total_processed": total_patients,
                "average_risk": total_risk / total_patients if total_patients > 0 else 0,
                "risk_distribution": risk_categories,
                "high_risk_count": risk_categories["high"],
                "high_risk_percentage": (risk_categories["high"] / total_patients * 100) if total_patients > 0 else 0
            }
            
            logger.info(f"Batch prediction completed - {total_patients} patients processed")
            
            return {
                "total_patients": total_patients,
                "predictions": predictions,
                "summary": summary
            }
            
        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            raise
    
    def _get_risk_category(self, probability: float) -> RiskCategory:
        """Determine risk category based on probability."""
        if probability < 0.3:
            return RiskCategory.LOW
        elif probability < 0.7:
            return RiskCategory.MEDIUM
        else:
            return RiskCategory.HIGH
    
    def _generate_recommendations(self, patient_data: PatientInput, risk_probability: float) -> List[str]:
        """Generate personalized health recommendations."""
        recommendations = []
        
        # Age-based recommendations
        if patient_data.age > 65:
            recommendations.append("Regular cardiac check-ups recommended due to advanced age")
        
        # Cholesterol recommendations
        if patient_data.chol > 240:
            recommendations.append("Consider cholesterol-lowering diet and medication consultation")
        elif patient_data.chol > 200:
            recommendations.append("Monitor cholesterol levels and maintain healthy diet")
        
        # Blood pressure recommendations
        if patient_data.trestbps > 140:
            recommendations.append("High blood pressure detected - consult physician for management")
        elif patient_data.trestbps > 120:
            recommendations.append("Monitor blood pressure regularly and reduce sodium intake")
        
        # Exercise recommendations
        if patient_data.exang == 1:
            recommendations.append("Exercise-induced chest pain noted - medical evaluation recommended")
        else:
            recommendations.append("Regular moderate exercise beneficial for heart health")
        
        # Chest pain recommendations
        if patient_data.cp in [1, 2, 3]:
            recommendations.append("Chest pain symptoms present - medical consultation advised")
        
        # Risk-based recommendations
        if risk_probability > 0.7:
            recommendations.append("High risk detected - immediate medical consultation recommended")
            recommendations.append("Consider comprehensive cardiac evaluation")
        elif risk_probability > 0.3:
            recommendations.append("Moderate risk - lifestyle modifications and regular monitoring advised")
        else:
            recommendations.append("Low risk - maintain healthy lifestyle for prevention")
        
        # General recommendations
        recommendations.extend([
            "Maintain healthy diet rich in fruits and vegetables",
            "Avoid smoking and limit alcohol consumption",
            "Manage stress through relaxation techniques"
        ])
        
        return recommendations[:6]  # Limit to 6 most relevant recommendations
    
    def get_service_info(self) -> Dict[str, Any]:
        """Get information about the prediction service."""
        uptime = time.time() - self.service_start_time
        
        return {
            "status": "healthy" if self.model is not None else "unhealthy",
            "model_loaded": self.model is not None,
            "preprocessor_loaded": self.preprocessor is not None,
            "validator_loaded": self.validator is not None,
            "uptime_seconds": uptime,
            "model_info": self.model_info,
            "feature_count": len(self.preprocessor.get_feature_names()) if self.preprocessor else 0
        }
    
    def get_model_details(self) -> Dict[str, Any]:
        """Get detailed model information."""
        if not self.model:
            return {"error": "Model not loaded"}
        
        return {
            "model_name": type(self.model).__name__,
            "model_version": settings.model_version,
            "training_date": self.model_info.get("training_date", "unknown"),
            "performance_metrics": self.model_info.get("performance", {}),
            "feature_names": self.preprocessor.get_feature_names() if self.preprocessor else [],
            "total_features": len(self.preprocessor.get_feature_names()) if self.preprocessor else 0
        }


# Global prediction service instance
prediction_service = None

def get_prediction_service() -> HeartDiseasePredictionService:
    """Get or create prediction service instance."""
    global prediction_service
    if prediction_service is None:
        prediction_service = HeartDiseasePredictionService()
    return prediction_service