"""Pydantic schemas for API request/response validation."""

from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, validator
from enum import Enum


class RiskCategory(str, Enum):
    """Risk categories for heart disease."""
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"


class PatientInput(BaseModel):
    """Schema for patient input data."""
    age: int = Field(..., ge=18, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="Sex (0=female, 1=male)")
    cp: int = Field(..., ge=0, le=3, description="Chest pain type (0-3)")
    trestbps: int = Field(..., ge=80, le=250, description="Resting blood pressure (mm Hg)")
    chol: int = Field(..., ge=100, le=600, description="Serum cholesterol (mg/dl)")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar >120 mg/dl (0=false, 1=true)")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG results (0-2)")
    thalach: int = Field(..., ge=60, le=220, description="Maximum heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise induced angina (0=no, 1=yes)")
    oldpeak: float = Field(..., ge=0.0, le=10.0, description="ST depression induced by exercise")
    slope: int = Field(..., ge=0, le=2, description="Slope of peak exercise ST segment")
    ca: int = Field(..., ge=0, le=4, description="Number of major vessels colored by fluoroscopy")
    thal: int = Field(..., ge=1, le=3, description="Thalassemia type")

    @validator('age')
    def validate_age(cls, v):
        if v < 18 or v > 120:
            raise ValueError('Age must be between 18 and 120')
        return v
    
    @validator('trestbps')
    def validate_blood_pressure(cls, v):
        if v < 70 or v > 300:
            raise ValueError('Blood pressure seems unrealistic')
        return v

    class Config:
        schema_extra = {
            "example": {
                "age": 63,
                "sex": 1,
                "cp": 3,
                "trestbps": 145,
                "chol": 233,
                "fbs": 1,
                "restecg": 0,
                "thalach": 150,
                "exang": 0,
                "oldpeak": 2.3,
                "slope": 0,
                "ca": 0,
                "thal": 1
            }
        }


class PredictionResponse(BaseModel):
    """Schema for prediction response."""
    prediction: int = Field(..., description="Predicted class (0=no disease, 1=disease)")
    probability: float = Field(..., description="Probability of heart disease (0-1)")
    risk_percentage: float = Field(..., description="Risk percentage (0-100)")
    risk_category: RiskCategory = Field(..., description="Risk category")
    confidence: float = Field(..., description="Model confidence score")
    interpretation: Dict[str, str] = Field(..., description="Medical interpretation")
    recommendations: List[str] = Field(..., description="Health recommendations")


class BatchPatientInput(BaseModel):
    """Schema for batch prediction requests."""
    patients: List[PatientInput] = Field(..., description="List of patient data")
    return_detailed: bool = Field(default=True, description="Return detailed analysis")


class BatchPredictionResponse(BaseModel):
    """Schema for batch prediction responses."""
    total_patients: int = Field(..., description="Total number of patients processed")
    predictions: List[PredictionResponse] = Field(..., description="Individual predictions")
    summary: Dict[str, Any] = Field(..., description="Batch summary statistics")


class HealthResponse(BaseModel):
    """Schema for health check response."""
    status: str = Field(..., description="Service status")
    timestamp: str = Field(..., description="Check timestamp")
    version: str = Field(..., description="API version")
    model_loaded: bool = Field(..., description="Whether ML model is loaded")
    database_connected: bool = Field(..., description="Database connection status")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")


class ModelInfoResponse(BaseModel):
    """Schema for model information response."""
    model_name: str = Field(..., description="Name of the model")
    model_version: str = Field(..., description="Model version")
    training_date: str = Field(..., description="When the model was trained")
    performance_metrics: Dict[str, float] = Field(..., description="Model performance")
    feature_names: List[str] = Field(..., description="Model feature names")
    total_features: int = Field(..., description="Number of features")


class ErrorResponse(BaseModel):
    """Schema for error responses."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: str = Field(..., description="Error timestamp")