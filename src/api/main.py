"""FastAPI application for heart disease prediction."""

import time
from datetime import datetime
from typing import List

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from ..config import settings
from .schemas import (
    PatientInput, PredictionResponse, BatchPatientInput, 
    BatchPredictionResponse, HealthResponse, ModelInfoResponse, ErrorResponse
)
from .prediction_service import get_prediction_service, HeartDiseasePredictionService
from ..utils.logger import get_logger

logger = get_logger("fastapi_app")

# Create FastAPI application
app = FastAPI(
    title="Heart Disease Risk Prediction API",
    description="Professional API for heart disease risk assessment using machine learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Application startup time
app_start_time = time.time()


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for better error responses."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="internal_server_error",
            message="An internal server error occurred",
            details={"exception": str(exc)},
            timestamp=datetime.now().isoformat()
        ).dict()
    )


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("Starting Heart Disease Prediction API...")
    try:
        # Initialize prediction service
        get_prediction_service()
        logger.info("API startup completed successfully")
    except Exception as e:
        logger.error(f"Failed to start API: {e}")
        raise


@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Heart Disease Risk Prediction API",
        "version": "1.0.0",
        "status": "healthy",
        "endpoints": {
            "prediction": "/predict",
            "batch_prediction": "/predict/batch", 
            "health": "/health",
            "model_info": "/model/info",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check(service: HeartDiseasePredictionService = Depends(get_prediction_service)):
    """Health check endpoint."""
    try:
        service_info = service.get_service_info()
        uptime = time.time() - app_start_time
        
        return HealthResponse(
            status="healthy" if service_info["model_loaded"] else "degraded",
            timestamp=datetime.now().isoformat(),
            version="1.0.0",
            model_loaded=service_info["model_loaded"],
            database_connected=True,  # Assuming SQLite is always available
            uptime_seconds=uptime
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            timestamp=datetime.now().isoformat(),
            version="1.0.0",
            model_loaded=False,
            database_connected=False,
            uptime_seconds=0
        )


@app.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info(service: HeartDiseasePredictionService = Depends(get_prediction_service)):
    """Get model information and performance metrics."""
    try:
        model_details = service.get_model_details()
        
        if "error" in model_details:
            raise HTTPException(status_code=503, detail="Model not available")
        
        return ModelInfoResponse(
            model_name=model_details["model_name"],
            model_version=model_details["model_version"],
            training_date=model_details["training_date"],
            performance_metrics=model_details["performance_metrics"],
            feature_names=model_details["feature_names"],
            total_features=model_details["total_features"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve model information")


@app.post("/predict", response_model=PredictionResponse)
async def predict_heart_disease(
    patient_data: PatientInput,
    service: HeartDiseasePredictionService = Depends(get_prediction_service)
):
    """
    Predict heart disease risk for a single patient.
    
    Returns detailed prediction with medical interpretation and recommendations.
    """
    try:
        logger.info(f"Prediction request received for patient age {patient_data.age}")
        
        prediction = service.predict_single(patient_data)
        
        logger.info(f"Prediction completed - Risk: {prediction.risk_percentage:.1f}%")
        
        return prediction
        
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Prediction service unavailable")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch_heart_disease(
    batch_data: BatchPatientInput,
    service: HeartDiseasePredictionService = Depends(get_prediction_service)
):
    """
    Predict heart disease risk for multiple patients.
    
    Processes multiple patients and returns detailed analysis with summary statistics.
    """
    try:
        if len(batch_data.patients) == 0:
            raise HTTPException(status_code=400, detail="No patient data provided")
        
        if len(batch_data.patients) > 100:  # Limit batch size
            raise HTTPException(status_code=400, detail="Batch size limited to 100 patients")
        
        logger.info(f"Batch prediction request for {len(batch_data.patients)} patients")
        
        results = service.predict_batch(batch_data.patients)
        
        response = BatchPredictionResponse(
            total_patients=results["total_patients"],
            predictions=results["predictions"],
            summary=results["summary"]
        )
        
        logger.info(f"Batch prediction completed - {results['total_patients']} patients processed")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Batch prediction service unavailable")


@app.get("/predict/sample")
async def get_sample_prediction():
    """Get a sample prediction for testing purposes."""
    sample_patient = PatientInput(
        age=63,
        sex=1,
        cp=3,
        trestbps=145,
        chol=233,
        fbs=1,
        restecg=0,
        thalach=150,
        exang=0,
        oldpeak=2.3,
        slope=0,
        ca=0,
        thal=1
    )
    
    service = get_prediction_service()
    prediction = service.predict_single(sample_patient)
    
    return {
        "sample_input": sample_patient.dict(),
        "prediction": prediction.dict()
    }


if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        log_level=settings.log_level.lower()
    )