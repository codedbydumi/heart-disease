"""Comprehensive error handling for the application."""

import traceback
import pandas as pd
from typing import Dict, Any, Optional
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from src.utils.logger import get_logger

logger = get_logger("error_handler")


class ErrorHandler:
    """Centralized error handling with medical context awareness."""
    
    @staticmethod
    def categorize_error(error: Exception) -> str:
        """Categorize error type for appropriate handling."""
        error_msg = str(error).lower()
        
        if any(keyword in error_msg for keyword in ['not fitted', 'model', 'prediction']):
            return 'model_error'
        elif any(keyword in error_msg for keyword in ['validation', 'invalid', 'range']):
            return 'validation_error'
        elif any(keyword in error_msg for keyword in ['connection', 'timeout', 'network']):
            return 'connection_error'
        elif any(keyword in error_msg for keyword in ['database', 'sql']):
            return 'database_error'
        elif isinstance(error, HTTPException):
            return 'api_error'
        else:
            return 'unknown_error'
    
    @staticmethod
    def handle_model_error(e: Exception) -> Dict[str, Any]:
        """Handle ML model related errors with medical context."""
        error_msg = str(e)
        
        if "not fitted" in error_msg.lower():
            return {
                "error_code": "MODEL_NOT_READY",
                "message": "Medical prediction model is not properly initialized",
                "user_message": "Our prediction service is temporarily unavailable. Please try again in a few minutes.",
                "technical_details": "Model not fitted - requires retraining",
                "suggested_action": "contact_support",
                "retry_possible": False
            }
        elif "shape" in error_msg.lower() or "feature" in error_msg.lower():
            return {
                "error_code": "INVALID_INPUT_FORMAT", 
                "message": "Patient data format is incompatible with prediction model",
                "user_message": "Please ensure all required medical parameters are provided with valid values.",
                "technical_details": f"Input shape mismatch: {error_msg}",
                "suggested_action": "check_input_format",
                "retry_possible": True
            }
        elif "memory" in error_msg.lower():
            return {
                "error_code": "INSUFFICIENT_RESOURCES",
                "message": "Insufficient system resources for prediction",
                "user_message": "Our servers are experiencing high load. Please try again in a moment.",
                "suggested_action": "retry_later",
                "retry_possible": True
            }
        else:
            return {
                "error_code": "MODEL_PREDICTION_FAILED",
                "message": "Model prediction encountered an unexpected error",
                "user_message": "Unable to complete risk assessment. Please verify your input data and try again.",
                "technical_details": error_msg,
                "suggested_action": "retry_with_different_data",
                "retry_possible": True
            }
    
    @staticmethod
    def handle_validation_error(e: Exception) -> Dict[str, Any]:
        """Handle data validation errors with medical guidance."""
        error_msg = str(e)
        
        # Extract field name if present
        field_name = None
        for field in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']:
            if field in error_msg.lower():
                field_name = field
                break
        
        medical_guidance = ErrorHandler._get_medical_guidance(field_name, error_msg)
        
        return {
            "error_code": "MEDICAL_VALIDATION_FAILED",
            "message": error_msg,
            "user_message": f"Please check your medical parameters. {medical_guidance}",
            "field": field_name,
            "medical_guidance": medical_guidance,
            "suggested_action": "correct_input_values",
            "retry_possible": True
        }
    
    @staticmethod
    def _get_medical_guidance(field_name: Optional[str], error_msg: str) -> str:
        """Provide medical guidance for validation errors."""
        medical_ranges = {
            'age': "Age should be between 18-120 years",
            'trestbps': "Resting blood pressure should be 80-300 mm Hg (typical range: 90-140)",
            'chol': "Cholesterol should be 100-800 mg/dl (normal: <200, high: >240)",
            'thalach': "Maximum heart rate should be 60-220 bpm (varies by age and fitness)",
            'oldpeak': "ST depression should be 0-10 mm (typically 0-4 mm)"
        }
        
        if field_name and field_name in medical_ranges:
            return medical_ranges[field_name]
        elif "age" in error_msg.lower():
            return medical_ranges['age']
        elif "blood pressure" in error_msg.lower() or "trestbps" in error_msg.lower():
            return medical_ranges['trestbps']
        elif "cholesterol" in error_msg.lower() or "chol" in error_msg.lower():
            return medical_ranges['chol']
        else:
            return "Please ensure all values are within realistic medical ranges."
    
    @staticmethod
    def handle_connection_error(e: Exception) -> Dict[str, Any]:
        """Handle connection and network errors."""
        return {
            "error_code": "SERVICE_UNAVAILABLE",
            "message": "Unable to connect to prediction service",
            "user_message": "Our prediction service is temporarily unavailable. Please try again in a few moments.",
            "technical_details": str(e),
            "suggested_action": "retry_later",
            "retry_possible": True,
            "retry_delay": 30  # seconds
        }
    
    @staticmethod
    def handle_api_error(e: Exception) -> Dict[str, Any]:
        """Handle API-specific errors."""
        if isinstance(e, HTTPException):
            status_code = e.status_code
            
            if status_code == 400:
                return {
                    "error_code": "BAD_REQUEST",
                    "message": e.detail,
                    "user_message": "Please check your input data and try again.",
                    "suggested_action": "verify_input",
                    "retry_possible": True
                }
            elif status_code == 429:
                return {
                    "error_code": "RATE_LIMIT_EXCEEDED",
                    "message": "Too many requests",
                    "user_message": "You've made too many requests. Please wait a moment before trying again.",
                    "suggested_action": "wait_and_retry",
                    "retry_possible": True,
                    "retry_delay": 60
                }
            elif status_code == 500:
                return {
                    "error_code": "INTERNAL_SERVER_ERROR",
                    "message": "Internal server error",
                    "user_message": "We're experiencing technical difficulties. Please try again later.",
                    "suggested_action": "contact_support",
                    "retry_possible": True
                }
            else:
                return {
                    "error_code": f"HTTP_{status_code}",
                    "message": e.detail,
                    "user_message": "An error occurred while processing your request.",
                    "suggested_action": "contact_support",
                    "retry_possible": False
                }
        else:
            return ErrorHandler.handle_unknown_error(e)
    
    @staticmethod
    def handle_unknown_error(e: Exception) -> Dict[str, Any]:
        """Handle unexpected errors."""
        return {
            "error_code": "UNEXPECTED_ERROR",
            "message": "An unexpected error occurred",
            "user_message": "We encountered an unexpected issue. Our team has been notified.",
            "technical_details": str(e),
            "suggested_action": "contact_support",
            "retry_possible": False
        }
    
    @staticmethod
    def format_error_response(error_info: Dict[str, Any], request_path: str = "") -> Dict[str, Any]:
        """Format error information into standardized response."""
        return {
            "success": False,
            "error": {
                "code": error_info.get("error_code", "UNKNOWN_ERROR"),
                "message": error_info.get("user_message", "An error occurred"),
                "technical_message": error_info.get("message"),
                "details": error_info.get("technical_details"),
                "field": error_info.get("field"),
                "medical_guidance": error_info.get("medical_guidance"),
                "suggested_action": error_info.get("suggested_action