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
                "suggested_action": error_info.get("suggested_action"),
                "retry_possible": error_info.get("retry_possible", False),
                "retry_delay": error_info.get("retry_delay")
            },
            "timestamp": pd.Timestamp.now().isoformat(),
            "request_path": request_path,
            "help": {
                "documentation": "/docs",
                "support": "Contact system administrator",
                "retry_guidance": ErrorHandler._get_retry_guidance(error_info)
            }
        }
    
    @staticmethod
    def _get_retry_guidance(error_info: Dict[str, Any]) -> str:
        """Provide guidance on how to retry the request."""
        action = error_info.get("suggested_action", "")
        
        guidance_map = {
            "check_input_format": "Verify all required medical fields are provided with valid numeric values",
            "correct_input_values": "Ensure all medical parameters are within realistic ranges",
            "retry_later": "Wait a few minutes and try your request again",
            "wait_and_retry": "Wait for the rate limit to reset before making another request",
            "contact_support": "This error requires administrator assistance",
            "verify_input": "Double-check your patient data format and values"
        }
        
        return guidance_map.get(action, "Check your input data and try again")


async def global_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler for FastAPI with medical context.
    
    This handler processes all unhandled exceptions and returns
    user-friendly error responses with appropriate medical guidance.
    """
    logger.error(f"Unhandled exception on {request.url}: {exc}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    
    # Categorize and handle the error
    error_category = ErrorHandler.categorize_error(exc)
    
    if error_category == 'model_error':
        error_info = ErrorHandler.handle_model_error(exc)
        status_code = 503  # Service Unavailable
    elif error_category == 'validation_error':
        error_info = ErrorHandler.handle_validation_error(exc)
        status_code = 400  # Bad Request
    elif error_category == 'connection_error':
        error_info = ErrorHandler.handle_connection_error(exc)
        status_code = 503  # Service Unavailable
    elif error_category == 'api_error':
        error_info = ErrorHandler.handle_api_error(exc)
        status_code = getattr(exc, 'status_code', 500)
    else:
        error_info = ErrorHandler.handle_unknown_error(exc)
        status_code = 500  # Internal Server Error
    
    # Format the response
    response_content = ErrorHandler.format_error_response(
        error_info, 
        str(request.url.path)
    )
    
    return JSONResponse(
        status_code=status_code,
        content=response_content
    )


def handle_dashboard_error(error_context: str = ""):
    """
    Decorator for dashboard error handling with medical context.
    
    Args:
        error_context: Context description for better error messages
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                import streamlit as st
                
                logger.error(f"Dashboard error in {func.__name__} ({error_context}): {e}")
                
                # Categorize error
                error_category = ErrorHandler.categorize_error(e)
                
                if error_category == 'connection_error':
                    st.error("🚫 Cannot connect to prediction service")
                    st.info("Please ensure the API server is running:")
                    st.code("python scripts/start_api.py", language="bash")
                    
                elif error_category == 'validation_error':
                    st.error("❌ Invalid medical data")
                    error_info = ErrorHandler.handle_validation_error(e)
                    st.error(error_info.get('user_message', str(e)))
                    if error_info.get('medical_guidance'):
                        st.info(f"💡 {error_info['medical_guidance']}")
                        
                elif error_category == 'model_error':
                    st.error("🤖 Prediction model error")
                    error_info = ErrorHandler.handle_model_error(e)
                    st.error(error_info.get('user_message', 'Model prediction failed'))
                    
                    if not error_info.get('retry_possible', True):
                        st.warning("Please contact system administrator")
                        
                else:
                    st.error("⚠️ Something went wrong!")
                    st.error(f"Error: {str(e)}")
                
                # Show debug info in development mode
                if st.session_state.get("debug_mode", False):
                    with st.expander("🔧 Debug Information"):
                        st.code(f"Error Type: {type(e).__name__}")
                        st.code(f"Error Message: {str(e)}")
                        st.code(f"Function: {func.__name__}")
                        if error_context:
                            st.code(f"Context: {error_context}")
                        
                return None
        return wrapper
    return decorator


class StreamlitErrorHandler:
    """Specialized error handling for Streamlit dashboard."""
    
    @staticmethod
    def show_api_connection_error():
        """Display API connection error with troubleshooting."""
        import streamlit as st
        
        st.error("🚫 API Connection Failed")
        
        with st.expander("🔧 Troubleshooting Steps"):
            st.markdown("""
            **Step 1**: Check if API server is running
            ```bash
            # In terminal, run:
            python scripts/start_api.py
            ```
            
            **Step 2**: Verify API health
            ```bash
            # Test with curl:
            curl http://localhost:8000/health
            ```
            
            **Step 3**: Check port conflicts
            - API should run on port 8000
            - Dashboard should run on port 8501
            
            **Step 4**: Restart both services
            1. Stop API (Ctrl+C)
            2. Stop Dashboard (Ctrl+C) 
            3. Start API first, then Dashboard
            """)
    
    @staticmethod
    def show_validation_error(error_msg: str, field: str = None):
        """Display validation error with medical guidance."""
        import streamlit as st
        
        st.error(f"❌ Invalid Input: {error_msg}")
        
        # Provide field-specific guidance
        if field:
            guidance = {
                'age': "Age must be between 18-120 years",
                'trestbps': "Blood pressure: 80-300 mm Hg (normal: 90-140)",  
                'chol': "Cholesterol: 100-800 mg/dl (normal: <200)",
                'thalach': "Heart rate: 60-220 bpm (varies by age)",
                'oldpeak': "ST depression: 0-10 mm (typically 0-4)"
            }
            
            if field in guidance:
                st.info(f"💡 {guidance[field]}")
        
        st.info("Please correct the highlighted values and try again.")
    
    @staticmethod
    def show_prediction_error(risk_percentage: float = None):
        """Display prediction error with context."""
        import streamlit as st
        
        st.error("🤖 Prediction Failed")
        
        if risk_percentage is not None and risk_percentage < 0:
            st.warning("Received invalid risk score - please check model configuration")
        
        st.info("Please try with different patient data or contact support if the issue persists.")
    
    @staticmethod
    def enable_debug_mode():
        """Enable debug mode for detailed error information."""
        import streamlit as st
        
        if st.sidebar.checkbox("🔧 Debug Mode"):
            st.session_state.debug_mode = True
            st.sidebar.success("Debug mode enabled")
        else:
            st.session_state.debug_mode = False