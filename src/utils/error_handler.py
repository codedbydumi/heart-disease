"""Comprehensive error handling for the application."""

import traceback
from typing import Dict, Any
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from src.utils.logger import get_logger

logger = get_logger("error_handler")


class ErrorHandler:
    """Centralized error handling."""
    
    @staticmethod
    def handle_model_error(e: Exception) -> Dict[str, Any]:
        """Handle ML model related errors."""
        
        error_msg = str(e)
        
        if "not fitted" in error_msg.lower():
            return {
                "error": "model_not_ready",
                "message": "Machine learning model is not properly loaded",
                "suggestion": "Please contact administrator"
            }
        elif "shape" in error_msg.lower():
            return {
                "error": "invalid_input_format", 
                "message": "Input data format is incorrect",
                "suggestion": "Check that all required fields are provided"
            }
        else:
            logger.error(f"Model error: {error_msg}")
            return {
                "error": "model_error",
                "message": "Model prediction failed",
                "suggestion": "Please try again or contact support"
            }
    
    @staticmethod
    def handle_data_error(e: Exception) -> Dict[str, Any]:
        """Handle data processing errors."""
        
        error_msg = str(e)
        
        if "validation" in error_msg.lower():
            return {
                "error": "validation_error",
                "message": error_msg,
                "suggestion": "Please check input values are within medical ranges"
            }
        elif "missing" in error_msg.lower():
            return {
                "error": "missing_data",
                "message": error_msg,
                "suggestion": "Please provide all required patient information"
            }
        else:
            return {
                "error": "data_error", 
                "message": "Data processing failed",
                "suggestion": "Please check input format and try again"
            }
    
    @staticmethod
    def handle_api_error(e: Exception) -> Dict[str, Any]:
        """Handle API related errors."""
        
        if isinstance(e, HTTPException):
            return {
                "error": "api_error",
                "message": e.detail,
                "status_code": e.status_code
            }
        else:
            logger.error(f"API error: {str(e)}")
            return {
                "error": "internal_error",
                "message": "Internal server error",
                "suggestion": "Please try again later"
            }


async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for FastAPI."""
    
    logger.error(f"Unhandled exception: {exc}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    
    # Determine error type and create appropriate response
    if "model" in str(exc).lower():
        error_info = ErrorHandler.handle_model_error(exc)
    elif "validation" in str(exc).lower() or "data" in str(exc).lower():
        error_info = ErrorHandler.handle_data_error(exc)
    else:
        error_info = ErrorHandler.handle_api_error(exc)
    
    return JSONResponse(
        status_code=error_info.get("status_code", 500),
        content={
            "success": False,
            "error": error_info,
            "timestamp": str(pd.Timestamp.now()),
            "path": str(request.url)
        }
    )


def handle_dashboard_error(func):
    """Decorator for dashboard error handling."""
    
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            import streamlit as st
            
            logger.error(f"Dashboard error in {func.__name__}: {e}")
            
            st.error("Something went wrong!")
            
            if "connection" in str(e).lower():
                st.error("Cannot connect to API server. Please ensure it's running.")
                st.code("python scripts/start_api.py")
            elif "model" in str(e).lower():
                st.error("Model prediction failed. Please try different values.")
            else:
                st.error(f"Error: {str(e)}")
            
            # Show detailed error in development
            if st.session_state.get("show_debug", False):
                st.exception(e)
    
    return wrapper