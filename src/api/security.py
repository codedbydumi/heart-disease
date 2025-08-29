"""Security implementations for the API."""

import re
import time
from typing import Dict, Any, Optional
from datetime import datetime

from fastapi import HTTPException
from src.utils.logger import get_logger

logger = get_logger("security")
audit_logger = get_logger("audit")


class SecurityMiddleware:
    """Security validation and sanitization."""
    
    # Medical parameter ranges for validation
    MEDICAL_RANGES = {
        'age': (18, 120),
        'sex': (0, 1), 
        'cp': (0, 3),
        'trestbps': (80, 300),
        'chol': (100, 800),
        'fbs': (0, 1),
        'restecg': (0, 2),
        'thalach': (60, 220),
        'exang': (0, 1),
        'oldpeak': (0.0, 10.0),
        'slope': (0, 2),
        'ca': (0, 4),
        'thal': (1, 3)
    }
    
    @classmethod
    def sanitize_string_input(cls, value: str) -> str:
        """Sanitize string inputs to prevent XSS and injection attacks."""
        if not isinstance(value, str):
            return str(value)
        
        # Remove potentially harmful characters
        sanitized = re.sub(r'[<>"\'\(\)\[\]{}|\\^$*+?.]', '', value)
        
        # Limit length to prevent buffer overflow
        sanitized = sanitized[:100]
        
        # Remove multiple spaces
        sanitized = re.sub(r'\s+', ' ', sanitized).strip()
        
        return sanitized
    
    @classmethod
    def validate_numeric_range(cls, field: str, value: float) -> float:
        """Validate numeric values are within medical ranges."""
        if field not in cls.MEDICAL_RANGES:
            return value
        
        min_val, max_val = cls.MEDICAL_RANGES[field]
        
        if not (min_val <= value <= max_val):
            raise ValueError(
                f"Invalid {field}: {value}. Must be between {min_val} and {max_val}. "
                f"Please check that this is a realistic medical value."
            )
        
        return value
    
    @classmethod
    def validate_required_fields(cls, data: Dict[str, Any]) -> None:
        """Validate all required fields are present."""
        required_fields = list(cls.MEDICAL_RANGES.keys())
        
        missing_fields = []
        for field in required_fields:
            if field not in data or data[field] is None:
                missing_fields.append(field)
        
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")


def validate_patient_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Comprehensive validation and sanitization of patient input data.
    
    Args:
        data: Raw patient data dictionary
        
    Returns:
        Validated and sanitized data dictionary
        
    Raises:
        ValueError: If validation fails
    """
    try:
        # Check for required fields
        SecurityMiddleware.validate_required_fields(data)
        
        validated_data = {}
        
        for field, value in data.items():
            # Handle string inputs (shouldn't be any in medical data, but safety check)
            if isinstance(value, str):
                try:
                    # Try to convert to number
                    if '.' in value:
                        value = float(value)
                    else:
                        value = int(value)
                except ValueError:
                    raise ValueError(f"Invalid numeric value for {field}: {value}")
            
            # Validate numeric ranges
            validated_value = SecurityMiddleware.validate_numeric_range(field, value)
            validated_data[field] = validated_value
        
        # Additional medical logic validations
        _validate_medical_logic(validated_data)
        
        logger.info(f"Patient data validation passed for age {validated_data.get('age')}")
        
        return validated_data
        
    except ValueError as e:
        logger.warning(f"Patient data validation failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected validation error: {e}")
        raise ValueError(f"Data validation failed: {str(e)}")


def _validate_medical_logic(data: Dict[str, Any]) -> None:
    """Validate medical logic relationships between parameters."""
    
    # Age-related validations
    age = data.get('age', 0)
    thalach = data.get('thalach', 0)
    
    # Maximum heart rate should generally decrease with age
    # Rough formula: max_hr = 220 - age, but allow some flexibility
    expected_max_hr = 220 - age
    if thalach > expected_max_hr + 30:
        logger.warning(f"Unusually high heart rate {thalach} for age {age}")
    
    # Blood pressure and cholesterol relationship checks
    trestbps = data.get('trestbps', 0)
    chol = data.get('chol', 0)
    
    # Very high cholesterol with normal BP might be unusual
    if chol > 400 and trestbps < 120:
        logger.warning(f"High cholesterol {chol} with low BP {trestbps} - unusual combination")
    
    # Sex-specific validations could be added here
    # (e.g., certain conditions are more common in males vs females)


def log_prediction_audit(
    client_ip: str, 
    patient_data: Dict[str, Any], 
    prediction_result: Dict[str, Any],
    request_type: str = "single"
) -> None:
    """
    Log prediction requests for audit trail and monitoring.
    
    Args:
        client_ip: Client IP address
        patient_data: Input patient data (sanitized)
        prediction_result: Model prediction result
        request_type: Type of request (single, batch)
    """
    try:
        # Create audit log entry
        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "client_ip": client_ip,
            "request_type": request_type,
            "patient_age": patient_data.get('age'),
            "patient_sex": "M" if patient_data.get('sex') == 1 else "F",
            "risk_percentage": prediction_result.get('risk_percentage'),
            "risk_category": prediction_result.get('risk_category'),
            "model_confidence": prediction_result.get('confidence')
        }
        
        # Log the audit entry
        audit_logger.info(f"PREDICTION_AUDIT: {audit_entry}")
        
        # Additional monitoring for high-risk predictions
        if prediction_result.get('risk_percentage', 0) > 80:
            audit_logger.warning(
                f"HIGH_RISK_PREDICTION: IP={client_ip}, Age={patient_data.get('age')}, "
                f"Risk={prediction_result.get('risk_percentage'):.1f}%"
            )
        
    except Exception as e:
        logger.error(f"Audit logging failed: {e}")
        # Don't fail the request if audit logging fails


class RateLimitTracker:
    """Simple in-memory rate limit tracking."""
    
    def __init__(self):
        self.requests = {}
    
    def is_rate_limited(self, client_ip: str, limit: int = 10, window: int = 60) -> bool:
        """
        Check if client is rate limited.
        
        Args:
            client_ip: Client IP address
            limit: Number of requests allowed
            window: Time window in seconds
            
        Returns:
            True if rate limited, False otherwise
        """
        now = time.time()
        
        # Clean old entries
        if client_ip in self.requests:
            self.requests[client_ip] = [
                req_time for req_time in self.requests[client_ip] 
                if now - req_time < window
            ]
        else:
            self.requests[client_ip] = []
        
        # Check if limit exceeded
        if len(self.requests[client_ip]) >= limit:
            return True
        
        # Add current request
        self.requests[client_ip].append(now)
        return False


# Global rate limiter instance
rate_limiter = RateLimitTracker()


def check_rate_limit(client_ip: str, endpoint: str = "default") -> None:
    """
    Check rate limit for a client.
    
    Args:
        client_ip: Client IP address
        endpoint: Endpoint being accessed
        
    Raises:
        HTTPException: If rate limited
    """
    # Different limits for different endpoints
    limits = {
        "predict": (10, 60),  # 10 requests per minute
        "batch": (2, 60),     # 2 requests per minute
        "default": (20, 60)   # 20 requests per minute for other endpoints
    }
    
    limit, window = limits.get(endpoint, limits["default"])
    
    if rate_limiter.is_rate_limited(client_ip, limit, window):
        logger.warning(f"Rate limit exceeded for {client_ip} on {endpoint}")
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Maximum {limit} requests per {window} seconds."
        )


def validate_batch_size(batch_size: int, max_size: int = 100) -> None:
    """
    Validate batch processing size limits.
    
    Args:
        batch_size: Number of patients in batch
        max_size: Maximum allowed batch size
        
    Raises:
        HTTPException: If batch too large
    """
    if batch_size <= 0:
        raise HTTPException(
            status_code=400,
            detail="Batch size must be greater than 0"
        )
    
    if batch_size > max_size:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size {batch_size} exceeds maximum {max_size}. "
                   f"Please split your data into smaller batches."
        )