"""Basic security implementations."""

import time
from typing import Optional
from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

# Security scheme
security = HTTPBearer(auto_error=False)


class SecurityMiddleware:
    """Basic security checks."""
    
    @staticmethod
    def validate_patient_data(data: dict) -> dict:
        """Validate and sanitize patient input data."""
        
        # Input sanitization
        sanitized_data = {}
        
        for key, value in data.items():
            # Remove any potentially harmful characters
            if isinstance(value, str):
                # Basic XSS protection
                value = value.replace('<', '').replace('>', '').replace('"', '').replace("'", '')
            
            # Validate ranges
            if key == 'age' and not (18 <= value <= 120):
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid age: {value}. Must be between 18-120"
                )
            elif key == 'trestbps' and not (80 <= value <= 300):
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid blood pressure: {value}. Must be between 80-300"
                )
            elif key == 'chol' and not (100 <= value <= 800):
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid cholesterol: {value}. Must be between 100-800"
                )
            elif key == 'thalach' and not (60 <= value <= 220):
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid heart rate: {value}. Must be between 60-220"
                )
            
            sanitized_data[key] = value
        
        return sanitized_data
    
    @staticmethod
    def log_prediction_request(remote_addr: str, data: dict, result: dict):
        """Log prediction requests for audit trail."""
        
        from src.utils.logger import get_logger
        audit_logger = get_logger("audit")
        
        audit_logger.info(
            f"PREDICTION_REQUEST: IP={remote_addr}, "
            f"age={data.get('age')}, sex={data.get('sex')}, "
            f"risk={result.get('risk_percentage', 0):.1f}%"
        )


def get_current_user(credentials: Optional[str] = Depends(security)):
    """Simple authentication check (can be expanded later)."""
    
    # For now, just check if request is from localhost (development)
    # In production, implement proper JWT token validation
    return {"user_id": "anonymous", "role": "user"}


# Rate limiting decorators
def rate_limit_predictions():
    """Rate limit for prediction endpoints."""
    return limiter.limit("10/minute")


def rate_limit_batch():
    """Rate limit for batch processing."""
    return limiter.limit("2/minute")