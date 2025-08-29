"""Dashboard configuration and styling."""

import streamlit as st
from typing import Dict, Any

# API Configuration
API_BASE_URL = "http://localhost:8000"
API_ENDPOINTS = {
    "health": f"{API_BASE_URL}/health",
    "predict": f"{API_BASE_URL}/predict",
    "batch_predict": f"{API_BASE_URL}/predict/batch",
    "model_info": f"{API_BASE_URL}/model/info",
    "sample": f"{API_BASE_URL}/predict/sample"
}

# Medical color scheme
COLORS = {
    "primary": "#2E86AB",        # Medical blue
    "secondary": "#A23B72",      # Medical burgundy
    "success": "#F18F01",        # Medical orange
    "warning": "#C73E1D",        # Medical red
    "low_risk": "#28a745",       # Green
    "medium_risk": "#ffc107",    # Yellow
    "high_risk": "#dc3545",      # Red
    "background": "#f8f9fa",     # Light gray
    "card_bg": "#ffffff",        # White
    "text_primary": "#2c3e50",  # Dark blue-gray
    "text_secondary": "#6c757d"  # Gray
}

# Dashboard configuration
DASHBOARD_CONFIG = {
    "title": "Heart Disease Risk Assessment",
    "subtitle": "Professional ML-powered Cardiac Risk Evaluation",
    "icon": "🫀",
    "layout": "wide",
    "sidebar_state": "expanded"
}

# Form field configurations
FORM_FIELDS = {
    "age": {
        "label": "Age (years)",
        "min_value": 18,
        "max_value": 120,
        "value": 50,
        "help": "Patient's age in years"
    },
    "sex": {
        "label": "Biological Sex",
        "options": ["Female", "Male"],
        "help": "Biological sex affects heart disease risk patterns"
    },
    "cp": {
        "label": "Chest Pain Type",
        "options": [
            "Typical Angina",
            "Atypical Angina", 
            "Non-anginal Pain",
            "Asymptomatic"
        ],
        "help": "Type of chest pain experienced"
    },
    "trestbps": {
        "label": "Resting Blood Pressure (mm Hg)",
        "min_value": 80,
        "max_value": 250,
        "value": 120,
        "help": "Blood pressure while at rest"
    },
    "chol": {
        "label": "Serum Cholesterol (mg/dl)",
        "min_value": 100,
        "max_value": 600,
        "value": 200,
        "help": "Total cholesterol level in blood"
    },
    "fbs": {
        "label": "Fasting Blood Sugar",
        "options": ["≤ 120 mg/dl", "> 120 mg/dl"],
        "help": "Blood sugar level after fasting"
    },
    "restecg": {
        "label": "Resting ECG Results",
        "options": [
            "Normal",
            "ST-T wave abnormality",
            "Left ventricular hypertrophy"
        ],
        "help": "Electrocardiogram results at rest"
    },
    "thalach": {
        "label": "Maximum Heart Rate Achieved",
        "min_value": 60,
        "max_value": 220,
        "value": 150,
        "help": "Highest heart rate during exercise test"
    },
    "exang": {
        "label": "Exercise Induced Angina",
        "options": ["No", "Yes"],
        "help": "Chest pain during exercise"
    },
    "oldpeak": {
        "label": "ST Depression (mm)",
        "min_value": 0.0,
        "max_value": 10.0,
        "value": 0.0,
        "step": 0.1,
        "help": "ST depression induced by exercise relative to rest"
    },
    "slope": {
        "label": "ST Segment Slope",
        "options": ["Upsloping", "Flat", "Downsloping"],
        "help": "Slope of peak exercise ST segment"
    },
    "ca": {
        "label": "Major Vessels (0-4)",
        "min_value": 0,
        "max_value": 4,
        "value": 0,
        "help": "Number of major vessels colored by fluoroscopy"
    },
    "thal": {
        "label": "Thalassemia",
        "options": ["Normal", "Fixed Defect", "Reversible Defect"],
        "help": "Thalassemia test results"
    }
}

# Risk interpretation
RISK_INTERPRETATION = {
    "low": {
        "color": COLORS["low_risk"],
        "icon": "✅",
        "title": "Low Risk",
        "description": "Low probability of heart disease",
        "action": "Continue healthy lifestyle"
    },
    "medium": {
        "color": COLORS["medium_risk"],
        "icon": "⚠️",
        "title": "Medium Risk", 
        "description": "Moderate probability of heart disease",
        "action": "Consider lifestyle changes and regular monitoring"
    },
    "high": {
        "color": COLORS["high_risk"],
        "icon": "🚨",
        "title": "High Risk",
        "description": "High probability of heart disease",
        "action": "Immediate medical consultation recommended"
    }
}

def get_custom_css() -> str:
    """Get custom CSS for medical styling."""
    return f"""
    <style>
    /* Import medical fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Main container styling */
    .main {{
        font-family: 'Inter', sans-serif;
        background-color: {COLORS["background"]};
    }}
    
    /* Header styling */
    .main-header {{
        background: linear-gradient(135deg, {COLORS["primary"]}, {COLORS["secondary"]});
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }}
    
    .main-header h1 {{
        font-size: 2.5rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }}
    
    .main-header p {{
        font-size: 1.2rem;
        opacity: 0.9;
        margin: 0;
    }}
    
    /* Card styling */
    .medical-card {{
        background: {COLORS["card_bg"]};
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        border-left: 4px solid {COLORS["primary"]};
        margin-bottom: 1.5rem;
    }}
    
    /* Risk assessment styling */
    .risk-low {{
        background: linear-gradient(135deg, {COLORS["low_risk"]}, #20c997);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }}
    
    .risk-medium {{
        background: linear-gradient(135deg, {COLORS["medium_risk"]}, #fd7e14);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }}
    
    .risk-high {{
        background: linear-gradient(135deg, {COLORS["high_risk"]}, #e74c3c);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }}
    
    /* Button styling */
    .stButton > button {{
        background: linear-gradient(135deg, {COLORS["primary"]}, {COLORS["secondary"]});
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        font-weight: 500;
        font-size: 1rem;
        transition: all 0.3s ease;
        width: 100%;
    }}
    
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(46, 134, 171, 0.3);
    }}
    
    /* Hide Streamlit branding */
    #MainMenu {{visibility: hidden;}}
    .stDeployButton {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    
    /* Recommendation styling */
    .alert-info {{
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }}
    
    .alert-warning {{
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }}
    
    .alert-danger {{
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }}
    </style>
    """