"""Docker-compatible dashboard configuration."""

import os

# Get API URL from environment variable or default
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# API Configuration with Docker service name
API_ENDPOINTS = {
    "health": f"{API_BASE_URL}/health",
    "predict": f"{API_BASE_URL}/predict",
    "batch_predict": f"{API_BASE_URL}/predict/batch",
    "model_info": f"{API_BASE_URL}/model/info",
    "sample": f"{API_BASE_URL}/predict/sample"
}

# Medical color scheme
COLORS = {
    "primary": "#2E86AB",        
    "secondary": "#A23B72",      
    "success": "#F18F01",        
    "warning": "#C73E1D",        
    "low_risk": "#28a745",       
    "medium_risk": "#ffc107",    
    "high_risk": "#dc3545",      
    "background": "#f8f9fa",     
    "card_bg": "#ffffff",        
    "text_primary": "#2c3e50",  
    "text_secondary": "#6c757d"  
}

# Dashboard configuration
DASHBOARD_CONFIG = {
    "title": "Heart Disease Risk Assessment",
    "subtitle": "Professional ML-powered Cardiac Risk Evaluation",
    "icon": "🫀",
    "layout": "wide",
    "sidebar_state": "expanded"
}

def get_custom_css() -> str:
    """Get custom CSS for medical styling."""
    return f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .main {{
        font-family: 'Inter', sans-serif;
        background-color: {COLORS["background"]};
    }}
    
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
    
    .medical-card {{
        background: {COLORS["card_bg"]};
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        border-left: 4px solid {COLORS["primary"]};
        margin-bottom: 1.5rem;
    }}
    
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
    
    .stButton > button {{
        background: linear-gradient(135deg, {COLORS["primary"]}, {COLORS["secondary"]});
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        font-weight: 500;
        font-size: 1rem;
        width: 100%;
    }}
    
    .recommendation {{
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid {COLORS["primary"]};
    }}

    #MainMenu {{visibility: hidden;}}
    .stDeployButton {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    </style>
    """