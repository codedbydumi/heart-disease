"""Dashboard utility functions with Docker support."""

import requests
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, Any, List, Optional, Tuple
import streamlit as st
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.dashboard.config import API_ENDPOINTS, COLORS, FORM_FIELDS


def make_api_request(endpoint: str, method: str = "GET", data: Optional[Dict] = None) -> Tuple[bool, Any]:
    """
    Make API request with error handling and Docker support.
    
    Args:
        endpoint: API endpoint URL
        method: HTTP method
        data: Request data for POST requests
        
    Returns:
        Tuple of (success, response_data)
    """
    try:
        # Debug logging
        st.sidebar.info(f"Connecting to: {endpoint}")
        
        if method == "GET":
            response = requests.get(endpoint, timeout=30)
        elif method == "POST":
            response = requests.post(endpoint, json=data, timeout=30)
        else:
            return False, f"Unsupported HTTP method: {method}"
        
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, f"API error: {response.status_code} - {response.text}"
            
    except requests.RequestException as e:
        return False, f"Connection error: {str(e)}"
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"


def convert_form_data_to_api(form_data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert Streamlit form data to API format."""
    # Map form selections to API values
    mappings = {
        "sex": {"Female": 0, "Male": 1},
        "cp": {
            "Typical Angina": 0,
            "Atypical Angina": 1,
            "Non-anginal Pain": 2,
            "Asymptomatic": 3
        },
        "fbs": {"≤ 120 mg/dl": 0, "> 120 mg/dl": 1},
        "restecg": {
            "Normal": 0,
            "ST-T wave abnormality": 1,
            "Left ventricular hypertrophy": 2
        },
        "exang": {"No": 0, "Yes": 1},
        "slope": {"Upsloping": 0, "Flat": 1, "Downsloping": 2},
        "thal": {"Normal": 1, "Fixed Defect": 2, "Reversible Defect": 3}
    }
    
    api_data = form_data.copy()
    
    for field, mapping in mappings.items():
        if field in api_data and api_data[field] in mapping:
            api_data[field] = mapping[api_data[field]]
    
    return api_data


def create_simple_risk_chart(risk_percentage: float, risk_category: str) -> go.Figure:
    """Create a simple risk chart that works reliably."""
    colors = {
        "low": COLORS["low_risk"],
        "medium": COLORS["medium_risk"],
        "high": COLORS["high_risk"]
    }
    
    fig = go.Figure()
    
    # Add risk bar
    fig.add_trace(go.Bar(
        x=['Risk Level'],
        y=[risk_percentage],
        marker_color=colors.get(risk_category, COLORS["primary"]),
        text=[f"{risk_percentage:.1f}%"],
        textposition="outside",
        name="Your Risk",
        showlegend=False
    ))
    
    # Add reference lines
    fig.add_hline(y=30, line_dash="dash", line_color=COLORS["low_risk"], 
                  annotation_text="Low Risk Threshold (30%)")
    fig.add_hline(y=70, line_dash="dash", line_color=COLORS["high_risk"], 
                  annotation_text="High Risk Threshold (70%)")
    
    fig.update_layout(
        title=f"Heart Disease Risk: {risk_category.title()}",
        yaxis_title="Risk Percentage (%)",
        yaxis_range=[0, 100],
        height=400,
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12)
    )
    
    return fig


def create_risk_comparison_chart(patient_risk: float) -> go.Figure:
    """Create a risk comparison chart."""
    population_avg = 15.0  # Example population average
    
    categories = ['Your Risk', 'Population Avg', 'Low Threshold', 'High Threshold']
    values = [patient_risk, population_avg, 30, 70]
    colors = [COLORS["primary"], COLORS["secondary"], COLORS["low_risk"], COLORS["high_risk"]]
    
    fig = go.Figure(go.Bar(
        x=categories,
        y=values,
        marker_color=colors,
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
        showlegend=False
    ))
    
    fig.update_layout(
        title="Risk Comparison",
        yaxis_title="Risk Percentage (%)",
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    
    return fig


def display_recommendations(recommendations: List[str], risk_category: str):
    """Display personalized recommendations."""
    if risk_category == "low":
        alert_class = "alert-info"
        icon = "✅"
    elif risk_category == "medium":
        alert_class = "alert-warning"
        icon = "⚠️"
    else:
        alert_class = "alert-danger"
        icon = "🚨"
    
    st.markdown(f"### 📋 Personalized Recommendations {icon}")
    
    for i, recommendation in enumerate(recommendations[:6], 1):  # Limit to 6
        st.markdown(f"""
        <div class="{alert_class}" style="margin: 0.5rem 0;">
            <strong>{i}.</strong> {recommendation}
        </div>
        """, unsafe_allow_html=True)


def check_api_health() -> bool:
    """Check if API is healthy and responsive with Docker support."""
    try:
        success, response = make_api_request(API_ENDPOINTS["health"])
        
        if success and isinstance(response, dict) and response.get("status") == "healthy":
            return True
        return False
    except Exception as e:
        st.sidebar.error(f"Health check error: {str(e)}")
        return False


def load_sample_data() -> Optional[Dict[str, Any]]:
    """Load sample patient data for testing."""
    try:
        success, response = make_api_request(API_ENDPOINTS["sample"])
        
        if success and isinstance(response, dict):
            return response.get("sample_input")
    except Exception:
        pass
    return None


def format_patient_data_for_display(patient_data: Dict[str, Any]) -> pd.DataFrame:
    """Format patient data for display in a table."""
    formatted_data = []
    
    for field, value in patient_data.items():
        if field in FORM_FIELDS:
            field_config = FORM_FIELDS[field]
            
            # Convert numeric values back to display format
            if field == "sex":
                display_value = "Male" if value == 1 else "Female"
            elif field == "cp":
                cp_options = ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"]
                display_value = cp_options[value] if 0 <= value < len(cp_options) else str(value)
            elif field == "fbs":
                display_value = "> 120 mg/dl" if value == 1 else "≤ 120 mg/dl"
            elif field == "restecg":
                restecg_options = ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"]
                display_value = restecg_options[value] if 0 <= value < len(restecg_options) else str(value)
            elif field == "exang":
                display_value = "Yes" if value == 1 else "No"
            elif field == "slope":
                slope_options = ["Upsloping", "Flat", "Downsloping"]
                display_value = slope_options[value] if 0 <= value < len(slope_options) else str(value)
            elif field == "thal":
                thal_options = {1: "Normal", 2: "Fixed Defect", 3: "Reversible Defect"}
                display_value = thal_options.get(value, str(value))
            else:
                display_value = str(value)
            
            formatted_data.append({
                "Parameter": field_config["label"],
                "Value": display_value,
                "Description": field_config.get("help", "")
            })
    
    return pd.DataFrame(formatted_data)


def export_results_to_csv(patient_data: Dict, prediction_result: Dict) -> str:
    """Export patient data and results to CSV format."""
    # Combine patient data and results
    export_data = {
        **patient_data,
        "prediction": prediction_result["prediction"],
        "risk_percentage": prediction_result["risk_percentage"],
        "risk_category": prediction_result["risk_category"],
        "confidence": prediction_result["confidence"]
    }
    
    # Convert to DataFrame
    df = pd.DataFrame([export_data])
    
    # Convert to CSV string
    return df.to_csv(index=False)


def get_system_info() -> Dict[str, Any]:
    """Get system information for debugging."""
    return {
        "api_base_url": API_ENDPOINTS["health"].replace("/health", ""),
        "environment": os.getenv("ENVIRONMENT", "development"),
        "docker_mode": "API_BASE_URL" in os.environ,
        "endpoints": API_ENDPOINTS
    }