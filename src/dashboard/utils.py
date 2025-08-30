"""Dashboard utility functions - Clean version."""

import requests
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, Any, List, Optional, Tuple
import streamlit as st
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.dashboard.config import API_ENDPOINTS, COLORS, FORM_FIELDS


def make_api_request(endpoint: str, method: str = "GET", data: Optional[Dict] = None) -> Tuple[bool, Any]:
    """Make API request with error handling."""
    try:
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
    """Create a professional risk visualization chart."""
    colors = {
        "low": COLORS["low_risk"],
        "medium": COLORS["medium_risk"],
        "high": COLORS["high_risk"]
    }
    
    fig = go.Figure()
    
    # Main risk bar
    fig.add_trace(go.Bar(
        x=['Your Risk'],
        y=[risk_percentage],
        marker_color=colors.get(risk_category, COLORS["primary"]),
        text=[f"{risk_percentage:.1f}%"],
        textposition="outside",
        name="Risk Level",
        showlegend=False
    ))
    
    # Reference lines for context
    fig.add_hline(y=30, line_dash="dash", line_color=COLORS["low_risk"], 
                  annotation_text="Low Risk Threshold")
    fig.add_hline(y=70, line_dash="dash", line_color=COLORS["high_risk"], 
                  annotation_text="High Risk Threshold")
    
    fig.update_layout(
        title=f"Heart Disease Risk Assessment: {risk_category.title()}",
        yaxis_title="Risk Percentage (%)",
        yaxis_range=[0, 100],
        height=400,
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12, family="Inter, sans-serif")
    )
    
    return fig


def create_risk_comparison_chart(patient_risk: float) -> go.Figure:
    """Create a risk comparison visualization."""
    categories = ['Your Risk', 'Population Avg', 'Low Threshold', 'High Threshold']
    values = [patient_risk, 15.0, 30, 70]
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
        title="Risk Comparison Analysis",
        yaxis_title="Risk Percentage (%)",
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12, family="Inter, sans-serif")
    )
    
    return fig


def display_recommendations(recommendations: List[str], risk_category: str):
    """Display personalized recommendations with proper styling."""
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
    
    for i, recommendation in enumerate(recommendations[:6], 1):
        st.markdown(f"""
        <div class="{alert_class}" style="margin: 0.5rem 0;">
            <strong>{i}.</strong> {recommendation}
        </div>
        """, unsafe_allow_html=True)


def check_api_health() -> bool:
    """Check if API is healthy and responsive."""
    try:
        success, response = make_api_request(API_ENDPOINTS["health"])
        return success and isinstance(response, dict) and response.get("status") == "healthy"
    except:
        return False


def load_sample_data() -> Optional[Dict[str, Any]]:
    """Load sample patient data for testing."""
    try:
        success, response = make_api_request(API_ENDPOINTS["sample"])
        if success and isinstance(response, dict):
            return response.get("sample_input")
    except:
        pass
    return None


def export_results_to_csv(patient_data: Dict, prediction_result: Dict) -> str:
    """Export patient data and results to CSV format."""
    export_data = {
        **patient_data,
        "prediction": prediction_result["prediction"],
        "risk_percentage": prediction_result["risk_percentage"],
        "risk_category": prediction_result["risk_category"],
        "confidence": prediction_result["confidence"]
    }
    
    df = pd.DataFrame([export_data])
    return df.to_csv(index=False)