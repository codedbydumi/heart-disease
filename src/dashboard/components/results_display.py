"""Results display component - Simplified for compatibility."""

import streamlit as st
from typing import Dict, Any
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.dashboard.config import RISK_INTERPRETATION, COLORS
from src.dashboard.utils import (
    create_simple_risk_chart, display_recommendations, export_results_to_csv
)


def render_prediction_results(
    patient_data: Dict[str, Any], 
    prediction_result: Dict[str, Any]
):
    """
    Render prediction results with visualizations.
    
    Args:
        patient_data: Original patient input data
        prediction_result: API prediction response
    """
    risk_category = prediction_result["risk_category"]
    risk_info = RISK_INTERPRETATION[risk_category]
    
    # Main risk assessment card
    st.markdown(f"""
    <div class="medical-card">
        <div class="medical-icon">{risk_info["icon"]}</div>
        <h2 style="text-align: center; color: {risk_info["color"]};">
            {risk_info["title"]}
        </h2>
        <h3 style="text-align: center; color: {COLORS["text_primary"]};">
            {prediction_result["risk_percentage"]:.1f}% Risk of Heart Disease
        </h3>
        <p style="text-align: center; font-size: 1.1rem; color: {COLORS["text_secondary"]};">
            {risk_info["description"]}
        </p>
        <p style="text-align: center; font-weight: 500; color: {COLORS["text_primary"]};">
            <strong>Recommended Action:</strong> {risk_info["action"]}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create three columns for metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Risk Probability",
            value=f"{prediction_result['probability']:.3f}",
            delta=f"{prediction_result['risk_percentage']:.1f}%"
        )
    
    with col2:
        st.metric(
            label="Model Confidence",
            value=f"{prediction_result['confidence']:.1f}%",
            delta="High" if prediction_result['confidence'] > 80 else "Medium"
        )
    
    with col3:
        prediction_text = "Disease Likely" if prediction_result['prediction'] == 1 else "No Disease Detected"
        st.metric(
            label="Prediction",
            value=prediction_text,
            delta=risk_category.title()
        )
    
    # Simple risk visualization
    st.markdown("---")
    st.markdown("### 📊 Risk Assessment")
    
    try:
        # Try to create the chart
        risk_chart = create_simple_risk_chart(
            prediction_result["risk_percentage"], 
            risk_category
        )
        st.plotly_chart(risk_chart, use_container_width=True)
    except Exception as e:
        # Fallback to simple progress bar
        st.markdown("#### Risk Level")
        progress_value = prediction_result["risk_percentage"] / 100
        st.progress(progress_value)
        st.markdown(f"**{prediction_result['risk_percentage']:.1f}%** - {risk_category.title()} Risk")
    
    # Medical interpretation
    if prediction_result.get("interpretation"):
        st.markdown("### 🩺 Medical Interpretation")
        
        interpretation_items = list(prediction_result["interpretation"].items())
        
        # Display in a nice format
        for key, value in interpretation_items:
            st.markdown(f"""
            <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; margin: 0.5rem 0; border-left: 4px solid {COLORS["primary"]};">
                <h4 style="color: {COLORS["primary"]}; margin: 0 0 0.5rem 0;">{key.replace('_', ' ').title()}</h4>
                <p style="margin: 0; color: {COLORS["text_primary"]};">{value}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Recommendations
    if prediction_result.get("recommendations"):
        display_recommendations(
            prediction_result["recommendations"], 
            risk_category
        )
    
    # Export functionality
    st.markdown("---")
    st.markdown("### 📁 Export Results")
    
    export_col1, export_col2 = st.columns(2)
    
    with export_col1:
        csv_data = export_results_to_csv(patient_data, prediction_result)
        st.download_button(
            label="📊 Download CSV Report",
            data=csv_data,
            file_name=f"heart_risk_assessment_{prediction_result['risk_category']}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with export_col2:
        if st.button("📋 View Detailed Report", use_container_width=True):
            show_detailed_report(patient_data, prediction_result)


def show_detailed_report(patient_data: Dict[str, Any], prediction_result: Dict[str, Any]):
    """Show detailed medical report in expandable section."""
    with st.expander("📋 Detailed Medical Report", expanded=True):
        
        # Patient summary
        st.markdown("#### Patient Summary")
        st.write(f"**Age:** {patient_data['age']} years")
        st.write(f"**Sex:** {patient_data['sex']}")
        st.write(f"**Assessment Date:** {st.session_state.get('assessment_date', 'Today')}")
        
        # Risk assessment
        st.markdown("#### Risk Assessment")
        st.write(f"**Primary Risk:** {prediction_result['risk_percentage']:.1f}%")
        st.write(f"**Risk Category:** {prediction_result['risk_category'].title()}")
        st.write(f"**Model Confidence:** {prediction_result['confidence']:.1f}%")
        st.write(f"**Prediction:** {'Positive for Heart Disease Risk' if prediction_result['prediction'] == 1 else 'Negative for Heart Disease Risk'}")
        
        # Clinical parameters
        st.markdown("#### Clinical Parameters")
        clinical_data = {
            "Blood Pressure": f"{patient_data['trestbps']} mm Hg",
            "Cholesterol": f"{patient_data['chol']} mg/dl",
            "Max Heart Rate": f"{patient_data['thalach']} bpm",
            "ST Depression": f"{patient_data['oldpeak']} mm",
            "Major Vessels": patient_data['ca']
        }
        
        for param, value in clinical_data.items():
            st.write(f"**{param}:** {value}")
        
        # Recommendations
        st.markdown("#### Medical Recommendations")
        for i, rec in enumerate(prediction_result.get("recommendations", []), 1):
            st.write(f"{i}. {rec}")
        
        # Disclaimer
        st.markdown("---")
        st.markdown("#### Important Disclaimer")
        st.warning("""
        This assessment is based on machine learning analysis and should not replace professional medical advice. 
        Please consult with a qualified healthcare provider for proper medical diagnosis and treatment recommendations.
        """)