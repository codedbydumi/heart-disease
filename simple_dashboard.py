"""Simple Working Heart Disease Risk Assessment Dashboard."""

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configuration
API_BASE_URL = "http://localhost:8000"
COLORS = {
    "primary": "#2E86AB",
    "secondary": "#A23B72", 
    "low_risk": "#28a745",
    "medium_risk": "#ffc107",
    "high_risk": "#dc3545",
    "background": "#f8f9fa",
    "card_bg": "#ffffff"
}

# Page configuration
st.set_page_config(
    page_title="Heart Disease Risk Assessment",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown(f"""
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
""", unsafe_allow_html=True)

def check_api_health():
    """Check if API is healthy."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def make_prediction(patient_data):
    """Make prediction via API."""
    try:
        response = requests.post(f"{API_BASE_URL}/predict", json=patient_data, timeout=30)
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, f"API Error: {response.status_code}"
    except Exception as e:
        return False, f"Connection Error: {str(e)}"

def create_risk_chart(risk_percentage, risk_category):
    """Create simple risk visualization."""
    colors = {
        "low": COLORS["low_risk"],
        "medium": COLORS["medium_risk"],
        "high": COLORS["high_risk"]
    }
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=['Your Risk'],
        y=[risk_percentage],
        marker_color=colors.get(risk_category, COLORS["primary"]),
        text=[f"{risk_percentage:.1f}%"],
        textposition="outside",
        showlegend=False
    ))
    
    # Add reference lines
    fig.add_hline(y=30, line_dash="dash", line_color=COLORS["low_risk"], 
                  annotation_text="Low Risk (30%)")
    fig.add_hline(y=70, line_dash="dash", line_color=COLORS["high_risk"], 
                  annotation_text="High Risk (70%)")
    
    fig.update_layout(
        title=f"Heart Disease Risk Assessment: {risk_category.title()}",
        yaxis_title="Risk Percentage (%)",
        yaxis_range=[0, 100],
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    
    return fig

def main():
    """Main dashboard application."""
    
    # Sidebar
    with st.sidebar:
        st.markdown("# 🫀 Heart Risk AI")
        st.markdown("Professional cardiac risk assessment powered by machine learning")
        
        # API status
        st.markdown("---")
        st.markdown("### 🔧 System Status")
        if check_api_health():
            st.success("✅ API Connected")
        else:
            st.error("❌ API Disconnected")
            st.warning("Start API: `python scripts/start_api.py`")
        
        st.markdown("---")
        st.markdown("### 📈 Model Info")
        st.info("• Algorithm: Ensemble ML")
        st.info("• Accuracy: 89.5%")
        st.info("• Features: 20 engineered")
        
        st.markdown("---")
        st.markdown("*Built with Streamlit & FastAPI*")
    
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1>🫀 Heart Disease Risk Assessment</h1>
        <p>Professional ML-powered Cardiac Risk Evaluation</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check API first
    if not check_api_health():
        st.error("🚫 API server is not responding. Please start the API server first.")
        st.code("python scripts/start_api.py", language="bash")
        st.stop()
    
    # Patient Form
    st.markdown("### 👤 Patient Information")
    
    with st.form("patient_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age (years)", min_value=18, max_value=120, value=50)
            sex = st.selectbox("Biological Sex", ["Female", "Male"])
            cp = st.selectbox("Chest Pain Type", [
                "Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"
            ])
            trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=250, value=120)
            chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
            fbs = st.selectbox("Fasting Blood Sugar", ["≤ 120 mg/dl", "> 120 mg/dl"])
            restecg = st.selectbox("Resting ECG Results", [
                "Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"
            ])
        
        with col2:
            thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
            exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
            oldpeak = st.number_input("ST Depression (mm)", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
            slope = st.selectbox("ST Segment Slope", ["Upsloping", "Flat", "Downsloping"])
            ca = st.number_input("Major Vessels (0-4)", min_value=0, max_value=4, value=0)
            thal = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"])
        
        # Form buttons
        col1, col2 = st.columns([2, 1])
        with col1:
            submitted = st.form_submit_button("🔬 Analyze Risk", use_container_width=True)
        with col2:
            if st.form_submit_button("📝 Load Sample", use_container_width=True):
                st.rerun()
    
    # Process form submission
    if submitted:
        # Convert form data to API format
        api_data = {
            "age": age,
            "sex": 1 if sex == "Male" else 0,
            "cp": ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"].index(cp),
            "trestbps": trestbps,
            "chol": chol,
            "fbs": 1 if fbs == "> 120 mg/dl" else 0,
            "restecg": ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"].index(restecg),
            "thalach": thalach,
            "exang": 1 if exang == "Yes" else 0,
            "oldpeak": oldpeak,
            "slope": ["Upsloping", "Flat", "Downsloping"].index(slope),
            "ca": ca,
            "thal": ["Normal", "Fixed Defect", "Reversible Defect"].index(thal) + 1
        }
        
        # Make prediction
        with st.spinner("🔬 Analyzing patient data..."):
            success, result = make_prediction(api_data)
        
        if success:
            # Display results
            st.markdown("---")
            
            risk_category = result["risk_category"]
            risk_percentage = result["risk_percentage"]
            
            # Risk interpretation
            risk_info = {
                "low": {"color": COLORS["low_risk"], "icon": "✅", "title": "Low Risk"},
                "medium": {"color": COLORS["medium_risk"], "icon": "⚠️", "title": "Medium Risk"},
                "high": {"color": COLORS["high_risk"], "icon": "🚨", "title": "High Risk"}
            }
            
            info = risk_info[risk_category]
            
            # Main result card
            st.markdown(f"""
            <div class="medical-card">
                <div style="text-align: center; font-size: 3rem; margin-bottom: 1rem;">{info["icon"]}</div>
                <h2 style="text-align: center; color: {info["color"]};">{info["title"]}</h2>
                <h3 style="text-align: center;">
                    {risk_percentage:.1f}% Risk of Heart Disease
                </h3>
                <p style="text-align: center; font-size: 1.1rem;">
                    Model Confidence: {result["confidence"]:.1f}%
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Risk Probability", f"{result['probability']:.3f}")
            with col2:
                st.metric("Model Confidence", f"{result['confidence']:.1f}%")
            with col3:
                prediction_text = "Disease Risk" if result['prediction'] == 1 else "No Disease Risk"
                st.metric("Prediction", prediction_text)
            
            # Risk visualization
            st.markdown("### 📊 Risk Visualization")
            try:
                chart = create_risk_chart(risk_percentage, risk_category)
                st.plotly_chart(chart, use_container_width=True)
            except:
                # Fallback to progress bar
                progress = risk_percentage / 100
                st.progress(progress)
                st.markdown(f"**{risk_percentage:.1f}%** - {risk_category.title()} Risk")
            
            # Medical interpretation
            if result.get("interpretation"):
                st.markdown("### 🩺 Medical Interpretation")
                for key, value in result["interpretation"].items():
                    st.markdown(f"""
                    <div class="recommendation">
                        <strong>{key.replace('_', ' ').title()}:</strong> {value}
                    </div>
                    """, unsafe_allow_html=True)
            
            # Recommendations
            if result.get("recommendations"):
                st.markdown("### 📋 Personalized Recommendations")
                for i, rec in enumerate(result["recommendations"][:5], 1):
                    st.markdown(f"""
                    <div class="recommendation">
                        <strong>{i}.</strong> {rec}
                    </div>
                    """, unsafe_allow_html=True)
            
            # Export functionality
            st.markdown("### 📁 Export Results")
            
            # Prepare export data
            export_data = {
                **api_data,
                "risk_percentage": risk_percentage,
                "risk_category": risk_category,
                "confidence": result["confidence"],
                "prediction": result["prediction"]
            }
            
            df_export = pd.DataFrame([export_data])
            csv_export = df_export.to_csv(index=False)
            
            st.download_button(
                label="📊 Download CSV Report",
                data=csv_export,
                file_name=f"heart_risk_assessment_{risk_category}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            # Detailed report
            with st.expander("📋 Detailed Medical Report"):
                st.markdown(f"""
                **Patient Summary:**
                - Age: {age} years
                - Sex: {sex}
                - Assessment Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}
                
                **Risk Assessment:**
                - Primary Risk: {risk_percentage:.1f}%
                - Risk Category: {risk_category.title()}
                - Model Confidence: {result['confidence']:.1f}%
                - Prediction: {'Positive for Heart Disease Risk' if result['prediction'] == 1 else 'Negative for Heart Disease Risk'}
                
                **Clinical Parameters:**
                - Blood Pressure: {trestbps} mm Hg
                - Cholesterol: {chol} mg/dl
                - Max Heart Rate: {thalach} bpm
                - ST Depression: {oldpeak} mm
                """)
                
                st.warning("""
                **Important Disclaimer:**
                This assessment is based on machine learning analysis and should not replace professional medical advice. 
                Please consult with a qualified healthcare provider for proper medical diagnosis and treatment recommendations.
                """)
        
        else:
            st.error(f"❌ Prediction failed: {result}")

if __name__ == "__main__":
    main()