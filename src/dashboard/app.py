"""Main Streamlit dashboard application with Docker support."""

import streamlit as st
from datetime import datetime
import sys
import os
from pathlib import Path

# Add project root to path and set working directory
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

# Import with absolute imports
from src.dashboard.config import DASHBOARD_CONFIG, get_custom_css, get_debug_info
from src.dashboard.utils import make_api_request, convert_form_data_to_api, check_api_health, get_system_info
from src.dashboard.components.patient_form import render_patient_form
from src.dashboard.components.results_display import render_prediction_results
from src.dashboard.pages.batch_processing import render_batch_processing_page


def main():
    """Main dashboard application."""
    
    # Page configuration
    st.set_page_config(
        page_title=DASHBOARD_CONFIG["title"],
        page_icon=DASHBOARD_CONFIG["icon"],
        layout=DASHBOARD_CONFIG["layout"],
        initial_sidebar_state=DASHBOARD_CONFIG["sidebar_state"]
    )
    
    # Apply custom CSS
    st.markdown(get_custom_css(), unsafe_allow_html=True)
    
    # Initialize session state
    if 'assessment_date' not in st.session_state:
        st.session_state.assessment_date = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown(f"# {DASHBOARD_CONFIG['icon']} Heart Risk AI")
        st.markdown("Professional cardiac risk assessment powered by machine learning")
        
        # Navigation
        page = st.selectbox(
            "Select Page",
            ["🏠 Risk Calculator", "📊 Batch Processing", "ℹ️ About"],
            key="page_selection"
        )
        
        # System Status
        st.markdown("---")
        st.markdown("### 🔧 System Status")
        
        # Get system info
        system_info = get_system_info()
        st.markdown(f"**API URL:** `{system_info['api_base_url']}`")
        
        if system_info["docker_mode"]:
            st.info("🐳 Running in Docker")
        else:
            st.info("💻 Running locally")
        
        if check_api_health():
            st.success("✅ API Connected")
        else:
            st.error("❌ API Disconnected")
            if not system_info["docker_mode"]:
                st.warning("Start API: `python scripts/start_api.py`")
            else:
                st.warning("Check Docker containers")
        
        # Model Information
        st.markdown("---")
        st.markdown("### 📈 Model Info")
        st.info("• Algorithm: Ensemble ML")
        st.info("• Accuracy: 86.89%")
        st.info("• Features: 20 engineered")
        
        # Debug info (only show if needed)
        debug_info = get_debug_info()
        if debug_info.get("docker_mode"):
            st.markdown("---")
            st.markdown("### 🔍 Debug Info")
            st.code(f"Environment: {debug_info['environment']}")
        
        # Footer
        st.markdown("---")
        st.markdown("*Built with Streamlit & FastAPI*")
    
    # Main header
    st.markdown(f"""
    <div class="main-header">
        <h1>{DASHBOARD_CONFIG["icon"]} {DASHBOARD_CONFIG["title"]}</h1>
        <p>{DASHBOARD_CONFIG["subtitle"]}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Route to selected page
    if page == "🏠 Risk Calculator":
        render_risk_calculator_page()
    elif page == "📊 Batch Processing":
        render_batch_processing_page()
    elif page == "ℹ️ About":
        render_about_page()


def render_risk_calculator_page():
    """Render the main risk calculator page."""
    
    # Check API health
    if not check_api_health():
        st.error("🚫 API server is not responding.")
        system_info = get_system_info()
        
        if not system_info["docker_mode"]:
            st.code("python scripts/start_api.py")
        else:
            st.code("docker-compose up -d")
            
        st.info("The API must be running for predictions to work.")
        return
    
    # Patient form
    form_data = render_patient_form()
    
    if form_data:
        # Convert form data to API format
        api_data = convert_form_data_to_api(form_data)
        
        # Make prediction request
        with st.spinner("🔬 Analyzing patient data..."):
            success, response = make_api_request(
                get_system_info()["endpoints"]["predict"],
                method="POST",
                data=api_data
            )
        
        if success:
            # Display results
            st.markdown("---")
            render_prediction_results(form_data, response)
            
            # Store in session state for potential export
            st.session_state.last_assessment = {
                'patient_data': form_data,
                'results': response,
                'timestamp': datetime.now().isoformat()
            }
            
        else:
            st.error(f"❌ Prediction failed: {response}")


def render_about_page():
    """Render the about/information page."""
    
    st.markdown("### 🎯 About Heart Disease Risk Assessment")
    
    st.markdown("""
    This professional-grade application uses advanced machine learning to assess heart disease risk
    based on clinical parameters and patient history.
    """)
    
    # Model information
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### 🤖 Machine Learning Model
        - **Algorithm**: Ensemble (Random Forest + XGBoost + Logistic Regression)
        - **Accuracy**: 86.89%
        - **Features**: 20 engineered features
        - **Training Data**: UCI Heart Disease Dataset (303 patients)
        - **Performance**: AUC-ROC 95.35%
        """)
    
    with col2:
        st.markdown("""
        #### 🏥 Clinical Features
        - Age, sex, chest pain type
        - Blood pressure and cholesterol
        - ECG results and heart rate
        - Exercise stress test results
        - Cardiac catheterization data
        """)
    
    # Risk categories
    st.markdown("### 🚦 Risk Categories")
    
    risk_col1, risk_col2, risk_col3 = st.columns(3)
    
    with risk_col1:
        st.markdown("""
        <div class="risk-low">
            <h4>✅ Low Risk</h4>
            <p><strong>< 30%</strong></p>
            <p>Continue healthy lifestyle and regular check-ups</p>
        </div>
        """, unsafe_allow_html=True)
    
    with risk_col2:
        st.markdown("""
        <div class="risk-medium">
            <h4>⚠️ Medium Risk</h4>
            <p><strong>30% - 70%</strong></p>
            <p>Consider lifestyle changes and closer monitoring</p>
        </div>
        """, unsafe_allow_html=True)
    
    with risk_col3:
        st.markdown("""
        <div class="risk-high">
            <h4>🚨 High Risk</h4>
            <p><strong>> 70%</strong></p>
            <p>Immediate medical consultation recommended</p>
        </div>
        """, unsafe_allow_html=True)
    
    # System Architecture
    st.markdown("### 🏗️ System Architecture")
    
    system_info = get_system_info()
    
    arch_col1, arch_col2 = st.columns(2)
    
    with arch_col1:
        st.markdown("""
        #### Frontend
        - **Framework**: Streamlit with custom CSS
        - **Visualization**: Plotly interactive charts
        - **Forms**: Professional medical styling
        - **Responsive**: Works on mobile/tablet
        """)
    
    with arch_col2:
        st.markdown("""
        #### Backend
        - **API**: FastAPI with async support
        - **ML Pipeline**: scikit-learn ensemble
        - **Database**: SQLite with proper schema
        - **Deployment**: Docker containerization
        """)
    
    # Deployment info
    if system_info["docker_mode"]:
        st.markdown("### 🐳 Docker Deployment")
        st.success("This instance is running in Docker containers")
        st.markdown(f"**API Endpoint**: `{system_info['api_base_url']}`")
    else:
        st.markdown("### 💻 Local Development")
        st.info("This instance is running in local development mode")
    
    # Disclaimer
    st.markdown("---")
    st.markdown("### ⚠️ Important Medical Disclaimer")
    
    st.warning("""
    **This tool is for educational and research purposes only.**
    
    - This assessment should not replace professional medical advice
    - Always consult with qualified healthcare providers for diagnosis and treatment
    - The predictions are based on statistical models and may not reflect individual cases
    - Emergency symptoms require immediate medical attention
    """)
    
    # Technical details
    with st.expander("🔧 Technical Details"):
        st.markdown("""
        #### Architecture
        - **Frontend**: Streamlit with custom CSS styling
        - **Backend**: FastAPI with Pydantic validation
        - **ML Pipeline**: scikit-learn with ensemble methods
        - **Database**: SQLite for data persistence
        - **Deployment**: Docker containerization
        
        #### Features
        - Real-time risk assessment
        - Batch processing capabilities
        - Interactive visualizations
        - Export functionality
        - Professional medical styling
        - Docker support for easy deployment
        """)


if __name__ == "__main__":
    main()