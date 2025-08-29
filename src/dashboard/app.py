"""Main Streamlit dashboard application with improved error handling."""

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
from src.dashboard.config import DASHBOARD_CONFIG, get_custom_css
from src.dashboard.utils import make_api_request, convert_form_data_to_api, check_api_health
from src.dashboard.components.patient_form import render_patient_form
from src.dashboard.components.results_display import render_prediction_results
from src.dashboard.pages.batch_processing import render_batch_processing_page
from src.utils.error_handler import handle_dashboard_error, StreamlitErrorHandler


@handle_dashboard_error("main_application")
def main():
    """Main dashboard application with comprehensive error handling."""
    
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
    if 'api_health_checked' not in st.session_state:
        st.session_state.api_health_checked = False
    
    # Sidebar configuration
    render_sidebar()
    
    # Main header
    st.markdown(f"""
    <div class="main-header">
        <h1>{DASHBOARD_CONFIG["icon"]} {DASHBOARD_CONFIG["title"]}</h1>
        <p>{DASHBOARD_CONFIG["subtitle"]}</p>
        <p style="font-size: 0.9rem; opacity: 0.8;">Enhanced with Security & Error Handling | v1.1</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get selected page from session state
    page = st.session_state.get("page_selection", "🏠 Risk Calculator")
    
    # Route to selected page with error handling
    try:
        if page == "🏠 Risk Calculator":
            render_risk_calculator_page()
        elif page == "📊 Batch Processing":
            render_batch_processing_page()
        elif page == "ℹ️ About":
            render_about_page()
        else:
            st.error(f"Unknown page: {page}")
            
    except Exception as e:
        st.error("Page loading failed")
        StreamlitErrorHandler.show_prediction_error()
        
        if st.session_state.get("debug_mode", False):
            st.exception(e)


@handle_dashboard_error("sidebar")
def render_sidebar():
    """Render sidebar with system status and navigation."""
    
    with st.sidebar:
        st.markdown(f"# {DASHBOARD_CONFIG['icon']} Heart Risk AI")
        st.markdown("Professional cardiac risk assessment powered by machine learning")
        
        # Navigation
        page = st.selectbox(
            "Select Page",
            ["🏠 Risk Calculator", "📊 Batch Processing", "ℹ️ About"],
            key="page_selection"
        )
        
        # System status with better error handling
        st.markdown("---")
        st.markdown("### 🔧 System Status")
        
        try:
            api_healthy = check_api_health()
            if api_healthy:
                st.success("✅ API Connected")
                st.session_state.api_health_checked = True
            else:
                st.error("❌ API Disconnected")
                st.warning("Start API: `python scripts/start_api.py`")
                st.session_state.api_health_checked = False
        except Exception as e:
            st.error("❌ API Status Unknown")
            st.session_state.api_health_checked = False
        
        # Enhanced system info
        st.markdown("---")
        st.markdown("### 📈 System Info")
        st.info("• Model: Ensemble ML")
        st.info("• Security: Rate Limited")
        st.info("• Data: Real UCI Dataset")
        st.info("• Accuracy: 85%+ Expected")
        
        # Debug mode toggle
        StreamlitErrorHandler.enable_debug_mode()
        
        # System improvements indicator
        st.markdown("---")
        st.markdown("### 🚀 Improvements")
        st.success("✅ Real Medical Data")
        st.success("✅ Input Validation")
        st.success("✅ Error Handling")
        st.success("✅ Audit Logging")
        
        st.markdown("---")
        st.markdown("*Enhanced Professional Version 1.1*")


@handle_dashboard_error("risk_calculator")
def render_risk_calculator_page():
    """Render the main risk calculator page with error handling."""
    
    # API health check with user-friendly error
    if not st.session_state.get("api_health_checked", False):
        if not check_api_health():
            StreamlitErrorHandler.show_api_connection_error()
            return
    
    # Patient form with error handling
    try:
        form_data = render_patient_form()
    except Exception as e:
        st.error("Form loading failed")
        if st.session_state.get("debug_mode", False):
            st.exception(e)
        return
    
    if form_data:
        # Convert form data with validation
        try:
            api_data = convert_form_data_to_api(form_data)
        except Exception as e:
            StreamlitErrorHandler.show_validation_error(str(e))
            return
        
        # Make prediction request with comprehensive error handling
        with st.spinner("🔬 Analyzing patient data..."):
            try:
                success, response = make_api_request(
                    "http://localhost:8000/predict",
                    method="POST",
                    data=api_data
                )
                
                if success:
                    # Validate response structure
                    required_fields = ['risk_percentage', 'risk_category', 'confidence', 'prediction']
                    if not all(field in response for field in required_fields):
                        st.error("Invalid prediction response format")
                        return
                    
                    # Display results with error protection
                    st.markdown("---")
                    render_prediction_results(form_data, response)
                    
                    # Store in session state for potential export
                    st.session_state.last_assessment = {
                        'patient_data': form_data,
                        'results': response,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                else:
                    # Handle API errors gracefully
                    error_msg = str(response)
                    
                    if "rate limit" in error_msg.lower():
                        st.error("⏱️ Rate limit exceeded")
                        st.warning("Please wait a minute before making another prediction.")
                        
                    elif "validation" in error_msg.lower():
                        StreamlitErrorHandler.show_validation_error(error_msg)
                        
                    elif "connection" in error_msg.lower():
                        StreamlitErrorHandler.show_api_connection_error()
                        
                    else:
                        st.error(f"❌ Prediction failed: {response}")
                        
            except Exception as e:
                st.error("Prediction request failed")
                StreamlitErrorHandler.show_prediction_error()
                
                if st.session_state.get("debug_mode", False):
                    st.exception(e)


@handle_dashboard_error("about_page")
def render_about_page():
    """Render the about/information page with system improvements info."""
    
    st.markdown("### 🎯 About Heart Disease Risk Assessment")
    
    st.markdown("""
    This professional-grade application uses advanced machine learning to assess heart disease risk
    based on clinical parameters and patient history. **Version 1.1** includes significant improvements
    in model accuracy, security, and error handling.
    """)
    
    # System improvements section
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### 🚀 Version 1.1 Improvements
        - **Real Medical Data**: UCI Heart Disease dataset (85%+ accuracy)
        - **Security Enhancement**: Rate limiting and input validation
        - **Error Handling**: Comprehensive error management
        - **Audit Logging**: Request tracking and monitoring
        - **Medical Validation**: Realistic parameter ranges
        """)
    
    with col2:
        st.markdown("""
        #### 🤖 Machine Learning Model
        - **Algorithm**: Ensemble (Random Forest + XGBoost + Logistic Regression)
        - **Expected Accuracy**: 85%+ (improved from 60%)
        - **Features**: 20 engineered features
        - **Training Data**: Real UCI Heart Disease Dataset
        - **Validation**: Cross-validation with medical expertise
        """)
    
    # Technical improvements
    st.markdown("### 🔧 Technical Enhancements")
    
    improvements_col1, improvements_col2 = st.columns(2)
    
    with improvements_col1:
        st.markdown("""
        **Data Quality Improvements:**
        - Real UCI medical dataset (303 patients)
        - Medical parameter validation
        - Outlier detection and handling
        - Missing value imputation strategies
        """)
    
    with improvements_col2:
        st.markdown("""
        **Security & Reliability:**
        - Rate limiting (10 requests/minute)
        - Input sanitization and validation
        - Comprehensive error handling
        - Audit trail logging
        """)
    
    # Risk categories with updated guidance
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
    
    # Error handling demonstration
    with st.expander("🔧 Error Handling Features"):
        st.markdown("""
        **Comprehensive Error Management:**
        - Medical parameter validation with guidance
        - Rate limiting protection
        - API connection monitoring
        - User-friendly error messages
        - Debug mode for troubleshooting
        - Graceful failure handling
        
        **Example Error Scenarios Handled:**
        - Invalid age range (shows medical guidance)
        - API server offline (shows restart instructions)
        - Rate limit exceeded (shows wait time)
        - Model prediction failures (shows retry options)
        """)
    
    # Disclaimer with enhanced warnings
    st.markdown("---")
    st.markdown("### ⚠️ Important Medical Disclaimer")
    
    st.warning("""
    **This tool is for educational and research purposes only.**
    
    - This assessment should not replace professional medical advice
    - Always consult with qualified healthcare providers for diagnosis and treatment
    - The predictions are based on statistical models and may not reflect individual cases
    - Emergency symptoms require immediate medical attention
    - Model accuracy has been improved but is not 100% reliable
    """)
    
    # System status and troubleshooting
    with st.expander("🔍 System Status & Troubleshooting"):
        st.markdown("""
        **If you encounter issues:**
        
        1. **API Connection Problems:**
           ```bash
           # Start API server
           python scripts/start_api.py
           ```
        
        2. **Validation Errors:**
           - Check that all values are within medical ranges
           - Age: 18-120 years
           - Blood Pressure: 80-300 mm Hg
           - Cholesterol: 100-800 mg/dl
        
        3. **Rate Limiting:**
           - Wait 1 minute between prediction requests
           - Batch processing limited to 2 requests per minute
        
        4. **Debug Mode:**
           - Enable in sidebar for detailed error information
           - Shows technical details for troubleshooting
        """)


if __name__ == "__main__":
    main()