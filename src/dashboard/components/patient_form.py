"""Patient input form component."""

import streamlit as st
from typing import Dict, Any, Optional
from ..config import FORM_FIELDS
from ..utils import load_sample_data


def render_patient_form() -> Optional[Dict[str, Any]]:
    """
    Render the patient input form.
    
    Returns:
        Dictionary with patient data if form is submitted, None otherwise
    """
    st.markdown("### 👤 Patient Information")
    
    with st.form("patient_form"):
        # Create form columns
        col1, col2 = st.columns(2)
        
        form_data = {}
        
        # Age and Sex
        with col1:
            form_data["age"] = st.number_input(
                **FORM_FIELDS["age"]
            )
            
            form_data["sex"] = st.selectbox(
                FORM_FIELDS["sex"]["label"],
                FORM_FIELDS["sex"]["options"],
                help=FORM_FIELDS["sex"]["help"]
            )
            
            form_data["cp"] = st.selectbox(
                FORM_FIELDS["cp"]["label"],
                FORM_FIELDS["cp"]["options"],
                help=FORM_FIELDS["cp"]["help"]
            )
            
            form_data["trestbps"] = st.number_input(
                **FORM_FIELDS["trestbps"]
            )
            
            form_data["chol"] = st.number_input(
                **FORM_FIELDS["chol"]
            )
            
            form_data["fbs"] = st.selectbox(
                FORM_FIELDS["fbs"]["label"],
                FORM_FIELDS["fbs"]["options"],
                help=FORM_FIELDS["fbs"]["help"]
            )
            
            form_data["restecg"] = st.selectbox(
                FORM_FIELDS["restecg"]["label"],
                FORM_FIELDS["restecg"]["options"],
                help=FORM_FIELDS["restecg"]["help"]
            )
        
        with col2:
            form_data["thalach"] = st.number_input(
                **FORM_FIELDS["thalach"]
            )
            
            form_data["exang"] = st.selectbox(
                FORM_FIELDS["exang"]["label"],
                FORM_FIELDS["exang"]["options"],
                help=FORM_FIELDS["exang"]["help"]
            )
            
            form_data["oldpeak"] = st.number_input(
                **FORM_FIELDS["oldpeak"]
            )
            
            form_data["slope"] = st.selectbox(
                FORM_FIELDS["slope"]["label"],
                FORM_FIELDS["slope"]["options"],
                help=FORM_FIELDS["slope"]["help"]
            )
            
            form_data["ca"] = st.number_input(
                **FORM_FIELDS["ca"]
            )
            
            form_data["thal"] = st.selectbox(
                FORM_FIELDS["thal"]["label"],
                FORM_FIELDS["thal"]["options"],
                help=FORM_FIELDS["thal"]["help"]
            )
        
        # Form buttons
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            submitted = st.form_submit_button("🔬 Analyze Risk", use_container_width=True)
        
        with col2:
            load_sample = st.form_submit_button("📝 Load Sample", use_container_width=True)
        
        with col3:
            clear_form = st.form_submit_button("🗑️ Clear Form", use_container_width=True)
        
        # Handle form actions
        if load_sample:
            sample_data = load_sample_data()
            if sample_data:
                st.success("Sample data loaded! Please refresh to see the values.")
                st.experimental_rerun()
            else:
                st.error("Failed to load sample data")
        
        if clear_form:
            st.experimental_rerun()
        
        if submitted:
            # Validate form data
            validation_errors = validate_form_data(form_data)
            
            if validation_errors:
                for error in validation_errors:
                    st.error(error)
                return None
            
            return form_data
    
    return None


def validate_form_data(form_data: Dict[str, Any]) -> List[str]:
    """Validate form data and return list of errors."""
    errors = []
    
    # Age validation
    if form_data["age"] < 18 or form_data["age"] > 120:
        errors.append("Age must be between 18 and 120 years")
    
    # Blood pressure validation
    if form_data["trestbps"] < 80 or form_data["trestbps"] > 250:
        errors.append("Blood pressure seems unrealistic (should be 80-250 mm Hg)")
    
    # Cholesterol validation
    if form_data["chol"] < 100 or form_data["chol"] > 600:
        errors.append("Cholesterol level seems unrealistic (should be 100-600 mg/dl)")
    
    # Heart rate validation
    if form_data["thalach"] < 60 or form_data["thalach"] > 220:
        errors.append("Maximum heart rate seems unrealistic (should be 60-220 bpm)")
    
    # ST depression validation
    if form_data["oldpeak"] < 0 or form_data["oldpeak"] > 10:
        errors.append("ST depression should be between 0 and 10 mm")
    
    return errors