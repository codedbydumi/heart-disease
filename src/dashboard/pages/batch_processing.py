"""Batch processing page for multiple patients."""

import streamlit as st
import pandas as pd
import io
from typing import List, Dict, Any

# Fix imports
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.dashboard.config import API_ENDPOINTS
from src.dashboard.utils import make_api_request, convert_form_data_to_api, format_patient_data_for_display


def render_batch_processing_page():
    """Render the batch processing interface."""
    st.markdown("### 📊 Batch Risk Assessment")
    st.markdown("Process multiple patients at once using CSV upload")
    
    # Instructions
    with st.expander("📋 Instructions", expanded=False):
        st.markdown("""
        **How to use Batch Processing:**
        
        1. **Download Template**: Use the template CSV with correct column names
        2. **Fill Patient Data**: Add your patient data to the template
        3. **Upload File**: Upload your completed CSV file
        4. **Process**: Review and process all patients
        5. **Download Results**: Get detailed results for all patients
        
        **CSV Format Requirements:**
        - All numeric values must be within valid medical ranges
        - Required columns: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal
        - Maximum 100 patients per batch
        """)
    
    # Template download
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📥 Download CSV Template", use_container_width=True):
            template_data = get_csv_template()
            st.download_button(
                label="Download Template",
                data=template_data,
                file_name="heart_disease_assessment_template.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("📝 Download Sample Data", use_container_width=True):
            sample_data = get_sample_csv_data()
            st.download_button(
                label="Download Sample",
                data=sample_data,
                file_name="heart_disease_sample_data.csv",
                mime="text/csv"
            )
    
    st.markdown("---")
    
    # File upload
    st.markdown("### 📤 Upload Patient Data")
    uploaded_file = st.file_uploader(
        "Choose CSV file",
        type=['csv'],
        help="Upload a CSV file with patient data using the template format"
    )
    
    if uploaded_file is not None:
        try:
            # Read and validate CSV
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ File uploaded successfully: {len(df)} patients found")
            
            # Display data preview
            st.markdown("### 👀 Data Preview")
            st.dataframe(df.head(), use_container_width=True)
            
            # Validate data
            validation_results = validate_batch_data(df)
            
            if validation_results["errors"]:
                st.error("❌ Data validation failed:")
                for error in validation_results["errors"]:
                    st.error(f"• {error}")
                return
            
            if validation_results["warnings"]:
                st.warning("⚠️ Data validation warnings:")
                for warning in validation_results["warnings"]:
                    st.warning(f"• {warning}")
            
            # Process data
            if st.button("🔬 Process All Patients", use_container_width=True, type="primary"):
                process_batch_data(df)
                
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
            st.error("Please ensure your CSV file follows the template format")


def get_csv_template() -> str:
    """Generate CSV template with headers and example row."""
    template_data = {
        'age': [63],
        'sex': [1],  # 0=Female, 1=Male
        'cp': [3],   # 0-3 chest pain types
        'trestbps': [145],
        'chol': [233],
        'fbs': [1],  # 0=≤120, 1=>120
        'restecg': [0],  # 0-2
        'thalach': [150],
        'exang': [0],  # 0=No, 1=Yes
        'oldpeak': [2.3],
        'slope': [0],  # 0-2
        'ca': [0],  # 0-4
        'thal': [1]  # 1-3
    }
    
    df = pd.DataFrame(template_data)
    return df.to_csv(index=False)


def get_sample_csv_data() -> str:
    """Generate sample CSV data with multiple patients."""
    sample_data = {
        'age': [63, 37, 41, 56, 57],
        'sex': [1, 1, 0, 1, 0],
        'cp': [3, 2, 1, 1, 0],
        'trestbps': [145, 130, 130, 120, 120],
        'chol': [233, 250, 204, 236, 354],
        'fbs': [1, 0, 0, 0, 0],
        'restecg': [0, 1, 0, 1, 1],
        'thalach': [150, 187, 172, 178, 163],
        'exang': [0, 0, 0, 0, 1],
        'oldpeak': [2.3, 3.5, 1.4, 0.8, 0.6],
        'slope': [0, 0, 2, 2, 2],
        'ca': [0, 0, 0, 0, 0],
        'thal': [1, 2, 2, 2, 2]
    }
    
    df = pd.DataFrame(sample_data)
    return df.to_csv(index=False)


def validate_batch_data(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Validate batch data and return errors/warnings."""
    errors = []
    warnings = []
    
    required_columns = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
        'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
    ]
    
    # Check required columns
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        errors.append(f"Missing required columns: {missing_columns}")
    
    # Check data types and ranges
    if len(errors) == 0:  # Only if columns exist
        
        # Age validation
        age_issues = (df['age'] < 18) | (df['age'] > 120)
        if age_issues.any():
            errors.append(f"Age values out of range (18-120): {age_issues.sum()} patients")
        
        # Blood pressure validation
        bp_issues = (df['trestbps'] < 80) | (df['trestbps'] > 250)
        if bp_issues.any():
            warnings.append(f"Blood pressure values seem unusual: {bp_issues.sum()} patients")
        
        # Cholesterol validation
        chol_issues = (df['chol'] < 100) | (df['chol'] > 600)
        if chol_issues.any():
            warnings.append(f"Cholesterol values seem unusual: {chol_issues.sum()} patients")
        
        # Categorical validations
        categorical_validations = {
            'sex': [0, 1],
            'cp': [0, 1, 2, 3],
            'fbs': [0, 1],
            'restecg': [0, 1, 2],
            'exang': [0, 1],
            'slope': [0, 1, 2],
            'ca': [0, 1, 2, 3, 4],
            'thal': [1, 2, 3]
        }
        
        for col, valid_values in categorical_validations.items():
            if col in df.columns:
                invalid = ~df[col].isin(valid_values)
                if invalid.any():
                    errors.append(f"Invalid values in {col}: {invalid.sum()} patients")
    
    # Check batch size
    if len(df) > 100:
        errors.append(f"Batch size too large: {len(df)} patients (maximum: 100)")
    
    return {"errors": errors, "warnings": warnings}


def process_batch_data(df: pd.DataFrame):
    """Process batch data through API and display results."""
    st.markdown("### 🔄 Processing Patients...")
    
    # Convert DataFrame to API format
    patients_data = []
    for _, row in df.iterrows():
        patient_data = row.to_dict()
        patients_data.append(patient_data)
    
    # Prepare API request
    batch_request = {
        "patients": patients_data,
        "return_detailed": True
    }
    
    # Make API request
    with st.spinner("Processing patients..."):
        success, response = make_api_request(
            API_ENDPOINTS["batch_predict"], 
            method="POST", 
            data=batch_request
        )
    
    if not success:
        st.error(f"❌ Batch processing failed: {response}")
        return
    
    # Display results
    st.success(f"✅ Processing completed: {response['total_patients']} patients processed")
    
    # Summary statistics
    summary = response['summary']
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Patients", summary['total_processed'])
    
    with col2:
        st.metric("Average Risk", f"{summary['average_risk']:.1f}%")
    
    with col3:
        st.metric("High Risk Count", summary['high_risk_count'])
    
    with col4:
        st.metric("High Risk %", f"{summary['high_risk_percentage']:.1f}%")
    
    # Risk distribution
    st.markdown("### 📊 Risk Distribution")
    
    risk_dist = summary['risk_distribution']
    
    # Create simple risk distribution display
    st.markdown(f"""
    - **Low Risk:** {risk_dist.get('low', 0)} patients
    - **Medium Risk:** {risk_dist.get('medium', 0)} patients  
    - **High Risk:** {risk_dist.get('high', 0)} patients
    """)
    
    # Detailed results table
    st.markdown("### 📋 Detailed Results")
    
    results_data = []
    for i, prediction in enumerate(response['predictions']):
        patient_result = {
            'Patient': i + 1,
            'Risk %': f"{prediction['risk_percentage']:.1f}%",
            'Category': prediction['risk_category'].title(),
            'Prediction': 'Disease Risk' if prediction['prediction'] == 1 else 'No Disease',
            'Confidence': f"{prediction['confidence']:.1f}%"
        }
        results_data.append(patient_result)
    
    results_df = pd.DataFrame(results_data)
    st.dataframe(results_df, use_container_width=True)
    
    # Export results
    st.markdown("### 📁 Export Results")
    
    # Prepare detailed export data
    detailed_export = prepare_detailed_export(df, response['predictions'])
    
    st.download_button(
        label="📊 Download Detailed Results",
        data=detailed_export,
        file_name="batch_heart_risk_results.csv",
        mime="text/csv",
        use_container_width=True
    )


def prepare_detailed_export(original_df: pd.DataFrame, predictions: List[Dict]) -> str:
    """Prepare detailed export combining original data with predictions."""
    export_data = []
    
    for i, (_, patient) in enumerate(original_df.iterrows()):
        if i < len(predictions):
            prediction = predictions[i]
            
            export_row = {
                **patient.to_dict(),
                'prediction': prediction['prediction'],
                'risk_percentage': prediction['risk_percentage'],
                'risk_category': prediction['risk_category'],
                'confidence': prediction['confidence'],
                'top_recommendation': prediction['recommendations'][0] if prediction['recommendations'] else ''
            }
            
            export_data.append(export_row)
    
    export_df = pd.DataFrame(export_data)
    return export_df.to_csv(index=False)