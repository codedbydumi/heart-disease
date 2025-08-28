# Heart Disease Risk Prediction System

A professional machine learning system for predicting heart disease risk using advanced ensemble methods and interpretable AI.

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.85+-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.15+-red.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 🎯 Features

- **Advanced ML Pipeline**: Ensemble models (Random Forest + XGBoost + Logistic Regression)
- **Interactive Dashboard**: Professional Streamlit interface with medical styling
- **REST API**: FastAPI backend for integration and batch processing
- **Model Interpretability**: SHAP values for explainable predictions
- **Professional Deployment**: Docker containerization and cloud-ready
- **Data Validation**: Comprehensive input validation and error handling
- **Monitoring**: Structured logging and health monitoring

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- pip package manager

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/heart-disease-prediction.git
cd heart-disease-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp .env.example .env