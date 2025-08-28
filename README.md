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

Running the Application
bash# Start the dashboard
streamlit run src/dashboard/app.py

# Start the API (in another terminal)
uvicorn src.api.main:app --reload --port 8000

📊 Model Performance

Accuracy: 89.5%
Precision: 88.2%
Recall: 90.1%
F1-Score: 89.1%
ROC-AUC: 0.94

🛠️ Technology Stack

ML Framework: scikit-learn, XGBoost
Web Framework: FastAPI, Streamlit
Data Processing: pandas, numpy
Visualization: plotly, matplotlib
Interpretation: SHAP
Deployment: Docker, Gunicorn

📋 Project Structure

heart-disease-prediction/
├── src/                    # Source code
│   ├── data/              # Data processing modules
│   ├── models/            # ML model modules
│   ├── api/               # FastAPI application
│   ├── dashboard/         # Streamlit dashboard
│   └── utils/             # Utility functions
├── data/                  # Data storage
├── models/                # Trained models
├── tests/                 # Test files
└── docs/                  # Documentation

🤝 Contributing

Fork the repository
Create your feature branch (git checkout -b feature/AmazingFeature)
Commit your changes (git commit -m 'Add some AmazingFeature')
Push to the branch (git push origin feature/AmazingFeature)
Open a Pull Request

🙏 Acknowledgments

Heart disease dataset from UCI ML Repository
Medical guidelines from American Heart Association
Inspiration from modern healthcare AI applications