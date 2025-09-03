# 🫀 Heart Disease Risk Assessment System

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.85+-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.15+-red.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Live Demo](https://img.shields.io/badge/demo-live-brightgreen.svg)](https://heartdisease.duminduthushan.com/)

> A professional machine learning system for cardiovascular risk assessment using advanced ensemble methods and explainable AI.

🔗 **Live Application**: https://heartdisease.duminduthushan.com/

## ✨ Features

### 🔬 Advanced Machine Learning
- **Ensemble Models**: Random Forest + XGBoost + Logistic Regression
- **High Performance**: 86.89% accuracy, 95.35% AUC-ROC
- **Real Dataset**: Trained on UCI Heart Disease dataset (303 patients)
- **Feature Engineering**: 20 engineered features from 13 base parameters

### 🏥 Medical Grade Interface
- **Professional UI**: Medical-themed design with clinical color scheme
- **Risk Stratification**: Low/Medium/High risk categorization
- **Medical Interpretations**: SHAP-based explainable predictions
- **Personalized Recommendations**: Actionable health advice

### ⚡ Production Features
- **REST API**: FastAPI backend with OpenAPI documentation
- **Interactive Dashboard**: Streamlit web interface
- **Batch Processing**: CSV upload for multiple patients
- **Real-time Predictions**: Sub-2 second response times
- **Export Capabilities**: Detailed CSV reports

### 🛡️ Enterprise Ready
- **Docker Containerization**: Multi-service architecture
- **SSL Security**: HTTPS with automatic certificate management
- **Health Monitoring**: Comprehensive logging and status checks
- **Scalable Deployment**: Nginx reverse proxy with load balancing

## 🎯 Use Cases

### 🏥 Healthcare Providers
- **Primary Care Screening**: Quick cardiovascular risk assessment
- **Decision Support**: Evidence-based referral recommendations
- **Population Health**: Batch analysis of patient cohorts
- **Telemedicine**: Remote risk evaluation

### 📊 Research & Analytics
- **Clinical Studies**: Risk stratification for research
- **Health Insurance**: Underwriting and wellness programs  
- **Corporate Wellness**: Employee health assessments
- **Public Health**: Population screening initiatives

## 🚀 Quick Start

### 🌐 Web Access
Simply visit: **https://heartdisease.duminduthushan.com/**

### 💻 Local Development

```bash
# Clone repository
git clone https://github.com/yourusername/heart-disease-prediction
cd heart-disease-prediction

# Setup environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Start services
python scripts/train_model.py    # Train ML models
python scripts/start_api.py      # Start API server
streamlit run scripts/start_dashboard.py # Start dashboard
```

### 🐳 Docker Deployment

```bash
# Production deployment
docker-compose up -d

# Access services
# Dashboard: http://localhost:8501
# API: http://localhost:8000/docs
```

## 📊 Model Performance

| Metric | Value | Benchmark |
|--------|--------|-----------|
| **Accuracy** | 86.89% | >80% (Clinical Standard) |
| **AUC-ROC** | 95.35% | >90% (Excellent) |
| **Precision** | 81.25% | High Specificity |
| **Recall** | 92.86% | High Sensitivity |
| **F1-Score** | 86.67% | Balanced Performance |

### 📈 Cross-Validation Results
- **Logistic Regression**: 81.39% ± 5.95%
- **Random Forest**: 80.16% ± 4.98%  
- **XGBoost**: 78.90% ± 8.21%

## 🏗️ Architecture

```mermaid
graph TB
    subgraph "Frontend Layer"
        A[Streamlit Dashboard] --> B[Nginx Reverse Proxy]
        B --> C[SSL Certificate]
    end
    
    subgraph "API Layer"  
        D[FastAPI Backend] --> E[Pydantic Validation]
        E --> F[Health Monitoring]
    end
    
    subgraph "ML Pipeline"
        G[Data Preprocessing] --> H[Feature Engineering]
        H --> I[Ensemble Models]
        I --> J[SHAP Interpretation]
    end
    
    subgraph "Data Layer"
        K[SQLite Database] --> L[Model Artifacts]
        L --> M[Training Data]
    end
    
    A -.->|API Calls| D
    D --> G
    G --> K
```

## 🛠️ Technology Stack

### 🧠 Machine Learning
- **Framework**: scikit-learn 1.1+
- **Algorithms**: Random Forest, XGBoost, Logistic Regression
- **Interpretability**: SHAP 0.41+
- **Validation**: Stratified K-Fold Cross-Validation

### 🌐 Web Services
- **API Framework**: FastAPI 0.85+
- **Frontend**: Streamlit 1.15+
- **Validation**: Pydantic 1.10+
- **Visualization**: Plotly 5.10+

### 🐳 Infrastructure
- **Containerization**: Docker & Docker Compose
- **Web Server**: Nginx (Alpine)
- **Process Management**: Uvicorn ASGI Server
- **SSL/TLS**: Let's Encrypt Certbot

### 📊 Data Processing
- **Data Manipulation**: Pandas 1.5+, NumPy 1.21+
- **Feature Engineering**: Custom medical domain features
- **Scaling**: RobustScaler for outlier handling
- **Validation**: Medical parameter range checking

## 📁 Project Structure

```
heart-disease-prediction/
├── 📄 README.md                   # Project documentation
├── 🐳 docker-compose.yml          # Multi-service orchestration
├── 🐳 Dockerfile.api              # API container definition
├── 🐳 Dockerfile.dashboard        # Dashboard container definition
├── ⚙️ nginx.conf                  # Reverse proxy configuration
├── 📦 requirements.txt            # Python dependencies
├── 📊 start_dashboard.py          # Streamlit application or scripts/start_dashboard.py
│
├── 📁 src/                        # Source code
│   ├── 🔧 api/                    # FastAPI backend
│   │   ├── main.py                 # API entry point
│   │   ├── prediction_service.py   # ML prediction service
│   │   └── schemas.py              # Pydantic models
│   ├── 📊 data/                   # Data processing
│   │   ├── data_loader.py         # Data loading utilities
│   │   ├── preprocessing.py       # Feature engineering
│   │   └── validation.py          # Data validation
│   ├── 🤖 models/                # ML model management
│   │   ├── model_trainer.py       # Training pipeline
│   │   └── ensemble_model.py      # Ensemble implementation
│   ├── 🖥️ dashboard/              # Streamlit components
│   └── ⚙️ utils/                  # Utility functions
│
├── 🎯 models/                     # Trained models
│   ├── trained_models/            # Serialized ML models
│   ├── scalers/                   # Feature scalers
│   └── metadata/                  # Model performance data
│
├── 📊 data/                       # Data storage
│   ├── raw/                       # Original datasets
│   ├── processed/                 # Cleaned datasets
│   └── schemas/                   # Data validation schemas
│
├── 🧪 tests/                      # Test suites
├── 📜 scripts/                    # Automation scripts
└── 📚 docs/                       # Documentation (User Guide And other data)
```

## 🎮 API Documentation

### 🔗 Endpoints

#### 🩺 Health Check
```http
GET /health
```
Returns service health status and system metrics.

#### 🔮 Single Prediction
```http
POST /predict
Content-Type: application/json

{
  "age": 63,
  "sex": 1,
  "cp": 3,
  "trestbps": 145,
  "chol": 233,
  "fbs": 1,
  "restecg": 0,
  "thalach": 150,
  "exang": 0,
  "oldpeak": 2.3,
  "slope": 0,
  "ca": 0,
  "thal": 1
}
```

#### 📊 Batch Prediction
```http
POST /predict/batch
Content-Type: application/json

{
  "patients": [
    { /* patient 1 data */ },
    { /* patient 2 data */ }
  ]
}
```

#### 📖 Interactive Documentation


## 📊 Clinical Parameters

| Parameter | Description | Range | Unit |
|-----------|-------------|-------|------|
| **Age** | Patient age | 18-120 | years |
| **Sex** | Biological sex | 0=Female, 1=Male | - |
| **CP** | Chest pain type | 0-3 | category |
| **RestBPS** | Resting blood pressure | 80-250 | mm Hg |
| **Chol** | Serum cholesterol | 100-600 | mg/dl |
| **FBS** | Fasting blood sugar | 0=≤120, 1=>120 | mg/dl |
| **RestECG** | Resting ECG results | 0-2 | category |
| **Thalach** | Max heart rate achieved | 60-220 | bpm |
| **Exang** | Exercise induced angina | 0=No, 1=Yes | boolean |
| **Oldpeak** | ST depression | 0.0-10.0 | mm |
| **Slope** | ST segment slope | 0-2 | category |
| **CA** | Major vessels | 0-4 | count |
| **Thal** | Thalassemia | 1-3 | category |

## 🎯 Risk Stratification

### 🟢 Low Risk (< 30%)
- **Interpretation**: Low probability of heart disease
- **Action**: Continue healthy lifestyle and regular check-ups
- **Follow-up**: Annual cardiovascular screening

### 🟡 Medium Risk (30-70%)  
- **Interpretation**: Moderate probability of heart disease
- **Action**: Lifestyle modifications and closer monitoring
- **Follow-up**: Semi-annual assessment, consider stress testing

### 🔴 High Risk (> 70%)
- **Interpretation**: High probability of heart disease
- **Action**: Immediate medical consultation recommended
- **Follow-up**: Urgent cardiology referral, comprehensive evaluation

## 🚀 Deployment Guide

### 🌐 Production Deployment

The system is deployed on a Contabo VPS with the following configuration:

- **Server**: Ubuntu 22.04 LTS
- **Domain**: heartdisease.duminduthushan.com
- **SSL**: Let's Encrypt with automatic renewal
- **Monitoring**: Docker health checks and logging
- **Backup**: Automated model and data backups

### 🔧 Environment Variables

```bash
# Production Configuration
ENVIRONMENT=production
LOG_LEVEL=INFO
SECRET_KEY=your-production-secret-key

# API Settings
API_HOST=0.0.0.0
API_PORT=8000

# Database
DATABASE_URL=sqlite:///./heart_disease.db

# Model Configuration
MODEL_VERSION=v1.0.0
MODEL_PATH=./models/trained_models/
```

## 🧪 Testing

### ✅ Test Coverage
- **Unit Tests**: Individual component testing
- **Integration Tests**: API endpoint validation
- **End-to-End Tests**: Complete workflow verification
- **Load Tests**: Performance under concurrent users

### 🔍 Quality Assurance
- **Data Validation**: Medical parameter range checking
- **Model Validation**: Cross-validation with hold-out test set
- **API Testing**: Comprehensive endpoint testing
- **UI Testing**: Dashboard functionality verification

## 📈 Performance Metrics

### ⚡ Response Times
- **Single Prediction**: < 2 seconds
- **Batch Processing**: < 1 second per patient
- **Dashboard Loading**: < 3 seconds
- **API Documentation**: < 1 second

### 🎯 Accuracy Benchmarks
- **Clinical Standard**: >80% accuracy required
- **Our Performance**: 86.89% accuracy achieved
- **Industry Comparison**: Top quartile performance
- **Validation Method**: Stratified 5-fold cross-validation

## ⚠️ Important Disclaimers

### 🏥 Medical Disclaimer
This system is designed for educational and research purposes. **It should not replace professional medical advice, diagnosis, or treatment.** Always consult qualified healthcare providers for medical decisions.

### 🔒 Privacy & Security
- Patient data is processed locally and not stored permanently
- All communications encrypted with SSL/TLS
- HIPAA-aware data handling practices
- No personal health information retained

### 📊 Model Limitations
- Based on UCI dataset with specific demographic characteristics
- Performance may vary across different populations
- Requires validation in clinical settings before medical use
- Regular model updates recommended with new data

## 🤝 Contributing

### 🔧 Development Setup
```bash
# Fork the repository
git clone https://github.com/yourusername/heart-disease-prediction
cd heart-disease-prediction

# Create development branch
git checkout -b feature/your-feature-name

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Submit pull request
```

### 📋 Contribution Guidelines
- Follow PEP 8 code styling
- Add tests for new features
- Update documentation for API changes
- Ensure medical accuracy for clinical features

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

### 📊 Data Sources
- **UCI Machine Learning Repository**: Heart Disease Dataset
- **American Heart Association**: Clinical guidelines and risk factors
- **Mayo Clinic**: Medical parameter interpretations

### 🛠️ Technologies
- **Streamlit Team**: Amazing framework for ML applications
- **FastAPI Team**: High-performance web framework
- **scikit-learn Contributors**: Comprehensive ML library
- **Docker Team**: Containerization platform

### 🎓 Research References
- Detrano et al. (1989) - Original UCI Heart Disease study
- American Heart Association Guidelines
- European Society of Cardiology Risk Assessment

## 📞 Contact & Support

### 👨‍💻 Developer
- **Name**: Duminda Thushan
- **Portfolio**: https://duminduthushan.com
- **LinkedIn**: [Connect on LinkedIn](https://linkedin.com/in/dumindu-thushan)
- **Email**: duminduthushan9@gmail.com

### 🔗 Project Links
- **Live Demo**: https://heartdisease.duminduthushan.com/
- **API Documentation**: https://._______________./docs
- **Source Code**: https://github.com/codedbydumi/heart-disease/tree/main/src
- **Issue Tracker**: https://github.com/codedbydumi/heart-disease/issues

### 📈 Project Status
- **Status**: ✅ Production Ready
- **Version**: v1.0.0
- **Last Updated**: August 2025
- **Maintenance**: Actively maintained

---

<div align="center">
  <h3>🫀 Built with ❤️ by Codedbydumi for Better Healthcare</h3>
  <p>Empowering healthcare professionals with AI-driven insights for cardiovascular risk assessment.</p>
  
  [![Made with Python](https://img.shields.io/badge/Made%20with-Python-blue?style=for-the-badge&logo=python)](https://python.org)
  [![Powered by ML](https://img.shields.io/badge/Powered%20by-Machine%20Learning-green?style=for-the-badge&logo=scikitlearn)](https://scikit-learn.org)
  [![Deployed with Docker](https://img.shields.io/badge/Deployed%20with-Docker-blue?style=for-the-badge&logo=docker)](https://docker.com)
</div>
