# Heart Disease Risk Prediction System

A professional machine learning system for predicting heart disease risk using advanced ensemble methods and interpretable AI.

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.85+-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.15+-red.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)



<div align="center">




[Live Demo](https://heartdisease.duminduthushan.com) • [API Documentation](https://heartdisease.duminduthushan.com/docs) • [User Guide](docs/user-guide.md)

</div>

## Overview

An enterprise-grade machine learning system for predicting heart disease risk using advanced ensemble methods and medical domain expertise. This system combines multiple ML algorithms to provide accurate risk assessments with medical interpretations and personalized recommendations.

### Key Features

- **Advanced ML Pipeline**: Ensemble models (Random Forest + XGBoost + Logistic Regression)
- **Interactive Dashboard**: Professional Streamlit interface with medical styling
- **REST API**: Comprehensive FastAPI backend for integration
- **Batch Processing**: CSV upload for multiple patient assessments
- **Medical Interpretations**: SHAP-powered explainable AI with clinical insights
- **Professional Deployment**: Docker containerization with production-ready infrastructure
- **Real-time Analytics**: Performance monitoring and data quality tracking

## Performance Metrics

- **Test Accuracy**: 86.89%
- **AUC-ROC**: 95.35%
- **Cross-validation**: 81.39%
- **Precision**: 81.25%
- **Recall**: 92.86%

*Trained on UCI Heart Disease Dataset (303 patients) with comprehensive validation*

## Quick Start

### Prerequisites

- Python 3.9+
- Docker & Docker Compose (for containerized deployment)
- 4GB RAM minimum

### Local Development

```bash
# Clone the repository
git clone https://github.com/duminduthushan/heart-disease-prediction-system.git
cd heart-disease-prediction-system

# Set up virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment configuration
cp .env.example .env

# Train the model (optional - pre-trained models included)
python scripts/train_model.py

# Start the API server
python scripts/start_api.py

# Start the dashboard (in another terminal)
python scripts/start_dashboard.py
```

Visit:
- **Dashboard**: http://localhost:8501
- **API Documentation**: http://localhost:8000/docs

### Docker Deployment

```bash
# Clone and navigate to repository
git clone https://github.com/duminduthushan/heart-disease-prediction-system.git
cd heart-disease-prediction-system

# Start with Docker Compose
docker-compose up -d

# Check status
docker-compose ps
```

## System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit     │    │    FastAPI      │    │   ML Pipeline   │
│   Dashboard     │◄──►│   Backend       │◄──►│   Ensemble      │
│                 │    │                 │    │   Models        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│     Nginx       │    │   SQLite DB     │    │   SHAP/LIME     │
│  Reverse Proxy  │    │   Data Store    │    │  Interpretability│
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Features in Detail

### 1. Risk Assessment Engine

- **Ensemble Learning**: Combines Random Forest, XGBoost, and Logistic Regression
- **Feature Engineering**: 20 engineered features including medical risk scores
- **Risk Stratification**: Low (<30%), Medium (30-70%), High (>70%) categories
- **Confidence Scoring**: Model uncertainty quantification

### 2. Professional Dashboard

- **Medical-Grade UI**: Professional healthcare interface design
- **Interactive Forms**: Comprehensive patient data input with validation
- **Real-time Analysis**: Instant risk assessment with visual feedback
- **Batch Processing**: CSV upload for multiple patients
- **Export Capabilities**: Detailed CSV reports with recommendations

### 3. Enterprise API

- **RESTful Design**: Clean, documented API endpoints
- **Data Validation**: Pydantic schemas with medical range checking
- **Batch Operations**: Process multiple patients simultaneously
- **Health Monitoring**: System status and performance metrics
- **Auto Documentation**: Interactive OpenAPI/Swagger interface

### 4. Medical Intelligence

- **Clinical Interpretations**: Convert ML features to medical insights
- **Personalized Recommendations**: Actionable health advice
- **Risk Factor Analysis**: Identify key contributors to patient risk
- **Medical Compliance**: Follows standard medical parameter ranges

## API Usage

### Single Patient Prediction

```python
import requests

patient_data = {
    "age": 63,
    "sex": 1,  # 1=Male, 0=Female
    "cp": 3,   # Chest pain type (0-3)
    "trestbps": 145,  # Blood pressure
    "chol": 233,      # Cholesterol
    "fbs": 1,         # Fasting blood sugar
    "restecg": 0,     # ECG results
    "thalach": 150,   # Max heart rate
    "exang": 0,       # Exercise angina
    "oldpeak": 2.3,   # ST depression
    "slope": 0,       # ST slope
    "ca": 0,          # Major vessels
    "thal": 1         # Thalassemia
}

response = requests.post(
    "https://heartdisease.duminduthushan.com/predict",
    json=patient_data
)

result = response.json()
print(f"Risk: {result['risk_percentage']:.1f}%")
print(f"Category: {result['risk_category']}")
```

### Batch Processing

```python
batch_data = {
    "patients": [patient_data_1, patient_data_2, ...],
    "return_detailed": True
}

response = requests.post(
    "https://heartdisease.duminduthushan.com/predict/batch",
    json=batch_data
)
```

## Development

### Project Structure

```
src/
├── api/                 # FastAPI backend
├── data/               # Data processing modules
├── models/             # ML model implementations
├── dashboard/          # Streamlit frontend
└── utils/              # Shared utilities

scripts/                # Utility scripts
tests/                  # Test suites
deployment/             # Docker and deployment configs
```

### Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Test specific components
python -m pytest tests/test_api.py -v
python -m pytest tests/test_models.py -v

# Test with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

### Model Training

```bash
# Train with default settings
python scripts/train_model.py

# Custom training configuration
python scripts/train_model.py --data-path data/custom_dataset.csv --test-size 0.2
```

## Deployment

### Production Deployment

The system is deployed at [heartdisease.duminduthushan.com](https://heartdisease.duminduthushan.com) using:

- **Infrastructure**: Contabo VPS
- **Containerization**: Docker + Docker Compose
- **Web Server**: Nginx reverse proxy
- **SSL**: Let's Encrypt certificates
- **Monitoring**: Health checks and logging

### Deployment Guide

See [deployment documentation](docs/deployment.md) for detailed production setup instructions.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt

# Install pre-commit hooks
pre-commit install

# Run code formatting
black src/ tests/
isort src/ tests/

# Run linting
flake8 src/ tests/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Medical Disclaimer

This system is for educational and research purposes only. It should not replace professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare providers for medical decisions.

## Acknowledgments

- **Dataset**: UCI Heart Disease Dataset
- **Medical Guidelines**: American Heart Association standards
- **ML Framework**: scikit-learn, XGBoost ecosystem
- **Web Framework**: FastAPI and Streamlit communities

## Citation

If you use this system in your research, please cite:

```bibtex
@software{heart_disease_prediction_system,
  author = {Dumindu Thushan},
  title = {Heart Disease Risk Assessment System},
  url = {https://github.com/duminduthushan/heart-disease-prediction-system},
  year = {2025}
}
```

## Support

- **Documentation**: [docs/](docs/)
- **Issues**: [GitHub Issues](https://github.com/duminduthushan/heart-disease-prediction-system/issues)
- **Discussions**: [GitHub Discussions](https://github.com/duminduthushan/heart-disease-prediction-system/discussions)

---

<div align="center">

**Built with ❤️ for better healthcare through AI**

[⭐ Star this repository](https://github.com/duminduthushan/heart-disease-prediction-system) if you found it helpful!

</div>
