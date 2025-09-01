# Heart Disease Risk Assessment - Technical Specifications

## Machine Learning Model Specifications

### Ensemble Architecture

The system employs a sophisticated ensemble approach combining three complementary algorithms to maximize prediction accuracy and reliability.

#### Primary Models
- **Logistic Regression**: Linear model providing interpretable baseline predictions
- **Random Forest**: Tree-based ensemble handling non-linear relationships
- **XGBoost**: Gradient boosting for complex pattern recognition

#### Ensemble Method
- **Strategy**: Soft voting with probability averaging
- **Weight Distribution**: Equal weighting across all models
- **Final Prediction**: Aggregated probability from all three models
- **Performance**: Best individual model (Logistic Regression: 81.39% CV accuracy)

### Dataset and Training Specifications

#### Training Dataset
- **Source**: UCI Heart Disease Dataset (Cleveland Clinic Foundation)
- **Sample Size**: 303 patients with complete clinical data
- **Feature Count**: 13 original clinical parameters
- **Target Variable**: Binary classification (0=No disease, 1=Disease presence)
- **Class Distribution**: 138 negative, 165 positive cases

#### Data Quality Metrics
- **Completeness**: 100% complete cases (no missing values)
- **Data Integrity**: Medical parameter validation applied
- **Temporal Consistency**: Single time-point clinical assessments
- **Demographic Distribution**: Mixed age, gender representation

### Feature Engineering Pipeline

#### Original Clinical Parameters
1. **age**: Patient age (18-120 years)
2. **sex**: Biological sex (0=Female, 1=Male)
3. **cp**: Chest pain type (0-3 categorical)
4. **trestbps**: Resting blood pressure (80-250 mm Hg)
5. **chol**: Serum cholesterol (100-600 mg/dl)
6. **fbs**: Fasting blood sugar (0=≤120, 1=>120 mg/dl)
7. **restecg**: Resting ECG results (0-2 categorical)
8. **thalach**: Maximum heart rate achieved (60-220 bpm)
9. **exang**: Exercise induced angina (0=No, 1=Yes)
10. **oldpeak**: ST depression induced by exercise (0.0-10.0 mm)
11. **slope**: Peak exercise ST segment slope (0-2 categorical)
12. **ca**: Number of major vessels (0-4)
13. **thal**: Thalassemia (1-3 categorical)

#### Engineered Features (7 Additional)
1. **age_group**: Age stratification (0-3 categories)
   - 0: 18-40 years (Low risk)
   - 1: 41-55 years (Moderate risk)
   - 2: 56-65 years (Higher risk)
   - 3: >65 years (High risk)

2. **bp_category**: Blood pressure classification
   - 0: Normal (<120 mm Hg)
   - 1: Elevated (120-139 mm Hg)
   - 2: Stage 1 HTN (140-179 mm Hg)
   - 3: Stage 2 HTN (≥180 mm Hg)

3. **chol_category**: Cholesterol risk levels
   - 0: Desirable (<200 mg/dl)
   - 1: Borderline (200-239 mg/dl)
   - 2: High (240-299 mg/dl)
   - 3: Very High (≥300 mg/dl)

4. **hr_reserve**: Heart rate reserve calculation
   - Formula: max_hr_achieved - (220 - age)
   - Range: Typically -70 to +50 bpm

5. **age_chol_interaction**: Age-cholesterol interaction term
   - Formula: (age × cholesterol) / 1000
   - Captures combined age-lipid risk

6. **bp_age_interaction**: Blood pressure-age interaction
   - Formula: (blood_pressure × age) / 1000
   - Models age-related hypertension risk

7. **composite_risk**: Weighted composite risk score
   - Formula: (age × 0.1) + (chol × 0.01) + (bp × 0.1) + (cp × 10) + (exang × 20)
   - Integrates multiple risk factors

### Model Performance Metrics

#### Cross-Validation Results (5-Fold Stratified)
```
Individual Model Performance:
├── Logistic Regression: 81.39% ± 5.95%
├── Random Forest: 80.16% ± 4.98%
└── XGBoost: 78.90% ± 8.21%

Best Model Selection: Logistic Regression
```

#### Independent Test Set Performance
```
Final Model Performance (n=61):
├── Accuracy: 86.89%
├── Precision: 81.25%
├── Recall: 92.86%
├── F1-Score: 86.67%
├── AUC-ROC: 95.35%
├── Specificity: 78.79%
└── NPV: 92.86%
```

#### Clinical Performance Interpretation
- **Sensitivity (Recall)**: 92.86% - Excellent detection of disease cases
- **Specificity**: 78.79% - Good identification of healthy patients
- **PPV (Precision)**: 81.25% - Strong positive prediction accuracy
- **NPV**: 92.86% - Excellent negative prediction reliability

### Preprocessing Pipeline Specifications

#### Data Validation
- **Parameter Range Checking**: Medical validity constraints applied
- **Outlier Detection**: Statistical and clinical outlier identification
- **Data Type Validation**: Ensure appropriate numeric/categorical types
- **Missing Value Strategy**: Median imputation for continuous, mode for categorical

#### Feature Scaling
- **Algorithm**: RobustScaler from scikit-learn
- **Rationale**: Less sensitive to outliers than StandardScaler
- **Quartile Range**: 25th to 75th percentile scaling
- **Median Centering**: Features centered on median value

#### Feature Selection Validation
- **Correlation Analysis**: Multicollinearity assessment
- **Feature Importance**: Random Forest-based importance scoring
- **Clinical Relevance**: Medical domain expert validation
- **Variance Analysis**: Low-variance feature identification

## System Architecture Specifications

### Backend Infrastructure

#### API Framework
- **Framework**: FastAPI 0.101.1
- **ASGI Server**: Uvicorn with uvloop for high performance
- **Request Validation**: Pydantic 2.1.1 for schema validation
- **Async Support**: Full asynchronous request handling
- **Documentation**: Automatic OpenAPI 3.0 generation

#### Data Processing Stack
- **Core Libraries**:
  - pandas 2.0.3: Data manipulation and analysis
  - NumPy 1.24.3: Numerical computing foundation
  - scikit-learn 1.3.0: Machine learning algorithms
  - XGBoost 1.7.6: Gradient boosting implementation
  - SHAP 0.42.1: Model interpretability

#### Model Serving Architecture
```python
Model Loading Strategy:
├── Pickle Serialization: Primary model storage format
├── Lazy Loading: Models loaded on first request
├── Memory Caching: Models cached in application memory
├── Version Management: Model versioning support
└── Fallback Strategy: Default model for failures
```

### Frontend Infrastructure

#### Web Framework
- **Framework**: Streamlit 1.25.0
- **Python Version**: 3.9+ required
- **Session Management**: Streamlit session state
- **Component Architecture**: Modular page components
- **Styling**: Custom CSS with medical theme

#### Visualization Stack
- **Primary Library**: Plotly 5.10+ for interactive charts
- **Chart Types**:
  - Risk gauge visualization
  - Bar charts for comparisons
  - Progress bars for risk levels
  - Interactive scatter plots
- **Responsive Design**: Mobile-optimized layouts

### Database and Storage

#### Data Storage Strategy
- **Configuration**: SQLite for development, PostgreSQL-ready
- **Model Artifacts**: File system storage with backup
- **Session Data**: Memory-based temporary storage
- **No PHI Storage**: Patient data not persisted
- **Backup Strategy**: Automated model and config backup

#### File System Organization
```
Storage Hierarchy:
├── models/trained_models/: Serialized ML models
├── models/scalers/: Feature preprocessing objects
├── models/metadata/: Performance metrics and info
├── data/schemas/: Data validation schemas
├── logs/: Application and error logs
└── config/: Environment configurations
```

## Infrastructure and Deployment

### Container Specifications

#### Base Images
- **API Container**: python:3.9-slim (Debian-based)
- **Dashboard Container**: python:3.9-slim
- **Proxy Container**: nginx:alpine (Minimal footprint)

#### Container Resource Requirements
```yaml
API Service:
├── CPU: 0.5-2.0 vCPU cores
├── Memory: 1-4 GB RAM
├── Storage: 2 GB for application and models
└── Network: HTTP/HTTPS ports

Dashboard Service:
├── CPU: 0.25-1.0 vCPU cores
├── Memory: 512 MB - 2 GB RAM
├── Storage: 1 GB for application
└── Network: Streamlit port 8501

Nginx Proxy:
├── CPU: 0.1-0.5 vCPU cores
├── Memory: 128-512 MB RAM
├── Storage: 100 MB for configs
└── Network: Ports 80, 443
```

#### Docker Image Optimization
- **Multi-stage Builds**: Minimize production image size
- **Layer Caching**: Efficient build process
- **Security Scanning**: Vulnerability assessment
- **Health Checks**: Container health monitoring

### Production Environment

#### Operating System
- **Distribution**: Ubuntu 22.04 LTS
- **Kernel**: Linux 5.15+ recommended
- **Container Runtime**: Docker 24.0.5+
- **Orchestration**: Docker Compose 2.20+

#### Network Configuration
- **Domain**: heartdisease.duminduthushan.com
- **SSL/TLS**: Let's Encrypt with automatic renewal
- **Ports**: 80 (HTTP), 443 (HTTPS), 8501 (Direct access)
- **Proxy Configuration**: Nginx reverse proxy

#### Security Specifications
- **Encryption**: TLS 1.3 for all communications
- **Certificates**: 256-bit RSA with SHA-256
- **Security Headers**: HSTS, X-Frame-Options, CSP
- **Rate Limiting**: 100 requests/minute per IP

## Performance Specifications

### Response Time Requirements

#### API Performance Targets
- **Single Prediction**: <2 seconds (95th percentile)
- **Batch Processing**: <1 second per patient
- **Health Check**: <100 milliseconds
- **Model Info**: <500 milliseconds
- **Documentation**: <1 second

#### Dashboard Performance
- **Initial Load**: <3 seconds for first render
- **Form Submission**: <2 seconds for processing
- **Result Display**: <1 second for visualization
- **CSV Export**: <5 seconds for file generation

### Scalability Specifications

#### Concurrent User Support
- **Simultaneous Users**: 100+ active users
- **Request Throughput**: 1,000+ requests per minute
- **Batch Capacity**: 10,000 patients per hour
- **Memory Efficiency**: Linear scaling with user load

#### Resource Utilization Targets
- **CPU Usage**: <80% under normal load
- **Memory Usage**: <2 GB per service container
- **Disk I/O**: Minimal for stateless operation
- **Network Bandwidth**: <10 Mbps per 100 users

### Availability and Reliability

#### Uptime Requirements
- **Target Availability**: 99.9% (8.77 hours downtime/year)
- **Planned Maintenance**: <4 hours/month
- **Recovery Time**: <5 minutes for service restart
- **Backup Frequency**: Daily automated backups

#### Monitoring and Alerting
- **Health Monitoring**: Continuous service health checks
- **Performance Metrics**: Response time and throughput tracking
- **Error Tracking**: Structured error logging and alerting
- **Resource Monitoring**: CPU, memory, and disk usage alerts

## Security and Compliance

### Data Protection Standards

#### Encryption Specifications
- **Data in Transit**: TLS 1.3 with perfect forward secrecy
- **API Communications**: HTTPS mandatory
- **Certificate Management**: Automated Let's Encrypt renewal
- **Cipher Suites**: Modern, secure cipher suite selection

#### Privacy Protection
- **Data Minimization**: Only essential data processed
- **No Persistent Storage**: Patient data not retained
- **Session Management**: Secure session handling
- **Data Anonymization**: No personally identifiable information stored

### Compliance Considerations

#### Healthcare Regulations
- **HIPAA Readiness**: Architecture supports PHI handling requirements
- **Data Processing**: Transparent data flow documentation
- **Audit Trails**: Comprehensive logging for compliance reporting
- **Access Controls**: Role-based access control framework

#### International Standards
- **GDPR Compliance**: Data processing transparency and consent
- **ISO 27001**: Information security management alignment
- **SOC 2**: Security, availability, and confidentiality controls
- **Medical Device**: Class I software considerations (non-diagnostic)

## Testing and Quality Assurance

### Testing Framework

#### Automated Testing
- **Unit Tests**: pytest framework with >90% coverage
- **Integration Tests**: End-to-end API testing
- **Load Testing**: Concurrent user simulation
- **Security Testing**: Vulnerability scanning

#### Performance Testing
- **Stress Testing**: Maximum load capacity determination
- **Endurance Testing**: Long-duration stability validation
- **Volume Testing**: Large batch processing verification
- **Scalability Testing**: Concurrent user load testing

### Quality Metrics

#### Code Quality Standards
- **Code Coverage**: >90% test coverage required
- **Static Analysis**: Flake8 linting and MyPy type checking
- **Security Scanning**: Bandit security analysis
- **Documentation**: Comprehensive docstring coverage

#### Model Quality Assurance
- **Cross-Validation**: Stratified k-fold validation
- **Bias Testing**: Demographic fairness assessment
- **Calibration**: Probability calibration validation
- **Drift Detection**: Model performance monitoring

This technical specification provides comprehensive details for implementing, deploying, and maintaining the Heart Disease Risk Assessment System in production environments while ensuring clinical accuracy, security, and regulatory compliance.