"""Railway deployment main application - Combined API and simple frontend."""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from datetime import datetime

# Import your existing API components
from src.api.schemas import PatientInput, PredictionResponse
from src.api.prediction_service import HeartDiseasePredictionService

# Create FastAPI app
app = FastAPI(
    title="Heart Disease Risk Assessment API",
    description="Professional API for heart disease risk assessment",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize prediction service
try:
    prediction_service = HeartDiseasePredictionService()
except Exception as e:
    print(f"Warning: Could not initialize prediction service: {e}")
    prediction_service = None

@app.get("/")
async def root():
    """Root endpoint with simple HTML dashboard."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Heart Disease Risk Assessment</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: #333;
                min-height: 100vh;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                border-radius: 15px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.3);
                overflow: hidden;
            }
            .header {
                background: linear-gradient(135deg, #2E86AB, #A23B72);
                color: white;
                padding: 30px;
                text-align: center;
            }
            .header h1 {
                margin: 0;
                font-size: 2.5rem;
                font-weight: 600;
            }
            .header p {
                margin: 10px 0 0 0;
                font-size: 1.2rem;
                opacity: 0.9;
            }
            .content {
                padding: 30px;
            }
            .api-section {
                background: #f8f9fa;
                padding: 25px;
                border-radius: 10px;
                margin-bottom: 20px;
            }
            .api-section h2 {
                color: #2E86AB;
                margin-top: 0;
            }
            .endpoint {
                background: white;
                padding: 15px;
                border-radius: 8px;
                margin: 10px 0;
                border-left: 4px solid #2E86AB;
            }
            .method {
                display: inline-block;
                padding: 4px 12px;
                border-radius: 4px;
                font-weight: bold;
                margin-right: 10px;
            }
            .get { background: #28a745; color: white; }
            .post { background: #007bff; color: white; }
            .test-form {
                background: #e9ecef;
                padding: 20px;
                border-radius: 8px;
                margin-top: 20px;
            }
            .form-group {
                margin-bottom: 15px;
            }
            .form-group label {
                display: block;
                margin-bottom: 5px;
                font-weight: 500;
            }
            .form-group input, .form-group select {
                width: 100%;
                padding: 8px;
                border: 1px solid #ddd;
                border-radius: 4px;
                box-sizing: border-box;
            }
            .btn {
                background: linear-gradient(135deg, #2E86AB, #A23B72);
                color: white;
                padding: 12px 24px;
                border: none;
                border-radius: 6px;
                cursor: pointer;
                font-size: 16px;
                font-weight: 500;
                transition: transform 0.2s;
            }
            .btn:hover {
                transform: translateY(-2px);
            }
            .result {
                margin-top: 20px;
                padding: 15px;
                border-radius: 6px;
                background: white;
                border: 1px solid #ddd;
            }
            .risk-low { border-left: 5px solid #28a745; }
            .risk-medium { border-left: 5px solid #ffc107; }
            .risk-high { border-left: 5px solid #dc3545; }
            .grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin-top: 20px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🫀 Heart Disease Risk Assessment</h1>
                <p>Professional ML-powered Cardiac Risk Evaluation API</p>
            </div>
            
            <div class="content">
                <div class="api-section">
                    <h2>📡 API Endpoints</h2>
                    
                    <div class="endpoint">
                        <span class="method get">GET</span>
                        <strong>/health</strong> - API health check
                    </div>
                    
                    <div class="endpoint">
                        <span class="method get">GET</span>
                        <strong>/docs</strong> - Interactive API documentation
                    </div>
                    
                    <div class="endpoint">
                        <span class="method post">POST</span>
                        <strong>/predict</strong> - Single patient risk assessment
                    </div>
                    
                    <div class="endpoint">
                        <span class="method get">GET</span>
                        <strong>/model/info</strong> - Model information and metrics
                    </div>
                </div>

                <div class="api-section">
                    <h2>🔬 Quick Risk Assessment</h2>
                    <p>Enter patient data to get an instant risk assessment:</p>
                    
                    <form id="riskForm" class="test-form">
                        <div class="grid">
                            <div class="form-group">
                                <label for="age">Age (18-120):</label>
                                <input type="number" id="age" name="age" min="18" max="120" value="65" required>
                            </div>
                            
                            <div class="form-group">
                                <label for="sex">Sex:</label>
                                <select id="sex" name="sex" required>
                                    <option value="0">Female</option>
                                    <option value="1" selected>Male</option>
                                </select>
                            </div>
                            
                            <div class="form-group">
                                <label for="cp">Chest Pain Type:</label>
                                <select id="cp" name="cp" required>
                                    <option value="0">Typical Angina</option>
                                    <option value="1">Atypical Angina</option>
                                    <option value="2">Non-anginal Pain</option>
                                    <option value="3" selected>Asymptomatic</option>
                                </select>
                            </div>
                            
                            <div class="form-group">
                                <label for="trestbps">Resting BP (mm Hg):</label>
                                <input type="number" id="trestbps" name="trestbps" min="80" max="250" value="160" required>
                            </div>
                            
                            <div class="form-group">
                                <label for="chol">Cholesterol (mg/dl):</label>
                                <input type="number" id="chol" name="chol" min="100" max="600" value="280" required>
                            </div>
                            
                            <div class="form-group">
                                <label for="fbs">Fasting Blood Sugar >120:</label>
                                <select id="fbs" name="fbs" required>
                                    <option value="0">No</option>
                                    <option value="1" selected>Yes</option>
                                </select>
                            </div>
                        </div>
                        
                        <button type="submit" class="btn">🔬 Assess Risk</button>
                    </form>
                    
                    <div id="result" class="result" style="display: none;"></div>
                </div>

                <div class="api-section">
                    <h2>📚 Documentation</h2>
                    <p>For complete API documentation with interactive testing:</p>
                    <a href="/docs" class="btn" target="_blank">📖 View API Documentation</a>
                </div>
            </div>
        </div>

        <script>
            document.getElementById('riskForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                
                const formData = new FormData(e.target);
                const data = {};
                
                // Convert form data to API format
                for (let [key, value] of formData.entries()) {
                    data[key] = key === 'oldpeak' ? parseFloat(value) : parseInt(value);
                }
                
                // Add default values for missing fields
                data.restecg = 0;
                data.thalach = 150;
                data.exang = 0;
                data.oldpeak = 2.0;
                data.slope = 1;
                data.ca = 0;
                data.thal = 2;
                
                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify(data)
                    });
                    
                    if (!response.ok) {
                        throw new Error(`HTTP error! status: ${response.status}`);
                    }
                    
                    const result = await response.json();
                    displayResult(result);
                    
                } catch (error) {
                    document.getElementById('result').innerHTML = 
                        `<h3 style="color: #dc3545;">Error</h3><p>Failed to get prediction: ${error.message}</p>`;
                    document.getElementById('result').style.display = 'block';
                }
            });
            
            function displayResult(result) {
                const resultDiv = document.getElementById('result');
                const riskClass = `risk-${result.risk_category}`;
                const riskIcon = result.risk_category === 'low' ? '✅' : 
                                result.risk_category === 'medium' ? '⚠️' : '🚨';
                
                resultDiv.innerHTML = `
                    <h3>${riskIcon} ${result.risk_category.toUpperCase()} RISK</h3>
                    <p><strong>Risk Percentage:</strong> ${result.risk_percentage.toFixed(1)}%</p>
                    <p><strong>Model Confidence:</strong> ${result.confidence.toFixed(1)}%</p>
                    <p><strong>Prediction:</strong> ${result.prediction === 1 ? 'Heart Disease Risk Detected' : 'No Heart Disease Risk Detected'}</p>
                    
                    ${result.recommendations ? `
                        <h4>📋 Recommendations:</h4>
                        <ul>
                            ${result.recommendations.slice(0, 3).map(rec => `<li>${rec}</li>`).join('')}
                        </ul>
                    ` : ''}
                `;
                
                resultDiv.className = `result ${riskClass}`;
                resultDiv.style.display = 'block';
                resultDiv.scrollIntoView({ behavior: 'smooth' });
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "model_loaded": prediction_service is not None,
        "environment": "railway-production"
    }

@app.get("/model/info")
async def get_model_info():
    """Get model information."""
    if not prediction_service:
        raise HTTPException(status_code=503, detail="Model service not available")
    
    model_details = prediction_service.get_model_details()
    return model_details

@app.post("/predict", response_model=PredictionResponse)
async def predict_heart_disease(patient_data: PatientInput):
    """Predict heart disease risk for a single patient."""
    if not prediction_service:
        raise HTTPException(status_code=503, detail="Prediction service not available")
    
    try:
        prediction = prediction_service.predict_single(patient_data)
        return prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)