"""Test script for FastAPI endpoints."""

import sys
import os
from pathlib import Path
import requests
import json

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from src.utils.logger import get_logger

logger = get_logger("test_api")

# API base URL
BASE_URL = "http://localhost:8000"


def test_health_endpoint():
    """Test health check endpoint."""
    try:
        response = requests.get(f"{BASE_URL}/health")
        
        if response.status_code == 200:
            data = response.json()
            logger.info(f"✅ Health check passed - Status: {data['status']}")
            return True
        else:
            logger.error(f"❌ Health check failed - Status: {response.status_code}")
            return False
            
    except requests.RequestException as e:
        logger.error(f"❌ Health check failed - Error: {e}")
        return False


def test_model_info_endpoint():
    """Test model info endpoint."""
    try:
        response = requests.get(f"{BASE_URL}/model/info")
        
        if response.status_code == 200:
            data = response.json()
            logger.info(f"✅ Model info retrieved - Model: {data['model_name']}, Features: {data['total_features']}")
            return True
        else:
            logger.error(f"❌ Model info failed - Status: {response.status_code}")
            return False
            
    except requests.RequestException as e:
        logger.error(f"❌ Model info failed - Error: {e}")
        return False


def test_single_prediction():
    """Test single patient prediction."""
    patient_data = {
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
    
    try:
        response = requests.post(f"{BASE_URL}/predict", json=patient_data)
        
        if response.status_code == 200:
            data = response.json()
            logger.info(f"✅ Single prediction successful - Risk: {data['risk_percentage']:.1f}%, Category: {data['risk_category']}")
            logger.info(f"   Recommendations: {len(data['recommendations'])} provided")
            return True
        else:
            logger.error(f"❌ Single prediction failed - Status: {response.status_code}")
            logger.error(f"   Response: {response.text}")
            return False
            
    except requests.RequestException as e:
        logger.error(f"❌ Single prediction failed - Error: {e}")
        return False


def test_batch_prediction():
    """Test batch prediction."""
    batch_data = {
        "patients": [
            {
                "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
                "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
                "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
            },
            {
                "age": 37, "sex": 1, "cp": 2, "trestbps": 130, "chol": 250,
                "fbs": 0, "restecg": 1, "thalach": 187, "exang": 0,
                "oldpeak": 3.5, "slope": 0, "ca": 0, "thal": 2
            }
        ]
    }
    
    try:
        response = requests.post(f"{BASE_URL}/predict/batch", json=batch_data)
        
        if response.status_code == 200:
            data = response.json()
            logger.info(f"✅ Batch prediction successful - {data['total_patients']} patients processed")
            logger.info(f"   Average risk: {data['summary']['average_risk']:.1f}%")
            logger.info(f"   High risk count: {data['summary']['high_risk_count']}")
            return True
        else:
            logger.error(f"❌ Batch prediction failed - Status: {response.status_code}")
            logger.error(f"   Response: {response.text}")
            return False
            
    except requests.RequestException as e:
        logger.error(f"❌ Batch prediction failed - Error: {e}")
        return False


def test_sample_endpoint():
    """Test sample prediction endpoint."""
    try:
        response = requests.get(f"{BASE_URL}/predict/sample")
        
        if response.status_code == 200:
            data = response.json()
            prediction = data['prediction']
            logger.info(f"✅ Sample prediction successful - Risk: {prediction['risk_percentage']:.1f}%")
            return True
        else:
            logger.error(f"❌ Sample prediction failed - Status: {response.status_code}")
            return False
            
    except requests.RequestException as e:
        logger.error(f"❌ Sample prediction failed - Error: {e}")
        return False


def run_all_tests():
    """Run all API tests."""
    logger.info("🚀 Starting API tests...")
    logger.info(f"   Testing API at: {BASE_URL}")
    
    tests = [
        ("Health Check", test_health_endpoint),
        ("Model Info", test_model_info_endpoint),
        ("Single Prediction", test_single_prediction),
        ("Batch Prediction", test_batch_prediction),
        ("Sample Endpoint", test_sample_endpoint)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n🧪 Testing {test_name}...")
        if test_func():
            passed += 1
        
    logger.info(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All API tests passed successfully!")
        return True
    else:
        logger.error(f"❌ {total - passed} tests failed")
        return False


if __name__ == "__main__":
    print("🔧 Make sure the API server is running:")
    print("   python -m uvicorn src.api.main:app --reload --port 8000")
    print("\nPress Enter to start tests...")
    input()
    
    success = run_all_tests()
    exit(0 if success else 1)