"""API client utilities for the dashboard."""

import os
import requests
import streamlit as st
from typing import Dict, Any, Optional
import json

class APIClient:
    """API client for dashboard-API communication."""
    
    def __init__(self):
        self.base_url = os.getenv("API_BASE_URL", "http://localhost/api")
        self.endpoints = {
            "health": f"{self.base_url}/health",
            "predict": f"{self.base_url}/predict",
            "batch_predict": f"{self.base_url}/predict/batch",
            "model_info": f"{self.base_url}/model/info",
            "sample": f"{self.base_url}/predict/sample"
        }
        
    def check_health(self) -> Dict[str, Any]:
        """Check API health status."""
        try:
            response = requests.get(self.endpoints["health"], timeout=5)
            if response.status_code == 200:
                return {
                    "status": "connected",
                    "message": "API is healthy",
                    "data": response.json() if response.text else {}
                }
            else:
                return {
                    "status": "error",
                    "message": f"API returned status code: {response.status_code}",
                    "data": {}
                }
        except requests.exceptions.ConnectionError:
            return {
                "status": "disconnected",
                "message": "Cannot connect to API server",
                "data": {}
            }
        except requests.exceptions.Timeout:
            return {
                "status": "timeout",
                "message": "API request timed out",
                "data": {}
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Unexpected error: {str(e)}",
                "data": {}
            }
    
    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Make prediction request."""
        try:
            response = requests.post(
                self.endpoints["predict"],
                json=features,
                timeout=10
            )
            if response.status_code == 200:
                return {
                    "status": "success",
                    "data": response.json()
                }
            else:
                return {
                    "status": "error",
                    "message": f"Prediction failed with status: {response.status_code}",
                    "data": {}
                }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Prediction error: {str(e)}",
                "data": {}
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        try:
            response = requests.get(self.endpoints["model_info"], timeout=5)
            if response.status_code == 200:
                return {
                    "status": "success",
                    "data": response.json()
                }
            else:
                return {
                    "status": "error",
                    "message": f"Failed to get model info: {response.status_code}",
                    "data": {}
                }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Model info error: {str(e)}",
                "data": {}
            }

def display_api_status():
    """Display API connection status in sidebar."""
    client = APIClient()
    health = client.check_health()
    
    if health["status"] == "connected":
        st.sidebar.success("🟢 API Connected")
    elif health["status"] == "disconnected":
        st.sidebar.error("🔴 API Disconnected")
        st.sidebar.warning("Please ensure the API server is running")
        st.sidebar.code(f"Expected API URL: {client.base_url}")
    else:
        st.sidebar.warning(f"⚠️ API Issue: {health['message']}")
    
    # Debug info in expander
    with st.sidebar.expander("🔍 Debug Info"):
        st.write(f"**API Base URL:** `{client.base_url}`")
        st.write(f"**Health Endpoint:** `{client.endpoints['health']}`")
        st.write(f"**Environment API_BASE_URL:** `{os.getenv('API_BASE_URL', 'Not set')}`")
        st.write(f"**Status:** {health['status']}")
        st.write(f"**Message:** {health['message']}")
    
    return client, health

# Global API client instance
api_client = APIClient()