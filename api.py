"""
FastAPI REST API for Stroke Prediction Model
Provides programmatic access to stroke risk predictions with security and monitoring
"""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import pandas as pd
import joblib
from pydantic import BaseModel, Field, field_validator
from fastapi import FastAPI, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from utils import load_config, resolve_path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

API_VERSION = "1.0.1"
DEFAULT_CONFIG = {
    'paths': {'models_dir': 'models', 'model_file': 'best_stroke_model.joblib'},
    'validation': {
        'age_min': 0, 'age_max': 120,
        'bmi_min': 10.0, 'bmi_max': 100.0,
        'glucose_min': 0.0, 'glucose_max': 300.0
    },
    'api': {'host': '127.0.0.1', 'port': 8000, 'reload': True}
}

VALID_GENDERS = {'Male', 'Female'}
VALID_BINARY_VALUES = {0, 1}
VALID_MARRIAGE_STATUS = {'Yes', 'No'}
VALID_WORK_TYPES = {'Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'}
VALID_RESIDENCE_TYPES = {'Urban', 'Rural'}
VALID_SMOKING_STATUS = {'never smoked', 'formerly smoked', 'smokes', 'Unknown'}


def get_model_path() -> Path:
    """Resolve the configured model artifact path."""
    paths_config = CONFIG.get('paths', {})
    model_dir = paths_config.get('models_dir', 'models')
    model_file = paths_config.get('model_file', 'best_stroke_model.joblib')
    return resolve_path(Path(model_dir) / model_file)


# Load configuration
try:
    CONFIG = load_config('config.yaml')
except FileNotFoundError:
    logger.warning("config.yaml not found, using built-in defaults")
    CONFIG = DEFAULT_CONFIG

# ═══════════════════════════════════════════════════════════
# PYDANTIC MODELS
# ═══════════════════════════════════════════════════════════

class PatientData(BaseModel):
    """Patient input data for stroke prediction."""
    gender: str = Field(..., description="Patient gender: 'Male' or 'Female'")
    age: float = Field(..., description="Patient age in years")
    hypertension: int = Field(..., description="Hypertension status: 0 or 1")
    heart_disease: int = Field(..., description="Heart disease status: 0 or 1")
    ever_married: str = Field(..., description="Marriage status: 'Yes' or 'No'")
    work_type: str = Field(..., description="Work type: Private, Self-employed, Govt_job, children, Never_worked")
    Residence_type: str = Field(..., description="Residence type: 'Urban' or 'Rural'")
    avg_glucose_level: float = Field(..., description="Average glucose level in mg/dL")
    bmi: float = Field(..., description="Body Mass Index")
    smoking_status: str = Field(..., description="Smoking status: never smoked, formerly smoked, smokes, Unknown")

    @field_validator('gender')
    @classmethod
    def validate_gender(cls, value: str) -> str:
        if value not in VALID_GENDERS:
            raise ValueError("Gender must be 'Male' or 'Female'")
        return value

    @field_validator('age')
    @classmethod
    def validate_age(cls, value: float) -> float:
        min_age = CONFIG['validation']['age_min']
        max_age = CONFIG['validation']['age_max']
        if not (min_age <= value <= max_age):
            raise ValueError(f"Age must be between {min_age} and {max_age}")
        return value

    @field_validator('bmi')
    @classmethod
    def validate_bmi(cls, value: float) -> float:
        min_bmi = CONFIG['validation']['bmi_min']
        max_bmi = CONFIG['validation']['bmi_max']
        if not (min_bmi <= value <= max_bmi):
            raise ValueError(f"BMI must be between {min_bmi} and {max_bmi}")
        return value

    @field_validator('avg_glucose_level')
    @classmethod
    def validate_glucose(cls, value: float) -> float:
        min_glucose = CONFIG['validation']['glucose_min']
        max_glucose = CONFIG['validation']['glucose_max']
        if not (min_glucose <= value <= max_glucose):
            raise ValueError(f"Glucose must be between {min_glucose} and {max_glucose}")
        return value

    @field_validator('hypertension', 'heart_disease')
    @classmethod
    def validate_binary_fields(cls, value: int) -> int:
        if value not in VALID_BINARY_VALUES:
            raise ValueError("Binary clinical fields must be 0 or 1")
        return value

    @field_validator('ever_married')
    @classmethod
    def validate_marriage_status(cls, value: str) -> str:
        if value not in VALID_MARRIAGE_STATUS:
            raise ValueError("Marriage status must be 'Yes' or 'No'")
        return value

    @field_validator('work_type')
    @classmethod
    def validate_work_type(cls, value: str) -> str:
        if value not in VALID_WORK_TYPES:
            raise ValueError(
                "Work type must be one of: Private, Self-employed, Govt_job, children, Never_worked"
            )
        return value

    @field_validator('Residence_type')
    @classmethod
    def validate_residence_type(cls, value: str) -> str:
        if value not in VALID_RESIDENCE_TYPES:
            raise ValueError("Residence type must be 'Urban' or 'Rural'")
        return value

    @field_validator('smoking_status')
    @classmethod
    def validate_smoking_status(cls, value: str) -> str:
        if value not in VALID_SMOKING_STATUS:
            raise ValueError(
                "Smoking status must be one of: never smoked, formerly smoked, smokes, Unknown"
            )
        return value


class PredictionResponse(BaseModel):
    """API response for stroke prediction."""
    patient_id: Optional[str] = None
    prediction: int = Field(..., description="0 = No stroke, 1 = Stroke")
    probability: float = Field(..., description="Probability of stroke (0-1)")
    risk_level: str = Field(..., description="Risk level: LOW, MODERATE, HIGH")
    confidence: float = Field(..., description="Model confidence score (0-1)")
    timestamp: str = Field(..., description="Prediction timestamp in ISO format")
    recommendations: List[str] = Field(..., description="Clinical recommendations")


class HealthCheckResponse(BaseModel):
    """API health check response."""
    status: str = Field(..., description="API status")
    version: str = Field(..., description="API version")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    timestamp: str = Field(..., description="Timestamp in ISO format")


class BatchPredictionItem(BaseModel):
    """Prediction result for a single patient inside a batch request."""
    patient_index: int
    prediction: int
    probability: float
    confidence: float
    risk_level: str


class BatchPredictionResponse(BaseModel):
    """API response for batch prediction requests."""
    total_patients: int
    predictions: List[BatchPredictionItem]
    timestamp: str


def get_risk_assessment(probability: float) -> tuple[str, List[str]]:
    """Map a probability score to a risk band and recommendations."""
    if probability > 0.4:
        return "HIGH", [
            "Immediate medical consultation recommended",
            "Schedule comprehensive health screening",
            "Monitor blood pressure regularly",
            "Consider preventive medications"
        ]

    if probability > 0.15:
        return "MODERATE", [
            "Regular health monitoring advised",
            "Maintain healthy lifestyle habits",
            "Schedule routine check-ups",
            "Monitor glucose levels regularly"
        ]

    return "LOW", [
        "Continue healthy lifestyle",
        "Regular exercise recommended",
        "Maintain balanced diet",
        "Annual health check-ups"
    ]


def run_model_inference(patient_data: Dict[str, Any]) -> tuple[int, float, float]:
    """
    Run prediction and probability scoring for a single patient.
    
    Args:
        patient_data: Dictionary containing patient features
        
    Returns:
        Tuple of (prediction, probability, confidence)
        
    Raises:
        RuntimeError: If model is not loaded
        ValueError: If input data is invalid
    """
    if model is None:
        raise RuntimeError("Model not loaded")

    # Secondary validation (defense in depth)
    age = patient_data.get('age', 0)
    if not (0 <= age <= 120):
        raise ValueError(f"Invalid age: {age}")
        
    bmi = patient_data.get('bmi', 0)
    if not (10 <= bmi <= 100):
        raise ValueError(f"Invalid BMI: {bmi}")

    patient_df = pd.DataFrame([patient_data])
    probabilities = model.predict_proba(patient_df)[0]
    prediction = int(model.predict(patient_df)[0])
    probability = float(probabilities[1])
    confidence = float(max(probabilities))
    return prediction, probability, confidence


def run_batch_inference(patients_data: List[Dict[str, Any]]) -> List[Tuple[int, float, float]]:
    """
    Optimized batch inference for multiple patients.
    
    Args:
        patients_data: List of patient feature dictionaries
        
    Returns:
        List of tuples containing (prediction, probability, confidence)
        
    Raises:
        RuntimeError: If model is not loaded
    """
    if model is None:
        raise RuntimeError("Model not loaded")
    
    # Create DataFrame from all patients (vectorized operation)
    patients_df = pd.DataFrame(patients_data)
    
    # Batch prediction (much faster than individual predictions)
    predictions = model.predict(patients_df)
    probabilities = model.predict_proba(patients_df)
    
    results = []
    for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
        prediction = int(pred)
        probability = float(probs[1])
        confidence = float(max(probs))
        results.append((prediction, probability, confidence))
    
    return results


# ═══════════════════════════════════════════════════════════
# FASTAPI APP INITIALIZATION
# ═══════════════════════════════════════════════════════════

model = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load and release application resources during API lifecycle."""
    global model
    try:
        model_path = get_model_path()
        if model_path.exists():
            model = joblib.load(model_path)
            logger.info(f"Model loaded from {model_path}")
        else:
            logger.warning(f"Model file not found at {model_path}")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
    try:
        yield
    finally:
        logger.info("API shutting down")


app = FastAPI(
    title="Stroke Prediction API",
    description="REST API for stroke risk prediction",
    version=API_VERSION,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan
)

# CORS configuration - restrict to specific origins
CORS_ORIGINS = os.getenv(
    "CORS_ORIGINS", 
    "http://localhost:8501,http://localhost:3000,http://127.0.0.1:8501"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,  # Restricted origins for security
    allow_credentials=False,
    allow_methods=["GET", "POST"],  # Only allow necessary methods
    allow_headers=["Content-Type", "Accept"],  # Only allow necessary headers
)


# ═══════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════

@app.get("/health", response_model=HealthCheckResponse, tags=["Health"])
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint.

    Returns:
        HealthCheckResponse with API status
    """
    return {
        "status": "healthy" if model is not None else "degraded",
        "version": API_VERSION,
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict_stroke(patient: PatientData) -> Dict[str, Any]:
    """
    Predict stroke risk for a patient.

    Args:
        patient: Patient data

    Returns:
        PredictionResponse with prediction and recommendations

    Raises:
        HTTPException: If model is not loaded or prediction fails
    """
    if model is None:
        logger.error("Model not loaded")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please try again later."
        )

    try:
        # Convert input to DataFrame
        patient_dict = patient.model_dump()
        prediction, probability, confidence = run_model_inference(patient_dict)
        risk_level, recommendations = get_risk_assessment(probability)

        logger.info(f"Prediction made: {probability:.2%} stroke probability, {risk_level} risk")

        return {
            "prediction": int(prediction),
            "probability": float(probability),
            "risk_level": risk_level,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat(),
            "recommendations": recommendations
        }

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Validation error: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error during prediction. Please try again later."
        )


@app.post("/batch-predict", response_model=BatchPredictionResponse, tags=["Predictions"])
async def batch_predict(patients: List[PatientData]) -> Dict[str, Any]:
    """
    Batch predict stroke risk for multiple patients (optimized vectorized inference).

    Args:
        patients: List of patient data

    Returns:
        Dictionary with predictions for all patients

    Raises:
        HTTPException: If model is not loaded or prediction fails
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )

    try:
        # Convert all patients to dictionaries
        patients_data = [patient.model_dump() for patient in patients]
        
        # Optimized batch inference (vectorized operation)
        batch_results = run_batch_inference(patients_data)
        
        # Build response
        predictions = []
        for idx, (prediction, probability, confidence) in enumerate(batch_results):
            risk_level, _ = get_risk_assessment(probability)
            predictions.append({
                "patient_index": idx,
                "prediction": int(prediction),
                "probability": float(probability),
                "confidence": confidence,
                "risk_level": risk_level,
            })

        logger.info(f"Batch prediction completed for {len(patients)} patients")
        return {
            "total_patients": len(patients),
            "predictions": predictions,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error during batch prediction"
        )


@app.get("/info", tags=["Info"])
async def api_info() -> Dict[str, Any]:
    """
    Get API information.

    Returns:
        Dictionary with API metadata
    """
    return {
        "name": "Stroke Prediction REST API",
        "version": API_VERSION,
        "description": "REST API for stroke risk assessment using machine learning",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "batch_predict": "/batch-predict",
            "info": "/info",
            "docs": "/api/docs"
        }
    }


if __name__ == "__main__":
    import uvicorn
    api_config = CONFIG.get('api', {})
    uvicorn.run(
        app,
        host=api_config.get('host', '127.0.0.1'),
        port=api_config.get('port', 8000),
        reload=api_config.get('reload', True)
    )
