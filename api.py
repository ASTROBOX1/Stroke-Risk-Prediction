"""
FastAPI REST API for Stroke Prediction Model
Provides programmatic access to stroke risk predictions with security and monitoring.
"""

import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from sklearn.base import clone as sk_clone

from utils import load_config, resolve_path

# ── Structured Logger ──────────────────────────────────────
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("api")

# ── Constants ───────────────────────────────────────────────
API_VERSION = "1.1.0"
API_TITLE = "Stroke Prediction API"
MAX_BATCH_SIZE = 500  # Prevent OOM on huge batches

DEFAULT_CONFIG: dict[str, Any] = {
    "paths": {
        "models_dir": "models",
        "model_file": "best_stroke_model.joblib",
    },
    "validation": {
        "age_min": 0,
        "age_max": 120,
        "bmi_min": 10.0,
        "bmi_max": 100.0,
        "glucose_min": 0.0,
        "glucose_max": 300.0,
    },
    "api": {"host": "127.0.0.1", "port": 8000, "reload": True},
}

# ── Validation Sets ─────────────────────────────────────────
_VALID_GENDERS = {"Male", "Female"}
_VALID_BINARY = {0, 1}
_VALID_MARRIAGE = {"Yes", "No"}
_VALID_WORK_TYPES = {"Private", "Self-employed", "Govt_job", "children", "Never_worked"}
_VALID_RESIDENCE_TYPES = {"Urban", "Rural"}
_VALID_SMOKING_STATUS = {"never smoked", "formerly smoked", "smokes", "Unknown"}

# ── CORS Validation ─────────────────────────────────────────
_cors_env = os.getenv("CORS_ORIGINS", "").strip()
if not _cors_env:
    raise RuntimeError(
        "CORS_ORIGINS environment variable must be set. "
        "Got empty string. Configure it to comma-separated origins, e.g.: "
        "http://localhost:8501,http://localhost:3000"
    )
CORS_ORIGINS = [origin.strip() for origin in _cors_env.split(",") if origin.strip()]

# ── Load Configuration ──────────────────────────────────────
try:
    CONFIG = load_config("config.yaml")
except FileNotFoundError:
    logger.warning("config.yaml not found — using built-in defaults")
    CONFIG = DEFAULT_CONFIG

VALIDATION = CONFIG.get("validation", DEFAULT_CONFIG["validation"])

# ═══════════════════════════════════════════════════════════
# PROMETHEUS METRICS (must be imported before app creation)
# ═══════════════════════════════════════════════════════════
from metrics import (
    PrometheusMiddleware,
    metrics_endpoint,
    PREDICTION_COUNT,
    set_model_info,
    BATCH_SIZE,
)

# ═══════════════════════════════════════════════════════════
# REQUEST ID MIDDLEWARE
# ═══════════════════════════════════════════════════════════


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Attach a unique request ID to every request for audit tracing."""
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = request_id

    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


# ═══════════════════════════════════════════════════════════
# PYDANTIC MODELS
# ═══════════════════════════════════════════════════════════


class PatientData(BaseModel):
    """Patient input data for stroke risk prediction."""

    gender: str = Field(..., description="Patient gender: 'Male' or 'Female'")
    age: float = Field(..., ge=0, le=120, description="Patient age in years")
    hypertension: int = Field(..., description="Hypertension status: 0 or 1")
    heart_disease: int = Field(..., description="Heart disease status: 0 or 1")
    ever_married: str = Field(..., description="Marriage status: 'Yes' or 'No'")
    work_type: str = Field(
        ...,
        description="Work type: Private, Self-employed, Govt_job, children, Never_worked",
    )
    Residence_type: str = Field(..., description="Residence type: 'Urban' or 'Rural'")
    avg_glucose_level: float = Field(
        ..., ge=0, le=500, description="Average glucose level in mg/dL"
    )
    bmi: float = Field(..., ge=5, le=150, description="Body Mass Index")
    smoking_status: str = Field(
        ...,
        description="Smoking status: never smoked, formerly smoked, smokes, Unknown",
    )
    patient_id: str | None = Field(
        None, description="Optional patient identifier for audit trails"
    )

    @field_validator("gender")
    @classmethod
    def _validate_gender(cls, v: str) -> str:
        if v not in _VALID_GENDERS:
            raise ValueError(f"Gender must be one of {_VALID_GENDERS}")
        return v

    @field_validator("hypertension", "heart_disease")
    @classmethod
    def _validate_binary(cls, v: int) -> int:
        if v not in _VALID_BINARY:
            raise ValueError(f"Value must be 0 or 1, got {v}")
        return v

    @field_validator("ever_married")
    @classmethod
    def _validate_marriage(cls, v: str) -> str:
        if v not in _VALID_MARRIAGE:
            raise ValueError(f"ever_married must be one of {_VALID_MARRIAGE}")
        return v

    @field_validator("work_type")
    @classmethod
    def _validate_work_type(cls, v: str) -> str:
        if v not in _VALID_WORK_TYPES:
            raise ValueError(f"work_type must be one of {_VALID_WORK_TYPES}")
        return v

    @field_validator("Residence_type")
    @classmethod
    def _validate_residence(cls, v: str) -> str:
        if v not in _VALID_RESIDENCE_TYPES:
            raise ValueError(f"Residence_type must be one of {_VALID_RESIDENCE_TYPES}")
        return v

    @field_validator("smoking_status")
    @classmethod
    def _validate_smoking(cls, v: str) -> str:
        if v not in _VALID_SMOKING_STATUS:
            raise ValueError(f"smoking_status must be one of {_VALID_SMOKING_STATUS}")
        return v


class PredictionResponse(BaseModel):
    """Structured response for a single stroke risk prediction."""

    request_id: str
    patient_id: str | None
    prediction: int = Field(..., description="0 = No stroke, 1 = Stroke")
    probability: float = Field(..., ge=0, le=1, description="Probability of stroke")
    risk_level: str = Field(..., description="Risk level: LOW, MODERATE, HIGH")
    confidence: float = Field(..., ge=0, le=1, description="Model confidence score")
    model_version: str
    timestamp: str
    recommendations: list[str]


class BatchPredictionItem(BaseModel):
    """Prediction result for a single patient in a batch."""

    patient_index: int
    patient_id: str | None
    prediction: int
    probability: float
    confidence: float
    risk_level: str


class BatchPredictionResponse(BaseModel):
    """Structured response for batch stroke risk predictions."""

    request_id: str
    total_patients: int
    high_risk_count: int
    model_version: str
    timestamp: str
    predictions: list[BatchPredictionItem]


class HealthResponse(BaseModel):
    """API health check response with model metadata."""

    status: str
    version: str
    model_loaded: bool
    model_path: str | None
    request_id: str


class ErrorResponse(BaseModel):
    """Standardized error response format."""

    request_id: str
    error: str
    detail: str | None
    timestamp: str


# ═══════════════════════════════════════════════════════════
# RISK ASSESSMENT
# ═══════════════════════════════════════════════════════════

RISK_THRESHOLDS: dict[str, float] = {
    "high": float(os.getenv("RISK_THRESHOLD_HIGH", "0.4")),
    "moderate": float(os.getenv("RISK_THRESHOLD_MODERATE", "0.15")),
}

_HIGH_RECOMMENDATIONS = [
    "Immediate medical consultation recommended",
    "Schedule comprehensive health screening within 48–72 hours",
    "Monitor blood pressure regularly at home",
    "Consider preventive medication review with physician",
]

_MODERATE_RECOMMENDATIONS = [
    "Schedule routine check-up with primary care provider",
    "Maintain healthy lifestyle: balanced diet and regular exercise",
    "Monitor glucose levels regularly",
    "Review cardiovascular risk factors with healthcare provider",
]

_LOW_RECOMMENDATIONS = [
    "Continue healthy lifestyle habits",
    "Annual preventive health check-ups recommended",
    "Maintain regular physical activity",
    "Balanced diet rich in vegetables and whole grains",
]


def get_risk_assessment(probability: float) -> tuple[str, list[str]]:
    """Map a probability score to a risk band and evidence-based recommendations."""
    if probability > RISK_THRESHOLDS["high"]:
        return "HIGH", _HIGH_RECOMMENDATIONS
    if probability > RISK_THRESHOLDS["moderate"]:
        return "MODERATE", _MODERATE_RECOMMENDATIONS
    return "LOW", _LOW_RECOMMENDATIONS


# ═══════════════════════════════════════════════════════════
# INFERENCE ENGINE
# ═══════════════════════════════════════════════════════════

_model: Any = None
_model_path: Path | None = None
_model_metadata: dict[str, Any] = {}


def _get_model_path() -> Path:
    paths_cfg = CONFIG.get("paths", DEFAULT_CONFIG["paths"])
    return resolve_path(Path(paths_cfg["models_dir"]) / paths_cfg["model_file"])


def _load_model() -> bool:
    """Load model from disk and extract metadata. Returns True on success."""
    global _model, _model_path, _model_metadata

    path = _get_model_path()
    _model_path = path

    if not path.exists():
        logger.error(f"Model file not found: {path}")
        _model_metadata = {"loaded": False, "path": str(path), "error": "File not found"}
        set_model_info(None, str(path))
        return False

    try:
        _model = joblib.load(path)

        step_names = list(_model.named_steps.keys())
        classifier_name = next(
            (s for s in step_names if s != "preprocessor" and s != "smote"), "unknown"
        )
        _model_metadata = {
            "loaded": True,
            "path": str(path),
            "classifier": classifier_name,
            "pipeline_steps": step_names,
        }
        set_model_info(classifier_name, str(path))
        logger.info(f"Model loaded: {classifier_name} from {path}")
        return True

    except Exception as exc:
        logger.error(f"Failed to load model from {path}: {exc}")
        _model_metadata = {"loaded": False, "path": str(path), "error": str(exc)}
        set_model_info(None, str(path))
        return False


def _run_inference_sync(patients_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Run synchronous batch inference against the loaded model.
    Uses a cloned pipeline so fit state is never modified.
    """
    global _model

    if _model is None:
        raise RuntimeError("Model not loaded")

    model_clone = sk_clone(_model)
    predictions = model_clone.predict(patients_df)
    probabilities = model_clone.predict_proba(patients_df)

    return predictions, probabilities


def _build_batch_results(
    patients: list[PatientData],
    predictions: np.ndarray,
    probabilities: np.ndarray,
    request_id: str,
) -> BatchPredictionResponse:
    """
    Vectorized construction of batch response.
    No per-patient Python loops in the hot path.
    """
    n = len(patients)
    probs_arr = np.asarray(probabilities)

    stroke_probs = probs_arr[:, 1]
    confidences = probs_arr.max(axis=1)

    high_mask = stroke_probs > RISK_THRESHOLDS["high"]
    mod_mask = ~high_mask & (stroke_probs > RISK_THRESHOLDS["moderate"])
    risk_levels = np.where(high_mask, "HIGH", np.where(mod_mask, "MODERATE", "LOW"))

    items: list[BatchPredictionItem] = [
        BatchPredictionItem(
            patient_index=i,
            patient_id=patients[i].patient_id,
            prediction=int(predictions[i]),
            probability=float(stroke_probs[i]),
            confidence=float(confidences[i]),
            risk_level=str(risk_levels[i]),
        )
        for i in range(n)
    ]

    return BatchPredictionResponse(
        request_id=request_id,
        total_patients=n,
        high_risk_count=int(high_mask.sum()),
        model_version=_model_metadata.get("classifier", API_VERSION),
        timestamp=datetime.now(timezone.utc).isoformat(),
        predictions=items,
    )


# ═══════════════════════════════════════════════════════════
# A/B TESTING
# ═══════════════════════════════════════════════════════════
from ab_testing import ModelRegistry, ModelVariant, get_registry

_ab_registry: ModelRegistry | None = None


def _init_ab_testing() -> None:
    """Register model variants for A/B testing if configured."""
    global _ab_registry

    if not os.getenv("AB_TESTING_ENABLED", "").lower() in ("1", "true", "yes"):
        return

    registry = get_registry()
    model_dir = resolve_path(Path(CONFIG.get("paths", {}).get("models_dir", "models")))

    # Load variant A (primary model — weight from env or 80)
    primary_weight = int(os.getenv("AB_PRIMARY_WEIGHT", "80"))
    variant_a_path = model_dir / CONFIG.get("paths", {}).get("model_file", "best_stroke_model.joblib")

    if variant_a_path.exists():
        model_a = joblib.load(variant_a_path)
        registry.register(
            name="primary",
            model=model_a,
            weight=primary_weight,
            description="Primary production model",
            metadata={"path": str(variant_a_path)},
        )
        logger.info(f"A/B variant A registered: primary weight={primary_weight}%")

    # Load variant B if model file B is configured
    variant_b_path = os.getenv("AB_VARIANT_B_PATH", "").strip()
    if variant_b_path:
        variant_b_weight = int(os.getenv("AB_VARIANT_B_WEIGHT", str(100 - primary_weight)))
        if variant_b_weight > 0:
            registry.register(
                name="variant_b",
                model=joblib.load(resolve_path(variant_b_path)),
                weight=variant_b_weight,
                description=" challenger model for A/B test",
                metadata={"path": variant_b_path},
            )
            logger.info(f"A/B variant B registered: weight={variant_b_weight}%")

    _ab_registry = registry


# ═══════════════════════════════════════════════════════════
# MODEL MONITORING (Evidently)
# ═══════════════════════════════════════════════════════════
from monitoring import DriftMonitor, init_monitor, get_monitor

_monitor: DriftMonitor | None = None


def _init_monitoring() -> None:
    """Initialize Evidently drift monitor with reference data if available."""
    global _monitor

    if not os.getenv("DRIFT_MONITORING_ENABLED", "").lower() in ("1", "true", "yes"):
        logger.info("Drift monitoring is disabled (DRIFT_MONITORING_ENABLED != true)")
        return

    try:
        from monitoring import FEATURE_COLUMNS, COLUMN_MAPPING

        data_path = resolve_path(CONFIG.get("paths", {}).get("data", "data/healthcare-dataset-stroke-data.csv"))
        if not data_path.exists():
            logger.warning(f"Reference data not found at {data_path} — drift monitoring disabled")
            return

        df = pd.read_csv(data_path, low_memory=False)

        # Preprocess to match training pipeline
        if "id" in df.columns:
            df = df.drop("id", axis=1)
        df = df[df["gender"] != "Other"]
        df["bmi"] = df["bmi"].fillna(df["bmi"].median())

        # Build feature set matching what the model sees
        ref_df = df[FEATURE_COLUMNS].copy()

        _monitor = init_monitor(
            reference_df=ref_df,
            model_version=_model_metadata.get("classifier", "unknown"),
        )
        logger.info(f"Drift monitoring initialized with {len(ref_df)} reference samples")

    except Exception as exc:
        logger.warning(f"Failed to initialize drift monitoring: {exc}")


# ═══════════════════════════════════════════════════════════
# APP LIFECYCLE
# ═══════════════════════════════════════════════════════════


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup; log shutdown on teardown."""
    logger.info("API starting up...")
    _load_model()
    _init_ab_testing()
    _init_monitoring()
    yield
    logger.info("API shutting down")


# ── FastAPI App ────────────────────────────────────────────
app = FastAPI(
    title=API_TITLE,
    description="REST API for stroke risk prediction using ensemble ML models.",
    version=API_VERSION,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan,
)

# Prometheus middleware must be added BEFORE any routes
app.add_middleware(PrometheusMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Accept", "X-Request-ID"],
)


# ═══════════════════════════════════════════════════════════
# EXCEPTION HANDLER
# ═══════════════════════════════════════════════════════════


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all — ensures no raw traceback leaks to clients."""
    request_id = getattr(request.state, "request_id", "unknown")
    logger.exception(f"[{request_id}] Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "request_id": request_id,
            "error": "InternalServerError",
            "detail": "An unexpected error occurred. Please try again later.",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )


# ═══════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check(request: Request) -> dict[str, Any]:
    """Liveness probe — checks API and model availability."""
    request_id = getattr(request.state, "request_id", "unknown")
    model_loaded = _model_metadata.get("loaded", False)

    return {
        "status": "healthy" if model_loaded else "degraded",
        "version": API_VERSION,
        "model_loaded": model_loaded,
        "model_path": _model_metadata.get("path"),
        "request_id": request_id,
    }


@app.get("/ready", tags=["Health"])
async def readiness_check(request: Request) -> Response:
    """Readiness probe — returns 200 only if model is loaded."""
    request_id = getattr(request.state, "request_id", "unknown")

    if not _model_metadata.get("loaded"):
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "request_id": request_id,
                "ready": False,
                "reason": "Model not loaded",
            },
        )

    return Response(
        content=f'{{"request_id":"{request_id}","ready":true}}',
        media_type="application/json",
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict_stroke(request: Request, patient: PatientData) -> dict[str, Any]:
    """
    Predict stroke risk for a single patient.
    Runs inference in a thread pool to avoid blocking the async event loop.
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))

    if not _model_metadata.get("loaded"):
        logger.warning(f"[{request_id}] Prediction attempted but model not loaded")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please try again later.",
        )

    try:
        import asyncio
        from concurrent.futures import ThreadPoolExecutor

        patient_dict = patient.model_dump()
        loop = asyncio.get_event_loop()

        with ThreadPoolExecutor(max_workers=4) as executor:
            predictions, probabilities = await loop.run_in_executor(
                executor,
                lambda: _run_inference_sync(pd.DataFrame([patient_dict])),
            )

        pred = int(predictions[0])
        probability = float(probabilities[0, 1])
        confidence = float(probabilities[0].max())
        risk_level, recommendations = get_risk_assessment(probability)
        model_version = _model_metadata.get("classifier", API_VERSION)

        logger.info(
            f"[{request_id}] prediction=stroke:{pred} prob={probability:.4f} "
            f"risk={risk_level} patient_id={patient.patient_id} model={model_version}"
        )

        # Prometheus: record prediction
        PREDICTION_COUNT.labels(risk_level=risk_level, model_version=model_version).inc()

        # Drift monitoring: record input for drift detection
        if _monitor is not None:
            should_check = _monitor.record_prediction(patient_dict)
            if should_check:
                _monitor.check_drift()

        # A/B: record assignment if active
        if _ab_registry is not None:
            _ab_registry.record_prediction(
                variant_name=model_version,
                patient_id=patient.patient_id,
                probability=probability,
                risk_level=risk_level,
            )

        return {
            "request_id": request_id,
            "patient_id": patient.patient_id,
            "prediction": pred,
            "probability": probability,
            "risk_level": risk_level,
            "confidence": confidence,
            "model_version": model_version,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "recommendations": recommendations,
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception(f"[{request_id}] Prediction error: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error during prediction. Please try again later.",
        )


@app.post("/batch-predict", response_model=BatchPredictionResponse, tags=["Predictions"])
async def batch_predict(request: Request, patients: list[PatientData]) -> dict[str, Any]:
    """
    Batch stroke risk prediction for multiple patients.
    Vectorized numpy operations, single thread pool call.
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))

    if not _model_metadata.get("loaded"):
        logger.warning(f"[{request_id}] Batch prediction attempted but model not loaded")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded.",
        )

    n = len(patients)
    if n == 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Request body must contain at least 1 patient record.",
        )
    if n > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Batch size {n} exceeds maximum allowed ({MAX_BATCH_SIZE}).",
        )

    try:
        import asyncio
        from concurrent.futures import ThreadPoolExecutor

        patients_data = [p.model_dump() for p in patients]
        patients_df = pd.DataFrame(patients_data)

        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=4) as executor:
            predictions, probabilities = await loop.run_in_executor(
                executor, lambda: _run_inference_sync(patients_df)
            )

        response = _build_batch_results(patients, predictions, probabilities, request_id)
        model_version = _model_metadata.get("classifier", API_VERSION)

        # Prometheus: record batch size and per-risk-level counts
        BATCH_SIZE.observe(n)
        probs_arr = np.asarray(probabilities)
        stroke_probs = probs_arr[:, 1]
        for level in ["LOW", "MODERATE", "HIGH"]:
            mask = (
                (stroke_probs > RISK_THRESHOLDS["high"]) if level == "HIGH"
                else ((stroke_probs > RISK_THRESHOLDS["moderate"]) & (stroke_probs <= RISK_THRESHOLDS["high"])) if level == "MODERATE"
                else (stroke_probs <= RISK_THRESHOLDS["moderate"])
            )
            count = int(mask.sum())
            if count > 0:
                PREDICTION_COUNT.labels(risk_level=level, model_version=model_version).inc(count)

        logger.info(
            f"[{request_id}] batch_size={n} "
            f"high_risk={response.high_risk_count} "
            f"model={model_version}"
        )

        return response.model_dump()

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception(f"[{request_id}] Batch prediction error: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error during batch prediction.",
        )


@app.get("/info", tags=["Info"])
async def api_info(request: Request) -> dict[str, Any]:
    """API metadata and available endpoints."""
    request_id = getattr(request.state, "request_id", "unknown")
    return {
        "name": API_TITLE,
        "version": API_VERSION,
        "model_version": _model_metadata.get("classifier"),
        "model_loaded": _model_metadata.get("loaded"),
        "cors_origins": CORS_ORIGINS,
        "max_batch_size": MAX_BATCH_SIZE,
        "risk_thresholds": RISK_THRESHOLDS,
        "ab_testing": {
            "enabled": _ab_registry is not None,
            "variants": _ab_registry.traffic_split if _ab_registry else {},
        },
        "drift_monitoring": {
            "enabled": _monitor is not None and _monitor.is_available,
            "window_size": _monitor.window_size if _monitor else 0,
            "total_predictions": _monitor.total_predictions if _monitor else 0,
        },
        "endpoints": {
            "health": "/health",
            "ready": "/ready",
            "predict": "/predict",
            "batch_predict": "/batch-predict",
            "info": "/info",
            "metrics": "/metrics",
            "monitoring": "/monitoring/drift",
            "docs": "/api/docs",
        },
        "request_id": request_id,
    }


# ── Prometheus Metrics Endpoint ───────────────────────────────
app.add_route("/metrics", metrics_endpoint)


# ── Evidently Monitoring Endpoints ───────────────────────────


@app.get("/monitoring/drift", tags=["Monitoring"])
async def get_drift_report(request: Request) -> JSONResponse:
    """
    Returns the latest Evidently drift report if monitoring is active.
    """
    request_id = getattr(request.state, "request_id", "unknown")

    if _monitor is None:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "request_id": request_id,
                "enabled": False,
                "detail": "Drift monitoring is not enabled. "
                           "Set DRIFT_MONITORING_ENABLED=true to activate.",
            },
        )

    report = _monitor.check_drift()

    if report is None:
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "request_id": request_id,
                "enabled": True,
                "window_populated": _monitor.window_size > 0,
                "total_predictions": _monitor.total_predictions,
                "detail": "Not enough data in window to produce a report yet.",
            },
        )

    return JSONResponse(content={"request_id": request_id, **report.to_dict()})


@app.get("/monitoring/status", tags=["Monitoring"])
async def get_monitoring_status(request: Request) -> dict[str, Any]:
    """Lightweight monitoring health check."""
    request_id = getattr(request.state, "request_id", "unknown")

    return {
        "request_id": request_id,
        "drift_monitoring": {
            "enabled": _monitor is not None and _monitor.is_available,
            "window_size": _monitor.window_size if _monitor else 0,
            "total_predictions_seen": _monitor.total_predictions if _monitor else 0,
        },
        "ab_testing": {
            "enabled": _ab_registry is not None,
            "variants": _ab_registry.traffic_split if _ab_registry else {},
        },
    }


# ── A/B Testing Endpoints ─────────────────────────────────────


@app.get("/ab/stats", tags=["A/B Testing"])
async def get_ab_stats(request: Request) -> dict[str, Any]:
    """Return A/B test registration status and variant stats."""
    request_id = getattr(request.state, "request_id", "unknown")

    if _ab_registry is None:
        return {
            "request_id": request_id,
            "enabled": False,
            "detail": "A/B testing is not enabled. Set AB_TESTING_ENABLED=true to activate.",
        }

    return {
        "request_id": request_id,
        "enabled": True,
        **_ab_registry.get_stats(),
    }


# ═══════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════


if __name__ == "__main__":
    import uvicorn

    api_cfg = CONFIG.get("api", DEFAULT_CONFIG["api"])
    uvicorn.run(
        "api:app",
        host=api_cfg.get("host", "127.0.0.1"),
        port=api_cfg.get("port", 8000),
        reload=api_cfg.get("reload", False),
    )
