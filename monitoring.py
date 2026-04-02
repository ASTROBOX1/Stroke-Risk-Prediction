"""
Model monitoring with Evidently — tracks data drift and prediction drift.

Architecture:
  - On API startup, a "reference snapshot" is captured from the training data.
  - Every N predictions, a "current window" is evaluated against the reference.
  - Drift results are exposed via /monitoring/drift endpoint.

Usage:
  monitor = DriftMonitor(reference_data=df_train, column_mapping=column_mapping)
  monitor.check_drift(current_data=recent_predictions_df)
"""

import hashlib
import logging
import os
import threading
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger("monitoring")

# Suppress Evidently warnings that are non-actionable
warnings.filterwarnings("ignore", category=UserWarning, module="evidently")


# ── Column Mapping ───────────────────────────────────────────
# Tells Evidently which column is the target and which are features.

FEATURE_COLUMNS = [
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "ever_married",
    "work_type",
    "Residence_type",
    "avg_glucose_level",
    "bmi",
    "smoking_status",
]

COLUMN_MAPPING = {
    "target": "stroke",
    "numerical_features": ["age", "hypertension", "heart_disease", "avg_glucose_level", "bmi"],
    "categorical_features": [
        "gender",
        "ever_married",
        "work_type",
        "Residence_type",
        "smoking_status",
    ],
}

# Default drift detection settings (can be overridden via env vars)
DRIFT_CHECK_INTERVAL = int(os.getenv("DRIFT_CHECK_INTERVAL", "100"))
DRIFT_WINDOW_SIZE = int(os.getenv("DRIFT_WINDOW_SIZE", "500"))
DRIFT_ALERT_THRESHOLD = float(os.getenv("DRIFT_ALERT_THRESHOLD", "0.5"))


# ── Drift Report Result ──────────────────────────────────────


@dataclass
class DriftReport:
    """Lightweight snapshot of current drift state."""

    timestamp: str
    window_size: int
    total_predictions_processed: int
    data_drift_detected: bool
    data_drift_score: float  # 0.0 = identical, 1.0 = completely different
    target_drift_detected: bool
    target_drift_score: float
    n_features_drifted: int
    drifted_features: list[str]
    alert_level: str  # "OK" | "WARNING" | "CRITICAL"
    model_version: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "window_size": self.window_size,
            "total_predictions_processed": self.total_predictions_processed,
            "drift_detected": self.data_drift_detected,
            "data_drift_score": round(self.data_drift_score, 4),
            "target_drift_detected": self.target_drift_detected,
            "target_drift_score": round(self.target_drift_score, 4),
            "n_features_drifted": self.n_features_drifted,
            "drifted_features": self.drifted_features,
            "alert_level": self.alert_level,
            "model_version": self.model_version,
        }


# ── Prediction Window Buffer ─────────────────────────────────


class PredictionWindowBuffer:
    """
    Thread-safe circular buffer that collects prediction inputs
    for drift evaluation against the reference dataset.
    """

    def __init__(self, capacity: int = DRIFT_WINDOW_SIZE):
        self.capacity = capacity
        self._buffer: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._total: int = 0

    def push(self, patient_data: dict[str, Any]) -> None:
        with self._lock:
            if len(self._buffer) >= self.capacity:
                self._buffer.pop(0)
            self._buffer.append(patient_data)
            self._total += 1

    def get_dataframe(self) -> pd.DataFrame | None:
        with self._lock:
            if not self._buffer:
                return None
            return pd.DataFrame(self._buffer)

    def size(self) -> int:
        with self._lock:
            return len(self._buffer)

    def total_seen(self) -> int:
        with self._lock:
            return self._total


# ── Drift Monitor ────────────────────────────────────────────


class DriftMonitor:
    """
    Wraps Evidently Report to detect data and prediction drift.

    Usage:
        monitor = DriftMonitor(reference_df=train_df)
        monitor.check_drift(current_df=recent_df)  # returns DriftReport
    """

    def __init__(
        self,
        reference_df: pd.DataFrame | None = None,
        model_version: str = "unknown",
    ):
        self.model_version = model_version
        self._reference_df = reference_df
        self._report: Any = None  # Evidently Report — lazily imported
        self._available = self._check_evidently()

        if not self._available:
            logger.warning(
                "Evidently not available — drift detection is disabled. "
                "Install with: pip install evidently"
            )

        self._window_buffer = PredictionWindowBuffer(capacity=DRIFT_WINDOW_SIZE)
        self._last_check_total: int = 0

    def _check_evidently(self) -> bool:
        try:
            from evidently.dashboard import Dashboard
            from evidently.tabs import DataDriftTab, TargetDriftTab
            return True
        except ImportError:
            return False

    def set_reference(self, reference_df: pd.DataFrame) -> None:
        """Set the reference dataset for drift comparison."""
        self._reference_df = reference_df.copy()
        logger.info(
            f"Reference dataset set: {len(reference_df)} rows, "
            f"columns: {list(reference_df.columns)}"
        )

    def record_prediction(self, patient_data: dict[str, Any]) -> bool:
        """
        Add a prediction to the rolling window.
        Returns True if a drift check should be triggered.
        """
        self._window_buffer.push(patient_data)

        # Trigger check every DRIFT_CHECK_INTERVAL new predictions
        should_check = (
            self._window_buffer.total_seen() - self._last_check_total
            >= DRIFT_CHECK_INTERVAL
        )
        return should_check

    def check_drift(self) -> DriftReport | None:
        """
        Run Evidently drift report against the current window vs reference.
        Returns None if reference data is not set or window is empty.
        """
        if not self._available:
            return None

        if self._reference_df is None:
            logger.warning("No reference dataset set — skipping drift check")
            return None

        current_df = self._window_buffer.get_dataframe()
        if current_df is None or current_df.empty:
            logger.info("Prediction window empty — skipping drift check")
            return None

        self._last_check_total = self._window_buffer.total_seen()

        try:
            from evidently.dashboard import Dashboard
            from evidently.tabs import DataDriftTab, TargetDriftTab

            # Build Evidently report
            report = Dashboard(
                tabs=[DataDriftTab(), TargetDriftTab()],
                options=[
                    ("data_drift", {"threshold": DRIFT_ALERT_THRESHOLD}),
                    ("target_drift", {"threshold": DRIFT_ALERT_THRESHOLD}),
                ],
            )

            report.run(
                reference_data=self._reference_df,
                current_data=current_df,
                column_mapping=COLUMN_MAPPING,
            )

            # Extract simple metrics from report
            # Evidently stores results as JSON in report object
            result = report.as_dict() if hasattr(report, "as_dict") else {}

            # Parse Evidently output
            data_drift_score = result.get("metrics", {}).get(
                "data_drift", {}
            ).get("score", 0.0)
            data_drift_detected = bool(
                result.get("metrics", {})
                .get("data_drift", {})
                .get("data_drift_detected", False)
            )
            target_drift_score = result.get("metrics", {}).get(
                "target_drift", {}
            ).get("score", 0.0)
            target_drift_detected = bool(
                result.get("metrics", {})
                .get("target_drift", {})
                .get("target_drift_detected", False)
            )

            # Identify drifted features
            drifted_features = []
            drift_by_column = (
                result.get("metrics", {})
                .get("data_drift", {})
                .get("drift_by_columns", {})
            )
            if isinstance(drift_by_column, dict):
                drifted_features = [
                    col
                    for col, meta in drift_by_column.items()
                    if isinstance(meta, dict) and meta.get("drifted", False)
                ]

            # Alert level
            if data_drift_detected and target_drift_detected:
                alert_level = "CRITICAL"
            elif data_drift_detected or target_drift_detected:
                alert_level = "WARNING"
            else:
                alert_level = "OK"

            report_obj = DriftReport(
                timestamp=datetime.now(timezone.utc).isoformat(),
                window_size=len(current_df),
                total_predictions_processed=self._window_buffer.total_seen(),
                data_drift_detected=data_drift_detected,
                data_drift_score=float(data_drift_score),
                target_drift_detected=target_drift_detected,
                target_drift_score=float(target_drift_score),
                n_features_drifted=len(drifted_features),
                drifted_features=drifted_features,
                alert_level=alert_level,
                model_version=self.model_version,
            )

            logger.info(
                f"Drift check complete: alert={alert_level} "
                f"data_drift_score={data_drift_score:.3f} "
                f"n_features_drifted={len(drifted_features)}"
            )
            return report_obj

        except Exception as exc:
            logger.error(f"Drift check failed: {exc}")
            return None

    @property
    def is_available(self) -> bool:
        return self._available

    @property
    def window_size(self) -> int:
        return self._window_buffer.size()

    @property
    def total_predictions(self) -> int:
        return self._window_buffer.total_seen()


# ── Module-level singleton ───────────────────────────────────

_monitor: DriftMonitor | None = None


def get_monitor() -> DriftMonitor | None:
    return _monitor


def init_monitor(reference_df: pd.DataFrame | None, model_version: str) -> DriftMonitor:
    global _monitor
    _monitor = DriftMonitor(reference_df=reference_df, model_version=model_version)
    return _monitor
