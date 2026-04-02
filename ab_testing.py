"""
A/B Testing infrastructure for the Stroke Prediction API.

Structure:
  - Multiple model variants are registered with traffic weights.
  - Incoming requests are routed to a variant based on weighted random assignment.
  - Assignment is sticky — same patient_id always gets the same variant.
  - Results are tracked in Prometheus metrics for statistical analysis.

Usage:
    ab_registry = ModelRegistry()
    ab_registry.register("model_v1", model_v1_pipeline, weight=80)  # 80%
    ab_registry.register("model_v2", model_v2_pipeline, weight=20)  # 20%

    variant, model = ab_registry.get_variant(patient_id="patient_123", patient_data={...})
    ab_registry.record_outcome(variant, patient_id, prediction, actual_outcome=None)
"""

import hashlib
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

from sklearn.base import BaseEstimator

from metrics import PREDICTION_COUNT

logger = logging.getLogger("ab_testing")

# ── Variant Registry ─────────────────────────────────────────


@dataclass
class ModelVariant:
    """A single model variant registered in the A/B test."""

    name: str
    model: Any  # Fitted sklearn pipeline
    weight: int  # Traffic percentage (0-100)
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def weight_float(self) -> float:
        return self.weight / 100.0


class ModelRegistry:
    """
    Weighted model registry for A/B testing.

    Assignment uses deterministic hashing (patient_id → float in [0,1))
    so that the same patient always hits the same variant — no session drift.
    """

    def __init__(self):
        self._variants: list[ModelVariant] = []
        self._weights: list[float] = []  # Cumulative normalized weights
        self._lock = threading.RLock()
        self._total_weight: int = 0
        self._assignments: dict[str, str] = {}  # patient_id → variant_name (sticky)
        self._assignment_lock = threading.Lock()

    def register(
        self,
        name: str,
        model: Any,
        weight: int,
        description: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Register a model variant with a traffic weight (percentage).

        Args:
            name: Unique identifier for this variant (e.g. 'logistic_regression_v2')
            model: Fitted sklearn-compatible pipeline
            weight: Integer weight 0-100 representing % of traffic
            description: Human-readable description
            metadata: Arbitrary extra info (e.g. training date, AUC)
        """
        if weight < 0 or weight > 100:
            raise ValueError(f"Weight must be 0-100, got {weight}")
        if any(v.name == name for v in self._variants):
            raise ValueError(f"Variant '{name}' is already registered")

        with self._lock:
            self._variants.append(
                ModelVariant(
                    name=name,
                    model=model,
                    weight=weight,
                    description=description,
                    metadata=metadata or {},
                )
            )
            self._recompute_weights()

        logger.info(f"A/B variant registered: {name} weight={weight}%")

    def _recompute_weights(self) -> None:
        """Rebuild cumulative weight boundaries after registration."""
        total = sum(v.weight for v in self._variants)
        self._total_weight = total

        if total == 0:
            self._weights = []
            return

        cumulative = 0.0
        self._weights = []
        for variant in self._variants:
            cumulative += variant.weight_float
            self._weights.append(cumulative)

    def get_variant(
        self,
        patient_id: str | None = None,
        patient_data: dict[str, Any] | None = None,
    ) -> tuple[ModelVariant, Any]:
        """
        Resolve which variant to use for a given request.

        Uses deterministic hashing on patient_id for sticky assignment.
        Falls back to data-based hashing if no patient_id provided.

        Returns:
            Tuple of (ModelVariant, model_instance)

        Raises:
            RuntimeError: If no variants are registered
        """
        if not self._variants:
            raise RuntimeError("No model variants registered for A/B test")

        # Deterministic variant selection
        if patient_id:
            hash_input = f"{patient_id}:{self._total_weight}".encode()
        elif patient_data:
            # Sort keys for deterministic hashing
            encoded = str(sorted(patient_data.items())).encode()
            hash_input = f"{encoded.hex()}:{self._total_weight}".encode()
        else:
            # Fallback: random assignment
            import random
            rand = random.random()
            variant_idx = self._select_by_cumulative(rand)
            return self._variants[variant_idx], self._variants[variant_idx].model

        hash_value = int(hashlib.sha256(hash_input).hexdigest(), 16)
        normalized = (hash_value % 10_000) / 10_000.0  # [0, 1)

        variant_idx = self._select_by_cumulative(normalized)

        # Track sticky assignment
        if patient_id:
            with self._assignment_lock:
                self._assignments[patient_id] = self._variants[variant_idx].name

        return self._variants[variant_idx], self._variants[variant_idx].model

    def _select_by_cumulative(self, normalized_value: float) -> int:
        """Binary-search-style selection from cumulative weights."""
        for i, boundary in enumerate(self._weights):
            if normalized_value <= boundary:
                return i
        return len(self._weights) - 1

    def record_prediction(
        self,
        variant_name: str,
        patient_id: str | None,
        probability: float,
        risk_level: str,
    ) -> None:
        """
        Record a prediction outcome for statistical tracking.

        In production this would write to a metrics store;
        here we emit Prometheus counters with variant labels.
        """
        PREDICTION_COUNT.labels(
            risk_level=risk_level,
            model_version=variant_name,
        ).inc()

        logger.debug(
            f"A/B recorded: variant={variant_name} "
            f"patient_id={patient_id} prob={probability:.4f}"
        )

    @property
    def variants(self) -> list[ModelVariant]:
        with self._lock:
            return list(self._variants)

    @property
    def traffic_split(self) -> dict[str, int]:
        """Return {variant_name: weight_percent} for all registered variants."""
        return {v.name: v.weight for v in self._variants}

    def get_stats(self) -> dict[str, Any]:
        """Return per-variant aggregate statistics."""
        return {
            "variants": [
                {
                    "name": v.name,
                    "weight": v.weight,
                    "description": v.description,
                    "metadata": v.metadata,
                }
                for v in self._variants
            ],
            "total_registered": len(self._variants),
            "total_weight": self._total_weight,
            "sticky_assignments": len(self._assignments),
        }


# ── Module-level singleton ────────────────────────────────────

_registry = ModelRegistry()


def get_registry() -> ModelRegistry:
    return _registry
