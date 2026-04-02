"""
Prometheus metrics for the Stroke Prediction API.

Metrics tracked:
  - Request counts and latencies by endpoint
  - Prediction counts by risk level (HIGH/MODERATE/LOW)
  - Batch size distributions
  - Model version info
  - Active requests (in-flight gauge)
"""

import time
from typing import Callable

from fastapi import Request, Response
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    Info,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.routing import Match

# ── Metric Definitions ───────────────────────────────────────

REQUEST_COUNT = Counter(
    "stroke_api_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status_code"],
)

REQUEST_LATENCY = Histogram(
    "stroke_api_request_duration_seconds",
    "HTTP request latency",
    ["method", "endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

ACTIVE_REQUESTS = Gauge(
    "stroke_api_active_requests",
    "Number of requests currently being processed",
)

PREDICTION_COUNT = Counter(
    "stroke_api_predictions_total",
    "Total predictions made",
    ["risk_level", "model_version"],
)

BATCH_SIZE = Histogram(
    "stroke_api_batch_size",
    "Batch size distribution",
    buckets=(1, 2, 5, 10, 25, 50, 100, 250, 500),
)

MODEL_INFO = Info(
    "stroke_api_model",
    "Currently loaded model metadata",
)


def set_model_info(classifier: str | None, path: str | None) -> None:
    """Call once after model loads to publish model metadata."""
    MODEL_INFO.info({
        "classifier": classifier or "unloaded",
        "model_path": path or "n/a",
        "version": classifier or "n/a",  # classifier name as version stand-in
    })


# ── Middleware ────────────────────────────────────────────────


class PrometheusMiddleware(BaseHTTPMiddleware):
    """
    Captures request count + latency for every route.

    Uses BaseHTTPMiddleware (not ASGI middleware) so it composes
    correctly with FastAPI's exception handler.
    """

    def __init__(self, app):
        super().__init__(app)
        self._route_path_cache: dict[int, str] = {}

    async def dispatch(
        self, request: Request, call_next: Callable
    ) -> Response:
        # Skip the /metrics endpoint itself to avoid recursion
        if request.url.path == "/metrics":
            return await call_next(request)

        # Resolve stable endpoint label (normalize path params)
        endpoint = self._resolve_endpoint(request)

        method = request.method
        ACTIVE_REQUESTS.inc()

        start = time.perf_counter()
        try:
            response = await call_next(request)
            status_code = str(response.status_code)
        finally:
            duration = time.perf_counter() - start
            ACTIVE_REQUESTS.dec()

        REQUEST_COUNT.labels(
            method=method, endpoint=endpoint, status_code=status_code
        ).inc()

        REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(duration)

        return response

    def _resolve_endpoint(self, request: Request) -> str:
        """Return the route template string (e.g. '/predict') not '/predict/123'."""
        scope = request.scope
        for route in self.app.routes:
            match, _ = route.matches(scope)
            if match == Match.FULL:
                # Store by route id to avoid repeated lookup
                route_id = id(route)
                self._route_path_cache[route_id] = route.path
                return route.path

        # Fallback: use the raw path (will include path params)
        return request.url.path


# ── Metrics Endpoint ─────────────────────────────────────────


def metrics_endpoint(request: Request) -> Response:
    """
    GET /metrics
    Returns Prometheus-formatted metrics.
    """
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )
