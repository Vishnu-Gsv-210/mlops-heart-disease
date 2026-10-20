"""Prometheus metrics and structured logging setup for the prediction API."""
import logging
import os

from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter("prediction_requests_total", "Total prediction requests", ["status"])
REQUEST_LATENCY = Histogram(
    "prediction_request_duration_seconds", "Prediction request duration"
)


def setup_logging(level: str | None = None) -> logging.Logger:
    """Configure root logging once and return the API logger."""
    logging.basicConfig(
        level=level or os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    return logging.getLogger("heart-disease-api")
