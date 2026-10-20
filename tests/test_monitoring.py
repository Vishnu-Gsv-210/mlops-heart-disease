"""Tests for Prometheus metrics and logging setup."""
from prometheus_client import generate_latest

from src.heart_disease.monitoring import REQUEST_COUNT, REQUEST_LATENCY, setup_logging

METRIC_TOTAL = "prediction_requests_total"
METRIC_LATENCY = "prediction_request_duration_seconds"


def _exported_value(target: str) -> float | None:
    for line in generate_latest().decode().splitlines():
        if line.startswith(target + " ") or line.startswith(target + "\t"):
            return float(line.rsplit(" ", 1)[1])
    return None


def test_metric_names_are_exported():
    text = generate_latest().decode()
    assert METRIC_TOTAL in text
    assert METRIC_LATENCY in text


def test_request_counter_has_status_labels():
    text = generate_latest().decode()
    assert '# TYPE prediction_requests_total counter' in text
    assert 'status=' in text


def test_counter_increments_on_observe():
    before = _exported_value(f'{METRIC_TOTAL}{{status="test_probe"}}') or 0.0
    REQUEST_COUNT.labels(status="test_probe").inc()
    after = _exported_value(f'{METRIC_TOTAL}{{status="test_probe"}}')
    assert after == before + 1


def test_histogram_count_increments_on_observe():
    before = _exported_value(f"{METRIC_LATENCY}_count") or 0.0
    REQUEST_LATENCY.observe(0.25)
    after = _exported_value(f"{METRIC_LATENCY}_count")
    assert after == before + 1


def test_setup_logging_returns_named_logger():
    logger = setup_logging("INFO")
    assert logger.name == "heart-disease-api"
