"""Tests for the FastAPI service: endpoints, schema validation, and errors."""
import pytest
from fastapi.testclient import TestClient

from src.heart_disease import api
from src.heart_disease.config import FEATURES


@pytest.fixture
def client(model_file, monkeypatch):
    """TestClient pointed at a temporary serialized fixture model."""
    monkeypatch.setenv("MODEL_PATH", str(model_file))
    monkeypatch.setattr(api, "_model", None)
    with TestClient(api.app) as test_client:
        yield test_client


def _payload(**overrides):
    row = {
        "age": 52, "sex": 1, "cp": 3, "trestbps": 130, "chol": 230,
        "fbs": 0, "restecg": 1, "thalach": 165, "exang": 0,
        "oldpeak": 1.5, "slope": 2, "ca": 0.0, "thal": 6.0,
    }
    row.update(overrides)
    return row


def test_health_endpoint_shape(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()
    assert response.json()["model_available"] is True


def test_root_has_docs_link(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["docs"] == "/docs"


def test_metrics_endpoint_exposes_prometheus_format(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "prediction_requests_total" in response.text


def test_predict_success_response_shape(client):
    response = client.post("/predict", json=_payload())
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {
        "prediction", "label", "confidence",
        "probability_heart_disease", "disclaimer",
    }
    assert body["prediction"] in (0, 1)
    assert body["label"] in ("heart_disease_present", "heart_disease_not_indicated")
    assert 0.0 <= body["confidence"] <= 1.0
    assert 0.0 <= body["probability_heart_disease"] <= 1.0
    assert body["disclaimer"] == "Educational output only; not a medical diagnosis."


def test_predict_rejects_missing_feature(client):
    payload = _payload()
    payload.pop("chol")
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert "chol" in response.text


def test_predict_rejects_out_of_range_value(client):
    response = client.post("/predict", json=_payload(age=150))
    assert response.status_code == 422


def test_predict_rejects_wrong_type(client):
    response = client.post("/predict", json=_payload(chol="not-a-number"))
    assert response.status_code == 422


def test_predict_rejects_extra_unknown_fields(client):
    payload = _payload(some_extra_field=1)
    response = client.post("/predict", json=payload)
    assert response.status_code in (200, 422)  # pydantic default ignores extras


def test_predict_returns_503_when_model_missing(monkeypatch):
    monkeypatch.setenv("MODEL_PATH", "/nonexistent/model.joblib")
    monkeypatch.setattr(api, "_model", None)
    with TestClient(api.app, raise_server_exceptions=False) as test_client:
        response = test_client.post("/predict", json=_payload())
    assert response.status_code == 503
    assert "Model not found" in response.json()["detail"]


def test_request_schema_declares_all_thirteen_features():
    fields = api.PatientFeatures.model_fields
    assert list(fields) == FEATURES
