"""FastAPI prediction service with structured request logs and Prometheus metrics."""
import json
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from src.heart_disease import monitoring, predict

logger = monitoring.setup_logging()
REQUEST_COUNT = monitoring.REQUEST_COUNT
REQUEST_LATENCY = monitoring.REQUEST_LATENCY

app = FastAPI(
    title="Heart Disease Classification API",
    description="Educational MLOps demo. Not a medical device; do not use for diagnosis.",
    version="1.0.0",
)
_model = None


class PatientFeatures(BaseModel):
    age: float = Field(..., ge=1, le=120)
    sex: int = Field(..., ge=0, le=1)
    cp: int = Field(..., ge=1, le=4)
    trestbps: float = Field(..., ge=40, le=300)
    chol: float = Field(..., ge=50, le=800)
    fbs: int = Field(..., ge=0, le=1)
    restecg: int = Field(..., ge=0, le=2)
    thalach: float = Field(..., ge=30, le=250)
    exang: int = Field(..., ge=0, le=1)
    oldpeak: float = Field(..., ge=-5, le=15)
    slope: int = Field(..., ge=1, le=3)
    ca: float = Field(..., ge=0, le=3)
    thal: float = Field(..., ge=3, le=7)


def get_model():
    global _model
    if _model is None:
        _model = predict.load_model()
    return _model


@app.get("/")
def root():
    return {"service": "heart-disease-classification", "docs": "/docs",
            "health": "/health", "metrics": "/metrics"}


@app.get("/health")
def health():
    model_path = predict.resolve_model_path()
    return {"status": "ok", "model_available": model_path.exists()}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict")
def predict_endpoint(payload: PatientFeatures, request: Request):
    start = time.perf_counter()
    try:
        model = get_model()
        result = predict.predict_single(payload.model_dump(), model=model)
        result["disclaimer"] = "Educational output only; not a medical diagnosis."
        REQUEST_COUNT.labels(status="success").inc()
        logger.info(json.dumps({
            "event": "prediction", "status": "success",
            "client": request.client.host if request.client else "unknown",
            "prediction": result["prediction"],
            "duration_ms": round((time.perf_counter() - start) * 1000, 2)
        }))
        return result
    except FileNotFoundError as exc:
        REQUEST_COUNT.labels(status="error").inc()
        logger.exception("Model unavailable")
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception:
        REQUEST_COUNT.labels(status="error").inc()
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed; inspect server logs.")
    finally:
        REQUEST_LATENCY.observe(time.perf_counter() - start)
