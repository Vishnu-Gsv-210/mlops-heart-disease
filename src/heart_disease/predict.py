"""Model loading and single-patient prediction helpers."""
import os
from pathlib import Path

import joblib
import pandas as pd

from src.heart_disease.config import FEATURES, MODEL_PATH, TARGET_MAPPING


def resolve_model_path(path: Path | str | None = None) -> Path:
    """Resolve the model location: explicit path, $MODEL_PATH, or the default."""
    return Path(path or os.getenv("MODEL_PATH", str(MODEL_PATH)))


def load_model(path: Path | str | None = None):
    """Load the serialized pipeline from an explicit path, $MODEL_PATH, or the default."""
    path = resolve_model_path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}. Run `python -m src.heart_disease.train` first."
        )
    return joblib.load(path)


def predict_single(features: dict, model=None) -> dict:
    """Predict for one patient feature dictionary.

    Returns the prediction, human-readable label, predicted-class confidence,
    and the positive-class probability.
    """
    if model is None:
        model = load_model()
    data = pd.DataFrame([features], columns=FEATURES)
    prediction = int(model.predict(data)[0])
    probability = float(model.predict_proba(data)[0][1])
    return {
        "prediction": prediction,
        "label": TARGET_MAPPING[prediction],
        "confidence": round(probability if prediction == 1 else 1 - probability, 6),
        "probability_heart_disease": round(probability, 6),
    }
