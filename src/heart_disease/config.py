"""Shared configuration for the MLOps Heart Disease project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
ARTIFACTS_DIR = REPORTS_DIR / "artifacts"
MODEL_PATH = MODELS_DIR / "heart_disease_pipeline.joblib"
METRICS_PATH = ARTIFACTS_DIR / "test_metrics.json"
MLFLOW_DB_URI = f"sqlite:///{ROOT / 'mlflow.db'}"
MLFLOW_ARTIFACT_ROOT = ROOT / "mlruns"

SEED = 42
TEST_SIZE = 0.20
N_FOLDS = 5

FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]
TARGET = "target"
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = [
    "sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"
]
TARGET_MAPPING = {0: "heart_disease_not_indicated", 1: "heart_disease_present"}
