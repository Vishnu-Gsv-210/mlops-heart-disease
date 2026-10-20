"""Shared fixtures: synthetic dataset, fast trained pipeline, and model files."""
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.heart_disease.config import FEATURES, TARGET
from src.heart_disease.preprocessing import build_preprocessor

N_ROWS = 24


def make_synthetic_rows(n_rows: int = N_ROWS) -> list[dict]:
    """Deterministic 13-feature rows with a balanced raw UCI `num` column."""
    rows = []
    for i in range(n_rows):
        rows.append({
            "age": 40 + (i % 20),
            "sex": i % 2,
            "cp": 1 + (i % 4),
            "trestbps": 120 + (i % 15),
            "chol": 180 + (i % 40),
            "fbs": i % 2,
            "restecg": i % 3,
            "thalach": 170 - (i % 40),
            "exang": i % 2,
            "oldpeak": (i % 50) / 10.0,
            "slope": 1 + (i % 3),
            "ca": float(i % 4),
            "thal": float([3, 6, 7][i % 3]),
            "num": [0, 0, 1, 1][i % 4],
        })
    return rows


@pytest.fixture
def raw_df() -> pd.DataFrame:
    """Uncleaned UCI-style frame: raw `num` target plus '?'-style missing values."""
    df = pd.DataFrame(make_synthetic_rows())
    df.loc[0, "ca"] = np.nan  # the UCI '?' convention after na_values handling
    df.loc[1, "thal"] = np.nan
    return df


@pytest.fixture
def synthetic_df(raw_df) -> pd.DataFrame:
    """Cleaned frame with the binary target, ready for training."""
    df = raw_df.copy()
    df[TARGET] = (df["num"] > 0).astype(int)
    return df.drop(columns=["num"]).reset_index(drop=True)


@pytest.fixture
def patient_row() -> dict:
    """One valid 13-feature payload for prediction tests."""
    row = make_synthetic_rows(3)[2]
    row.pop("num")
    return row


@pytest.fixture
def trained_pipeline(synthetic_df):
    """A fast logistic-regression pipeline fitted on the synthetic data."""
    pipe = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", LogisticRegression(max_iter=2000, random_state=42)),
    ])
    pipe.fit(synthetic_df[FEATURES], synthetic_df[TARGET])
    return pipe


@pytest.fixture
def model_file(tmp_path, trained_pipeline):
    """The fitted pipeline serialized to a temporary joblib file."""
    path = tmp_path / "heart_disease_pipeline.joblib"
    joblib.dump(trained_pipeline, path)
    return path
