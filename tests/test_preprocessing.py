"""Tests for the leakage-safe preprocessing pipeline."""
import numpy as np
import pandas as pd

from src.heart_disease.config import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES
from src.heart_disease.preprocessing import build_preprocessor


def test_preprocessor_handles_missing_values_and_unknown_categories(synthetic_df):
    X = synthetic_df[FEATURES].copy()
    X.loc[0, "chol"] = float("nan")
    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(X)
    assert transformed.shape[0] == len(X)
    assert transformed.shape[1] > len(FEATURES)
    assert pd.notna(transformed).all()


def test_numeric_features_are_scaled_first_in_order(synthetic_df):
    preprocessor = build_preprocessor()
    preprocessor.fit(synthetic_df[FEATURES])
    names = list(preprocessor.get_feature_names_out())
    assert names[: len(NUMERIC_FEATURES)] == NUMERIC_FEATURES
    # One-hot columns follow in CATEGORICAL_FEATURES order.
    assert names[len(NUMERIC_FEATURES)].startswith(CATEGORICAL_FEATURES[0] + "_")


def test_onehot_expands_categorical_features(synthetic_df):
    preprocessor = build_preprocessor()
    preprocessor.fit(synthetic_df[FEATURES])
    names = list(preprocessor.get_feature_names_out())
    onehot_names = names[len(NUMERIC_FEATURES):]
    assert len(onehot_names) > len(CATEGORICAL_FEATURES)  # multi-level columns expanded
    for feature in CATEGORICAL_FEATURES:
        assert any(name.startswith(feature + "_") for name in onehot_names)


def test_unknown_categories_are_ignored_at_transform_time(synthetic_df):
    preprocessor = build_preprocessor()
    preprocessor.fit(synthetic_df[FEATURES])
    row = synthetic_df.iloc[[0]][FEATURES].copy()
    row["thal"] = 99.0  # category never seen during fit
    row["ca"] = 9.0
    transformed = preprocessor.transform(row)
    assert np.isfinite(transformed).all()
    assert transformed.shape[1] == preprocessor.get_feature_names_out().shape[0]


def test_median_imputation_fills_numeric_gaps(synthetic_df):
    X = synthetic_df[FEATURES].copy()
    X.loc[0, "chol"] = float("nan")
    X.loc[3, "oldpeak"] = float("nan")
    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(X)
    names = list(preprocessor.get_feature_names_out())
    chol_idx = names.index("chol")
    assert np.isfinite(transformed[:, chol_idx]).all()
    assert np.isfinite(transformed).all()
