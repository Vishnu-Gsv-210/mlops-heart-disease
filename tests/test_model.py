"""Tests for model training, prediction outputs, and serialization."""
import joblib
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from src.heart_disease.config import FEATURES, N_FOLDS, SEED, TARGET, TARGET_MAPPING
from src.heart_disease.predict import predict_single
from src.heart_disease.preprocessing import build_preprocessor
from src.heart_disease.train import make_candidates


def test_training_fits_and_predicts_binary_labels(synthetic_df, trained_pipeline):
    preds = trained_pipeline.predict(synthetic_df[FEATURES])
    assert set(preds).issubset({0, 1})
    assert len(preds) == len(synthetic_df)


def test_grid_search_selects_from_declared_grid(synthetic_df):
    name, (estimator, grid) = next(iter(make_candidates().items()))
    assert name == "logistic_regression"
    pipe = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", estimator),
    ])
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    search = GridSearchCV(pipe, grid, cv=cv, scoring="roc_auc", refit=True)
    search.fit(synthetic_df[FEATURES], synthetic_df[TARGET])
    assert set(search.best_params_) <= set(grid)
    assert search.best_estimator_.predict(synthetic_df[FEATURES][:2]).shape == (2,)


def test_predict_proba_shape_and_range(synthetic_df, trained_pipeline):
    probs = trained_pipeline.predict_proba(synthetic_df[FEATURES])
    assert probs.shape == (len(synthetic_df), 2)
    assert np.all(probs >= 0) and np.all(probs <= 1)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_predict_single_output_contract(trained_pipeline, patient_row):
    result = predict_single(patient_row, model=trained_pipeline)
    assert set(result) == {"prediction", "label", "confidence", "probability_heart_disease"}
    assert result["prediction"] in (0, 1)
    assert result["label"] == TARGET_MAPPING[result["prediction"]]
    probability = result["probability_heart_disease"]
    assert 0.0 <= probability <= 1.0
    expected_confidence = round(
        probability if result["prediction"] == 1 else 1 - probability, 6
    )
    assert result["confidence"] == expected_confidence


def test_predict_single_respects_feature_order(trained_pipeline, patient_row):
    shuffled = dict(reversed(list(patient_row.items())))
    assert predict_single(patient_row, model=trained_pipeline) == \
        predict_single(shuffled, model=trained_pipeline)


def test_serialization_reload_produces_identical_predictions(
    tmp_path, trained_pipeline, synthetic_df
):
    path = tmp_path / "pipeline.joblib"
    joblib.dump(trained_pipeline, path)
    reloaded = joblib.load(path)
    original = trained_pipeline.predict(synthetic_df[FEATURES])
    restored = reloaded.predict(synthetic_df[FEATURES])
    assert (original == restored).all()
    assert np.allclose(
        trained_pipeline.predict_proba(synthetic_df[FEATURES]),
        reloaded.predict_proba(synthetic_df[FEATURES]),
    )


def test_predict_single_with_reloaded_model_file(model_file, patient_row):
    model = joblib.load(model_file)
    result = predict_single(patient_row, model=model)
    assert set(result) == {"prediction", "label", "confidence", "probability_heart_disease"}


def test_declared_candidate_grids_are_unchanged():
    candidates = make_candidates()
    assert set(candidates) == {"logistic_regression", "random_forest"}
    assert candidates["logistic_regression"][1] == {"model__C": [0.1, 1.0, 10.0]}
    assert candidates["random_forest"][1] == {
        "model__n_estimators": [150, 300],
        "model__max_depth": [None, 5, 10],
        "model__min_samples_leaf": [1, 3],
    }
    assert N_FOLDS == 5 and SEED == 42


def test_select_best_model_prefers_higher_roc_auc():
    from src.heart_disease.evaluate import select_best_model

    candidates = [
        (0.90, "a", object(), {"roc_auc": 0.90}),
        (0.97, "b", object(), {"roc_auc": 0.97}),
        (0.93, "c", object(), {"roc_auc": 0.93}),
    ]
    winner_name, _, winner_metrics = select_best_model(candidates)
    assert winner_name == "b"
    assert winner_metrics["roc_auc"] == 0.97


def test_compute_test_metrics_keys():
    from src.heart_disease.evaluate import compute_test_metrics

    class FixedModel:
        def predict(self, X):
            return np.ones(len(X))

        def predict_proba(self, X):
            return np.array([[0.4, 0.6]] * len(X))

    y_test = np.array([1, 0, 1, 0])
    metrics = compute_test_metrics(FixedModel(), np.zeros((4, 1)), y_test)
    assert set(metrics) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert metrics["roc_auc"] == pytest.approx(roc_auc_score(y_test, [0.6, 0.6, 0.6, 0.6]))
