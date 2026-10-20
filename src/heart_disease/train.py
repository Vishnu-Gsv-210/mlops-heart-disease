"""Train, cross-validate, compare, track, and serialize classification pipelines."""
import argparse
import json

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from src.heart_disease import evaluate
from src.heart_disease.config import (
    ARTIFACTS_DIR,
    FEATURES,
    FIGURES_DIR,
    MLFLOW_ARTIFACT_ROOT,
    MLFLOW_DB_URI,
    MODEL_PATH,
    N_FOLDS,
    SEED,
    TARGET,
    TEST_SIZE,
)
from src.heart_disease.data import load_raw
from src.heart_disease.preprocessing import build_preprocessor

SCORING = {"accuracy": "accuracy", "precision": "precision", "recall": "recall",
           "f1": "f1", "roc_auc": "roc_auc"}


def make_candidates():
    return {
        "logistic_regression": (
            LogisticRegression(max_iter=3000, class_weight="balanced",
                               random_state=SEED),
            {"model__C": [0.1, 1.0, 10.0]}
        ),
        "random_forest": (
            RandomForestClassifier(random_state=SEED, class_weight="balanced"),
            {"model__n_estimators": [150, 300],
             "model__max_depth": [None, 5, 10],
             "model__min_samples_leaf": [1, 3]}
        ),
    }


def _prepare_experiment(experiment_name: str) -> None:
    """Point MLflow at the sqlite backend with artifacts under ./mlruns."""
    mlflow.set_tracking_uri(MLFLOW_DB_URI)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        artifact_location = (MLFLOW_ARTIFACT_ROOT / experiment_name).as_uri()
        mlflow.create_experiment(experiment_name, artifact_location=artifact_location)
    mlflow.set_experiment(experiment_name)


def train_models(experiment_name: str = "heart-disease-classification") -> dict:
    df = load_raw()
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    _prepare_experiment(experiment_name)
    results = {}
    fitted_candidates = []

    for name, (estimator, grid) in make_candidates().items():
        pipe = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("model", estimator),
        ])
        search = GridSearchCV(pipe, grid, cv=cv, scoring="roc_auc",
                              n_jobs=-1, refit=True, error_score="raise")
        with mlflow.start_run(run_name=name):
            mlflow.log_param("model_family", name)
            mlflow.log_param("test_size", TEST_SIZE)
            mlflow.log_param("random_state", SEED)
            mlflow.log_param("cv_folds", N_FOLDS)
            cv_result = cross_validate(pipe, X_train, y_train, cv=cv,
                                       scoring=SCORING, n_jobs=-1)
            for metric in SCORING:
                mlflow.log_metric(f"cv_{metric}_mean", float(np.mean(cv_result[f"test_{metric}"])))
                mlflow.log_metric(f"cv_{metric}_std", float(np.std(cv_result[f"test_{metric}"])))
            search.fit(X_train, y_train)
            model = search.best_estimator_
            metrics = evaluate.compute_test_metrics(model, X_test, y_test)
            metrics["cv_roc_auc_mean"] = float(np.mean(cv_result["test_roc_auc"]))
            metrics["cv_roc_auc_std"] = float(np.std(cv_result["test_roc_auc"]))
            mlflow.log_params({k: str(v) for k, v in search.best_params_.items()})
            mlflow.log_metrics(metrics)
            # MLflow >= 3.x audits serialized sklearn models with skops. numpy.dtype
            # is required by every sklearn pipeline, and sklearn.tree._tree.Tree is
            # the internal node storage of tree ensembles; both are trusted here
            # because the model is trained locally from the validated dataset.
            mlflow.sklearn.log_model(model, artifact_path="model",
                                     skops_trusted_types=["numpy.dtype",
                                                          "sklearn.tree._tree.Tree"])
            report_path = evaluate.write_classification_report(
                name, y_test, model.predict(X_test)
            )
            mlflow.log_artifact(str(report_path))
            results[name] = {"best_params": search.best_params_, **metrics}
            fitted_candidates.append((metrics["roc_auc"], name, model, metrics))
            print(f"{name}: {json.dumps(results[name], indent=2)}")

    # Select by held-out ROC-AUC; the test set is used once for model selection here.
    winner_name, winner, winner_metrics = evaluate.select_best_model(fitted_candidates)
    joblib.dump(winner, MODEL_PATH)
    evaluate.write_model_card(winner_name, winner_metrics, results[winner_name]["best_params"])
    (ARTIFACTS_DIR / "comparison.json").write_text(json.dumps(results, indent=2))
    evaluate.plot_diagnostics(winner, winner_name, X_test, y_test)

    output = {"selected_model": winner_name, "models": results, "test_size": len(y_test)}
    (ARTIFACTS_DIR / "training_summary.json").write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name", default="heart-disease-classification")
    args = parser.parse_args()
    train_models(args.experiment_name)
