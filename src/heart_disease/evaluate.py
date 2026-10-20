"""Test-set evaluation, diagnostic plots, metrics artifacts, selection, and model card."""
import json
from pathlib import Path

import matplotlib
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.heart_disease.config import ARTIFACTS_DIR, FEATURES, FIGURES_DIR, MODEL_PATH


def compute_test_metrics(model, X_test, y_test) -> dict:
    """Evaluate a fitted pipeline on the held-out test set."""
    probs = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "f1": float(f1_score(y_test, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probs)),
    }


def write_classification_report(name: str, y_test, preds) -> Path:
    """Persist the per-model classification report and return its path."""
    report = classification_report(y_test, preds, output_dict=True, zero_division=0)
    report_path = ARTIFACTS_DIR / f"{name}_classification_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    return report_path


def select_best_model(fitted_candidates: list[tuple]) -> tuple:
    """Select by held-out ROC-AUC from ``(roc_auc, name, model, metrics)`` entries.

    The test set is used once for model selection in this teaching task.
    For strict unbiased final estimation, reserve a second untouched holdout set.
    """
    ranked = sorted(fitted_candidates, key=lambda x: x[0], reverse=True)
    return ranked[0][1], ranked[0][2], ranked[0][3]


def plot_diagnostics(model, name: str, X_test, y_test) -> None:
    """Render the confusion matrix and ROC curve for the selected model."""
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap="Blues")
    plt.title(f"Confusion Matrix — {name}")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=180, bbox_inches="tight")
    plt.close("all")
    RocCurveDisplay.from_predictions(y_test, y_prob)
    plt.title(f"ROC Curve — {name}")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "roc_curve.png", dpi=180, bbox_inches="tight")
    plt.close("all")


def write_model_card(name: str, metrics: dict, best_params: dict) -> Path:
    """Record the selected model's card: order, target, metrics, limits."""
    model_card = {
        "model_name": name,
        "model_path": str(MODEL_PATH.relative_to(MODEL_PATH.parents[1])),
        "feature_order": FEATURES,
        "target": "binary heart disease indicator (UCI num > 0)",
        "metrics": metrics,
        "best_params": best_params,
        "limitations": [
            "Small historical Cleveland subset; not representative of all populations.",
            "Educational demonstration only; not for diagnosis or clinical decisions.",
            "Metrics are internal evaluation results, not evidence of clinical utility."
        ],
    }
    path = ARTIFACTS_DIR / "model_card.json"
    path.write_text(json.dumps(model_card, indent=2))
    return path
