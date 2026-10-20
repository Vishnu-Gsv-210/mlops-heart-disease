"""Generate a PDF report from actual artifacts and available screenshots.

Run from repository root: python scripts/generate_report.py
This generator does not invent missing metrics or screenshots; it lists missing evidence.
"""
import json
import logging
from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "reports" / "artifacts"
FIG = ROOT / "reports" / "figures"
SHOTS = ROOT / "screenshots"
OUT = ROOT / "report_generated.pdf"

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="CoverTitle2", parent=styles["Title"], fontSize=22, leading=27,
                          alignment=1, textColor=colors.HexColor("#17365D"), spaceAfter=14))
styles.add(ParagraphStyle(name="Section2", parent=styles["Heading1"], fontSize=15, leading=19,
                          textColor=colors.HexColor("#17365D"), spaceBefore=5, spaceAfter=8))
styles.add(ParagraphStyle(name="Body2", parent=styles["BodyText"], fontSize=9, leading=13, spaceAfter=6))
styles.add(ParagraphStyle(name="Small2", parent=styles["BodyText"], fontSize=7.8, leading=10, spaceAfter=4))
styles.add(ParagraphStyle(name="Code2", parent=styles["Code"], fontSize=7.5, leading=9,
                          backColor=colors.HexColor("#F2F4F7"), borderPadding=5))
styles.add(ParagraphStyle(name="Notice2", parent=styles["BodyText"], fontSize=9, leading=13,
                          backColor=colors.HexColor("#FFF7E6"), borderPadding=7, spaceAfter=8))


def p(s, style="Body2"):
    return Paragraph(str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                     if not (str(s).startswith("<") and str(s).endswith(">")) else str(s), styles[style])


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#D0D5DD"))
    canvas.line(18*mm, 14*mm, A4[0]-18*mm, 14*mm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#667085"))
    canvas.drawString(18*mm, 9*mm, "MLOps AIMLCZG523 | Generated from local run artifacts")
    canvas.drawRightString(A4[0]-18*mm, 9*mm, f"Page {doc.page}")
    canvas.restoreState()


def add_image(story, path, width=165*mm, height=95*mm):
    if not path.exists():
        return False
    try:
        img = Image(str(path))
        ratio = img.imageHeight / float(img.imageWidth)
        img.drawWidth = width
        img.drawHeight = min(height, width * ratio)
        story.extend([Spacer(1, 3*mm), img, Spacer(1, 3*mm)])
    except (OSError, ValueError) as exc:
        logging.getLogger(__name__).debug("Skipped image %s: %s", path.name, exc)
        return False
    return True


def build_report():
    summary = read_json(ART / "training_summary.json")
    eda = read_json(ART / "eda_summary.json")
    comparison = read_json(ART / "comparison.json")
    model_card = read_json(ART / "model_card.json")  # noqa: F841  (kept from the original)

    evidence_names = [
        "01_eda.png", "02_mlflow.png", "03_tests.png", "04_lint.png",
        "05_docker_build.png", "06_api_health.png", "07_api_predict.png",
        "08_kubernetes.png", "09_prometheus_metrics.png", "10_grafana.png",
        "11_github_actions.png"
    ]
    missing = [name for name in evidence_names if not (SHOTS / name).exists()]
    checks = {
        "dataset_summary": bool(eda),
        "training_summary": bool(summary),
        "comparison_metrics": bool(comparison),
        "serialized_model": (ROOT / "models" / "heart_disease_pipeline.joblib").exists(),
        "test_evidence_screenshot": (SHOTS / "03_tests.png").exists(),
        "docker_evidence_screenshot": (SHOTS / "05_docker_build.png").exists(),
        "kubernetes_evidence_screenshot": (SHOTS / "08_kubernetes.png").exists(),
        "github_actions_evidence_screenshot": (SHOTS / "11_github_actions.png").exists(),
    }
    status = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "checks": checks,
        "missing_screenshots": missing,
        "warning": "A file's existence does not independently prove its content or authenticity. Verify all artifacts before submission."
    }
    ART.mkdir(parents=True, exist_ok=True)
    (ART / "submission_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")

    story = [
        Spacer(1, 22*mm),
        Paragraph("END-TO-END MLOps", styles["CoverTitle2"]),
        Paragraph("Heart Disease Classification", styles["CoverTitle2"]),
        Spacer(1, 4*mm),
        p("Machine Learning Operations (MLOps) — AIMLCZG523", "Small2"),
        p("Assignment 01 | Total marks: 50", "Small2"),
        Spacer(1, 10*mm),
        p("This report is generated from local artifacts available at build time. Any absent metric or screenshot is marked as unavailable rather than fabricated.", "Notice2"),
        p("Health disclaimer: educational project only; not for diagnosis, treatment, triage, or clinical decisions.", "Notice2"),
        PageBreak(),
        p("1. Executive Summary", "Section2"),
        p("This project implements a reproducible binary classification workflow for the UCI Heart Disease processed Cleveland subset. It covers data acquisition, EDA, preprocessing, Logistic Regression and Random Forest model comparison, cross-validation and tuning, MLflow experiment tracking, model serialization, automated tests, CI configuration, FastAPI serving, Docker packaging, Kubernetes manifests, and Prometheus metrics."),
        p("The original UCI target `num` is converted to a binary target: 0 remains 0 and values above 0 become 1. The data is historical and small, so results do not establish clinical validity or generalization to other populations."),
        p("2. Runtime Status and Actual Results", "Section2"),
    ]
    if eda:
        story.append(p(f"Dataset rows: {eda.get('rows', 'not recorded')}; predictor features: {eda.get('features', 'not recorded')}; class 0: {eda.get('negative_class_count', 'not recorded')}; class 1: {eda.get('positive_class_count', 'not recorded')}.", "Body2"))
        story.append(p("Missing values by column: " + json.dumps(eda.get("missing_values_by_column", {}), sort_keys=True), "Small2"))
    else:
        story.append(p("EDA summary is not available. Run `make eda` to create it. No dataset counts are claimed in this report.", "Notice2"))
    if summary:
        story.append(p(f"Selected model: {summary.get('selected_model', 'not recorded')}. Test-set size: {summary.get('test_size', 'not recorded')}.", "Body2"))
        rows = [["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "CV ROC-AUC ± SD"]]
        for name, metrics in (summary.get("models") or {}).items():
            rows.append([name] + [f"{metrics.get(k, float('nan')):.3f}" if isinstance(metrics.get(k), (int, float)) else "n/a"
                                  for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]]
                        + [f"{metrics.get('cv_roc_auc_mean', float('nan')):.3f} ± {metrics.get('cv_roc_auc_std', float('nan')):.3f}"])
        story.append(Table(rows, repeatRows=1, colWidths=[35*mm, 19*mm, 19*mm, 17*mm, 15*mm, 19*mm, 39*mm],
                           style=TableStyle([
                               ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365D")),
                               ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                               ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                               ("FONTSIZE", (0, 0), (-1, -1), 7),
                               ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D0D5DD")),
                               ("VALIGN", (0, 0), (-1, -1), "TOP"),
                               ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
                               ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                               ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)
                           ])))
    else:
        story.append(p("Training summary is not available. Run `make train` to generate actual metrics. This report deliberately does not use stand-in values.", "Notice2"))
    story += [PageBreak(), p("3. Data Acquisition and EDA", "Section2"),
              p("The acquisition script downloads the processed Cleveland dataset from the UCI Machine Learning Repository when no local copy is present. The loader handles the original question-mark missing-value convention, normalizes numeric fields, removes duplicate rows, and derives the binary target. Missing feature values are imputed inside the model pipeline rather than before the split."),
              p("EDA checks include feature histograms, class balance, a correlation heatmap, and a JSON summary. Correlations describe association and do not imply causation.")]
    for fname, caption in [
        ("feature_distributions.png", "Figure 1. Feature distributions"),
        ("class_balance.png", "Figure 2. Binary class balance"),
        ("correlation_heatmap.png", "Figure 3. Feature/target correlation heatmap")
    ]:
        story.append(p(caption, "Small2"))
        if not add_image(story, FIG / fname, width=160*mm, height=80*mm):
            story.append(p(f"Figure unavailable: run `make eda` to generate {fname}.", "Notice2"))
    story += [PageBreak(), p("4. Feature Engineering and Model Development", "Section2"),
              p("Numeric features are median-imputed and standardized. Categorical-coded attributes are imputed with the most frequent value and one-hot encoded with unknown values ignored. Both transformations are fitted as part of a scikit-learn pipeline, reducing data leakage during cross-validation."),
              p("Logistic Regression provides a linear baseline. Random Forest can model nonlinear patterns and feature interactions. Grid search uses five-fold StratifiedKFold and ROC-AUC scoring. Reported metrics include accuracy, precision, recall, F1-score and ROC-AUC, alongside cross-validation ROC-AUC mean and standard deviation."),
              p("Evaluation caveat: the compact implementation uses the held-out split to compare tuned candidates. A rigorous unbiased final estimate should reserve an untouched test set after model selection.")]
    if summary:
        for fname, caption in [("confusion_matrix.png", "Figure 4. Selected model confusion matrix"),
                               ("roc_curve.png", "Figure 5. Selected model ROC curve")]:
            story.append(p(caption, "Small2"))
            if not add_image(story, FIG / fname, width=145*mm, height=80*mm):
                story.append(p(f"Figure unavailable: {fname}.", "Small2"))
    story += [PageBreak(), p("5. Experiment Tracking and Packaging", "Section2"),
              p("MLflow records model family, seed, split, fold count, cross-validation metric summaries, tuned parameters, held-out metrics, serialized sklearn model, and classification report. Start the local UI with `mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000` and inspect both runs."),
              p("The selected preprocessing-plus-estimator pipeline is saved with joblib. A model card records the selected model, target definition, metrics, parameters and limitations. Keep dependency versions reproducible by freezing the successfully tested environment if the instructor requires exact package versions."),
              p("6. Automated Testing and CI/CD", "Section2"),
              p("Pytest verifies preprocessing behavior and basic API endpoints. Ruff checks code style. The GitHub Actions workflow installs dependencies, validates the dataset, runs tests and EDA, trains the models, builds a Docker image, and uploads available run artifacts. It runs on pushes, pull requests, and manual dispatch."),
              p("GitHub Actions evidence is unavailable unless the repository has been pushed to GitHub and the workflow has run. Do not claim a passing workflow without checking its actual status.")]
    story += [PageBreak(), p("7. API and Containerization", "Section2"),
              p("FastAPI exposes `/`, `/health`, `/docs`, `/predict`, and `/metrics`. The prediction request schema validates the 13 required features and returns a binary prediction, label, predicted-class confidence and positive-class probability. These probabilities are model outputs, not calibrated clinical risk."),
              Preformatted("python -m src.heart_disease.train\nuvicorn src.heart_disease.api:app --host 127.0.0.1 --port 8000\ndocker build -t heart-disease-api:latest .\ndocker run --rm -p 8000:8000 heart-disease-api:latest", styles["Code2"]),
              p("The Docker image copies the source and serialized model into a Python 3.11 container. Build the image only after the model artifact exists. Test `/health`, `/docs`, `/predict` and `/metrics` on the running service."),
              p("8. Kubernetes Deployment", "Section2"),
              Preformatted("minikube start\ndocker build -t heart-disease-api:latest .\nminikube image load heart-disease-api:latest\nkubectl apply -f deployment/\nkubectl rollout status deployment/heart-disease-api\nkubectl get pods,svc", styles["Code2"]),
              p("The manifest defines two replicas, resource requests/limits, liveness/readiness probes and a LoadBalancer Service. Local Minikube may require `minikube service heart-disease-api --url` or port forwarding. No public endpoint is claimed unless a remote cluster is actually configured.")]
    story += [PageBreak(), p("9. Monitoring, Security, and Limitations", "Section2"),
              p("The API emits structured logs to stdout and exposes Prometheus counters for request outcome and a histogram for request duration. Prometheus can scrape `/metrics`; Grafana needs a configured Prometheus data source and dashboard. If those steps have not been run, the report records them as missing."),
              p("Before any real deployment, add authentication, TLS, rate limiting, network restrictions, secret management, log retention and vulnerability scanning. Do not log raw patient features or identifiable health information."),
              p("The dataset is small and historical. It cannot establish clinical validity, safety, fairness, or performance on current patient populations. The service is an educational demo only."),
              p("10. Evidence and Remaining Work", "Section2")]
    missing_table = [["Evidence item", "Status"]]
    for key, value in checks.items():
        missing_table.append([key.replace("_", " "), "Available" if value else "MISSING"])
    story.append(Table(missing_table, repeatRows=1, colWidths=[105*mm, 55*mm], style=TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365D")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D0D5DD")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)
    ])))
    story.append(Spacer(1, 3*mm))
    story.append(p("Missing screenshots: " + (", ".join(missing) if missing else "None of the 11 expected screenshot filenames are missing."), "Small2"))
    for filename in evidence_names:
        path = SHOTS / filename
        if path.exists():
            story.append(p(f"Evidence: {filename}", "Small2"))
            add_image(story, path, width=155*mm, height=82*mm)
    story += [PageBreak(), p("11. Conclusion and References", "Section2"),
              p("The repository provides an end-to-end implementation scaffold. Final submission should include only results from a genuine local run, verified screenshots, the student's own GitHub repository URL, and a short demo video. Any uncompleted deployment steps should be disclosed rather than represented as successful."),
              p("References", "Section2"),
              p("1. UCI Machine Learning Repository, Heart Disease, DOI: 10.24432/C52P4X — https://doi.org/10.24432/C52P4X<br/>2. Scikit-learn documentation — https://scikit-learn.org/<br/>3. MLflow documentation — https://mlflow.org/docs/latest/<br/>4. FastAPI documentation — https://fastapi.tiangolo.com/<br/>5. Docker documentation — https://docs.docker.com/<br/>6. Kubernetes documentation — https://kubernetes.io/docs/<br/>7. Prometheus documentation — https://prometheus.io/docs/", "Small2"),
              p("Repository URL: add the real URL after pushing to your own GitHub account.", "Notice2")]

    doc = SimpleDocTemplate(str(OUT), pagesize=A4, rightMargin=18*mm, leftMargin=18*mm,
                            topMargin=16*mm, bottomMargin=20*mm,
                            title="MLOps Assignment 01 — Generated from local results")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"Generated {OUT}")
    print(json.dumps(status, indent=2))
    return status


if __name__ == "__main__":
    build_report()
