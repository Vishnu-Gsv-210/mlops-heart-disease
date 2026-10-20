"""Generate the end-to-end MLOps architecture diagram for the heart-disease project.

Source-based and re-runnable: python reports/architecture/generate_architecture.py
Output: reports/architecture/architecture.png (~2200x1500 px at dpi 200)
Dependencies: matplotlib only (bundled with the project venv; no graphviz/networkx).
"""
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path as MplPath

OUT = Path(__file__).resolve().parent / "architecture.png"

# ---------------------------------------------------------------- palette ----
INK = "#0f172a"
MUTED = "#475569"
DATA_FACE, DATA_EDGE, DATA_TITLE = "#dbeafe", "#1d4ed8", "#1e3a8a"
ML_FACE, ML_EDGE, ML_TITLE = "#dcfce7", "#15803d", "#14532d"
SERVE_FACE, SERVE_EDGE, SERVE_TITLE = "#ffedd5", "#c2410c", "#7c2d12"
CI_FACE, CI_EDGE = "#ede9fe", "#6d28d9"
PROM_FACE, PROM_EDGE = "#e0f2fe", "#0284c7"
GRAF_FACE, GRAF_EDGE = "#fee2e2", "#b91c1c"
MLF_FACE, MLF_EDGE = "#fae8ff", "#a21caf"

TITLE = "End-to-End MLOps Architecture — Heart Disease Classification"
SUBTITLE = ("UCI Heart Disease (DOI 10.24432/C52P4X) · Python 3.11 · scikit-learn · "
            "MLflow · FastAPI · Docker · Kubernetes · Prometheus/Grafana")
FOOTER = "Vishnu Gopisetti  |  2025AE05666  |  AIMLCZG523 Assignment 01"

# ------------------------------------------------------------ draw helpers ----
def box(ax, x, y, w, h, title, lines, face, edge, title_color):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0,rounding_size=1.4",
                                facecolor=face, edgecolor=edge, linewidth=1.7))
    ax.text(x + w / 2, y + h - 1.9, title, ha="center", va="center",
            fontsize=10.5, fontweight="bold", color=title_color)
    body = "\n".join(lines)
    ax.text(x + w / 2, y + h - 4.1, body, ha="center", va="top",
            fontsize=7.6, color=MUTED, linespacing=1.45)

def lane(ax, xc, yc, label, color):
    ax.text(xc, yc, " ".join(label), ha="center", va="center", rotation=90,
            fontsize=10.5, fontweight="bold", color=color)

def arrow(ax, p0, p1, color, dashed=False, lw=1.8):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=15, linewidth=lw,
        color=color, shrinkA=2, shrinkB=2, joinstyle="round", capstyle="round",
        linestyle=(0, (5, 3)) if dashed else "solid"))

def elbow(ax, pts, color, dashed=False, lw=1.8):
    verts = list(pts)
    codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(verts) - 1)
    ax.add_patch(FancyArrowPatch(
        path=MplPath(verts, codes), arrowstyle="-|>", mutation_scale=15,
        linewidth=lw, color=color, shrinkA=2, shrinkB=2, joinstyle="round",
        capstyle="round", linestyle=(0, (5, 3)) if dashed else "solid"))

def tag(ax, x, y, text, color):
    ax.text(x, y, text, ha="center", va="center", fontsize=7.0,
            style="italic", color=color,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor=color, linewidth=0.7, alpha=0.95))

# ------------------------------------------------------------------ canvas ----
fig = plt.figure(figsize=(11, 7.5), dpi=200)
fig.patch.set_facecolor("white")
ax = fig.add_axes((0, 0, 1, 1))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")
ax.set_facecolor("white")

ax.text(50, 96.2, TITLE, ha="center", va="center",
        fontsize=20, fontweight="bold", color=INK)
ax.text(50, 92.6, SUBTITLE, ha="center", va="center", fontsize=8.6, color=MUTED)
ax.add_patch(plt.Rectangle((2, 91.2), 96, 0.25, facecolor="#cbd5e1", edgecolor="none"))

# ---- Row A · DATA (y 84.5-91.5) ----
A_Y, A_H = 84.5, 7.0
box(ax, 7, A_Y, 20, A_H, "UCI Repository",
    ["processed.cleveland.data", "DOI 10.24432/C52P4X"], DATA_FACE, DATA_EDGE, DATA_TITLE)
box(ax, 31, A_Y, 21, A_H, "Download & Validate",
    ["scripts/download_data.py →", "src.heart_disease.data.validate_schema", "na '?' → NaN · schema/range checks"],
    DATA_FACE, DATA_EDGE, DATA_TITLE)
box(ax, 56, A_Y, 17, A_H, "Cleaned CSV",
    ["data/processed/", "heart_disease_cleaned.csv", "303 × 13 · 0 duplicates"],
    DATA_FACE, DATA_EDGE, DATA_TITLE)
box(ax, 77, A_Y, 21, A_H, "EDA",
    ["reports/figures/*.png", "reports/artifacts/eda_summary.json", "139 pos / 164 neg · ca 4, thal 2 missing"],
    DATA_FACE, DATA_EDGE, DATA_TITLE)
for x0, x1 in ((27, 31), (52, 56), (73, 77)):
    arrow(ax, (x0, A_Y + A_H / 2), (x1, A_Y + A_H / 2), DATA_EDGE)

# ---- Row B · MODELING (y 63.5-71.5) ----
B_Y, B_H = 63.5, 8.0
box(ax, 7, B_Y, 17, B_H, "Preprocessing",
    ["numeric: median impute", "+ StandardScaler",
     "categorical: mode impute", "+ OneHotEncoder",
     "sklearn Pipeline +", "ColumnTransformer"],
    ML_FACE, ML_EDGE, ML_TITLE)
box(ax, 27, B_Y, 16, B_H, "Candidate Models",
    ["Logistic Regression", "(class_weight=balanced,",
     "max_iter=3000)", "Random Forest", "(class_weight=balanced)"],
    ML_FACE, ML_EDGE, ML_TITLE)
box(ax, 46, B_Y, 16, B_H, "GridSearchCV",
    ["StratifiedKFold(5, shuffle)", "scoring = ROC-AUC",
     "grid: C {0.1,1,10} /", "trees {150,300} ×", "depth {None,5,10} ×", "leaf {1,3}"],
    ML_FACE, ML_EDGE, ML_TITLE)
box(ax, 65, B_Y, 16, B_H, "Evaluation",
    ["held-out 20% (n=61)", "accuracy · precision", "recall · F1 · ROC-AUC",
     "confusion matrix ·", "ROC curve figures"],
    ML_FACE, ML_EDGE, ML_TITLE)
box(ax, 84, B_Y, 14, B_H, "Serialized Model",
    ["models/", "heart_disease_pipeline.", "joblib", "(full pipeline:", "preprocess + model)"],
    ML_FACE, ML_EDGE, ML_TITLE)
for x0, x1 in ((24, 27), (43, 46), (62, 65), (81, 84)):
    arrow(ax, (x0, B_Y + B_H / 2), (x1, B_Y + B_H / 2), ML_EDGE)

# ---- Row C · SERVING (y 43.5-51.5) ----
C_Y, C_H = 43.5, 8.0
box(ax, 7, C_Y, 29, C_H, "FastAPI Service  (uvicorn, port 8000)",
    ["POST /predict · GET /health · GET /metrics · GET /docs",
     "Pydantic feature validation (ranges per UCI coding)",
     "Prometheus counters/histograms + JSON request logs"],
    SERVE_FACE, SERVE_EDGE, SERVE_TITLE)
box(ax, 40, C_Y, 24, C_H, "Docker Image",
    ["python:3.11-slim", "models + reports baked in",
     "HEALTHCHECK /health · EXPOSE 8000"],
    SERVE_FACE, SERVE_EDGE, SERVE_TITLE)
box(ax, 68, C_Y, 30, C_H, "Kubernetes (minikube)",
    ["Deployment: 2 replicas · resources + probes",
     "readiness/liveness on /health",
     "Service: LoadBalancer 80 → 8000"],
    SERVE_FACE, SERVE_EDGE, SERVE_TITLE)
for x0, x1 in ((36, 40), (64, 68)):
    arrow(ax, (x0, C_Y + C_H / 2), (x1, C_Y + C_H / 2), SERVE_EDGE)

# ---- Row D · OPERATIONS (y 20.5-28.5) ----
D_Y, D_H = 20.5, 8.0
box(ax, 7, D_Y, 21, D_H, "GitHub Actions CI",
    ["on push / PR:", "ruff → pytest → download_data", "+ schema → src.heart_disease.train",
     "→ docker build · uploads artifacts"],
    CI_FACE, CI_EDGE, "#4c1d95")
box(ax, 32, D_Y, 20, D_H, "Prometheus",
    ["scrape /metrics every 15s", "prediction_requests_total",
     "prediction_request_duration_seconds"],
    PROM_FACE, PROM_EDGE, "#0c4a6e")
box(ax, 56, D_Y, 12, D_H, "Grafana",
    ["dashboards on", "Prometheus", "datasource"],
    GRAF_FACE, GRAF_EDGE, "#7f1d1d")
box(ax, 72, D_Y, 26, D_H, "MLflow Tracking",
    ["params · metrics · models · artifacts",
     "sqlite backend: mlflow.db",
     "experiment: heart-disease-classification"],
    MLF_FACE, MLF_EDGE, "#701a75")
arrow(ax, (52, D_Y + D_H / 2), (56, D_Y + D_H / 2), PROM_EDGE)

# ---- lane labels ----
lane(ax, 3.0, A_Y + A_H / 2, "STAGE · DATA", DATA_EDGE)
lane(ax, 3.0, B_Y + B_H / 2, "STAGE · MODELING", ML_EDGE)
lane(ax, 3.0, C_Y + C_H / 2, "STAGE · SERVING", SERVE_EDGE)
lane(ax, 3.0, D_Y + D_H / 2, "STAGE · OPERATIONS", CI_EDGE)

# ---- cross-row flows ----
# Cleaned CSV -> Preprocessing (fit on training folds only)
elbow(ax, [(64.5, A_Y), (64.5, 78.0), (15.5, 78.0), (15.5, B_Y + B_H)], DATA_EDGE)
tag(ax, 40, 78.0, "scripts/run_pipeline.py · fit transformers on training folds only (leakage-safe)", DATA_EDGE)
# Serialized model -> FastAPI
elbow(ax, [(91, B_Y), (91, 57.5), (21.5, 57.5), (21.5, C_Y + C_H)], ML_EDGE)
tag(ax, 56, 57.5, "joblib artifact loaded by the API at startup", ML_EDGE)
# FastAPI -> Prometheus (scrape)
elbow(ax, [(21.5, C_Y), (21.5, 36.5), (42, 36.5), (42, D_Y + D_H)], PROM_EDGE)
tag(ax, 30, 36.5, "scrape /metrics", PROM_EDGE)
# Training (GridSearchCV) -> MLflow
elbow(ax, [(54, B_Y), (54, 33.0), (85, 33.0), (85, D_Y + D_H)], MLF_EDGE)
tag(ax, 68, 33.0, "logs params / metrics / artifacts per run", MLF_EDGE)
# CI -> Modeling pipeline (dashed trigger, routed down the left margin)
elbow(ax, [(7, D_Y + 2.2), (4.9, D_Y + 2.2), (4.9, B_Y + 4.0), (7, B_Y + 4.0)], CI_EDGE, dashed=True)
tag(ax, 4.9, 37.0, "triggers pipeline on push / PR", CI_EDGE)
ax.texts[-1].set_rotation(90)

# ---- footer ----
ax.add_patch(plt.Rectangle((2, 9.2), 96, 0.25, facecolor="#cbd5e1", edgecolor="none"))
ax.text(50, 6.2, FOOTER, ha="center", va="center", fontsize=11.5,
        fontweight="bold", color=INK)
ax.text(50, 2.9, f"Generated {date.today().isoformat()} · source: reports/architecture/generate_architecture.py (matplotlib, re-runnable)",
        ha="center", va="center", fontsize=7.6, color=MUTED)

fig.savefig(OUT, dpi=200, bbox_inches=None, facecolor="white")
print(f"wrote {OUT}")
