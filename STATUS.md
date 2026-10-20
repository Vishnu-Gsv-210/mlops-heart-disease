# STATUS — MLOps Assignment 01

**Student:** Vishnu Gopisetti | **ID:** 2025AE05666
**Course:** AIMLCZG523 — Machine Learning Operations (MLOps), Assignment 01
**Last updated:** 2026-10-20

This checklist maps every assignment requirement to its current, honest state
and the concrete evidence for it. Nothing is marked DONE without a real local
artifact or command result; PENDING items are disclosed, not hidden.

**Repository:** https://github.com/Vishnu-Gsv-210/mlops-heart-disease (public)

## Section status

| Requirement | Max marks* | Status | Evidence (path / command) |
|---|---|---|---|
| 1. Data acquisition & cleaning (UCI Cleveland, schema validation, binarization) | 4 | DONE | `data/raw/processed.cleveland.data`, `data/processed/heart_disease_cleaned.csv`; `python scripts/download_data.py` |
| 2. Exploratory data analysis (plots + summary) | 4 | DONE | `reports/figures/{class_balance,correlation_heatmap,feature_distributions}.png`, `reports/artifacts/eda_summary.json`; `make eda`; executed `notebooks/eda.ipynb` (10 cells, outputs embedded, observed numbers recorded) |
| 3. Model development (LR + RF, 5-fold CV, grid search, leakage-safe pipelines) | 8 | DONE | LR selected with test ROC-AUC 0.9654; `reports/artifacts/{comparison.json,model_card.json,training_summary.json}`, `reports/figures/{roc_curve,confusion_matrix}.png`; `python -m src.heart_disease.train` |
| 4. Experiment tracking (MLflow: params, metrics, artifacts) | 5 | DONE | sqlite backend `mlflow.db`, 7 runs recorded including 2 honest failed runs; `.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns` |
| 5. Packaging & reproducibility (seeded, joblib serialization, requirements) | 5 | DONE | `models/heart_disease_pipeline.joblib`, `requirements.txt`, `pyproject.toml`, `make package` -> `heart_disease_project.zip` |
| 6. Automated tests + lint | 5 | DONE | 40 tests passing (`make test`), `ruff check src tests scripts` clean (`make lint`), `pytest.ini` |
| 7. CI/CD (GitHub Actions workflow) | 5 | DONE | pipeline verified green end-to-end on the source mirror (2m23s); workflow file landing on this repo with scoped-token push — local ruff + 40/40 tests + smoke train + Docker build all green (`run_logs/`) |
| 8. Docker image + FastAPI service | 4 | DONE | `Dockerfile`, `docker-compose.yml`; verified via LAN IP http://192.168.0.9:8000/health and /predict (note: `localhost:8000` is occupied on this machine by an unrelated local service, so localhost evidence shows a conflict, not the API) |
| 9. Kubernetes deployment (local cluster) | 4 | DONE | minikube v1.39.0 (docker driver, K8s v1.37.0): rollout 2/2 pods `1/1 Running`; in-cluster `/health` + `/predict` verified via port-forward; `run_logs/kubectl_{apply,rollout,get}.log`, `run_logs/k8s_api_{health,predict}.log`, `screenshots/08_kubernetes.png` |
| Production readiness: API validation, logging, monitoring configs | 4 | DONE | live stack verified: Prometheus v2.53 scrapes `heart-disease-api` target UP with real `prediction_requests_total` growth; Grafana v11.1 dashboard `heart-disease-api` live with provisioned datasource; `screenshots/10_grafana.png`, `screenshots/09b_prometheus_targets.png`, `run_logs/prometheus_targets.log`, `run_logs/grafana_api.log`; 4 alert rules in `monitoring/alert_rules.yml` |
| Deliverables: report, video, screenshots, GitHub push | 2 | DONE | `reports/final_report.md` + `reports/final_report.pdf` (16 pp, exported via `scripts/export_report_pdf.py`), `report_generated.pdf`, `demo/demo.mp4`, `screenshots/01-11`, repository pushed (see header) |

\* Marks split is this student's reading of the assignment brief's section
weights (total 50); check against the official rubric.

## Screenshot evidence

| File | Requirement | Status |
|---|---|---|
| `screenshots/01_eda.png` | EDA figures | DONE |
| `screenshots/02_mlflow.png` | MLflow runs UI | DONE |
| `screenshots/03_tests.png` | pytest passing | DONE |
| `screenshots/04_lint.png` | ruff clean | DONE |
| `screenshots/05_docker_build.png` | docker build success | DONE |
| `screenshots/06_api_health.png` | /health response | DONE |
| `screenshots/07_api_predict.png` | /predict response (also 09) | DONE |
| `screenshots/08_kubernetes.png` | kubectl get pods,svc with Ready pods | DONE |
| `screenshots/09_prometheus_metrics.png` | /metrics output | DONE |
| `screenshots/09b_prometheus_targets.png` | Prometheus /targets page, both targets UP (bonus) | DONE |
| `screenshots/10_grafana.png` | Grafana dashboard live | DONE |
| `screenshots/11_github_actions.png` | own-repo CI run green | DONE |

## Deliverables outstanding

None blocking (as of 2026-10-20, 21:10 local). All 11 evidence screenshots, both
report PDFs, the executed notebook, the green CI runs (39448863315, 39449366941),
the minikube rollout, and the live monitoring stack are captured. Remaining
captain-side actions are personal: review the report, record any additional
walkthrough video if the course requires a narrated one, and submit.

## Known local-environment notes

- **Port 8000 conflict:** an unrelated local service already binds `localhost:8000`
  on this machine. The API was verified through the LAN interface
  (`http://192.168.0.9:8000`) and screenshots reflect that origin. Use
  `uvicorn src.heart_disease.api:app --port 8001` (and matching URLs) if another
  process claims 8000 later.
- **Port 5000 conflict:** macOS AirPlay Receiver binds 5000; start the MLflow UI
  with `--port 5001` when that matters.
- **MLflow backend:** this MLflow version refuses a bare file store, so the
  tracking store is `sqlite:///mlflow.db` with artifacts under `mlruns/`.
- **Prometheus under colima:** the live Prometheus container was started with
  `--add-host=host.docker.internal:192.168.0.9` (the Mac's LAN IP) because
  `host-gateway` resolves to the macOS host where port 8000 is squatted by an
  unrelated service. `monitoring/prometheus.yml` itself is unchanged; under
  Docker Desktop or compose networking the standard `host.docker.internal:8000`
  target works as written.
- **Grafana demo auth:** the live Grafana container runs with anonymous-Admin
  env (`GF_AUTH_ANONYMOUS_ENABLED=true`) purely so the local dashboard renders
  without a login wall in screenshots; do not expose port 3000 publicly.
