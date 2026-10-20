# START HERE — MLOps Assignment 01 (1-hour runbook)

**Goal:** get the real dataset, run the ML pipeline, create a report from actual local outputs, and collect the evidence you can genuinely produce in the time available.

**Important:** this repository cannot honestly contain your personal GitHub Actions run, Kubernetes rollout, Grafana screenshot, screenshots, or video before you run them. Do not submit fabricated evidence. If a required step cannot be completed in one hour, record the true state in `STATUS.md` and ask your instructor whether late evidence can be supplied.

## Before starting (2 minutes)

1. Open the `mlops-heart-disease` folder in a terminal.
2. Ensure Python 3.11 is installed. Docker is needed for the container and monitoring sections. Minikube and `kubectl` are needed for Kubernetes.
3. Keep internet access available for package installation and the UCI dataset download.
4. **Port conflicts on this machine:** an unrelated local service occupies `localhost:8000` (the API was verified via the LAN IP, e.g. `http://192.168.0.9:8000`), and macOS AirPlay binds `5000` (start the MLflow UI on `--port 5001`). Workarounds are inline below.

## Fast path (about 35–50 minutes, if package/data downloads work)

### Using the existing virtualenv (recommended; macOS/Linux)

```bash
source .venv/bin/activate
python scripts/download_data.py      # fetch + validate the UCI data (or: make data)
python scripts/run_pipeline.py       # end-to-end: validate -> EDA -> train -> evaluate
make test                            # pytest (40 tests)
make lint                            # ruff check src tests scripts
python scripts/generate_report.py    # report_generated.pdf from real artifacts
```

### Fresh environment (Windows PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
make setup   # or: python -m pip install -r requirements.txt
python scripts\download_data.py
python scripts\run_pipeline.py
pytest
ruff check src tests scripts
python scripts\generate_report.py
```

### Fresh environment (macOS/Linux)

```bash
python3.11 -m venv .venv
source .venv/bin/activate
make setup
python scripts/download_data.py
python scripts/run_pipeline.py
pytest
ruff check src tests scripts
python scripts/generate_report.py
```

### What each command produces

- `python scripts/download_data.py`: downloads the official UCI Cleveland file into `data/raw/` and checks the schema contract.
- `python scripts/run_pipeline.py`: one-shot pipeline — download, validate, EDA figures + summary, train both model families, log MLflow runs, serialize the selected model.
- `make eda`: EDA figures + `reports/artifacts/eda_summary.json` only (`notebooks/eda.ipynb` is the narrated notebook version).
- `python -m src.heart_disease.train`: training step alone — metrics, model card, ROC/confusion plots, MLflow runs.
- `make test` / `make lint`: 40 pytest cases and ruff.
- `python scripts/generate_report.py`: builds `report_generated.pdf` from the actual local metrics, available figures/screenshots, and missing-evidence status.

## MLflow UI (SQLite backend)

The tracking store is `mlflow.db` (SQLite) with artifacts under `mlruns/` — this MLflow version refuses a bare file store.

```bash
.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --port 5001
```

Open `http://127.0.0.1:5001`. Port 5001 sidesteps the macOS AirPlay receiver on 5000. Current history: 7 runs, including 2 honest failed runs kept as real evidence.

## Evidence checklist (save only real screenshots)

Create these files in `screenshots/` with the names below. On Windows, use `Win+Shift+S`; on macOS, use `Shift+Cmd+4`. Take screenshots only after the matching command or service has actually run. Current status per file: `STATUS.md`.

1. `01_eda.png` — `reports/figures/` plots or the executed EDA notebook. (DONE)
2. `02_mlflow.png` — the MLflow UI command above, showing the runs and metrics. (DONE)
3. `03_tests.png` — terminal showing successful `pytest`. (DONE)
4. `04_lint.png` — terminal showing successful `ruff check src tests scripts`. (DONE)
5. `05_docker_build.png` — terminal showing successful `docker build`. (PENDING)
6. `06_api_health.png` — browser at the API's `/health` while the container/API runs. (DONE, captured via the LAN address because of the localhost:8000 conflict)
7. `07_api_predict.png` — `/docs` or a real `/predict` response using the example JSON in `README.md`. (DONE)
8. `08_kubernetes.png` — terminal showing `kubectl get pods,svc` with Ready pods. (PENDING)
9. `09_prometheus_metrics.png` — `/metrics` endpoint showing request metrics. (DONE)
10. `10_grafana.png` — Grafana dashboard; the "Heart Disease API" dashboard is now provisioned via `monitoring/` (below). (PENDING)
11. `11_github_actions.png` — GitHub Actions run in your own repository, with successful steps visible. (PENDING)

Then run `python scripts/generate_report.py` again to embed whatever evidence exists and list what is still missing.

## Monitoring stack (Prometheus + Grafana via docker compose)

```bash
uvicorn src.heart_disease.api:app --host 0.0.0.0 --port 8000   # terminal 1: the API
docker compose up -d prometheus grafana                        # terminal 2: the stack
```

Verify: Prometheus targets at `http://localhost:9090/targets` (`heart-disease-api` UP), Grafana at `http://localhost:3000` (admin/admin) with the pre-provisioned "Heart Disease API" dashboard. Full instructions and standalone `docker run` equivalents: `monitoring/README.md`.

## Docker (only if Docker Desktop is already installed; allow 5–10 minutes)

First ensure the model exists (`models/heart_disease_pipeline.joblib`).

```bash
make docker-build      # docker build -t heart-disease-api:latest .
make docker-run        # docker run --rm -p 8000:8000 heart-disease-api:latest
```

In another terminal:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/metrics
```

If `localhost:8000` answers with something that is not this API (known conflict on this machine), verify through the LAN IP (`http://192.168.0.9:8000`) or run the container on `--port 8001` style mappings and use those URLs.

For `/predict`, open `http://localhost:8000/docs` and use the sample JSON in `README.md`. Stop the server with `Ctrl+C`.

## Kubernetes (only if Minikube is already installed and working; allow 10+ minutes)

```bash
make docker-build
minikube start
minikube image load heart-disease-api:latest
kubectl apply -f deployment/kubernetes/
kubectl rollout status deployment/heart-disease-api -n heart-disease
kubectl get pods,svc -n heart-disease
minikube service heart-disease-api -n heart-disease --url
```

If Minikube is not installed or needs a large download, do not burn your entire remaining hour installing it. Capture only what you actually complete and record the limitation in `STATUS.md`.

## GitHub (5–10 minutes, account and Git installed required)

Create an empty repository on GitHub. Then run these commands from the project folder (replace the URL with your own repository URL):

```bash
git init
git add .
git commit -m "Complete MLOps heart disease assignment"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
git push -u origin main
```

Open the repository's **Actions** tab and wait for the workflow. If the workflow fails, inspect the log; do not claim it passed. A GitHub Actions run requires internet access and time to install packages/download data.

## Short demo video (2–3 minutes)

Follow `demo/recording_guide.md`: a scene-by-scene checklist with the exact commands per scene and which browser tabs to show. Record only steps that actually work; save as `screenshots/demo.mp4`.

## Troubleshooting

| Problem | What to do |
|---|---|
| UCI download fails / timeout | Retry on another network. If you have a legitimately downloaded Cleveland data file, place it at `data/raw/processed.cleveland.data`. Do not use fabricated stand-in data for final metrics. |
| `ModuleNotFoundError: src` / `src.heart_disease` | Run commands from the `mlops-heart-disease` project root, not from `src/`. |
| `localhost:8000` shows an unrelated service | Known conflict on this machine. Verify via the LAN IP (e.g. `http://192.168.0.9:8000`) or move the API to port 8001. |
| MLflow UI port busy / page blank | Use `--port 5001` (AirPlay holds 5000). For a blank page, confirm `python -m src.heart_disease.train` completed and that you launched with `--backend-store-uri sqlite:///mlflow.db` from the repository root. |
| PowerShell blocks activation | Use `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` for this terminal, then activate the venv. |
| Docker daemon unavailable | Start Docker Desktop and wait until it reports running. If not possible within your time, record that Docker verification is pending in `STATUS.md`. |
| Minikube image pull / LoadBalancer issue | Use `minikube image load heart-disease-api:latest`; for local access use `minikube service heart-disease-api -n heart-disease --url` or `kubectl port-forward svc/heart-disease-api 8080:80 -n heart-disease`. |
| Grafana dashboard missing | Confirm you launched the stack from the repo root so `monitoring/grafana/provisioning` is mounted; check `docker compose logs grafana`. |
| Report says evidence missing | Expected until the named screenshots/artifacts are created. Capture genuine evidence and rerun `python scripts/generate_report.py`. |
| Dependency install takes too long | Do not silently skip steps. Record the actual error and submit the generated status as evidence of the blocker. |

## What the design does / be ready to explain

- **Target conversion:** original UCI `num` values 0–4 become a binary label (`0` stays negative; `1–4` become positive).
- **Preprocessing inside a Pipeline:** imputation, scaling, and one-hot encoding are fitted within training folds to reduce data leakage.
- **Two model families:** Logistic Regression (selected, test ROC-AUC 0.9654) and Random Forest.
- **ROC-AUC tuning:** selected as the grid-search score; accuracy, precision, recall, F1 and ROC-AUC are also reported.
- **MLflow:** SQLite backend (`mlflow.db`) records parameters, metrics, model artifacts and classification reports; `mlruns/` holds artifacts.
- **API confidence:** this is a model probability, not a calibrated medical risk or diagnosis.
- **Monitoring:** API exposes `prediction_requests_total` and `prediction_request_duration_seconds`; Prometheus + Grafana come up with `docker compose up -d prometheus grafana` using `monitoring/` configs, alerts, and the provisioned dashboard.
- **Model selection caveat:** this compact assignment compares tuned candidates on the held-out partition. A rigorous unbiased final estimate would use a separate untouched test set after validation-based selection.
- **Health-data disclaimer:** historical Cleveland dataset; educational only, not for clinical use.

## Academic integrity and final check

Rewrite the report in your own words, personalize the implementation and explain it honestly. The assignment brief specifically warns about similar submissions. Never invent metrics, screenshots, a public API URL, a successful GitHub Actions run, or a Kubernetes deployment. If an item remains incomplete, disclose it in `STATUS.md` and ask the instructor whether you may provide it later.
