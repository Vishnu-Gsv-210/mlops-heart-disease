# Monitoring and logging

The FastAPI service emits structured JSON request logs to stdout and exposes
Prometheus metrics at `/metrics`. This directory contains the complete local
monitoring stack: Prometheus scrape configuration, alerting rules, and a
provisioned Grafana dashboard.

## Metric contract (emitted by `src/heart_disease/monitoring.py`)

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `prediction_requests_total` | counter | `status` (`success`/`error`) | Total `/predict` calls |
| `prediction_request_duration_seconds` | histogram | (none) | Per-request latency; exposes `_bucket`, `_sum`, `_count` |

## What each piece does

| File | Mounted at | Purpose |
|---|---|---|
| `prometheus.yml` | `/etc/prometheus/prometheus.yml` | Scrape config: Prometheus self-scrape + `heart-disease-api` job at `host.docker.internal:8000/metrics`, every 15s |
| `alert_rules.yml` | `/etc/prometheus/alert_rules.yml` | Alerts: `APIDown`, `HighPredictionErrorRate` (>5%), `HighLatencyP95` (>2s), `NoTraffic` |
| `grafana/provisioning/datasources/datasource.yml` | `/etc/grafana/provisioning/datasources/` | Auto-registers the Prometheus datasource (`uid: prometheus`, `http://prometheus:9090`) |
| `grafana/provisioning/dashboards/dashboards.yml` | `/etc/grafana/provisioning/dashboards/` | File-based dashboard provider watching `/var/lib/grafana/dashboards` |
| `grafana/dashboards/heart_disease_api.json` | `/var/lib/grafana/dashboards/` | "Heart Disease API" dashboard: request rate by status, p50/p95 latency, total requests, API up, error rate |

`docker-compose.yml` (repo root) already wires all of these mounts for the
`prometheus` (v2.53.0) and `grafana` (11.1.0) services.

## Run the stack

Prerequisite: a trained model exists at `models/heart_disease_pipeline.joblib`
(`make train` if not).

### Option A — everything with docker compose (recommended)

```bash
# terminal 1: the API on the host so Prometheus can reach it at host.docker.internal:8000
uvicorn src.heart_disease.api:app --host 0.0.0.0 --port 8000

# terminal 2: Prometheus + Grafana containers
docker compose up -d prometheus grafana
```

You can also run the API itself as a compose service (`docker compose up -d`),
since the prometheus container resolves the host through `host.docker.internal`.

### Option B — standalone containers (equivalent to the compose services)

```bash
docker network create mlops-mon

docker run -d --name prometheus --network mlops-mon -p 9090:9090 \
  --add-host=host.docker.internal:host-gateway \
  -v "$(pwd)/monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro" \
  -v "$(pwd)/monitoring/alert_rules.yml:/etc/prometheus/alert_rules.yml:ro" \
  prom/prometheus:v2.53.0

docker run -d --name grafana --network mlops-mon -p 3000:3000 \
  -v "$(pwd)/monitoring/grafana/provisioning:/etc/grafana/provisioning:ro" \
  -v "$(pwd)/monitoring/grafana/dashboards:/var/lib/grafana/dashboards:ro" \
  grafana/grafana:11.1.0
```

## Verify

```bash
# 1. API metrics are being served
curl http://localhost:8000/metrics | grep prediction_requests_total

# 2. Prometheus is up and its target for the API is healthy
open http://localhost:9090/targets        # heart-disease-api should show UP
# generate traffic, then query in the Prometheus UI:
#   rate(prediction_requests_total[5m])

# 3. Grafana dashboard is provisioned
open http://localhost:3000                # default login admin/admin
# Dashboard "Heart Disease API" appears under Dashboards; refresh is 10s over a 1h range.

# 4. Fire some predictions and watch the panels move
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"age":55,"sex":1,"cp":2,"trestbps":130,"chol":240,"fbs":0,"restecg":1,"thalach":150,"exang":0,"oldpeak":1.2,"slope":2,"ca":0,"thal":3}'

# 5. Alert rules are loaded
open http://localhost:9090/rules          # heart_disease_api group with 4 rules
```

Default Grafana credentials are `admin` / `admin`; Grafana asks you to set a new
password on first login.

## Logs

The API logs structured JSON lines (event, status, client, prediction,
duration_ms) to stdout. Collect them with `docker logs <container>` for the
containerized API or `kubectl logs deployment/heart-disease-api` in Kubernetes.
