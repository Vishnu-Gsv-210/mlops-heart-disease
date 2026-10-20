# Demo video recording guide (2–3 minutes)

**Rule: record only what actually works.** If a scene's command fails, fix it
first or drop the scene and say so in the report — never narrate a step you did
not run. Re-record rather than editing stills into a fake sequence.

**Setup before recording (one time):**
- Close unrelated tabs/windows; browser bookmarks bar hidden.
- Terminal: `.venv` activated, project root as cwd, reasonably large font.
- Browser tabs ready: `http://<your-host>:8000/docs`, `http://127.0.0.1:5000`
  (MLflow UI, or `:5001` if 5000 is busy), `http://localhost:9090` (Prometheus),
  `http://localhost:3000` (Grafana).
- Screen recorder: macOS `Cmd+Shift+5` (capture the whole screen or the app
  window), save as `screenshots/demo.mp4`.
- Do a dry run of every command below once before hitting record.

## Scene 1 — Project tour + EDA evidence (~30s)

Say: what the project is (UCI Cleveland heart disease, binary classification,
end-to-end MLOps) and what is on screen.

- Show the repo tree briefly (`ls` or Finder).
- Open `reports/figures/class_balance.png`, `correlation_heatmap.png`,
  `roc_curve.png` (or the executed `notebooks/eda.ipynb` with outputs).

## Scene 2 — MLflow tracking (~30s)

```bash
.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
```

- Switch to the MLflow tab; show the experiment table with the 7 runs
  (including the 2 failed runs — say honestly they were kept as real history).
- Click one run: show params, metrics, artifacts.

## Scene 3 — Tests + lint (~20s)

```bash
make test     # pytest: show "40 passed"
make lint     # ruff: show clean output
```

- Keep the terminal visible long enough to read the pass lines.

## Scene 4 — API live (~30s)

```bash
uvicorn src.heart_disease.api:app --host 0.0.0.0 --port 8000
```

- Browser: `/health` first, then `/docs`; execute `POST /predict` with the
  sample JSON from `README.md`; show the response
  (`{"prediction": ..., "label": ..., "confidence": ...}`).
- Note on camera if you use the LAN address (e.g. `http://192.168.0.9:8000`)
  because `localhost:8000` is occupied by another service on this machine.

## Scene 5 — Monitoring (only if the stack is actually up) (~30s)

```bash
docker compose up -d prometheus grafana
```

- Prometheus tab → Status ▸ Targets: `heart-disease-api` UP, rules loaded.
- Grafana tab → "Heart Disease API" dashboard with panels drawing.
- Skip this scene if the stack is not running and disclose that in the report.

## Scene 6 — Kubernetes (only if the rollout actually succeeded) (~30s)

```bash
kubectl apply -f deployment/kubernetes/
kubectl rollout status deployment/heart-disease-api -n heart-disease
kubectl get pods,svc -n heart-disease
```

- Show Ready `2/2` pods and the service; hit the service URL's `/health` if
  reachable. Skip and disclose if the rollout never went green.

## Scene 7 — Close (~10s)

- Show `STATUS.md` checklist; state what is done and what is pending, plainly.
- Stop recording.

## After recording

- Watch it once end-to-end: every command shown must be one you actually ran;
  no visible secrets, tokens, or unrelated personal tabs.
- Save as `screenshots/demo.mp4` (or `.mov`); keep it under ~50 MB if the
  submission is a zip (lower the resolution rather than trimming evidence).
- Update `STATUS.md` (video row) once the file exists.
