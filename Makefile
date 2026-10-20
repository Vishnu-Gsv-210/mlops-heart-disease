.PHONY: setup data eda train test lint report api docker-build docker-run deploy validate package clean

setup:
	python -m pip install --upgrade pip
	python -m pip install -r requirements.txt

data:
	python scripts/download_data.py

eda:
	python -c "from src.heart_disease.config import FIGURES_DIR; from src.heart_disease.data import generate_eda, load_clean_dataframe; generate_eda(load_clean_dataframe(), FIGURES_DIR)"

train:
	python -m src.heart_disease.train

test:
	pytest -v

lint:
	ruff check src tests scripts

report:
	python scripts/generate_report.py

api:
	uvicorn src.heart_disease.api:app --host 127.0.0.1 --port 8000

docker-build:
	docker build -t heart-disease-api:latest .

docker-run:
	docker run --rm -p 8000:8000 heart-disease-api:latest

deploy:
	kubectl apply -R -f deployment/

validate:
	python scripts/validate_deployment.py

package:
	python scripts/package_project.py

clean:
	rm -rf .pytest_cache .ruff_cache build dist heart_disease_project.zip
	find . -name "__pycache__" -type d -not -path "./.venv/*" -exec rm -rf {} +
