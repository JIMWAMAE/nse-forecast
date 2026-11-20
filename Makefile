# Run `make help` to see the commands.
PY ?= python3
VENV = .venv
BIN = $(VENV)/bin

help:
	@echo "make setup     - create .venv and install packages"
	@echo "make pipeline  - full pipeline: clean, tune, test on 2026, train final model"
	@echo "make quick     - same, but reuse saved best hyper-parameters"
	@echo "make test      - run the tests (uses synthetic data)"
	@echo "make serve     - start the API locally on http://127.0.0.1:8000/docs"
	@echo "make demo      - call the running API with a sample client"
	@echo "make docker    - build and run the API in Docker"

setup:
	$(PY) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -r requirements-dev.txt

pipeline:
	PYTHONPATH=src $(BIN)/python -m nse.pipeline

quick:
	PYTHONPATH=src $(BIN)/python -m nse.pipeline --skip-tuning

test:
	$(BIN)/python -m pytest -q

serve:
	PYTHONPATH=src:. $(BIN)/uvicorn api.main:app --reload --port 8000

demo:
	$(BIN)/python scripts/demo_client.py

docker:
	docker compose up --build api

.PHONY: help setup pipeline quick test serve demo docker
