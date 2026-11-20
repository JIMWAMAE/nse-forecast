# API image. Train first (make pipeline) so models/ contains the artifacts.
FROM python:3.12-slim

# libgomp1 is needed by XGBoost
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config.yaml pyproject.toml ./
COPY src/ src/
COPY api/ api/
COPY scripts/ scripts/
COPY models/ models/

ENV PYTHONPATH=/app/src:/app \
    MODELS_DIR=/app/models \
    PYTHONUNBUFFERED=1

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
