#!/usr/bin/env bash
set -e

# (Optional) build the RAG index on startup if you want.
# Better practice is to call POST /ingest once after deploy.
# python -c "import requests; requests.post('http://127.0.0.1:8003/ingest')" || true

exec uvicorn server:app --host 0.0.0.0 --port "${PORT:-8003}"

