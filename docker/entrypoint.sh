#!/bin/sh
set -e

echo "Running database migrations..."
/app/.venv/bin/python scripts/provision_db.py

echo "Configuring pgvector embedding dimensions..."
/app/.venv/bin/python scripts/configure_embeddings.py --yes

echo "Starting API server..."
exec /app/.venv/bin/fastapi run --host 0.0.0.0 src/main.py
