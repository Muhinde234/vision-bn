#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "  VisionDx API - Starting Development Server"
echo "  ============================================"
echo ""

if [ ! -f ".env" ]; then
    echo "[ERROR] .env not found. Copy .env.example first."
    exit 1
fi

if [ ! -f "visiondx_dev.db" ]; then
    echo "Creating database tables..."
    python create_tables.py
fi

echo "Starting FastAPI on http://localhost:8000"
echo "Swagger docs: http://localhost:8000/docs"
echo "Press Ctrl+C to stop."
echo ""

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
