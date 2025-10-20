#!/bin/bash

# Run the OUNASS API with uvicorn
# This script excludes venv from file watching to avoid unnecessary reloads

echo "Starting OUNASS Kubernetes Pod Forecasting API..."
echo "======================================================================="
echo ""
echo "API will be available at:"
echo "  - Main: http://127.0.0.1:8000"
echo "  - Docs: http://127.0.0.1:8000/docs"
echo "  - Health: http://127.0.0.1:8000/api/v1/health"
echo ""
echo "Press CTRL+C to stop"
echo "======================================================================="
echo ""

# Run uvicorn with reload, excluding venv directory
uvicorn src.main:app \
    --reload \
    --reload-exclude 'venv/*' \
    --reload-exclude '*.pyc' \
    --reload-exclude '__pycache__/*' \
    --host 0.0.0.0 \
    --port 8000
