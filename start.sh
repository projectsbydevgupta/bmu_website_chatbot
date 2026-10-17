#!/usr/bin/env bash
# ==============================================================================
# BMU Website Chatbot - Cloud Production Entrypoint (Render / Linux Container)
# ==============================================================================

# 1. Start the Retrieval Microservice in the background (Port 8001)
echo "Starting BMU Retrieval Microservice on port 8001..."
python -u retrieval_service.py &

# 2. Wait 2 seconds for microservice socket initialization
sleep 2

# 3. Start the Backend Orchestrator & Web Dashboard on Render public PORT
echo "Starting BMU Backend Orchestrator & Dashboard on port ${PORT:-8000}..."
python -u semantic.py
