#!/usr/bin/env bash
# ==============================================================================
# BMU Website Chatbot - Cloud Production Entrypoint (Render / Linux Container)
# ==============================================================================

# 1. Start the Retrieval Microservice in the background (Port 8001)
echo "Starting BMU Retrieval Microservice on port 8001..."
python -u retrieval_service.py &

# 2. Wait for Retrieval Microservice to be ready (up to 30s)
echo "Waiting for Retrieval Microservice to initialize..."
for i in $(seq 1 30); do
    if curl -s http://127.0.0.1:8001/health > /dev/null 2>&1; then
        echo "Retrieval Microservice is READY on port 8001 (took ${i}s)!"
        break
    fi
    sleep 1
done

# 3. Start the Backend Orchestrator & Web Dashboard on Render public PORT
echo "Starting BMU Backend Orchestrator & Dashboard on port ${PORT:-8000}..."
python -u semantic.py
