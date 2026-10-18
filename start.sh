#!/usr/bin/env bash
# ==============================================================================
# BMU Website Chatbot - Cloud Production Entrypoint (Render / Linux Container)
# ==============================================================================

# Memory & latency optimizations for cloud container
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export RETRIEVAL_SERVICE_URL=""
export ENABLE_LIVE_SCRAPING=false

# Pre-warm lightweight ONNX MiniLM embedder cache during boot (< 90MB RAM, Zero PyTorch)
echo "Pre-warming lightweight ONNX MiniLM embedder into memory..."
python -c "from retrieval_service import get_embedder; get_embedder().encode('bmu query')" || true


# Start the Unified Backend Orchestrator & Dashboard on Render public PORT
echo "Starting BMU Backend Orchestrator & Dashboard on port ${PORT:-8000}..."
python -u semantic.py


