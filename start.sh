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

# Pre-warm SentenceTransformer cache during boot so first user query is instant (< 3s)
echo "Pre-warming embedding model cache into memory..."
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" || true

# Start the Unified Backend Orchestrator & Dashboard on Render public PORT
echo "Starting BMU Backend Orchestrator & Dashboard on port ${PORT:-8000}..."
python -u semantic.py


