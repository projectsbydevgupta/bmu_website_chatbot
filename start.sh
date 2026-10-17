#!/usr/bin/env bash
# ==============================================================================
# BMU Website Chatbot - Cloud Production Entrypoint (Render / Linux Container)
# ==============================================================================

# Memory optimizations for cloud container (single-threaded PyTorch CPU)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

# Start the Unified Backend Orchestrator & Dashboard on Render public PORT
echo "Starting BMU Backend Orchestrator & Dashboard on port ${PORT:-8000}..."
python -u semantic.py

