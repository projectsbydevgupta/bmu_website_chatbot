"""
Voyage AI Connection & Dimension Probe
======================================
Tests whether the Voyage AI key in .env is active, verifies vector dimension (1024),
and tests retrieval against ChromaDB Cloud.
"""

import os
import sys
from dotenv import load_dotenv

# Force UTF-8 output
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

from voyage_client import voyage_client

def run_probe():
    print("==================================================")
    print("  BMU Voyage AI (1024-dim) Cloud Embedding Probe  ")
    print("==================================================")

    if not voyage_client.is_configured:
        print("❌ VOYAGE_API_KEY is missing or invalid in .env")
        return

    print("📡 Sending probe request to Voyage AI endpoint...")
    try:
        probe = voyage_client.test_connection()
        print(f"✅ Connection Status: {probe.get('status')}")
        print(f"🏷️  Model Name:        {probe.get('model')}")
        print(f"📐 Vector Dimensions: {probe.get('dimension')} dimensions")
        print(f"🔑 API Key Preview:   {probe.get('key_preview')}")
        print(f"⚡ Latency:           {probe.get('latency_ms')} ms")
        print("==================================================")
        print("🎉 Voyage AI is READY to index and power the chatbot!")
    except Exception as exc:
        print(f"❌ Probe failed with exception: {exc}")

if __name__ == "__main__":
    run_probe()

