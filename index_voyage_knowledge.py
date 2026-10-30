"""
BMU Voyage AI (1024-dim) Knowledge Base Indexer
================================================
Embeds verified BMU webpage records from bmu_knowledge_backup.json using
Voyage AI (voyage-4-large, 1024-dim) and stores them in ChromaDB Cloud
collection 'bmu_voyage_knowledge'.

Pacing:
- Batch size: 20 documents (~5,000 tokens, safely below 10,000 TPM limit).
- Inter-batch delay: 24.0 seconds (safely below 3 RPM limit).
- Pagination and resume-safe: skips already indexed documents.
- Optional --limit parameter for incremental verification.
"""

import argparse
import json
import os
import sys
import time
from dotenv import load_dotenv
import chromadb
from tqdm import tqdm

from voyage_client import voyage_client

# Force UTF-8 stdout
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(ENV_PATH)

BACKUP_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bmu_knowledge_backup.json")
COLLECTION_NAME = os.getenv("VOYAGE_COLLECTION_NAME", "bmu_voyage_knowledge")

def index_voyage_corpus(batch_size: int = 20, delay_seconds: float = 24.0, limit: int = None):
    if not os.path.exists(BACKUP_FILE):
        print(f"❌ Backup file not found: {BACKUP_FILE}")
        return

    if not voyage_client.is_configured:
        print("❌ Error: Voyage AI client not configured. Check VOYAGE_API_KEY in .env.")
        return

    print("🔌 Probing Voyage AI connection...")
    probe = voyage_client.test_connection()
    if probe.get("status") != "ok":
        print(f"❌ Voyage AI probe failed: {probe}")
        return
    print(f"✅ Voyage AI active: Model '{probe['model']}', Dimension {probe['dimension']}, Key: {probe['key_preview']}")

    api_key = os.getenv("CHROMA_API_KEY")
    tenant = os.getenv("CHROMA_TENANT")
    database = os.getenv("CHROMA_DATABASE")

    if not all([api_key, tenant, database]):
        print("❌ Error: ChromaDB Cloud credentials not configured in .env.")
        return

    print(f"🔌 Connecting to ChromaDB Cloud database '{database}'...")
    chroma_client = chromadb.CloudClient(
        api_key=api_key,
        tenant=tenant,
        database=database,
    )

    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "BMU website knowledge base indexed with Voyage AI voyage-4-large (1024-dim)"}
    )

    # Check already indexed records with pagination
    existing_count = collection.count()
    print(f"📦 Collection '{COLLECTION_NAME}' currently has {existing_count} records.")

    existing_ids = set()
    if existing_count > 0:
        offset = 0
        limit_fetch = 250
        while offset < existing_count:
            res = collection.get(limit=limit_fetch, offset=offset, include=[])
            fetched = res.get("ids", [])
            if not fetched:
                break
            existing_ids.update(fetched)
            offset += len(fetched)
        print(f"ℹ️ Found {len(existing_ids)} existing record IDs to skip.")

    with open(BACKUP_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_ids = data["ids"]
    all_docs = data["documents"]
    all_metas = data["metadatas"]
    total = len(all_ids)

    # Filter out already indexed
    records_to_index = [
        (doc_id, doc_text, meta)
        for doc_id, doc_text, meta in zip(all_ids, all_docs, all_metas)
        if doc_id not in existing_ids
    ]

    if limit and limit > 0:
        records_to_index = records_to_index[:limit]
        print(f"🎯 Limiting this run to {len(records_to_index)} records (for incremental verification).")

    print(f"📊 Total records in backup: {total}")
    print(f"🚀 Records to index in this run: {len(records_to_index)}")

    if not records_to_index:
        print("🎉 Target records are already indexed in ChromaDB Cloud!")
        return

    total_batches = (len(records_to_index) + batch_size - 1) // batch_size
    est_minutes = round(total_batches * (delay_seconds + 2) / 60, 1)
    print(f"⚡ Processing {len(records_to_index)} records across {total_batches} batches (Est: ~{est_minutes} min)...")

    pbar = tqdm(total=len(records_to_index), desc="Voyage AI Indexing")
    for b_idx in range(0, len(records_to_index), batch_size):
        batch = records_to_index[b_idx : b_idx + batch_size]
        b_ids = [item[0] for item in batch]
        b_texts = [item[1] for item in batch]
        b_metas = [item[2] for item in batch]

        # Generate embeddings with Voyage AI
        b_embeddings = voyage_client.embed_documents(b_texts, batch_size=batch_size)

        # Upload to ChromaDB Cloud
        collection.upsert(
            ids=b_ids,
            documents=b_texts,
            metadatas=b_metas,
            embeddings=b_embeddings,
        )

        pbar.update(len(batch))
        if b_idx + batch_size < len(records_to_index) and delay_seconds > 0:
            time.sleep(delay_seconds)

    pbar.close()
    final_count = collection.count()
    print(f"\n🎉 Progress saved! Collection '{COLLECTION_NAME}' now has {final_count} records.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BMU Voyage AI Indexer")
    parser.add_argument("--batch-size", type=int, default=20, help="Batch size (default: 20)")
    parser.add_argument("--delay", type=float, default=24.0, help="Seconds between batches (default: 24.0)")
    parser.add_argument("--limit", type=int, default=None, help="Max records to index in this run")
    args = parser.parse_args()

    index_voyage_corpus(batch_size=args.batch_size, delay_seconds=args.delay, limit=args.limit)
