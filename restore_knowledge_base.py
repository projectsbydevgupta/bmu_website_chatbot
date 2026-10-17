"""
BMU Knowledge Base Restore Utility
==================================
Restores documents, metadata, and embeddings from bmu_knowledge_backup.json
into your ChromaDB Cloud account (or local Chroma instance) defined in .env.
"""

import json
import os
import sys
from dotenv import load_dotenv
import chromadb
from tqdm import tqdm

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding="utf-8")

ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(ENV_PATH)

BACKUP_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bmu_knowledge_backup.json")

def restore_collection(collection_name: str = "bmu_summaries", batch_size: int = 64):
    if not os.path.exists(BACKUP_FILE):
        print(f"❌ Backup file not found: {BACKUP_FILE}")
        return

    api_key = os.getenv("CHROMA_API_KEY")
    tenant = os.getenv("CHROMA_TENANT")
    database = os.getenv("CHROMA_DATABASE")

    if not all([api_key, tenant, database]):
        print("❌ Error: ChromaDB credentials not configured in .env")
        print("   Please set CHROMA_API_KEY, CHROMA_TENANT, and CHROMA_DATABASE in .env first.")
        return

    print(f"📖 Reading backup from {BACKUP_FILE}...")
    with open(BACKUP_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    ids = data["ids"]
    docs = data["documents"]
    metas = data["metadatas"]
    embeddings = data.get("embeddings")

    total = len(ids)
    print(f"📦 Loaded backup with {total} records.")

    print(f"🔌 Connecting to destination ChromaDB ({database})...")
    client = chromadb.CloudClient(
        api_key=api_key,
        tenant=tenant,
        database=database,
    )

    coll = client.get_or_create_collection(name=collection_name)
    existing_count = coll.count()
    print(f"ℹ️ Collection '{collection_name}' currently has {existing_count} records.")

    print(f"🚀 Uploading in batches of {batch_size}...")
    for i in tqdm(range(0, total, batch_size), desc="Restoring", file=sys.stdout):
        b_ids = ids[i : i + batch_size]
        b_docs = docs[i : i + batch_size]
        b_metas = metas[i : i + batch_size]

        if embeddings and len(embeddings) == total:
            b_emb = embeddings[i : i + batch_size]
            coll.upsert(ids=b_ids, documents=b_docs, metadatas=b_metas, embeddings=b_emb)
        else:
            coll.upsert(ids=b_ids, documents=b_docs, metadatas=b_metas)

    final_count = coll.count()
    print(f"✅ Restore successful! Collection '{collection_name}' now contains {final_count} records.")

if __name__ == "__main__":
    restore_collection()
