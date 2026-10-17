"""
BMU Knowledge Base Backup Utility
=================================
Exports all documents, metadata, and embeddings from ChromaDB Cloud
to a local JSON file (bmu_knowledge_backup.json) so you never lose
the indexed website dataset when migrating to your own Chroma credentials.
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

def backup_collection(collection_name: str = "bmu_summaries", page_size: int = 150):
    api_key = os.getenv("CHROMA_API_KEY")
    tenant = os.getenv("CHROMA_TENANT")
    database = os.getenv("CHROMA_DATABASE")

    if not all([api_key, tenant, database]):
        print("❌ Error: ChromaDB credentials not found in .env")
        return

    print(f"🔌 Connecting to ChromaDB Cloud ({database})...")
    client = chromadb.CloudClient(
        api_key=api_key,
        tenant=tenant,
        database=database,
    )

    try:
        coll = client.get_collection(name=collection_name)
    except Exception as e:
        print(f"❌ Could not find collection '{collection_name}': {e}")
        return

    total = coll.count()
    print(f"📦 Found collection '{collection_name}' with {total} records.")

    if total == 0:
        print("⚠️ Collection is empty. Nothing to export.")
        return

    all_ids = []
    all_docs = []
    all_metas = []
    all_embs = []

    print(f"⏳ Downloading in batches of {page_size}...")
    for offset in tqdm(range(0, total, page_size), desc="Exporting", file=sys.stdout):
        batch = coll.get(
            limit=page_size,
            offset=offset,
            include=["documents", "metadatas", "embeddings"],
        )
        all_ids.extend(batch["ids"])
        all_docs.extend(batch["documents"])
        all_metas.extend(batch["metadatas"])
        if batch.get("embeddings") is not None:
            for emb in batch["embeddings"]:
                if hasattr(emb, "tolist"):
                    all_embs.append(emb.tolist())
                else:
                    all_embs.append(list(emb))

    backup_payload = {
        "collection_name": collection_name,
        "count": len(all_ids),
        "ids": all_ids,
        "documents": all_docs,
        "metadatas": all_metas,
        "embeddings": all_embs,
    }

    print(f"💾 Saving to {BACKUP_FILE}...")
    with open(BACKUP_FILE, "w", encoding="utf-8") as f:
        json.dump(backup_payload, f)

    size_mb = os.path.getsize(BACKUP_FILE) / (1024 * 1024)
    print(f"✅ Backup complete! Successfully exported {len(all_ids)} records ({size_mb:.2f} MB) to:")
    print(f"   {BACKUP_FILE}")

if __name__ == "__main__":
    backup_collection()
