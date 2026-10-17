import os
import sys
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from tqdm import tqdm

# ==============================
# 🔐 Load ENV
# ==============================
print("🔐 Loading environment variables...")
load_dotenv()

CHROMA_HOST = os.getenv("CHROMA_HOST")
CHROMA_API_KEY = os.getenv("CHROMA_API_KEY")
CHROMA_TENANT = os.getenv("CHROMA_TENANT")
CHROMA_DATABASE = os.getenv("CHROMA_DATABASE")

print("ENV CHECK →", CHROMA_HOST, CHROMA_API_KEY is not None)

if not all([CHROMA_HOST, CHROMA_API_KEY, CHROMA_TENANT, CHROMA_DATABASE]):
    raise ValueError("❌ One or more ENV variables are missing!")

# ==============================
# 🔌 Connect to Chroma
# ==============================
print("🔌 Connecting to Chroma DB...")

client =  chromadb.CloudClient(  
    api_key=os.getenv("CHROMA_API_KEY"),
    tenant=os.getenv("CHROMA_TENANT"),
    database=os.getenv("CHROMA_DATABASE"),
)
print("✅ Connected to Chroma")

# ==============================
# 📁 Load CSV
# ==============================
print("📁 Loading CSV...")

csv_path = os.getenv("CSV_PATH", "bmu_summaries.csv")
if not os.path.exists(csv_path):
    print(f"⚠️ CSV file not found at: {csv_path}")
    print("ℹ️ Note: If you have bmu_knowledge_backup.json, use 'restore_knowledge_base.py' instead.")
    sys.exit(0)

df = pd.read_csv(csv_path)
print(f"✅ Loaded {len(df)} rows")


# ==============================
# 🧠 Load Model
# ==============================
print("🧠 Loading embedding model (first time may take time)...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("✅ Model loaded")

# ==============================
# 📦 Create Collection
# ==============================
print("📦 Creating / loading collection...")

collection = client.get_or_create_collection(name="bmu_summaries")

print("✅ Collection ready")

# ==============================
# 🔄 Batch Processing
# ==============================
batch_size = 32  # change if needed

print("🚀 Starting embedding + upload...")

for i in tqdm(range(0, len(df), batch_size), desc="Processing", file=sys.stdout):
    batch = df.iloc[i:i+batch_size]

    documents = batch["Summary"].fillna("").tolist()
    urls = batch["URL"].fillna("").tolist()

    # 🧠 Encode
    embeddings = model.encode(documents).tolist()

    # 🆔 IDs
    ids = [f"id_{j}" for j in range(i, i + len(documents))]

    # 📎 Metadata
    metadatas = [{"url": url} for url in urls]

    # 📤 Upload
    collection.add(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )

print("✅ All data successfully stored in Chroma DB!")