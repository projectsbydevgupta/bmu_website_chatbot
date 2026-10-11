import chromadb
import os
from dotenv import load_dotenv

print("🔌 Connecting...")

load_dotenv()

client =  chromadb.CloudClient(  
    api_key=os.getenv("CHROMA_API_KEY"),
    tenant=os.getenv("CHROMA_TENANT"),
    database=os.getenv("CHROMA_DATABASE"),
)

print("✅ Connected!")
print(client.heartbeat())