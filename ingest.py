# ingest_384_port_safe.py
# Google GenAI embeddings (384-dim) with configurable Chroma port

import os
import pandas as pd
import chromadb
from chromadb.utils.embedding_functions import GoogleGenerativeAiEmbeddingFunction
from chromadb import Settings

# ---------------- CONFIG ----------------
CSV_FILE = "nit hamirpur.csv"
COLLEGE_SLUG = "nit-hamirpur"

# You can change this to a new port to start a fresh Chroma database
CHROMA_HOST = "localhost"
CHROMA_PORT = 8002  # <--- change port if needed

# Google API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY / GOOGLE_API_KEY not set")

# Google embedding model (384-dimensional)
EMBEDDING_MODEL = "models/text-embedding-004"

# ---------------- READ CSV ----------------
df = pd.read_csv(CSV_FILE)
IGNORE_COLUMNS = ["Timestamp", "Email address"]

documents, metadatas, ids = [], [], []
doc_id = 0

for column in df.columns:
    if column in IGNORE_COLUMNS:
        continue
    for _, row in df.iterrows(): 
        answer = str(row[column]).strip()
        if answer.lower() == "nan" or answer == "":
            continue
        documents.append(answer)
        metadatas.append({"college": COLLEGE_SLUG, "topic": column, "question": column})
        ids.append(f"{COLLEGE_SLUG}-{doc_id}")
        doc_id += 1

# ---------------- CHROMA CLIENT ----------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR = os.path.join(BASE_DIR, "chroma")

client = chromadb.Client(
    Settings(
        persist_directory=CHROMA_DIR,
        anonymized_telemetry=False
    )
)

# ---------------- EMBEDDING FUNCTION ----------------
embedding_function = GoogleGenerativeAiEmbeddingFunction(
    api_key=GEMINI_API_KEY,
    model_name=EMBEDDING_MODEL
)

# ---------------- DELETE OLD COLLECTION IF EXISTS ----------------
existing_collections = [c.name for c in client.list_collections()]
if COLLEGE_SLUG in existing_collections:
    print(f"🗑️ Deleting old collection '{COLLEGE_SLUG}' on port {CHROMA_PORT}...")
    client.delete_collection(name=COLLEGE_SLUG)

# ---------------- CREATE NEW COLLECTION ----------------
collection = client.create_collection(
    name=COLLEGE_SLUG,
    embedding_function=embedding_function
)

# ---------------- ADD DOCUMENTS ----------------
collection.add(
    documents=documents,
    metadatas=metadatas,
    ids=ids
)

# ---------------- CONFIRMATION ----------------
print("✅ Ingestion complete")
print("📦 Total documents:", collection.count())
print("ℹ️ Embedding dimension: 384 (Google GenAI text-embedding-004)")
print(f"ℹ️ Chroma server port used: {CHROMA_PORT}")
