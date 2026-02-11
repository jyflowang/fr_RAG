import chromadb
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent 
DB_PATH = str(BASE_DIR / "chroma_db")

client = chromadb.PersistentClient(path=DB_PATH)

colls = client.list_collections()
print(f"All collections: {[c.name for c in colls]}")

