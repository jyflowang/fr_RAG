import os
from dotenv import load_dotenv
from pathlib import Path
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from langchain.tools import tool

load_dotenv()
Settings.llm = GoogleGenAI(model="models/gemini-2.5-flash",
                           api_key = os.getenv("GOOGLE_API_KEY"))
Settings.embed_model = GoogleGenAIEmbedding(model_name="models/gemini-embedding-001",
                                            api_key = os.getenv("GOOGLE_API_KEY"))

BASE_DIR = Path(__file__).resolve().parent.parent 
DB_PATH = str(BASE_DIR / "chroma_db")

db = chromadb.PersistentClient(path=DB_PATH)
chroma_collection = db.get_or_create_collection("financial_reports")

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

index = VectorStoreIndex.from_vector_store(
    vector_store,
    embed_model=Settings.embed_model 
)

fast_hybrid_query_engine = index.as_query_engine(
    vector_store_query_mode="hybrid", 
    similarity_top_k=5,        
    sparse_top_k=5             
)

def get_raw_response(query: str):
    """
    Get the raw response from the fast hybrid query engine
    """
    return fast_hybrid_query_engine.query(query)

@tool
def fast_search_engine(query: str) -> str:
    """
    Search and answer the basic information according the financial report stored
    
    Args:
    query(str): The question that users asked
    
    Returns:
    str: The answer that agent returns
    """
    
    response = fast_hybrid_query_engine.query(query)
  #  return str(response.response)
    content = str(response.response).strip()

    if not content or "empty response" in content.lower():
        # Provide a structured signal that the Oracle can recognize
        return "ERROR_CODE: DATA_NOT_FOUND. The database search yielded no relevant snippets."

    return content