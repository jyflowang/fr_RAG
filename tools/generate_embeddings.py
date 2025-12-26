from docling.document_converter import DocumentConverter
from pathlib import Path
from llama_index.core import Document
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.node_parser import MarkdownElementNodeParser
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
import os
from dotenv import load_dotenv

def parse_file(file_name: str):
    """
    Parse the pdf file using Docling
    
    Args:
    file_name(str): The name of the file, should be written like "GOOG_2025Q3.pdf"
    
    Returns:
    Document: LlamaIndex Document, including the parsed text and meta data
    """
    
    # Get the path 
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent 
    source = project_root / "data" / file_name
    
    if not source.exists():
        raise FileNotFoundError(f"File not existed: {source}")
    
    converter = DocumentConverter()
    result = converter.convert(source)

    content_md = result.document.export_to_markdown()
    file_name = result.input.file.name

    meta_data = {
        "file_name": file_name,
        "company": file_name[:4], 
        "year": file_name[5:9]
    }
    doc = Document(text=content_md, metadata=meta_data, doc_id=file_name)
    
    return doc

def generate_store_embeddings(doc):
    
    BASE_DIR = Path(__file__).resolve().parent.parent 
    DB_PATH = str(BASE_DIR / "chroma_db")

    db = chromadb.PersistentClient(path=DB_PATH)
    chroma_collection = db.get_or_create_collection("financial_reports")

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Generate the embedding
    load_dotenv(override=True)
    llm = GoogleGenAI(
        model="gemini-2.5-flash",
        api_key=os.environ.get("GOOGLE_API_KEY")
    )

    embed_model = GoogleGenAIEmbedding(model_name="models/text-embedding-004")

    node_parser = MarkdownElementNodeParser(
        llm=llm, 
        num_workers=4  
    )

    nodes = node_parser.get_nodes_from_documents([doc])
    index = VectorStoreIndex(
        nodes, 
        embed_model=embed_model,
        storage_context=storage_context,
        show_progress=True
    )

    index.storage_context.persist(persist_dir=DB_PATH)
    print(f"Successfully stored {len(nodes)} of nodes in ChromaDB.")

    
    
