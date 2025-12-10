import pandas as pd
import numpy as np
from pypdf import PdfReader
import chromadb

def extract_text_from_pdf(file_path):
    """
    Extracts text from a pdf file
    
    Args:
    file_path(str): The path of the pdf file.
    
    Returns:
    str: The extracted text
    """
    
    text = []
    with open(file_path, "rb") as f:
        pdf = PdfReader(f)
        
        for page_num in range(pdf.get_num_pages()):
            page = pdf.get_page(page_num)
            text.append(page.extract_text())
            
        return "\n".join(text)

def load_chroma(file_name, collection_name, embedding_function):
    """
    Loads a document from pdf, generate embedding for the text, and stores it in ChromaDB collection
    
    Args:
    file_name(str): The path to the pdf file
    collection_name(str): The name of Chroma collection
    embedding_function(callable): The function to generate embeddings
    
    Returns:
    chroma.collection: The chroma collection with the generated embeddings
    """
    
    text = extract_text_from_pdf(file_name)
    paragraphs = text.split("\n\n")
    
    embeddings = [embedding_function(paragraph) for paragraph in paragraphs]
    data = {
        "text": paragraphs,
        "embeddings": embeddings
    }
    df = pd.DataFrame(data)
    
    collection = chromadb.Client().create_collection(collection_name)
    
    for ids, row in df.iterrows():
        collection.add(ids=[str(ids)], documents=[row["text"]], embeddings=[row["embeddings"]])
    
    return collection