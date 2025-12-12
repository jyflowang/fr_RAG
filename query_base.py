from pypdf import PdfReader
from helper_utils import extract_text_from_pdf, load_chroma
import chromadb
from sentence_transformers import SentenceTransformer

# The base scenario uses the function I defined in helper_utils.py. The file is roughly chunked
# by paragraph, which is not suitable for the financial file containing tables such as 10Q. I used
# MiniLM to generate embeddings for the file and query.

# generate embedding and load chromadb collection
model = SentenceTransformer("all-MiniLM-L6-v2")
report_collection = load_chroma("data/goog_2025Q3.pdf", "report_collection", 
                                 model.encode)

test_query = "What is the 2025 Q3 revenue?"
query_embedding = model.encode(test_query)
results = report_collection.query(query_embeddings = [query_embedding],
                                   n_results = 2)
retrieved_results = results["documents"][0]

# Print all the returned chunks
for i, doc in enumerate(retrieved_results):
    print(f"\n--- Chunk{i}-----\n")
    print(doc)


