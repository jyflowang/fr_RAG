from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# This file incorporated langchain framework and utilized very basic methods. For embeddings,
# I still used miniLM, the returned result is better than the base query scenario.
file_path = "data/goog_2025Q3.pdf"
loader = PyPDFLoader(file_path)

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)

all_splits = text_splitter.split_documents(docs)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store = Chroma(
    collection_name="fr_collection",
    embedding_function=embeddings
)
ids = vector_store.add_documents(documents=all_splits)

test_query = "What is the revenue for 2025Q3"
results = vector_store.similarity_search_with_score(test_query, k=4)

for i, result in enumerate(results):
    print(f"\n----The returned result {i} is: \n")
    print(result[0])
    print(f"\n---The score is: {result[1]}\n")