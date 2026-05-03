import os
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from langchain_ollama import OllamaEmbeddings

# Configuration
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "corporate_manual_hybrid"

def ingest_data():
    print("--- 1. Loading Word Document ---")
    loader = Docx2txtLoader("C:\ml\code\day9\my_word_files\HR_Policy.docx")
    data = loader.load()

    print("--- 2. Splitting into Semantic Chunks ---")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = text_splitter.split_documents(data)

    print("--- 3. Initializing Hybrid Embeddings ---")
    # Dense (Meaning)
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    # Sparse (Exact Keywords - BM25)
    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

    print(f"--- 4. Pushing to Docker Qdrant at {QDRANT_URL} ---")
    QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        sparse_embedding=sparse_embeddings,
        url=QDRANT_URL,
        collection_name=COLLECTION_NAME,
        retrieval_mode=RetrievalMode.HYBRID,
        force_recreate=True  # Fresh index every time we ingest
    )
    print("\nSUCCESS: Knowledge base is now live in Docker!")

if __name__ == "__main__":
    ingest_data()