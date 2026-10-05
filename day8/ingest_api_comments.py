import requests
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from langchain_ollama import OllamaEmbeddings

# Configuration
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "placeholder_comments"

def fetch_and_ingest():
    # 1. Fetch data from an API
    # Example: A public API for crypto news or space news
    print("--- Fetching data from API ---")
    url = "https://jsonplaceholder.typicode.com/comments"
    response = requests.get(url)
    data = response.json()

    # 2. Convert API JSON objects to LangChain Documents
    documents = []
    for item in data:
        # Create the text content for the vector
        content = f"PostID: {item['postId']}\nName: {item['name']}\Email: {item['email']}\Body: {item['body']}"
        
        # Keep original URL as metadata so the AI can cite its source
        metadata = {"source": "https://jsonplaceholder.typicode.com/comments"}
        
        doc = Document(page_content=content, metadata=metadata)
        documents.append(doc)

    # 3. Hybrid Indexing to Qdrant
    print(f"--- Indexing {len(documents)} API items to Qdrant ---")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

    QdrantVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        sparse_embedding=sparse_embeddings,
        url=QDRANT_URL,
        collection_name=COLLECTION_NAME,
        retrieval_mode=RetrievalMode.HYBRID,
        force_recreate=True
    )
    print("SUCCESS: API data is now searchable in Qdrant!")

if __name__ == "__main__":
    fetch_and_ingest()