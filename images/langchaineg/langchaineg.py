import os
from PIL import Image
from langchain_qdrant import QdrantVectorStore
from langchain_ollama import OllamaLLM
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient, models

# --- 1. SETUP ---
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "langchain_tiny_store"

# Smallest Embedding Model (CLIP via HuggingFace)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/clip-ViT-B-32")

# Smallest Vision Model (Ollama)
llm = OllamaLLM(model="moondream")

# --- 2. INITIALIZE VECTOR STORE ---
# Ensuring on_disk storage to save your 16GB RAM
client = QdrantClient(url=QDRANT_URL)
if not client.collection_exists(COLLECTION_NAME):
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=512, distance=models.Distance.COSINE, on_disk=True),
    )

vector_store = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings,
)

# --- 3. INDEX IMAGES ---
def add_images_to_langchain(folder_path):
    docs = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.png')):
            full_path = os.path.abspath(os.path.join(folder_path, filename))
            # In LangChain, we store the text description or path in page_content
            # and the rest in metadata
            doc = Document(
                page_content=f"Image file: {filename}",
                metadata={"path": full_path}
            )
            docs.append(doc)
    
    vector_store.add_documents(docs)
    print(f"Added {len(docs)} images to LangChain VectorStore.")

# --- 4. RETRIEVE & EXPLAIN ---
def search_and_explain(query_text):
    # Retrieve the top similar image document
    results = vector_store.similarity_search(query_text, k=1)
    
    if not results:
        return "No match found."
    
    img_path = results[0].metadata['path']
    
    # LangChain multi-modal invoke
    # Note: We pass the image path directly to Ollama
    print(f"Retrieved: {img_path}")
    
    # Using LangChain's Ollama interface to query the model
    # We bind the image to the LLM call
    response = llm.invoke(
        f"Briefly explain this image: {img_path}", 
        images=[img_path]
    )
    return response

# Example execution
add_images_to_langchain("C://ml//code//images//imagelangchain//my_images")
print(search_and_explain("A picture of a fox"))