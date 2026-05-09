import os
import ollama
from PIL import Image
from langchain_qdrant import QdrantVectorStore
from langchain_ollama import OllamaLLM
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient, models

# --- SETUP ---
client = QdrantClient(url="http://localhost:6333")
COLLECTION_NAME = "advanced_vision_rag"
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/clip-ViT-B-32")
vision_model = OllamaLLM(model="moondream")

# Ensure on-disk storage for 16GB RAM efficiency
if not client.collection_exists(COLLECTION_NAME):
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=512, distance=models.Distance.COSINE, on_disk=True),
    )

vector_store = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME, embedding=embeddings)
import base64

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
# --- ADVANCED INGESTION ---
def ingest_with_ai_descriptions(folder_path):
    """
    This 'enriches' the vector store by asking Ollama to describe 
    the image before we store it.
    """
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.png')):
            path = os.path.abspath(os.path.join(folder_path, filename))
            b64_img = encode_image(path)

            # 1. Ask Moondream to 'see' and 'describe' the image first
            print(f"AI is analyzing: {filename}...")
            description = vision_model.invoke(
                "Describe this image in detail for a search index. Mention objects, colors, and mood.",
                images=[b64_img]
            )
            print(filename,description)
            # 2. Store the description as 'page_content' and the path in metadata
            # This makes the image searchable by the AI's own words!
            doc = Document(
                page_content=description,
                metadata={"path": path, "filename": filename}
            )
            vector_store.add_documents([doc])


# --- ADVANCED QUERYING ---
def deep_image_query(user_query):
    """
    Finds the image and then uses the description to verify the answer.
    """
    # k=2 to find the top 2 matches
    docs = vector_store.similarity_search(user_query, k=2)
    
    if not docs:
        return "Nothing found."

    results = []
    for doc in docs:
        path = doc.metadata['path']
        b64_img = encode_image(path)
        # Verification Step: Ask Ollama if this retrieved image truly matches the user intent
        verification = vision_model.invoke(
            f"User is looking for: '{user_query}'. Does this image match? Why? Path: {path}",
            images=[b64_img]
        )
        results.append({
            "file": doc.metadata['filename'],
            "reasoning": verification
        })
    
    return results

# Example Usage:
ingest_with_ai_descriptions("C://ml//code//images//imagelangchain//my_images")
print(deep_image_query("Something that looks fox"))
