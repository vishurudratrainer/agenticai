import os
import ollama
from PIL import Image
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from io import BytesIO
import base64
# --- CONFIGURATION ---
client = QdrantClient(url="http://localhost:6333")
COLLECTION_NAME = "my_tiny_store"
IMAGE_DIR = "C://ml//code//images//my_images"  # Folder containing your .jpg or .png files

# Load the smallest embedding model (CLIP)
embed_model = SentenceTransformer('clip-ViT-B-32')

# --- 1. SETUP ON-DISK COLLECTION ---
if not client.collection_exists(COLLECTION_NAME):
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=512, 
            distance=models.Distance.COSINE,
            on_disk=True  # Vectors stay on disk, NOT in your 16GB RAM
        )
    )

# --- 2. BULK UPLOAD (Index Images) ---
def index_images(folder_path):
    points = []
    for idx, filename in enumerate(os.listdir(folder_path)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(folder_path, filename)
            img = Image.open(path)
            
            # Create vector
            vector = embed_model.encode(img).tolist()
            
            # Prepare Point
            points.append(models.PointStruct(
                id=idx,
                vector=vector,
                payload={"path": os.path.abspath(path), "name": filename}
            ))
    
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"Indexed {len(points)} images on disk.")

# --- 3. QUERY & EXPLAIN ---
def find_and_explain(search_text):
    # 1. Search Logic
    query_vector = embed_model.encode(search_text).tolist()
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=1
    ).points
    
    if not results:
        return "Qdrant found nothing."

    img_path = os.path.normpath(results[0].payload['path'])
    print(f"DEBUG: Found {results[0].payload['name']} at {img_path}")

    # 2. Encode Image to Base64 (The Windows Fix)
    with open(img_path, "rb") as img_file:
        base64_image = base64.b64encode(img_file.read()).decode('utf-8')

    # 3. Call Ollama with specific options to force a response
    print("DEBUG: Sending to Moondream (Waiting for AI)...")
    try:
        response = ollama.generate(
            model='moondream',
            prompt='Describe this image briefly.',
            images=[base64_image],
            options={
                'num_predict': 50,  # Limit length to save RAM
                'temperature': 0    # Make it deterministic
            }
        )
        
        output = response.get('response', '').strip()
        
        if not output:
            return "AI returned an empty string. Try restarting Ollama."
            
        return output

    except Exception as e:
        return f"Ollama Error: {str(e)}"
# --- EXECUTION ---
index_images(IMAGE_DIR)
print(find_and_explain("A photo of zebra"))