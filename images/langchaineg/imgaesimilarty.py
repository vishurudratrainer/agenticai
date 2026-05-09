import os
from PIL import Image
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

# 1. Setup
client = QdrantClient(url="http://localhost:6333")
COLLECTION_NAME = "similarity_store"
model = SentenceTransformer('clip-ViT-B-32')
embed_model = SentenceTransformer('clip-ViT-B-32')

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


def find_visual_lookalikes(input_image_path, limit=3):
    """
    Finds images in the store that look like the input_image_path.
    """
    # Load and encode the 'Query Image'
    query_img = Image.open(input_image_path)
    query_vector = model.encode(query_img).tolist()

    # Search Qdrant for the nearest neighbors
    search_results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        score_threshold=0.85, # Only return results that are 85%+ similar
        limit=limit,
        with_payload=True
    ).points

    print(f"--- Top {limit} Visual Matches for '{os.path.basename(input_image_path)}' ---")
    for res in search_results:
        score_percent = round(res.score * 100, 2)
        print(f"Match: {res.payload['name']} | Confidence: {score_percent}%")
        # print(f"Path: {res.payload['path']}")

# --- EXECUTION ---
# 1. First, index your folder (use the index_images function from the previous example)
index_images("C://ml//code//images//imagelangchain//my_images")

# 2. Query with a new image
find_visual_lookalikes("C://ml//code//images//my_images//img3.jpg")