from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION_NAME, VECTOR_SIZE
from tqdm import tqdm
from database.base import BaseVectorDB

def chunked_iterable(iterable, batch_size):
    """Chia iterable thành các batch có kích thước size"""
    for i in range(0, len(iterable), batch_size):
        yield iterable[i:i + batch_size]

class QDrantDB(BaseVectorDB):
    def __init__(self, url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=600):
        super().__init__()
        try:
            self.client = QdrantClient(
                url=url, 
                api_key=api_key, 
                timeout=timeout
            )
            print(f"🟢 Connected to Qdrant at {url}")
        except Exception as e:
            print(f"❌ Failed to connect to Qdrant: {e}")
            raise
        
    def collection_exists(self, collection_name: str) -> bool:
        try:
            self.client.get_collection(collection_name)
            return True
        except Exception:
            return False
        
    def create_collection(
            self, 
            vector_size: int = VECTOR_SIZE,
            distance: Distance = Distance.COSINE,
            recreate: bool = False,
            collection_name = QDRANT_COLLECTION_NAME
        ):
        """
        Tạo Qdrant collection.
        
        Args:
            vector_size (int): Kích thước vector
            distance (Distance): Distance metric (COSINE, EUCLID, DOT)
            recreate (bool): Xóa collection cũ nếu tồn tại
        """
        try:
            if recreate and self.collection_exists(collection_name):
                print(f"🗑️ Deleting existing collection '{collection_name}'")
                self.delete_collection()
            
            if not self.collection_exists(collection_name):
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=vector_size, 
                        distance=distance
                    )
                )
                print(f"✅ Created collection '{collection_name}' (size={vector_size}, distance={distance.name})")
            else:
                print(f"⚠️ Collection '{collection_name}' already exists")
                
        except Exception as e:
            print(f"❌ Error creating collection: {e}")
            raise
    
    def delete_collection(self, collection_name=QDRANT_COLLECTION_NAME):
        """Xóa collection"""
        try:
            if self.collection_exists():
                self.client.delete_collection(collection_name)
                print(f"🗑️ Deleted collection '{collection_name}'")
            else:
                print(f"Collection '{collection_name}' does not exist")
        except Exception as e:
            print(f"❌ Error deleting collection: {e}")
            raise

    def upsert_points(self, points, batch_size=500, max_workers=4, collection_name=QDRANT_COLLECTION_NAME):
    # Chuẩn hóa points thành PointStruct
        qdrant_points = [
            PointStruct(
                id=point['id'],
                vector=point["embedding"],
                payload={
                    "source_file": point["source_file"],
                    "category": point["category"],
                    "section": point["section"],
                    "content": point["content"],
                },
            )
            for point in points
        ]

        batches = list(chunked_iterable(qdrant_points, batch_size))
        total_batches = len(batches)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.client.upsert,
                                    collection_name=collection_name,
                                    points=batch)
                    for batch in batches]

            for _ in tqdm(as_completed(futures), total=total_batches,
                        desc="📤 Uploading to Qdrant"):
                pass  # chỉ để update progress bar

        print(f"✅ Đã upsert {len(qdrant_points)} points vào collection '{collection_name}'")

    def search(self, query_vector, collection_name, limit=5, with_vectors=True):
        return self.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=limit,
            with_vectors=with_vectors
        )
    
if __name__ == '__main__':
    import json
    db = QDrantDB()
    db.create_collection()
    with open('../chunks.json', 'r') as f:
        points = json.load(f)

    db.upsert_points(points)
