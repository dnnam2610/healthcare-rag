import os
import json
import numpy as np
from tqdm import tqdm
from typing import List
import pandas as pd

class MedicalFilterer:
    def __init__(self, embedding_model, 
                 embedding_path='/content/drive/MyDrive/tahospital_data/training_router_data/flag_routes_embedding.json'):
        self.embedding_path = embedding_path
        self.medicalEmbedding = None
        self.embedding = embedding_model  # SentenceTransformer hoặc model encode khác
        
        if os.path.exists(self.embedding_path):
            print(f'🔵 Loading embeddings from {self.embedding_path}...')
            self._load_embeddings()
            print('🟢 Loaded embeddings successfully!')
        else:
            raise FileNotFoundError(f'Embedding file not found at {self.embedding_path}')

    def _load_embeddings(self):
        """Load embeddings from JSON file"""
        with open(self.embedding_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for route_name, embeddings in data['routes'].items():
            if route_name == 'medical':
                self.medicalEmbedding = np.array(embeddings)
                break
        else:
            raise ValueError('🛑 Not found medical route in the embeddings file.')

    def filter(self, queries: List[str], batch_size: int = 256, output_path: str = "/content/drive/MyDrive/tahospital_data/training_router_data/similarity_results.csv"):
        """Tính độ tương đồng cosine giữa queries và bộ medical, sau đó lưu kết quả"""
        all_results = []
        num_queries = len(queries)
        
        print(f"⚙️ Encoding {num_queries} queries in batches of {batch_size}...")
        for i in tqdm(range(0, num_queries, batch_size), desc="🔹 Processing batches"):
            batch_queries = queries[i:i + batch_size]
            
            # Encode batch
            queriesEmbedding = self.embedding.encode(batch_queries)
            
            # Normalize medical embedding (đảm bảo unit vector)
            medical_norm = self.medicalEmbedding / np.linalg.norm(self.medicalEmbedding, axis=1, keepdims=True)
            
            # Cosine similarity: (num_medical, dim) ⋅ (dim, batch) => (num_medical, batch)
            similarities = np.dot(medical_norm, queriesEmbedding.T)
            
            # Lấy max similarity cho từng query (có thể thay = mean nếu muốn)
            max_similarities = np.max(similarities, axis=0)
            mean_similarities = np.mean(similarities, axis=0)

            # Lưu kết quả batch
            for q, max_s, mean_s in zip(batch_queries, max_similarities, mean_similarities):
                all_results.append({
                    "query": q,
                    "max_similarity": float(max_s),
                    "mean_similarity": float(mean_s)
                })

        # Chuyển sang DataFrame để dễ xử lý
        df = pd.DataFrame(all_results)
        df.sort_values(by="max_similarity", ascending=False, inplace=True)
        
        # Lưu ra file CSV
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"💾 Saved results to {output_path}")
        
        return df
