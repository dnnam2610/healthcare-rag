import os
import numpy as np
import json
from tqdm import tqdm
import faiss
import time

class SemanticRouter():
    def __init__(self, embedding, routes=None, batch=64, save_path='data/routingEmbedddings/bgem3_routing_embedding_1000.json'):
        self.routes = routes
        self.embedding = embedding
        self.routesEmbedding = {}
        self.save_path = save_path
        
        print(os.path.exists(self.save_path))
        # Kiểm tra xem có file embeddings đã lưu không
        if os.path.exists(self.save_path):
            print(f'🔵 Loading embeddings from {self.save_path}...')
            self._load_embeddings()
            
            print('🟢 Loaded embeddings successfully!')
            self._build_faiss_index()
        else:
            print('🔵 Generating new embeddings...')
            self._generate_embeddings(batch)
            self._save_embeddings()
            print('🟢 Encoded and saved successfully!')


    def _build_faiss_index(self):
        """
        Build FAISS index from self.routesEmbedding
        """
        all_embeddings = []
        self.id2route = []

        for route_name, emb in self.routesEmbedding.items():
            if emb is None or len(emb) == 0:
                continue

            # float32 + normalize (FAISS dùng inner product)
            emb = emb.astype("float32")
            faiss.normalize_L2(emb)

            all_embeddings.append(emb)
            self.id2route.extend([route_name] * len(emb))

        if not all_embeddings:
            raise RuntimeError("No embeddings to build FAISS index")

        all_embeddings = np.vstack(all_embeddings)

        dim = all_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # cosine similarity
        self.index.add(all_embeddings)

        print(f"🟢 FAISS index built with {self.index.ntotal} vectors")

    def _generate_embeddings(self, batch):
        """Generate embeddings for all routes"""
        for route in self.routes:
            embeddings = []
            progress_bar = tqdm(range(0, len(route.samples), batch), 
                              desc=f'Encoding {route.name}...')
            for i in progress_bar:
                batch_samples = route.samples[i:i+batch]
                batch_embeddings = self.embedding.encode(batch_samples)
                embeddings.append(batch_embeddings)
            self.routesEmbedding[route.name] = np.vstack(embeddings)

    def _save_embeddings(self):
        """Save embeddings to JSON file"""
        data = {
            'routes': {}
        }
        
        for route_name, embeddings in self.routesEmbedding.items():
            data['routes'][route_name] = embeddings.tolist()
        
        with open(self.save_path, 'w', encoding='utf-8') as f:
            json.dump(data, f)
        print(f'💾 Embeddings saved to {self.save_path}')

    def _load_embeddings(self):
        """Load embeddings from JSON file"""
        with open(self.save_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for route_name, embeddings in data['routes'].items():
            self.routesEmbedding[route_name] = np.array(embeddings)

    def force_regenerate(self, batch=64):
        """Force regenerate embeddings (bỏ qua cache)"""
        print('🔵 Force regenerating embeddings...')
        self.routesEmbedding = {}
        self._generate_embeddings(batch)
        self._save_embeddings()
        print('🟢 Regenerated and saved successfully!')

    def get_routes(self):
        return self.routes

    
    def guide(self, query, top_k=200, debug_time=True):
        """
        Route a single query using FAISS
        Trả về (best_score, best_route_name)
        """

        t0 = time.perf_counter()

        if not query:
            raise ValueError("Empty query provided.")

        # ================== 1. ENCODE ==================
        t_encode_start = time.perf_counter()
        q = self.embedding.encode([query])
        q = np.array(q, dtype="float32")
        faiss.normalize_L2(q)
        t_encode_end = time.perf_counter()

        # ================== 2. FAISS SEARCH ==================
        t_faiss_start = time.perf_counter()
        scores, indices = self.index.search(q, top_k)
        t_faiss_end = time.perf_counter()

        scores = scores.flatten()
        indices = indices.flatten()

        # ================== 3. GROUP BY ROUTE ==================
        t_group_start = time.perf_counter()
        route_sims = {}
        for sim, idx in zip(scores, indices):
            if idx < 0:
                continue
            route = self.id2route[idx]
            route_sims.setdefault(route, []).append(sim)
        t_group_end = time.perf_counter()

        if not route_sims:
            raise RuntimeError("No route matched")

        # ================== 4. SCORE CALCULATION ==================
        t_score_start = time.perf_counter()
        final_scores = []

        for route_name, sims in route_sims.items():
            sims = np.array(sims)

            if np.any(sims > 0.95):
                high_sim_values = sims[sims > 0.95]
                score = np.mean(high_sim_values)
            else:
                k = min(100, len(sims))
                top_similarities = np.partition(sims, -k)[-k:]
                top_similarities = np.where(
                    top_similarities > 0.75,
                    top_similarities ** 2,
                    top_similarities
                )
                score = np.mean(top_similarities)

            final_scores.append((float(score), route_name))

        final_scores.sort(reverse=True, key=lambda x: x[0])
        t_score_end = time.perf_counter()

        t_end = time.perf_counter()

        # ================== DEBUG OUTPUT ==================
        if debug_time:
            print({
                "encode_ms": round((t_encode_end - t_encode_start) * 1000, 2),
                "faiss_ms": round((t_faiss_end - t_faiss_start) * 1000, 2),
                "group_ms": round((t_group_end - t_group_start) * 1000, 2),
                "score_ms": round((t_score_end - t_score_start) * 1000, 2),
                "total_ms": round((t_end - t0) * 1000, 2),
                "routes_hit": len(route_sims),
            })

        return final_scores[0]
       
    def batch_guide(self, queries):
        """
        Route multiple queries in batch using vectorized operations
        (logic giống guide, nhưng xử lý nhiều queries)
        """
        try:
            if not queries:
                raise ValueError("Empty query list provided.")

            # Encode tất cả queries cùng lúc
            queries_embeddings = self.embedding.encode(queries)
            if queries_embeddings is None:
                raise ValueError("Embedding model returned None.")

            # Chuẩn hóa embedding (L2 normalization)
            queries_embeddings = queries_embeddings / np.linalg.norm(queries_embeddings, axis=1, keepdims=True)

            results = []
            route_names = list(self.routesEmbedding.keys())

            for route_name, route_emb in self.routesEmbedding.items():
                try:
                    if route_emb is None or len(route_emb) == 0:
                        raise ValueError(f"Route '{route_name}' has no embeddings.")

                    # Chuẩn hóa route embeddings
                    routes_embedding = route_emb / np.linalg.norm(route_emb, axis=1, keepdims=True)

                    # Tính cosine similarity cho toàn bộ batch queries
                    similarities = np.dot(queries_embeddings, routes_embedding.T)  # shape (n_queries, n_routesamples)

                    # Mảng điểm cho từng query
                    scores_per_query = np.zeros(similarities.shape[0])

                    for i in range(similarities.shape[0]):
                        sims = similarities[i]

                        # Nếu tồn tại similarity > 0.95 → dùng trung bình nhóm cao
                        if np.any(sims > 0.95):
                            high_sim_values = sims[sims > 0.95]
                            score = np.mean(high_sim_values)
                        else:
                            # Lấy top 100 similarity cao nhất
                            top_k = min(100, len(sims))
                            top_similarities = np.partition(sims, -top_k)[-top_k:]

                            # Tăng trọng số các giá trị > 0.75
                            top_similarities = np.where(top_similarities > 0.75,
                                                        top_similarities ** 2,
                                                        top_similarities)

                            # Trung bình sau hiệu chỉnh
                            score = np.mean(top_similarities)

                        scores_per_query[i] = score

                    results.append(scores_per_query)

                except Exception as e:
                    print(f"⚠️ Error processing route '{route_name}': {e}")
                    results.append(np.zeros(len(queries)))

            if not results:
                raise RuntimeError("No route embeddings processed successfully.")

            # Kết hợp kết quả: (n_routes, n_queries)
            results = np.array(results)

            # Chọn route tốt nhất cho từng query
            best_route_indices = np.argmax(results, axis=0)
            best_scores = np.max(results, axis=0)

            final_results = [
                (float(best_scores[i]), route_names[best_route_indices[i]])
                for i in range(len(queries))
            ]

            return final_results

        except Exception as e:
            print(f"❌ batch_guide failed: {e}")
            return [(None, "error")] * len(queries)

     
if __name__ == "__main__":
    import os
    import sys
    # Thêm thư mục cha (parent directory) vào sys.path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from embedders import FlagBaseEmbedding, EmbeddingConfig, SentenceTransformerEmbedding, GeminiEmbedding, APIEmbeddingConfig
    from config import EMBEDDING_MODEL_NAME
    from senmatic_router import Route


    with open('data/router/trainData/train_med.json', 'r') as f:
        med_samples = json.load(f)

    with open('data/router/trainData/train_non_med.json', 'r') as f:
        non_med_samples = json.load(f)
    MED_ROUTE_NAME = 'medical'
    NON_MED_ROUTE_NAME = 'non_medical'

    med_route = Route(MED_ROUTE_NAME, med_samples[:1000])
    non_med_route = Route(NON_MED_ROUTE_NAME, non_med_samples[:1000])
    routes = [med_route, non_med_route]

    embedding = SentenceTransformerEmbedding(
        config=EmbeddingConfig(name=EMBEDDING_MODEL_NAME)
    )
    # embedding = GeminiEmbedding(
    #     config=APIEmbeddingConfig(
    #         name='gemini-embedding-001')
    # )
    router = SemanticRouter(
        embedding=embedding,
        save_path='data/router/routingEmbedddings/bgem3_routing_embedding_2000.json'
    )

    decision = router.guide('xin chào bạn')
    print(decision)

        


