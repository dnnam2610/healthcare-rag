import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from typing import List, Dict, Any, Optional, Literal
import numpy as np
from retriever.base import BaseVectorRetriever
from config import QDRANT_API_KEY, QDRANT_URL, EMBEDDING_MODEL_NAME
from retriever.base import Candidate

class MMRRetriever(BaseVectorRetriever):
    """
    Maximal Marginal Relevance Retriever
    
    Implements MMR algorithm to balance relevance and diversity in search results.
    MMR selects documents that are relevant to the query while being dissimilar 
    to already selected documents.
    """
    
    def __init__(
        self,
        type: Literal['chromadb', 'mongodb', 'qdrant'],
        mongodbUri: Optional[str] = None,
        qdrant_api: Optional[str] = None,
        qdrant_url: Optional[str] = None,
        dbName: Optional[str] = None,
        dbCollection: Optional[str] = None,
        embeddingName: str = None,
        vector_size: int = None,
        llm=None,
    ):
        """
        Initialize MMR Retriever
        
        Args:
            type: Database type ('chromadb', 'mongodb', 'qdrant')
            lambda_param: Trade-off parameter between relevance and diversity (0-1)
                         1.0 = maximum relevance (no diversity)
                         0.0 = maximum diversity (no relevance consideration)
                         0.5 = balanced (recommended default)
            fetch_k: Number of documents to fetch initially for MMR selection
            Other parameters: inherited from BaseRetriever
        """
        super().__init__(
            type=type,
            mongodbUri=mongodbUri,
            qdrant_api=qdrant_api,
            qdrant_url=qdrant_url,
            dbName=dbName,
            dbCollection=dbCollection,
            embeddingName=embeddingName,
            vector_size=vector_size,
            llm=llm,
        )
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _compute_similarities(self,  embeddings):
        num_embeddings = embeddings.shape[0]
        similarities = np.zeros((num_embeddings, num_embeddings))

        for i in range(num_embeddings):
            for j in range(num_embeddings):
                similarities[i, j] = self._cosine_similarity(embeddings[i], embeddings[j])

        return similarities
    
    def _combine_similarity_matrices(self, candidates_similarities, scores_with_query):
        """
        Combine candidate-candidate similarities and query-candidate similarities
        into a single (N+1)x(N+1) similarity matrix.
        """
        num_candidates = len(scores_with_query)

        # Khởi tạo ma trận (N+1) x (N+1)
        sim_matrix = np.zeros((num_candidates + 1, num_candidates + 1))

        # Gán phần giữa các candidate
        sim_matrix[1:, 1:] = candidates_similarities

        # Gán độ tương đồng giữa query và từng candidate
        sim_matrix[0, 1:] = scores_with_query.flatten()
        sim_matrix[1:, 0] = scores_with_query.flatten()

        # Đặt độ tương đồng của query với chính nó = 1
        sim_matrix[0, 0] = 1.0

        return sim_matrix

    def _generate_similarity_matrices(
        self, 
        candidates: List[Candidate],
    ) -> np.ndarray:

        if not candidates:
            return []
        # Build candidate embeddings array efficiently
        candidate_embeddings = []
        scores_with_query = []
        for candidate in candidates:
            emb = candidate.embedding
            if emb:
                candidate_embeddings.append(np.array(emb))
            else:
                candidate_embeddings.append(np.zeros(self.vector_size))

            score = candidate.score
            if score:
                scores_with_query.append(np.array(score))
            else:
                scores_with_query.append(np.zeros(1))

        
        # Convert to numpy array for vectorized operations
        candidates_similarities = self._compute_similarities(np.array(candidate_embeddings))
        scores_with_query = np.array(scores_with_query)
        sim_matrix = self._combine_similarity_matrices(candidates_similarities, scores_with_query)
        return sim_matrix
       
    def _maximal_marginal_relevance(self, similarities:np.ndarray, num_to_select:int, lambda_param:float):
        if similarities.shape[0] <= 1 or num_to_select <= 0:
            return []

        most_similar = np.argmax(similarities[0, 1:])

        selected = [most_similar]
        candidates = set(range(len(similarities) - 1))
        candidates.remove(most_similar)

        while (len(selected) < num_to_select):
            if not candidates:
                break

            mmr_scores = {}
            for i in candidates:
                mmr_scores[i] = (lambda_param * similarities[i+1, 0] -
                    (1 - lambda_param) * max([similarities[i+1, j+1] for j in selected]))

            next_best = max(mmr_scores, key=mmr_scores.get)
            selected.append(next_best)
            candidates.remove(next_best)
        return selected
    
    def search(self, user_query:str, raw_limit:int=100, limit:int=10, lambda_param:float=0.5) -> List[Dict[str, Any]]:
        """
        Perform MMR-based vector search
        
        Args:
            user_query: User's search query
            limit: Number of final results to return (k in MMR)
            
        Returns:
            List of diverse and relevant documents
        """
        if not 0 <= lambda_param <= 1:
            raise ValueError("lambda_param must be between 0 and 1")
        
        print(f"\n🔍 MMR Search: query='{user_query[:50]}...', limit={raw_limit}, lambda_param={lambda_param}")
        
        # Step 1: Fetch initial candidates (fetch_k documents)
        candidates = self._raw_vector_search(user_query, limit=raw_limit, return_embedding=True)
        sim_matrix = self._generate_similarity_matrices(candidates=candidates)
        selected_indices = self._maximal_marginal_relevance(similarities=sim_matrix, num_to_select=limit, lambda_param=lambda_param)
        results = [c for idx, c in enumerate(candidates) if idx in selected_indices]

        return results
    
if __name__ == '__main__':
    mmr_retriever = MMRRetriever(
    type='qdrant',
    qdrant_api=QDRANT_API_KEY,
    qdrant_url=QDRANT_URL,
    embeddingName=EMBEDDING_MODEL_NAME,
    )
    results = mmr_retriever.vector_search("Tôi hay bị dau đầu vào ban đêm", raw_limit=100, limit=5)
    mmr_retriever.pprint(results)
    mmr_retriever.save(candidates=results, path_dir='test_mmr')