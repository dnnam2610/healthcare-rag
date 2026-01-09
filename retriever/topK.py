import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import List, Dict, Any
import numpy as np

from config import VECTOR_SIZE, QDRANT_API_KEY, QDRANT_URL
from retriever.base import BaseVectorRetriever

class TopKRetriever(BaseVectorRetriever):
    """Traditional Top-K retrieval strategy"""
    
    def search(self, user_query: str, limit: int = 4) -> List[Dict[str, Any]]:
        """
        Perform traditional top-k vector search.
        
        Args:
            user_query (str): The user's query string
            limit (int): Number of top results to return
            
        Returns:
            List[Dict]: Top-k matching documents
        """
        print(f"🔍 TopK Retrieval: Fetching top {limit} results")
        results = self._raw_vector_search(user_query, limit=limit)
        return results[:limit]

# Usage example:
"""
# Traditional Top-K retrieval
topk_retriever = TopKRetriever(
    type='qdrant',
    qdrant_api=QDRANT_API_KEY,
    qdrant_url=QDRANT_URL,
    embeddingName=EMBEDDING_MODEL_NAME
)
results = topk_retriever.vector_search("What are the symptoms?", limit=5)

# Kneedle-based retrieval
kneedle_retriever = KneedleRetriever(
    type='qdrant',
    qdrant_api=QDRANT_API_KEY,
    qdrant_url=QDRANT_URL,
    embeddingName=EMBEDDING_MODEL_NAME,
    curve_direction='decreasing',  # scores decrease with rank
    curve_nature='concave',        # typical for similarity scores
    sensitivity=1.0,               # adjust for more/less sensitive knee detection
    max_candidates=100             # fetch more candidates for knee detection
)
results = kneedle_retriever.vector_search("What are the symptoms?", limit=5)
"""
    
if __name__ == '__main__':
    retriever = TopKRetriever(
        type='qdrant',               
        embeddingName="BAAI/bge-m3",
        vector_size=VECTOR_SIZE,
        qdrant_api=QDRANT_API_KEY,
        qdrant_url=QDRANT_URL
    )
    query = "Tôi hay bị dau đầu vào ban đêm"
    results = retriever.vector_search(query, limit=9)
    retriever.pprint(results)