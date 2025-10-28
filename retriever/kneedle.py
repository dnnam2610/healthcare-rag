import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from kneed import KneeLocator
from typing import List, Dict, Any
from retriever.base import BaseRetriever
from config import QDRANT_API_KEY, QDRANT_URL, EMBEDDING_MODEL_NAME
import matplotlib.pyplot as plt
class KneedleRetriever(BaseRetriever):
    """Kneedle algorithm-based retrieval strategy"""
    
    def __init__(self, **kwargs):
        """
        Initialize KneedleRetriever with additional parameters for knee detection.
        
        Additional kwargs:
            curve_direction: 'decreasing' or 'increasing' (default: 'decreasing')
            curve_nature: 'concave' or 'convex' (default: 'concave')
            sensitivity: float, S parameter for knee detection (default: 1.0)
            max_candidates: int, maximum number of candidates to fetch (default: 100)
        """
        self.curve_direction = kwargs.pop('curve_direction', 'decreasing')
        self.sensitivity = kwargs.pop('sensitivity', 1.0)
        self.max_candidates = kwargs.pop('max_candidates', 100)
        super().__init__(**kwargs)
    

    def _detect_curve_nature(self, scores: list[float]) -> str:
        """Tự động phát hiện curve nature: concave hay convex"""
        y = np.array(scores)
        x = np.arange(len(y))
        
        # Đạo hàm bậc 1
        dy = np.gradient(y, x)
        # Đạo hàm bậc 2 (độ cong)
        d2y = np.gradient(dy, x)
        
        curvature_mean = np.mean(d2y)
        
        if curvature_mean < 0:
            nature = "concave"
        else:
            nature = "convex"
        
        print(f"🧮 Detected curve nature: {nature} (mean curvature={curvature_mean:.4f})")
        return nature
    
    def vector_search(self, user_query: str, limit: int = 4) -> List[Dict[str, Any]]:
        """
        Perform vector search using Kneedle algorithm to find optimal cutoff.
        
        Args:
            user_query (str): The user's query string
            limit (int): Minimum number of results to return (fallback)
            
        Returns:
            List[Dict]: Documents up to the knee point in similarity scores
        """
        print(f"🔍 Kneedle Retrieval: Fetching up to {self.max_candidates} candidates")
        
        # Fetch more candidates than limit for knee detection
        results = self._raw_vector_search(user_query, limit=self.max_candidates)
        
        if len(results) <= limit:
            print(f"⚠️ Only {len(results)} results found, returning all")
            return results
        
        # Extract scores
        scores = [r['score'] for r in results]
        indices = list(range(len(scores)))
        try:
            # Apply Kneedle algorithm
            kneedle = KneeLocator(
                indices,
                scores,
                curve=self._detect_curve_nature(scores),
                direction=self.curve_direction,
                S=self.sensitivity
            )

            kneedle.plot_knee_normalized()
            plt.show()  
            if kneedle.knee is not None:
                knee_index = kneedle.knee
                # Ensure we return at least 'limit' results
                cutoff_index = max(knee_index + 1, limit)
                print(f"📍 Knee detected at index {knee_index}, returning {cutoff_index} results")
                return results[:cutoff_index]

            else:
                print(f"⚠️ No knee detected, returning top {limit} results")
                return results[:limit]
                   
        except Exception as e:
            print(f"⚠️ Kneedle algorithm failed: {e}, returning top {limit} results")
            return results[:limit]
        
if __name__ == '__main__':
    kneedle_retriever = KneedleRetriever(
    type='qdrant',
    qdrant_api=QDRANT_API_KEY,
    qdrant_url=QDRANT_URL,
    embeddingName=EMBEDDING_MODEL_NAME,
    curve_direction='decreasing',  # scores decrease with rank
    sensitivity=1,        # adjust for more/less sensitive knee detection
    max_candidates=100             # fetch more candidates for knee detection
)
    results = kneedle_retriever.vector_search("Tôi hay bị dau đầu vào ban đêm", limit=5)

    print(results)
