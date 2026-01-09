from retriever.base import BaseRetriever, Candidate
from typing import List

class HybridRetriever(BaseRetriever):
    def __init__(self, vector_retriever, raw_retriever):
        self.vector_retriever = vector_retriever
        self.raw_retriver = raw_retriever
        
    def search(self, query: str, limit: int = 5) -> List[Candidate]:
        raw_results = self._rank(
            self.raw_retriver.search(query, limit=50),
            descent=True
        )
        self.raw_retriver.save(candidates=raw_results, path_dir='_raw')

        vector_results = self._rank(
            self.vector_retriever.search(query, limit=50),
            descent=True
        )
        self.vector_retriever.save(candidates=vector_results, path_dir='_vector')
        
        final_results = self._compute_RRF(
            [raw_results, vector_results],
            k=60,
            limit=limit
        )

        return final_results
    
    def _rank(self, candidates: List[Candidate], descent: bool = True) -> List[Candidate]:
        if not candidates:
            return []
        return sorted(
            candidates,
            key=lambda c: c.score,
            reverse=descent
        )

    def _compute_RRF(
        self,
        multi_ranks: List[List[Candidate]],
        k: int = 60,
        limit: int = 5
    ) -> List[Candidate]:
        from collections import defaultdict

        rrf_scores = defaultdict(float)
        candidate_map = {}

        for ranked_list in multi_ranks:
            for idx, candidate in enumerate(ranked_list):
                rank = idx + 1
                doc_id = candidate.id

                rrf_scores[doc_id] += 1.0 / (k + rank)

                if doc_id not in candidate_map:
                    candidate_map[doc_id] = candidate

        sorted_docs = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        final_candidates = []
        for doc_id, score in sorted_docs[:limit]:
            base = candidate_map[doc_id]
            final_candidates.append(
                Candidate(
                    id=doc_id,
                    category=base.category,
                    content=base.content,
                    score=score
                )
            )

        return final_candidates
