import os
from retriever.base import BaseRawRetriever, Candidate
from typing import Optional, List, Dict, Any, Literal
from data_generator.utils import preprocess_text
from collections import defaultdict
import math
import json
import py_vncorenlp
from vn_preprocess import VnTextProcessor
from transformers import AutoTokenizer
from utils.timing_utils import timeit
from time import time

phobert_tokenizer = AutoTokenizer.from_pretrained(
    "vinai/phobert-base"
)


class BM25Retriever(BaseRawRetriever):
    """
    BM25 retrieval implementation using inverted index.
    Content is fetched dynamically from database via BaseDBRetriever.
    """

    def __init__(
        self,
        type: Literal["chromadb", "mongodb", "qdrant"],                         # 'mongodb' | 'chromadb' | 'qdrant'
        index_path: Optional[str] = None,
        k1: float = 1.5,
        b: float = 0.75,
        use_segmentation: bool = True,
        segmenter_path: str = "../vncorenlp",
        **kwargs
    ):
        super().__init__(type=type, index_path=index_path, **kwargs)

        self.k1 = k1
        self.b = b
        self.use_segmentation = use_segmentation

        # BM25 index components
        self.N = 0
        self.avg_doc_len = 0.0
        self.doc_len = {}        # doc_id -> length
        self.postings = {}       # term -> {doc_id: tf}
        self.idf = {}            # term -> idf

        # Initialize segmenter safely
        self.segmenter = None
        if self.use_segmentation:
            cwd = os.getcwd()
            self.segmenter = VnTextProcessor(save_dir=segmenter_path)
            os.chdir(cwd)

        # Load index if provided
        if index_path:
            self.load_index(index_path)

    # ======================================================
    # INDEX
    # ======================================================
    def load_index(self, index_path: str) -> None:
        """
        Load BM25 inverted index from JSON file.
        """
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index file not found: {index_path}")

        print(f"📂 Loading BM25 index from {index_path}...")

        with open(index_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.N = data["meta"]["N"]
        self.avg_doc_len = data["meta"]["avg_doc_len"]
        self.k1 = data["meta"]["k1"]
        self.b = data["meta"]["b"]

        self.doc_len = {int(k): v for k, v in data["doc_len"].items()}
        self.idf = data["idf"]
        self.postings = {
            term: {int(doc_id): tf for doc_id, tf in docs.items()}
            for term, docs in data["postings"].items()
        }

        self.index_loaded = True

        print(f"✅ Index loaded")
        print(f"   Documents: {self.N}")
        print(f"   Vocabulary size: {len(self.postings)}")
        

    # ======================================================
    # TOKENIZATION
    # ======================================================
    @timeit("BM25::_tokenize")
    def _tokenize(self, text: str) -> List[str]:
        clean_text = preprocess_text(text).lower()

        if self.use_segmentation and self.segmenter:
            segmented = self.segmenter.preprocess(clean_text)
            tokenized = phobert_tokenizer.tokenize(segmented)
            print(tokenized)
            return tokenized
        
        tokenized = phobert_tokenizer.tokenize(clean_text)
        return tokenized

    # ======================================================
    # BM25 SCORING
    # ======================================================
    def _compute_score(self, query_terms: List[str], doc_id: int) -> float:
        score = 0.0
        dl = self.doc_len.get(doc_id, 0)

        for term in query_terms:
            if term not in self.postings:
                continue
            if doc_id not in self.postings[term]:
                continue

            tf = self.postings[term][doc_id]
            idf = self.idf.get(term, 0.0)

            denom = tf + self.k1 * (
                1 - self.b + self.b * dl / self.avg_doc_len
            )

            score += idf * (tf * (self.k1 + 1)) / denom

        return score

    # ======================================================
    # SEARCH
    # ======================================================
    @timeit("BM25::search")
    def search(self, user_query: str, limit: int = 5) -> List[Candidate]:
        """
        Perform BM25 search.
        Content is retrieved via BaseDBRetriever.search_by_id().
        """
        if not self.index_loaded:
            raise RuntimeError("BM25 index not loaded")

        query_terms = self._tokenize(user_query)

        # Candidate documents from inverted index
        candidate_docs = set()
        for term in query_terms:
            if term in self.postings:
                candidate_docs.update(self.postings[term].keys())

        if not candidate_docs:
            return []

        scored_docs = []
        
        start = time()
        for doc_id in candidate_docs:
            score = self._compute_score(query_terms, doc_id)
            if score > 0:
                scored_docs.append((doc_id, score))
        print(f"Score computing time: {time() - start}")

        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # Fetch content from DB via BaseDBRetriever
        candidates: List[Candidate] = []
        
        start = time()
        # for doc_id, score in scored_docs[:limit]:
        #     candidate = self.search_by_id(doc_id)   # từ BaseDBRetriever
        #     if candidate is None:
        #         continue

        #     candidate.score = score
        #     candidates.append(candidate)
        
        top = scored_docs[:limit]
        top_ids = [doc_id for doc_id, _ in top]

        candidates = self.search_by_ids(top_ids)

        score_map = dict(top)
        for c in candidates:
            c.score = score_map.get(c.id, c.score)

        print(f"Searching by id time: {time() - start}")
        return candidates
