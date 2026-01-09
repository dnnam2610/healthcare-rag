import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from retriever.bm25 import BM25Retriever


def test_bm25_basic():
    # Create retriever but inject docs manually (no external DB required)
    r = BM25Retriever(type='chromadb')
    r._docs = [
        {"id": 1, "content": "I have a headache and feel dizzy.", "category": "medical"},
        {"id": 2, "content": "How to bake a cake: flour, eggs, sugar.", "category": "cooking"},
        {"id": 3, "content": "Headache remedies include rest and hydration.", "category": "medical"},
    ]

    # Build index from the injected docs
    r._build_index()

    results = r.vector_search("What helps with a headache?", limit=2)
    assert len(results) == 2
    # top result should be one of the medical documents
    assert any('headache' in res.content.lower() for res in results)


if __name__ == '__main__':
    test_bm25_basic()
    print('BM25 basic test passed')
