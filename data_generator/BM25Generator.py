"""
{
  "meta": {
    "N": 1000,
    "avg_doc_len": 42.3,
    "k1": 1.5,
    "b": 0.75
  },
  "doc_len": {
    "1": 120,
    "2": 80
  },
  "idf": {
    "covid": 1.24
  },
  "postings": {
    "covid": {
      "1": 3,
      "5": 1
    }
  }
}
"""

import os
import json
import math
from collections import defaultdict
from datasets import load_dataset
from tqdm import tqdm
import py_vncorenlp
from data_generator.utils import preprocess_text
from transformers import AutoTokenizer
# =========================
# INIT VnCoreNLP
# =========================
# _ORIGINAL_CWD = os.getcwd()

rdrsegmenter = py_vncorenlp.VnCoreNLP(
    annotators=["wseg"],
    save_dir="/Users/nnam/Documents/Workspace/university/seminar/vncorenlp"
)

phobert_tokenizer = AutoTokenizer.from_pretrained(
    "vinai/phobert-base"
)

# os.chdir(_ORIGINAL_CWD)


def segment_word(text: str)->str:
    segmented_text = rdrsegmenter.word_segment(text)
    return " ".join([t for t in segmented_text]).lower()

def tokenize(text: str)->list:
    tokens = phobert_tokenizer.tokenize(text)
    return tokens

# =========================
# BM25 CLASS
# =========================
class BM25Index:
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b

        self.N = 0
        self.avg_doc_len = 0.0

        self.doc_len = {}           # doc_id -> length
        self.postings = {}          # term -> {doc_id: tf}
        self.idf = {}               # term -> idf

    # =========================
    # LOAD / SAVE
    # =========================
    def load(self, index_path):
        with open(index_path, encoding="utf-8") as f:
            index = json.load(f)

        self.N = index["meta"]["N"]
        self.avg_doc_len = index["meta"]["avg_doc_len"]
        self.k1 = index["meta"]["k1"]
        self.b = index["meta"]["b"]

        self.doc_len = {int(k): v for k, v in index["doc_len"].items()}
        self.idf = index["idf"]
        self.postings = {
            term: {int(doc_id): tf for doc_id, tf in docs.items()}
            for term, docs in index["postings"].items()
        }

        print(f"📂 Loaded index from {index_path}")

    def save(self, out_path):
        index = {
            "meta": {
                "N": self.N,
                "avg_doc_len": self.avg_doc_len,
                "k1": self.k1,
                "b": self.b
            },
            "doc_len": dict(self.doc_len),
            "idf": self.idf,
            "postings": {
                term: {str(doc_id): tf for doc_id, tf in docs.items()}
                for term, docs in self.postings.items()
            }
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=2)

        print(f"✅ Index saved to {out_path}")

    # =========================
    # BUILD FROM SCRATCH
    # =========================
    def build_from_json(self, json_path):
        dataset = load_dataset("json", data_files=json_path, split="train")

        self._add_documents(dataset, rebuild_idf=True)

    # =========================
    # ADD DOCUMENTS (INCREMENTAL)
    # =========================
    def add_documents(self, json_path):
        dataset = load_dataset("json", data_files=json_path, split="train")
        self._add_documents(dataset, rebuild_idf=True)

    def _add_documents(self, dataset, rebuild_idf=True):
        total_len = sum(self.doc_len.values())

        for ex in tqdm(dataset, desc="Indexing new docs"):
            doc_id = int(ex["id"])
            if doc_id in self.doc_len:
                continue  # skip existing docs

            raw_text = ex["content"]
            clean_text = preprocess_text(raw_text)
            segmented_words = segment_word(clean_text)
            tokens = tokenize(segmented_words)

            self.doc_len[doc_id] = len(tokens)

            tf_counter = defaultdict(int)
            for t in tokens:
                tf_counter[t] += 1

            for term, tf in tf_counter.items():
                if term not in self.postings:
                    self.postings[term] = {}
                self.postings[term][doc_id] = tf

            total_len += len(tokens)
            self.N += 1

        self.avg_doc_len = total_len / self.N

        if rebuild_idf:
            self._recompute_idf()

    def _recompute_idf(self):
        print("Recomputing IDF...")
        self.idf = {}
        for term, docs in self.postings.items():
            df = len(docs)
            self.idf[term] = math.log(
                (self.N - df + 0.5) / (df + 0.5) + 1
            )

    # =========================
    # SEARCH
    # =========================
    def _score(self, query_terms, doc_id):
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

    def search(self, query, top_k=5):
        segmented_query = segment_word(query)
        query_terms = tokenize(segmented_query)

        candidates = set()
        for term in query_terms:
            if term in self.postings:
                candidates.update(self.postings[term].keys())

        scores = []
        for doc_id in candidates:
            s = self._score(query_terms, doc_id)
            if s > 0:
                scores.append((doc_id, s))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# =========================
# USAGE EXAMPLE
# =========================
if __name__ == "__main__":

    INDEX_PATH = "/Users/nnam/Documents/Workspace/university/seminar/data/inverted_index/bm25_index_2.json"

    bm25 = BM25Index()

    if os.path.exists(INDEX_PATH):
        bm25.load(INDEX_PATH)
    else:
        bm25.build_from_json(
            "/Users/nnam/Documents/Workspace/university/seminar/data/chunks/chunks_with_id.json"
        )

    # ADD NEW DOCUMENTS LATER
    # bm25.add_documents("new_chunks.json")
    """
        [
    {
        "id": 12345,
        "content": "Nội dung văn bản tiếng Việt cần index"
    },
    {
        "id": 12346,
        "content": "Một đoạn văn bản khác"
    }
]
    """

    bm25.save(INDEX_PATH)

    print(bm25.search("covid 19", top_k=3))
