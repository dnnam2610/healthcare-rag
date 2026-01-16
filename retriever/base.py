from abc import ABC, abstractmethod
from typing import Optional, Literal, List, Dict, Any
import textwrap
from IPython.display import Markdown
from embedders import FlagBaseEmbedding, EmbeddingConfig
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams
from database import QDrantDB
from config import EMBEDDING_MODEL_NAME, VECTOR_SIZE
from datetime import datetime
import chromadb
import pymongo
import json
import os
import csv

from dataclasses import dataclass
from typing import Optional, Any, List


@dataclass
class Candidate:
    """Đại diện cho một kết quả truy vấn vector."""
    id: Any
    category: Optional[str]
    content: str
    score: float
    embedding: Optional[List[float]] = None

    def to_dict(self):
        """Chuyển Candidate sang dict (dùng để lưu JSON/CSV)."""
        return {
            "id": self.id,
            "category": self.category,
            "content": self.content,
            "score": self.score,
        }

class BaseRetriever(ABC):
    """
    Root base class for all retriever implementations.
    Provides common functionality for search, display, and saving results.
    """
    
    def __init__(self, **kwargs):
        """Initialize base retriever with common parameters."""
        pass
    
    @abstractmethod
    def search(self, query: str, limit: int = 5) -> List[Candidate]:
        """
        Abstract method for search functionality.
        Must be implemented by all subclasses.
        
        Args:
            query: Search query string
            top_k: Number of top results to return
            
        Returns:
            List of Candidate objects
        """
        pass
    
    @staticmethod
    def pprint(candidates: List[Candidate]) -> None:
        """Pretty print search results."""
        print("\n🔍 Search results:")
        for i, c in enumerate(candidates, 1):
            print(f"\n[{i}] Score: {c.score:.4f}")
            print(f"ID: {c.id} | Category: {c.category}")
            print(f"Content: {c.content[:200]}{'...' if len(c.content) > 200 else ''}")
            print("-" * 80)
    
    @staticmethod
    def save(
        query: str,
        candidates: List[Candidate],
        path_dir: str,
        format: str = "json"
    ) -> Optional[str]:
        """
        Save search results to file.

        Args:
            query: Original search query
            candidates: List of Candidate objects
            path_dir: Directory to save results
            format: Output format (json, txt)

        Returns:
            Path to saved file or None if failed
        """
        supported_formats = {"json", "txt"}
        format = format.lower()

        if format not in supported_formats:
            print(f"⚠️ Unsupported format '{format}'. Supported: {', '.join(supported_formats)}")
            return None

        os.makedirs(path_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_path = os.path.join(path_dir, f"results_{timestamp}.{format}")

        print(f"💾 Saving {len(candidates)} results to: {file_path}")

        try:
            if format == "json":
                data = {
                    "query": query,
                    "num_candidates": len(candidates),
                    "timestamp": timestamp,
                    "results": [c.to_dict() for c in candidates]
                }

                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)

            elif format == "txt":
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(f"Query: {query}\n")
                    f.write(f"Timestamp: {timestamp}\n")
                    f.write(f"Total results: {len(candidates)}\n")
                    f.write("=" * 80 + "\n\n")

                    for i, c in enumerate(candidates, 1):
                        f.write(f"[{i}] ID: {c.id}\n")
                        f.write(f"Category: {c.category}\n")
                        f.write(f"Score: {c.score:.4f}\n")
                        f.write(f"Content: {c.content}\n")
                        f.write("-" * 80 + "\n\n")

            print("✅ Save completed successfully.")
            return file_path

        except Exception as e:
            print(f"❌ Error saving file: {e}")
            return None
    
    @staticmethod
    def _to_markdown(text: str) -> Markdown:
        """Convert text to markdown format."""
        text = text.replace('•', '  *')
        return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

class BaseDBRetriever(ABC):
    """
    Base class handling ALL database logic.
    """

    def __init__(
        self,
        type: Literal["chromadb", "mongodb", "qdrant"],
        mongodbUri: Optional[str] = None,
        qdrant_api: Optional[str] = None,
        qdrant_url: Optional[str] = None,
        dbName: Optional[str] = None,
        dbCollection: Optional[str] = None,
        chroma_path: str = "./chroma_db",
        embedding_name: str = None,
        vector_size: int = None,
    ):
        self.type = type
        self.vector_size = vector_size
        self.embedding_name = embedding_name

        if self.type == "mongodb":
            self._init_mongodb(mongodbUri, dbName, dbCollection)
        elif self.type == "qdrant":
            self._init_qdrant(qdrant_api, qdrant_url)
        elif self.type == "chromadb":
            self._init_chromadb(chroma_path)
        else:
            raise ValueError(f"Unsupported DB type: {self.type}")

    # ---------- INIT ----------
    def _init_mongodb(self, uri, db, collection):
        self.client = pymongo.MongoClient(uri)
        self.collection = self.client[db][collection]

    def _init_qdrant(self, api_key, url):
        self.client = QdrantClient(url=url, api_key=api_key)
        self.collection_name = "ta_hospital"

    def _init_chromadb(self, path):
        self.client = chromadb.PersistentClient(path=path)
        self.collection_name = self.embedding_name.split("/")[-1]
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    # # ---------- CONTENT ----------
    # def search_by_id(self, doc_id: int) -> Optional[Candidate]:
    #     """
    #     Retrieve a document by ID and return as Candidate.
    #     """

    #     if self.type == "mongodb":
    #         doc = self.collection.find_one(
    #             {"id": doc_id},
    #             {"_id": 0}
    #         )
    #         if not doc:
    #             return None

    #         return Candidate(
    #             id=doc.get("id", doc_id),
    #             category=doc.get("category"),
    #             content=doc.get("content", ""),
    #             score=1.0
    #         )

    #     elif self.type == "chromadb":
    #         res = self.collection.get(
    #             ids=[str(doc_id)],
    #             include=["documents", "metadatas"]
    #         )
    #         if not res or not res.get("documents"):
    #             return None

    #         metadata = res["metadatas"][0] if res.get("metadatas") else {}

    #         return Candidate(
    #             id=doc_id,
    #             category=metadata.get("category"),
    #             content=res["documents"][0],
    #             score=1.0
    #         )

    #     elif self.type == "qdrant":
    #         res = self.client.retrieve(
    #             collection_name=self.collection_name,
    #             ids=[doc_id]
    #         )
    #         if not res:
    #             return None

    #         payload = res[0].payload or {}

    #         return Candidate(
    #             id=payload.get("id", doc_id),
    #             category=payload.get("category"),
    #             content=payload.get("content", ""),
    #             score=1.0
    #         )

    #     return None
    def search_by_ids(self, doc_ids: List[int]) -> List[Candidate]:
        """
        Batch retrieve documents by IDs.
        Order is NOT preserved.
        """

        if not doc_ids:
            return []

        # ======================
        # MongoDB
        # ======================
        if self.type == "mongodb":
            cursor = self.collection.find(
                {"id": {"$in": doc_ids}},
                {"_id": 0, "id": 1, "category": 1, "content": 1}
            )

            return [
                Candidate(
                    id=doc["id"],
                    category=doc.get("category"),
                    content=doc.get("content", ""),
                    score=1.0
                )
                for doc in cursor
            ]

        # ======================
        # ChromaDB
        # ======================
        elif self.type == "chromadb":
            res = self.collection.get(
                ids=[str(i) for i in doc_ids],
                include=["documents", "metadatas"]
            )

            if not res or not res.get("documents"):
                return []

            metadatas = res.get("metadatas") or [{}] * len(res["documents"])

            return [
                Candidate(
                    id=int(doc_ids[i]) if i < len(doc_ids) else i,
                    category=metadatas[i].get("category"),
                    content=res["documents"][i],
                    score=1.0
                )
                for i in range(len(res["documents"]))
            ]

        # ======================
        # Qdrant
        # ======================
        elif self.type == "qdrant":
            res = self.client.retrieve(
                collection_name=self.collection_name,
                ids=doc_ids
            )

            return [
                Candidate(
                    id=point.payload.get("id", point.id),
                    category=point.payload.get("category"),
                    content=point.payload.get("content", ""),
                    score=1.0
                )
                for point in res
            ]

        return []

class BaseRawRetriever(BaseDBRetriever, BaseRetriever):
    """
    Base class for raw text retrieval algorithms (BM25, TF-IDF, etc.).
    These algorithms don't use embeddings, only statistical methods.
    """
    
    def __init__(
        self, 
        index_path: Optional[str] = None, 
        **kwargs
        ):
        """
        Initialize raw retriever.
        
        Args:
            index_path: Path to pre-built index file
        """
        super().__init__(**kwargs)
        self.index_path = index_path
        self.index_loaded = False
    
    @abstractmethod
    def load_index(self, index_path: str) -> None:
        """
        Load pre-built index from file.
        
        Args:
            index_path: Path to index file
            
        """
        pass
    
class BaseVectorRetriever(BaseDBRetriever, BaseRetriever):
    """
    Base class for vector-based retrievers.
    """

    def __init__(
        self,
        type: Literal["chromadb", "mongodb", "qdrant"],
        embeddingName: str,
        vector_size: int,
        llm=None,
        **db_kwargs
    ):
        BaseDBRetriever.__init__(
            self,
            type=type,
            embedding_name=embeddingName,
            vector_size=vector_size,
            **db_kwargs
        )

        self.embedding_name = embeddingName
        self.embedding_model = FlagBaseEmbedding(
            EmbeddingConfig(name=embeddingName)
        )
        self.llm = llm

    def get_embedding(self, text: str):
        return self.embedding_model.encode(text).tolist()

    def _raw_vector_search(self, query: str, limit: int = 10) -> List[Candidate]:
        emb = self.get_embedding(query)[0]

        results = []

        if self.type == "qdrant":
            hits = self.client.query_points(
                collection_name=self.collection_name,
                query=emb,
                limit=limit,
            )
            for h in hits.points:
                results.append(
                    Candidate(
                        id=h.id,
                        category=h.payload.get("category"),
                        content=h.payload.get("content", ""),
                        score=h.score
                    )
                )

        elif self.type == "chromadb":
            hits = self.collection.query(
                query_embeddings=[emb],
                n_results=limit
            )
            for i in range(len(hits["ids"][0])):
                results.append(
                    Candidate(
                        id=hits["ids"][0][i],
                        category=None,
                        content=hits["documents"][0][i],
                        score=1 - hits["distances"][0][i]
                    )
                )

        elif self.type == "mongodb":
            pipeline = [{
                "$vectorSearch": {
                    "index": "vector_index",
                    "queryVector": emb,
                    "path": "embedding",
                    "limit": limit
                }
            }]
            for doc in self.collection.aggregate(pipeline):
                results.append(
                    Candidate(
                        id=doc["_id"],
                        category=None,
                        content=doc.get("content", ""),
                        score=doc["score"]
                    )
                )

        return results

    def enhance_prompt(self, query):
        """Enhance prompt for better retrieval"""
        pass

    def generate_content(self, prompt):
        """Generate content using LLM"""
        return self.llm.generate_content(prompt)