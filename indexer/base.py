from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


class IndexingStrategy(Enum):
    """Enumeration of available indexing strategies."""
    BASIC = "basic"
    PARENT_CHILD = "parent_child"
    SUMMARY = "summary"


@dataclass
class Document:
    """Represents a document to be indexed."""
    content: str
    metadata: Dict[str, Any]
    doc_id: Optional[str] = None
    
    def __post_init__(self):
        if self.doc_id is None:
            self.doc_id = str(uuid.uuid4())


@dataclass
class Chunk:
    """Represents a chunk of text."""
    content: str
    chunk_id: str
    parent_id: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class EmbeddingModel(ABC):
    """Abstract base class for embedding models."""
    
    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        pass
    
    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """Return the dimension of embeddings."""
        pass


class OpenAIEmbedding(EmbeddingModel):
    """OpenAI embedding implementation."""
    
    def __init__(self, model: str = "text-embedding-3-small"):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model
        self._dimension = 1536 if "3-small" in model else 3072
    
    def embed(self, text: str) -> List[float]:
        response = self.client.embeddings.create(input=text, model=self.model)
        return response.data[0].embedding
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        response = self.client.embeddings.create(input=texts, model=self.model)
        return [item.embedding for item in response.data]
    
    def get_dimension(self) -> int:
        return self._dimension


class Chunker(ABC):
    """Abstract base class for text chunking strategies."""
    
    @abstractmethod
    def chunk(self, text: str, doc_id: str) -> List[Chunk]:
        """Split text into chunks."""
        pass


class RecursiveChunker(Chunker):
    """Recursive character-based chunker."""
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk(self, text: str, doc_id: str) -> List[Chunk]:
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]
            
            chunk = Chunk(
                content=chunk_text,
                chunk_id=str(uuid.uuid4()),
                parent_id=doc_id,
                metadata={"start": start, "end": end}
            )
            chunks.append(chunk)
            start += self.chunk_size - self.overlap
        
        return chunks


class BaseIndexer(ABC):
    """Abstract base class for indexing strategies."""
    
    def __init__(
        self,
        collection_name: str,
        embedding_model: EmbeddingModel,
        qdrant_client: QdrantClient
    ):
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.client = qdrant_client
    
    @abstractmethod
    def index_document(self, document: Document) -> None:
        """Index a single document."""
        pass
    
    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant chunks."""
        pass
    
    def _create_collection(self, collection_name: str) -> None:
        """Create a Qdrant collection if it doesn't exist."""
        collections = self.client.get_collections().collections
        if collection_name not in [c.name for c in collections]:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_model.get_dimension(),
                    distance=Distance.COSINE
                )
            )


class BasicIndexer(BaseIndexer):
    """Simple chunking and indexing strategy."""
    
    def __init__(
        self,
        collection_name: str,
        embedding_model: EmbeddingModel,
        qdrant_client: QdrantClient,
        chunker: Chunker
    ):
        super().__init__(collection_name, embedding_model, qdrant_client)
        self.chunker = chunker
        self._create_collection(collection_name)
    
    def index_document(self, document: Document) -> None:
        chunks = self.chunker.chunk(document.content, document.doc_id)
        
        points = []
        for chunk in chunks:
            embedding = self.embedding_model.embed(chunk.content)
            
            point = PointStruct(
                id=chunk.chunk_id,
                vector=embedding,
                payload={
                    "content": chunk.content,
                    "doc_id": document.doc_id,
                    "metadata": {**document.metadata, **chunk.metadata}
                }
            )
            points.append(point)
        
        self.client.upsert(collection_name=self.collection_name, points=points)
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        query_vector = self.embedding_model.embed(query)
        
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k
        )
        
        return [
            {
                "content": hit.payload["content"],
                "score": hit.score,
                "metadata": hit.payload["metadata"]
            }
            for hit in results
        ]


class ParentChildIndexer(BaseIndexer):
    """Parent-child indexing: small chunks for retrieval, large chunks for context."""
    
    def __init__(
        self,
        collection_name: str,
        embedding_model: EmbeddingModel,
        qdrant_client: QdrantClient,
        child_chunker: Chunker,
        parent_chunker: Chunker
    ):
        super().__init__(collection_name, embedding_model, qdrant_client)
        self.child_chunker = child_chunker
        self.parent_chunker = parent_chunker
        self.parent_collection = f"{collection_name}_parents"
        
        self._create_collection(collection_name)
        self._create_collection(self.parent_collection)
    
    def index_document(self, document: Document) -> None:
        # Create parent chunks
        parent_chunks = self.parent_chunker.chunk(document.content, document.doc_id)
        
        # Store parent chunks
        parent_points = []
        for parent in parent_chunks:
            point = PointStruct(
                id=parent.chunk_id,
                vector=[0.0] * self.embedding_model.get_dimension(),  # Dummy vector
                payload={
                    "content": parent.content,
                    "doc_id": document.doc_id,
                    "metadata": parent.metadata
                }
            )
            parent_points.append(point)
        
        self.client.upsert(
            collection_name=self.parent_collection,
            points=parent_points
        )
        
        # Create and index child chunks
        child_points = []
        for parent in parent_chunks:
            child_chunks = self.child_chunker.chunk(parent.content, parent.chunk_id)
            
            for child in child_chunks:
                embedding = self.embedding_model.embed(child.content)
                
                point = PointStruct(
                    id=child.chunk_id,
                    vector=embedding,
                    payload={
                        "content": child.content,
                        "parent_id": parent.chunk_id,
                        "doc_id": document.doc_id,
                        "metadata": child.metadata
                    }
                )
                child_points.append(point)
        
        self.client.upsert(collection_name=self.collection_name, points=child_points)
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        query_vector = self.embedding_model.embed(query)
        
        # Search child chunks
        child_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k
        )
        
        # Retrieve parent chunks
        results = []
        for hit in child_results:
            parent_id = hit.payload["parent_id"]
            
            parent = self.client.retrieve(
                collection_name=self.parent_collection,
                ids=[parent_id]
            )[0]
            
            results.append({
                "content": parent.payload["content"],
                "child_content": hit.payload["content"],
                "score": hit.score,
                "metadata": parent.payload["metadata"]
            })
        
        return results


class SummaryIndexer(BaseIndexer):
    """Summary-based indexing: create summaries for retrieval."""
    
    def __init__(
        self,
        collection_name: str,
        embedding_model: EmbeddingModel,
        qdrant_client: QdrantClient,
        chunker: Chunker,
        summarizer  # LLM or summarization function
    ):
        super().__init__(collection_name, embedding_model, qdrant_client)
        self.chunker = chunker
        self.summarizer = summarizer
        self.content_collection = f"{collection_name}_content"
        
        self._create_collection(collection_name)
        self._create_collection(self.content_collection)
    
    def index_document(self, document: Document) -> None:
        chunks = self.chunker.chunk(document.content, document.doc_id)
        
        summary_points = []
        content_points = []
        
        for chunk in chunks:
            # Generate summary
            summary = self.summarizer(chunk.content)
            summary_embedding = self.embedding_model.embed(summary)
            
            # Store summary for retrieval
            summary_point = PointStruct(
                id=chunk.chunk_id,
                vector=summary_embedding,
                payload={
                    "summary": summary,
                    "chunk_id": chunk.chunk_id,
                    "doc_id": document.doc_id,
                    "metadata": chunk.metadata
                }
            )
            summary_points.append(summary_point)
            
            # Store full content
            content_point = PointStruct(
                id=chunk.chunk_id,
                vector=[0.0] * self.embedding_model.get_dimension(),
                payload={
                    "content": chunk.content,
                    "doc_id": document.doc_id,
                    "metadata": chunk.metadata
                }
            )
            content_points.append(content_point)
        
        self.client.upsert(collection_name=self.collection_name, points=summary_points)
        self.client.upsert(collection_name=self.content_collection, points=content_points)
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        query_vector = self.embedding_model.embed(query)
        
        # Search summaries
        summary_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k
        )
        
        # Retrieve full content
        results = []
        for hit in summary_results:
            chunk_id = hit.payload["chunk_id"]
            
            content = self.client.retrieve(
                collection_name=self.content_collection,
                ids=[chunk_id]
            )[0]
            
            results.append({
                "content": content.payload["content"],
                "summary": hit.payload["summary"],
                "score": hit.score,
                "metadata": content.payload["metadata"]
            })
        
        return results


class IndexerFactory:
    """Factory for creating indexers based on strategy."""
    
    @staticmethod
    def create_indexer(
        strategy: IndexingStrategy,
        collection_name: str,
        embedding_model: EmbeddingModel,
        qdrant_client: QdrantClient,
        **kwargs
    ) -> BaseIndexer:
        """Create an indexer based on the specified strategy."""
        
        if strategy == IndexingStrategy.BASIC:
            chunker = kwargs.get("chunker", RecursiveChunker())
            return BasicIndexer(collection_name, embedding_model, qdrant_client, chunker)
        
        elif strategy == IndexingStrategy.PARENT_CHILD:
            child_chunker = kwargs.get("child_chunker", RecursiveChunker(chunk_size=500))
            parent_chunker = kwargs.get("parent_chunker", RecursiveChunker(chunk_size=2000))
            return ParentChildIndexer(
                collection_name, embedding_model, qdrant_client,
                child_chunker, parent_chunker
            )
        
        elif strategy == IndexingStrategy.SUMMARY:
            chunker = kwargs.get("chunker", RecursiveChunker())
            summarizer = kwargs.get("summarizer")
            if summarizer is None:
                raise ValueError("Summarizer must be provided for SummaryIndexer")
            return SummaryIndexer(
                collection_name, embedding_model, qdrant_client,
                chunker, summarizer
            )
        
        else:
            raise ValueError(f"Unknown indexing strategy: {strategy}")


# Example usage
if __name__ == "__main__":
    # Initialize components
    client = QdrantClient(host="localhost", port=6333)
    embedding_model = OpenAIEmbedding()
    
    # Example: Basic Indexing
    basic_indexer = IndexerFactory.create_indexer(
        strategy=IndexingStrategy.BASIC,
        collection_name="my_documents",
        embedding_model=embedding_model,
        qdrant_client=client
    )
    
    doc = Document(
        content="Your long document text here...",
        metadata={"source": "example.pdf", "author": "John Doe"}
    )
    
    basic_indexer.index_document(doc)
    results = basic_indexer.search("What is this about?", top_k=3)
    
    # Example: Parent-Child Indexing
    parent_child_indexer = IndexerFactory.create_indexer(
        strategy=IndexingStrategy.PARENT_CHILD,
        collection_name="my_documents_pc",
        embedding_model=embedding_model,
        qdrant_client=client,
        child_chunker=RecursiveChunker(chunk_size=500, overlap=50),
        parent_chunker=RecursiveChunker(chunk_size=2000, overlap=200)
    )
    
    # Example: Summary Indexing
    def simple_summarizer(text: str) -> str:
        # Replace with actual LLM summarization
        return f"Summary of: {text[:100]}..."
    
    summary_indexer = IndexerFactory.create_indexer(
        strategy=IndexingStrategy.SUMMARY,
        collection_name="my_documents_summary",
        embedding_model=embedding_model,
        qdrant_client=client,
        summarizer=simple_summarizer
    )