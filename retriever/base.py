from abc import ABC, abstractmethod
from typing import Optional, Literal, List, Dict, Any
import textwrap
from IPython.display import Markdown
from embedders import FlagBaseEmbedding, EmbeddingConfig
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams
from config import EMBEDDING_MODEL_NAME, VECTOR_SIZE
import chromadb
import pymongo

class BaseRetriever(ABC):
    """Base class for all retriever implementations"""
    
    def __init__(
        self,
        type: Literal['chromadb', 'mongodb', 'qdrant'],
        mongodbUri: Optional[str] = None,
        qdrant_api: Optional[str] = None,
        qdrant_url: Optional[str] = None,
        dbName: Optional[str] = None,
        dbCollection: Optional[str] = None,
        embeddingName: str = EMBEDDING_MODEL_NAME,
        vector_size: int = VECTOR_SIZE,
        llm=None,
    ):
        self.type = type
        self.embedding_name = embeddingName
        self.llm = llm
        self.vector_size = vector_size
        
        # Initialize embedding model
        print(f"🔵 Loading embedding model: {embeddingName}")
        self.embedding_model = FlagBaseEmbedding(
            EmbeddingConfig(name=embeddingName)
        )
        
        # Initialize database client based on type
        if self.type == 'mongodb':
            self._init_mongodb(mongodbUri, dbName, dbCollection)
        elif self.type == 'qdrant':
            self._init_qdrant(qdrant_api, qdrant_url, embeddingName)
        else:
            self._init_chromadb(embeddingName)

    def _init_mongodb(self, uri, db_name, collection_name):
        """Initialize MongoDB connection"""
        try:
            self.client = pymongo.MongoClient(uri)
            self.db = self.client[db_name]
            self.collection = self.db[collection_name]
            print(f"🟢 Connected to MongoDB: {db_name}.{collection_name}")
        except Exception as e:
            print(f"❌ MongoDB connection failed: {e}")
            raise

    def _init_qdrant(self, api_key, url, embedding_name):
        """Initialize Qdrant connection"""
        self.qdrant_api = api_key
        self.qdrant_url = url
        self.qdrant_collection = 'ta_hospital'
        
        try:
            self.client = QdrantClient(url=self.qdrant_url, api_key=self.qdrant_api)
            print(self.client)
            print('🟢 Connected to Qdrant successfully')
        except Exception as e:
            print(f"❌ Qdrant connection failed: {e}")
            raise
        
        # Check and create collection if not exists
        if not self._collection_exists(self.qdrant_collection):
            print(f"Creating Qdrant collection: {self.qdrant_collection}")
            self._create_collection()
        else:
            print(f"🟢 Connected to Qdrant collection: {self.qdrant_collection}")

    def _init_chromadb(self, embedding_name):
        """Initialize ChromaDB connection"""
        try:
            self.client = chromadb.PersistentClient(path="./chroma_db")
            self.chromadb_collection_name = embedding_name.split('/')[-1]
            
            # Check and create collection if not exists
            if self._collection_exists(self.chromadb_collection_name):
                self.chromadb_collection = self.client.get_collection(
                    name=self.chromadb_collection_name
                )
                print(f"🟢 Connected to ChromaDB collection: {self.chromadb_collection_name}")
            else:
                print(f"Creating ChromaDB collection: {self.chromadb_collection_name}")
                self._create_collection()
        except Exception as e:
            print(f"❌ ChromaDB initialization failed: {e}")
            raise

    def _collection_exists(self, collection_name):
        """Check if collection exists"""
        if self.type == "qdrant":
            return self.client.collection_exists(collection_name=collection_name)
        elif self.type == "chromadb":
            try:
                self.client.get_collection(name=self.chromadb_collection_name)
                return True
            except ValueError:
                return False
            except Exception as e:
                print(f"⚠️ Error checking ChromaDB collection: {e}")
                return False
        return True

    def _create_collection(self):
        """Create new collection"""
        try:
            if self.type == "qdrant":
                self.client.create_collection(
                    collection_name=self.qdrant_collection,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE
                    )
                )
                print(f"✅ Qdrant collection '{self.qdrant_collection}' created successfully")
                print(f"   Vector size: {self.vector_size}, Distance: COSINE")
            elif self.type == "chromadb":
                self.chromadb_collection = self.client.create_collection(
                    name=self.chromadb_collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
                print(f"✅ ChromaDB collection '{self.chromadb_collection_name}' created successfully")
                print(f"   Vector size: {self.vector_size}, Distance: cosine")
            else:
                print("MongoDB collections are created automatically on first insert")
        except Exception as e:
            print(f"❌ Error creating collection: {e}")
            raise

    def get_embedding(self, text):
        """Generate embedding for text"""
        if not text.strip():
            return []
        embedding = self.embedding_model.encode(text)
        return embedding.tolist()

    def _raw_vector_search(self, user_query: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Perform raw vector search and return results with scores.
        This method fetches more results than needed for filtering.
        """
        query_embedding = self.get_embedding(user_query)[0]
        
        if query_embedding is None:
            return []

        if self.type == 'qdrant':
            if not self._collection_exists(self.qdrant_collection):
                print(f"Collection {self.qdrant_collection} does not exist")
                return []
            
            hits = self.client.search(
                collection_name=self.qdrant_collection,
                query_vector=query_embedding,
                limit=limit
            )
            results = []
            for hit in hits:
                results.append({
                    'id': hit.id,
                    'category': hit.payload['category'],
                    'content': hit.payload['content'],
                    'score': hit.score
                })
            return results
            
        elif self.type == 'mongodb':
            vector_search_stage = {
                "$vectorSearch": {
                    "index": "vector_index",
                    "queryVector": query_embedding,
                    "path": "embedding",
                    "numCandidates": 400,
                    "limit": limit,
                }
            }
            unset_stage = {"$unset": "embedding"}
            project_stage = {
                "$project": {
                    "_id": 1,
                    "title": 1,
                    "color_options": 1,
                    "current_price": 1,
                    "product_promotion": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
            pipeline = [vector_search_stage, unset_stage, project_stage]
            results = self.collection.aggregate(pipeline)
            return list(results)
            
        else:  # chromadb
            query_vector = self.get_embedding(user_query)
            hits = self.chromadb_collection.query(
                query_embeddings=[query_vector],
                n_results=limit
            )
            results = []
            for i in range(len(hits['ids'][0])):
                distance = hits['distances'][0][i]
                similarity = 1 - distance
                result = {
                    "_id": hits['ids'][0][i],
                    "combined_information": hits['documents'][0][i],
                    "score": similarity
                }
                results.append(result)
            return results

    @abstractmethod
    def vector_search(self, user_query: str, limit: int = 4) -> List[Dict[str, Any]]:
        """
        Abstract method for vector search with specific retrieval strategy.
        Must be implemented by subclasses.
        """
        pass

    def enhance_prompt(self, query):
        """Enhance prompt for better retrieval"""
        pass

    def generate_content(self, prompt):
        """Generate content using LLM"""
        return self.llm.generate_content(prompt)

    @staticmethod
    def _to_markdown(text):
        """Convert text to markdown format"""
        text = text.replace('•', '  *')
        return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

