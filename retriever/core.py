import pymongo
from IPython.display import Markdown
import textwrap

from embedders import FlagBaseEmbedding, EmbeddingConfig
from typing import Optional, Literal
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams
import chromadb

from config import VECTOR_SIZE, EMBEDDING_MODEL_NAME, QDRANT_API_KEY, QDRANT_URL

class Retriever():
    def __init__(self, 
            type: Literal['chromadb','mongodb', 'qdrant'],
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
        
        # Khởi tạo embedding model trước
        print(f"🔵 Loading embedding model: {embeddingName}")
        self.embedding_model = FlagBaseEmbedding(
            EmbeddingConfig(name=embeddingName)
        )
        
        self.vector_size = vector_size
        
        # Khởi tạo database client theo type
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
        # self.qdrant_collection = embedding_name.split('/')[-1]
        self.qdrant_collection = 'ta_hospital'
        
        try:
            self.client = QdrantClient(
                url=self.qdrant_url,
                api_key=self.qdrant_api
            )
            print(self.client)
            print('🟢 Connected to Qdrant successfully')
        except Exception as e:
            print(f"❌ Qdrant connection failed: {e}")
            raise
        
        # Kiểm tra và tạo collection nếu chưa có
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
            
            # Kiểm tra và tạo collection nếu chưa có
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
        """
        Check if collection exists (For Qdrant or ChromaDB)
        
        Returns:
            bool: True if collection exists, False otherwise
        """
        if self.type == "qdrant":
            # try:
            #     collections = self.client.get_collections().collections
            #     collection_names = [col.name for col in collections]
            #     exists = self.qdrant_collection in collection_names
            #     return exists
            # except Exception as e:
            #     print(f"⚠️ Error checking Qdrant collection: {e}")
            #     return False
            if self.client.collection_exists(collection_name=collection_name):
                return True    
            else:
                return False

        elif self.type == "chromadb":
            try:
                self.client.get_collection(name=self.chromadb_collection_name)
                return True
            except ValueError:
                return False
            except Exception as e:
                print(f"⚠️ Error checking ChromaDB collection: {e}")
                return False
        
        # MongoDB không cần check collection (tự động tạo khi insert)
        return True
            
    def _create_collection(self):
        """
        Create new collection (For Qdrant or ChromaDB)
        """
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
                # MongoDB không cần tạo collection trước
                print("MongoDB collections are created automatically on first insert")
                
        except Exception as e:
            print(f"❌ Error creating collection: {e}")
            raise

    def get_embedding(self, text):
        if not text.strip():
            return []

        embedding = self.embedding_model.encode(text)
        return embedding.tolist()
    
    def vector_search(
            self, 
            user_query: str, 
            limit=4):
        """
        Perform a vector search in the MongoDB collection or Qdrant collection based on the user query.

        Args:
        user_query (str): The user's query string.

        Returns:
        list: A list of matching documents.
        """

        # Generate embedding for the user query
        query_embedding = self.get_embedding(user_query)[0]

        if query_embedding is None:
            return "Invalid query or embedding generation failed."

        # Define the vector search pipeline
        if self.type == 'qdrant':
            if self._collection_exists:
                hits = self.client.search(
                    collection_name=self.qdrant_collection,
                    query_vector=query_embedding,
                    limit=limit
                )               
                results = []
                for hit in hits:
                    results.append({
                        'id':hit.id,
                        'category': hit.payload['category'], 
                        'content': hit.payload['content'], 
                        'score': hit.score
                        })
                return results
            else: 
                print(f"Collection {self.qdrant_collection} does not exist")
                return []
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

            unset_stage = {
                "$unset": "embedding" 
            }

            project_stage = {
                "$project": {
                    "_id": 1,  
                    "title": 1, 
                    # "product_specs": 1,
                    "color_options": 1,
                    "current_price": 1,
                    "product_promotion": 1,
                    "score": {
                        "$meta": "vectorSearchScore"
                    }
                }
            }

            pipeline = [vector_search_stage, unset_stage, project_stage]

            # Execute the search
            results = self.collection.aggregate(pipeline)
    
            return list(results)

        else:
            query_vector = self.get_embedding(user_query)
    
            hits = self.chromadb_collection.query(
                query_embeddings=[query_vector],
                n_results=limit
            )
                
            results = []
            for i in range(len(hits['ids'][0])):
                distance = hits['distances'][0][i]
                simlarity = 1 - distance 

                result = {
                    "_id": hits['ids'][0][i],
                    "combined_information": hits['documents'][0][i],
                    "score": simlarity
                }
                results.append(result)
            return results

    def enhance_prompt(self, query):
        pass

    def generate_content(self, prompt):
        return self.llm.generate_content(prompt)

    def _to_markdown(text):
        text = text.replace('•', '  *')
        return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))
    
if __name__ == '__main__':
    # 2️⃣ Tạo instance Retriever với ChromaDB (nhanh nhất để thử)
    retriever = Retriever(
        type='qdrant',               
        embeddingName="BAAI/bge-m3",
        vector_size=VECTOR_SIZE,
        qdrant_api=QDRANT_API_KEY,
        qdrant_url=QDRANT_URL
    )

    # 3️⃣ Tạo vài documents mẫu để insert vào ChromaDB
    sample_docs = [
        {"id": "1", "content": "Con mèo đang ngủ trên ghế sofa."},
        {"id": "2", "content": "Một chú chó chạy ngoài sân."},
        {"id": "3", "content": "Chiếc ô tô màu đỏ đỗ trước nhà."},
        {"id": "4", "content": "Con mèo thích ăn cá và sữa."}
    ]

    # 5️⃣ Thử truy vấn
    query = "tôi bị sổ mũi"
    results = retriever.vector_search(query, limit=3)

    print("\n🔍 Search results:")
    for res in results:
        print(f"ID: {res['id']} | Category: {res['category']} | Score: {res['score']:.4f}")
        print(f"Text: {res['content']}\n")