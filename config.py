import os
from dotenv import load_dotenv

load_dotenv()
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
RERANKER_MODEL_NAME = "namdp-ptit/ViRanker"
LLM_MODEL_NAME = "llama-3.1-8b-instant"


QDRANT_URL = "https://bdead5ce-9e87-4042-b61f-68b8c285551e.europe-west3-0.gcp.cloud.qdrant.io"
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION_NAME = "ta_hospital"
VECTOR_SIZE = 1024


# Groq
GROQ_API_KEY = os.getenv("GROQ_API_KEY")