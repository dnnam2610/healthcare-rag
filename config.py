import os
from dotenv import load_dotenv

load_dotenv()
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
RERANKER_MODEL_NAME = "namdp-ptit/ViRanker"
LLM_MODEL_NAME = "llama-3.1-8b-instant"


QDRANT_URL = "https://a22aee43-5786-400f-9ccf-e29b680714c0.europe-west3-0.gcp.cloud.qdrant.io:6333"
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION_NAME = "ta_hospital"
VECTOR_SIZE = 1024


# Groq
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Debug
DEBUG_TIMING = True