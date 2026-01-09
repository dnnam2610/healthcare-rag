import streamlit as st
import time
from embedders import SentenceTransformerEmbedding, EmbeddingConfig
from reflection import Reflection
from retriever import TopKRetriever
from reranker import Reranker
from senmatic_router import SemanticRouter
from llms import LLMs
from config import GROQ_API_KEY, QDRANT_API_KEY, QDRANT_URL, EMBEDDING_MODEL_NAME, VECTOR_SIZE
from prompts import ANSWER_WITH_RETRIVAL, ANSWER_WITHOUT_RETRIVAL

# --------------------------
# Init components
# --------------------------
@st.cache_resource
def init_components():
    reflection_llm = LLMs(
        type="online",
        model_name="chatgroq",
        api_key=GROQ_API_KEY,
        model_version="llama-3.1-8b-instant",
        base_url="https://api.groq.com"
    )
    reflector = Reflection(llm=reflection_llm)

    router = SemanticRouter(
        embedding=SentenceTransformerEmbedding(config=EmbeddingConfig(name=EMBEDDING_MODEL_NAME, device='mps')),
        save_path='data/router/routingEmbeddings/bgem3_routing_embedding_2000.json'
    )

    retriever = TopKRetriever(
        type='qdrant',               
        embeddingName="BAAI/bge-m3",
        vector_size=VECTOR_SIZE,
        qdrant_api=QDRANT_API_KEY,
        qdrant_url=QDRANT_URL
    )

    reranker = Reranker(device='mps')

    llm = LLMs(
        type="online",
        model_name="chatgroq",
        api_key=GROQ_API_KEY,
        model_version="qwen/qwen3-32b",
        base_url="https://api.groq.com"
    )
    return reflector, router, retriever, reranker, llm


def run_pipeline(user_input):
    initial_time = time.time()

    # Reflection
    reflection_text = reflector(st.session_state.messages)
    end_time = time.time()
    print(f'Reflection text: {reflection_text}')
    print(f'Reflection time: {end_time-initial_time}')
    start_time = end_time
    st.session_state.messages.append({"role": "user", "content": reflection_text})

    # Router
    decision = router.guide(str(reflection_text))
    end_time = time.time()
    print(f"Routing time: {end_time-start_time}")
    start_time = end_time

    retrieved_docs = []
    reranked_docs = []
    if decision[1] == "medical":
        end_time = time.time()
        retrieved_items = retriever.vector_search(reflection_text, limit=5)
        print(f"Retrival time: {end_time-start_time}")
        start_time = end_time

        retrieved_docs = [str(doc.content) for doc in retrieved_items]
        end_time = time.time()
        reranked_items = reranker.rerank(reflection_text, retrieved_docs)
        print(f"Reranked time: {end_time-start_time}")
        start_time = end_time
        _, reranked_docs = zip(*reranked_items)

    # Build prompt
    if reranked_docs:
        context_text = "\n".join([doc for doc in reranked_docs])
        prompt = [{"role": "system", "content": ANSWER_WITH_RETRIVAL.format(knowledge=context_text)}]
    else:
        prompt = [{"role": "system", "content": ANSWER_WITHOUT_RETRIVAL}]
    prompt.append({"role": "user", "content": reflection_text})
 
    answer = llm.generate_content(prompt)
    end_time = time.time()
    print(f"Generating time: {end_time-start_time}")

    elapsed = end_time-initial_time
    return answer, elapsed


# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="🤖 Chatbot", page_icon="💬", layout="wide")
st.title("🤖 Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Load all components
reflector, router, retriever, reranker, llm = init_components()

# --------------------------
# Render previous chat (ChatGPT-style)
# --------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --------------------------
# Chat input
# --------------------------
if user_input := st.chat_input("Nhập câu hỏi của bạn..."):
    # Add and render user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Assistant reply
    with st.chat_message("assistant"):
        with st.spinner("Đang suy nghĩ..."):
            answer, elapsed = run_pipeline(user_input)
        st.markdown(answer)
        st.caption(f"⏱️ {elapsed:.2f} giây")
    st.session_state.messages.append({"role": "assistant", "content": answer})
