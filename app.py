import streamlit as st
from pipeline import pipeline

st.set_page_config(page_title="Medical A2A Chat", layout="centered")

st.title("🩺 Trợ lý y tế thông minh")

# ===== SESSION STATE =====
if "messages" not in st.session_state:
    st.session_state.messages = []

# ===== CHAT HISTORY =====
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ===== USER INPUT =====
user_input = st.chat_input("Nhập câu hỏi của bạn...")

if user_input:
    # Lưu user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    # Gọi pipeline
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = pipeline(st.session_state.messages, use_reranker=False)

        st.markdown(answer)

    # Lưu assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })
