import streamlit as st
import requests
import uuid

st.title("Financial RAG Assistant")

if "sid" not in st.session_state:
    st.session_state.sid = str(uuid.uuid4())

user_input = st.chat_input("Ask me anything about financial reports")

if user_input:
    st.chat_message("user").write(user_input)
    
    # Send request to FastAPI
    try:
        resp = requests.post(
            "http://127.0.0.1:8000/chat",
            json={"query": user_input, "session_id": st.session_state.sid},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        st.chat_message("assistant").write(data.get("answer", "No answer returned."))
    except Exception as e:
        st.chat_message("assistant").write(f"Request failed: {e}")
