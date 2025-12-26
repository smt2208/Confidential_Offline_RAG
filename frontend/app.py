"""Streamlit chat interface for Confidential Interrogation Records RAG."""

import streamlit as st
import httpx
import uuid
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend import config

API_URL = os.getenv("API_URL", f"http://localhost:{config.API_PORT}")

# Page configuration
st.set_page_config(
    page_title="Confidential Records Assistant",
    page_icon="🔒",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        max-width: 900px;
        margin: 0 auto;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.title("🔒 Confidential Records Assistant")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# Display chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


def get_response(prompt: str, thread_id: str) -> str:
    """Send message to API and get response."""
    with httpx.Client(timeout=120.0) as client:
        response = client.post(
            f"{API_URL}/chat",
            json={"message": prompt, "thread_id": thread_id}
        )
        response.raise_for_status()
        return response.json()["response"]


# Chat input
if prompt := st.chat_input("Ask about interrogation records..."):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get AI response
    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                response = get_response(prompt, st.session_state.thread_id)
            st.markdown(response)
            st.session_state.messages.append({
                "role": "assistant",
                "content": response
            })
            st.rerun()
        except httpx.ConnectError:
            st.error(
                "Cannot connect to API. Please ensure the server is running:\n\n"
                "```bash\npython -m uvicorn backend.api:app --reload\n```"
            )
        except Exception as e:
            st.error(f"Error: {str(e)}")
