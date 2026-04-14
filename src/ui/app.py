import sys
import os
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import streamlit as st
from src.rag.graph import app
from src.rag.state import GraphState

st.set_page_config(page_title="Adaptive RAG", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .main {
        background-color: #0E1117;
    }
    h1 {
        color: #4CAF50;
        text-align: center;
        margin-bottom: 30px;
    }
    .stChatInput {
        padding-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

st.title("Adaptive RAG")
st.markdown("<p style='text-align: center;'>An intelligent system that dynamically routes between Vector Search and Web Search using LangGraph.</p>", unsafe_allow_html=True)

with st.sidebar:
    st.header("How it works")
    st.markdown("""
    1. **Router**: LLM decides if your query needs a web search or vector DB lookup.
    2. **Retrieval/Search**: It fetches data from the chosen source.
    3. **Grader**: It checks if the retrieved documents are relevant. If not, it falls back to Web Search.
    4. **Generator**: Formulates a concise answer.
    5. **Hallucination Check**: Ensures the answer is grounded in the retrieved facts.
    """)
    st.markdown("---")
    st.markdown("**Vector Store Topics:**\n- LLM Agents\n- Prompt Engineering\n- Adversarial Attacks on LLMs")


if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

def process_query(query: str):
    inputs = {"question": query}
    
    with st.status("Processing your query...", expanded=True) as status:
        for output in app.stream(inputs):
            for key, value in output.items():
                if key == "route_question":
                    st.write("🧭 **Router**: Analyzing query source...")
                elif key == "retrieve":
                    st.write("📚 **Retriever**: Fetching documents from FAISS Vector Store...")
                elif key == "websearch":
                    st.write("🌐 **Web Search**: Fetching real-time info from the web...")
                elif key == "rewrite":
                    st.write("🔄 **Rewriter**: Re-formulating the question for better results...")
                elif key == "grade_documents":
                    st.write("⚖️ **Grader**: Checking document relevance...")
                elif key == "generate":
                    st.write("🧠 **Generator**: Synthesizing the final answer...")
                    if value.get("generation"):
                        # Save final output
                        st.session_state.final_response = value["generation"]
        
        status.update(label="Done!", state="complete", expanded=False)
        
    return st.session_state.final_response

user_input = st.chat_input("Ask a question (e.g. 'What is prompt engineering?' or 'What are LLM Agents?')...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
        
    with st.chat_message("assistant"):
        response = process_query(user_input)
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
