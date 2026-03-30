import streamlit as st
import requests
import json
import os

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")

st.title("🤖 Retrieval-Augmented Generation System")
st.markdown("Ask questions about your documents - answers come with citations!")

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    api_url = st.text_input("API URL", value=os.getenv("API_URL", "http://localhost:8000"))
    k_documents = st.slider("Number of documents to retrieve", 1, 10, 3)

    st.header("📤 Add Document")
    uploaded_file = st.file_uploader("Upload a .txt or .pdf", type=["txt", "pdf"])
    if uploaded_file and st.button("Ingest Document"):
        try:
            files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
            response = requests.post(f"{api_url}/ingest", files=files)
            if response.status_code == 200:
                result = response.json()
                st.success(f"Added {result['file_name']} ({result['chunks_added']} chunks)")
            else:
                st.error(f"Ingest failed: {response.status_code} {response.text}")
        except Exception as e:
            st.error(f"Failed to upload: {e}")
    
    st.header("📚 Available Sources")
    try:
        response = requests.get(f"{api_url}/sources")
        if response.status_code == 200:
            sources = response.json().get("sources", [])
            for source in sources:
                st.write(f"- {source}")
    except:
        st.write("⚠️ API not connected")
    
    st.header("💡 Example Questions")
    examples = [
        "What is RAG?",
        "How to deploy AI to production?",
        "What Python libraries are used for AI?",
        "How does retrieval work in RAG?"
    ]
    for example in examples:
        if st.button(example):
            st.session_state.query = example

# Main chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "citations" in message and message["citations"]:
            with st.expander("📚 View Sources"):
                for citation in message["citations"]:
                    st.write(f"**Source:** {citation['source']}")
                    st.write(f"**Relevance:** {citation['relevance']:.2f}")
                    st.write(f"**Excerpt:** {citation['excerpt']}...")
                    st.divider()

# Input box
query = st.chat_input("Ask me anything about your documents...")

if query or ("query" in st.session_state and st.session_state.query):
    if "query" in st.session_state and st.session_state.query:
        query = st.session_state.query
        del st.session_state.query
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
    
    # Get response from API
    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating answer..."):
            try:
                response = requests.post(
                    f"{api_url}/ask",
                    json={"query": query, "k": k_documents}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Display answer
                    st.markdown(result["answer"])
                    
                    # Display citations
                    if result["citations"]:
                        with st.expander("📚 View Sources"):
                            for citation in result["citations"]:
                                st.write(f"**Source:** {citation['source']}")
                                st.write(f"**Relevance:** {citation['relevance']:.2f}")
                                st.write(f"**Excerpt:** {citation['excerpt']}...")
                                st.divider()
                    
                    # Add to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result["answer"],
                        "citations": result["citations"]
                    })
                    
                    # Show response time
                    st.caption(f"⏱️ Response time: {result['response_time_ms']}ms")
                else:
                    st.error(f"API Error: {response.status_code} {response.text}")
                    
            except Exception as e:
                st.error(f"Failed to connect to API: {e}")