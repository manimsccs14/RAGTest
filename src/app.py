import streamlit as st
from VectorStore import VectorStore
from FileLoader import FileLoader
from EmbeddingManager import EmbeddingManager
from RAGRetriever import RAGRetriever
from LLMProcess import rag_qa, llm
import os
from URLLoader import URLLoader
st.title("RAG Test Application")
st.write("This application demonstrates Retrieval-Augmented Generation (RAG) using LangChain and Groq LLM.")

# Initialize Vector Store, Embedding Manager, and RAG Retriever
@st.cache_resource
def load_or_create_vector_store(n: int = 1):
    # Check if vector store already exists
    print("Checking for existing vector store...")
    if not os.path.exists("./chroma_db"):
        fileLoader = FileLoader()
        urlLoader = URLLoader()
        fileDocuments = fileLoader.load_directory()
        urlDocuments = urlLoader.load_url("https://www.citigroup.com/global/about-us/leadership")
        documents = fileDocuments + urlDocuments
        print(f"Loaded {len(documents)} documents from directory.") 
        vector_store = VectorStore()
        embedding_manager = EmbeddingManager()
        chunks = embedding_manager.chunk_documents(documents)
        text = [chunk.page_content for chunk in chunks]
        embedding = embedding_manager.generate_embedding(text)
        vector_store.add_documents(chunks, embedding)
    else:
        print("Vector store found. Loading existing vector store...")
        vector_store = VectorStore()
        embedding_manager = EmbeddingManager()
    return vector_store, embedding_manager
    
vector_store, embedding_manager = load_or_create_vector_store(1)    
rag_retriever = RAGRetriever(vector_store, embedding_manager)

# Initialize the chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Disploay chat messages from history
for message in st.session_state.messages:
    if message["role"] == "user":
        st.chat_message("user").markdown(message["content"])
    else:
        st.chat_message("assistant").markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask me about Angular!"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate response from RAG QA system
    with st.chat_message("assistant"):
        response = rag_qa(prompt, rag_retriever, llm)
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})