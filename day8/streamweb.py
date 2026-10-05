import streamlit as st
import requests
from langchain_ollama import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from bs4 import BeautifulSoup
from functools import lru_cache

st.set_page_config(page_title="Ollama RAG Web Scraper", page_icon="🔗")

st.title("🔗 Ollama RAG Web Scraper")
st.caption("Fetch content from any URL, embed it locally with Ollama, and chat about it.")

# --- 1. Configuration Sidebar ---
st.sidebar.header("Ollama Model Configuration")
llm_model = st.sidebar.selectbox("LLM Model (Chat)", ["llama3", "mistral"], index=0)
embed_model = st.sidebar.text_input("Embedding Model", value="nomic-embed-text")
st.sidebar.caption("Ensure these models are running in Ollama.")

# --- 2. Data Loading and RAG Setup ---

@lru_cache(maxsize=1)
def fetch_web_content(url):
    """Fetches and cleans text content from a given URL."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        # Remove script and style elements
        for script_or_style in soup(['script', 'style']):
            script_or_style.decompose()
        # Extract meaningful text, ignoring scripts and styles
        text = soup.get_text(separator=' ', strip=True)
        return text
    except Exception as e:
        st.error(f"Error fetching web content from {url}: {e}")
        return None

def setup_rag_chain(docs, llm_model, embed_model):
    """Initializes embeddings, vectorstore, and the RAG chain."""
    try:
        # B. Chunking and Embedding
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(docs)

        ollama_embeddings = OllamaEmbeddings(model=embed_model)
        vectorstore = FAISS.from_documents(docs, ollama_embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        # C. RAG Chain Definition
        ollama_llm = ChatOllama(model=llm_model, temperature=0)

        RAG_PROMPT_TEMPLATE = """
        You are a documentation analysis assistant. Answer the user's question based ONLY on the provided web content. 
        CONTEXT:
        {context}
        QUESTION: {question}
        """
        rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | rag_prompt
            | ollama_llm
            | StrOutputParser()
        )
        return rag_chain
    except Exception as e:
        st.error(f"Error setting up RAG Chain: {e}")
        return None

# --- 3. Main Application Flow ---

# State management for the URL and the RAG chain
if 'rag_chain' not in st.session_state:
    st.session_state['rag_chain'] = None

# Input for URL
url = st.text_input(
    "Enter a URL to analyze:",
    value="https://python.langchain.com/v0.2/docs/concepts/#prompts",
    key="url_input"
)

# Button to process the URL
if st.button("Process URL and Initialize Chat"):
    with st.spinner(f"Fetching and embedding content from {url}..."):
        page_text = fetch_web_content(url)
        
        if page_text:
            web_documents = [Document(page_content=page_text, metadata={"source": url})]
            
            # Setup the RAG chain
            rag_chain = setup_rag_chain(web_documents, llm_model, embed_model)
            st.session_state['rag_chain'] = rag_chain
            st.session_state['messages'] = [{"role": "assistant", "content": f"Successfully loaded and embedded content from **{url}**. Ask me a question about it!"}]
            st.rerun()
        else:
            st.session_state['rag_chain'] = None
            st.session_state['messages'] = []

# --- 4. Chat Interface ---

if st.session_state['rag_chain']:
    # Display chat history
    if "messages" in st.session_state:
        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])

    # User Input & Query RAG System
    if prompt := st.chat_input("Ask a question about the documentation..."):
        # Add user message to state and display
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        # Invoke the RAG chain
        with st.chat_message("assistant"):
            try:
                # D. Query the RAG System
                with st.spinner("Retrieving context and generating answer..."):
                    # Using invoke is cleaner for direct Streamlit interaction
                    final_answer = st.session_state['rag_chain'].invoke(prompt)
                
                # Display final answer
                st.write(final_answer)
                
                # Save final answer to state
                st.session_state.messages.append({"role": "assistant", "content": final_answer})
                
            except Exception as e:
                st.error(f"An error occurred during chain execution: {e}")
else:
    st.info("Enter a URL and click 'Process URL and Initialize Chat' to start.")