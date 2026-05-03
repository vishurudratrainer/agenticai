import os
from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Configuration
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "corporate_manual_hybrid"

def start_chat():
    # 1. Connect to Qdrant
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

    vector_store = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        sparse_embedding=sparse_embeddings,
        url=QDRANT_URL,
        collection_name=COLLECTION_NAME,
        retrieval_mode=RetrievalMode.HYBRID
    )

    # 2. Setup LLM
    llm = ChatOllama(model="llama3", temperature=0)
    
    # 3. Create a Custom Prompt
    prompt = ChatPromptTemplate.from_template("""
    Answer the question based ONLY on the following context:
    {context}
    
    Question: {question}
    """)

    # 4. THE MODERN CHAIN (No 'langchain.chains' needed!)
    # This is called an LCEL Chain. It's faster and more reliable.
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # This 'pipe' | sequence is the modern 2026 standard
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # 5. Loop
    print(f"\n[SUCCESS] Connected to Qdrant Docker")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['exit', 'quit']: break
        
        # In LCEL, we use .invoke() directly on the chain
        response = rag_chain.invoke(user_input)
        print(f"\nAI: {response}")

if __name__ == "__main__":
    start_chat()