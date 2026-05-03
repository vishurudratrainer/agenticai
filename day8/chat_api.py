from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "api_news_data"

def start_api_chat():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

    vector_store = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        sparse_embedding=sparse_embeddings,
        url=QDRANT_URL,
        collection_name=COLLECTION_NAME,
        retrieval_mode=RetrievalMode.HYBRID
    )

    llm = ChatOllama(model="llama3", temperature=0)
    
    # Advanced Prompt that uses the metadata source
    prompt = ChatPromptTemplate.from_template("""
    Answer using the context. Cite the source URL at the end of your answer.
    
    Context: {context}
    
    Question: {question}
    """)

    retriever = vector_store.as_retriever(search_kwargs={"k": 2})

    def format_docs(docs):
        # This helper adds the source URL to the text the AI reads
        return "\n\n".join(f"{d.page_content}\n(Source: {d.metadata['source']})" for d in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print(f"\n[SYSTEM] API Chat Ready! Ask about recent space news.")
    while True:
        query = input("\nYou: ")
        if query.lower() in ['exit', 'quit']: break
        print(f"\nAI: {rag_chain.invoke(query)}")

if __name__ == "__main__":
    start_api_chat()
    
"""
: What is the most recent space mission mentioned in the data, and when was it published?
"""