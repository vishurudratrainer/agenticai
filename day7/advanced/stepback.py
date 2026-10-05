"""
Docstring for agentloop.advanced.stepback

This advanced technique prevents the LLM 
from getting bogged down in specifics. 
It forces the LLM to generate a generalized,
 "step-back" question first, 
 uses that to retrieve general context,
   and then answers the original,
     specific question. 
     This is combined with LangChain Routers.
"""

from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableSequence
from langchain_core.output_parsers import StrOutputParser

# --- A. Define Step-Back Prompt ---
STEP_BACK_PROMPT = ChatPromptTemplate.from_template(
    """
    You are an expert at simplification. 
    Given a highly specific question, generate a single, 
    more general,
     conceptual question that a human would ask to understand
       the core idea first.
    
    Example:
    Original: What are the latency benefits of using a CDN with HTTP/3?
    Step-Back: What is a CDN and how does it affect latency?
    
    Original Question: {question}
    Step-Back Question: 
    """
)

# --- B. Define Final Answer Prompt ---
FINAL_PROMPT = ChatPromptTemplate.from_template(
    """
    Answer the user's ORIGINAL question based on the CONTEXT provided. 
    Use the generalized answer only for conceptual framing, if necessary.
    
    ORIGINAL QUESTION: {original_question}
    GENERALIZED CONTEXT: {general_context}
    
    FINAL ANSWER:
    """
)

# 1. Initialize Ollama LLM (Used for both simplification and final answer)
ollama_llm = ChatOllama(model="llama3", temperature=0)

# --- C. Define Mock Retrieval Tool (In a real app, this would be a full RAG pipeline) ---
def mock_general_retrieval(step_back_question: str) -> str:
    """Simulates retrieving general conceptual knowledge."""
    if "neural network" in step_back_question.lower():
        return "A neural network is a series of algorithms that model data, helping to identify complex patterns. Activation functions are crucial components."
    return "No general concept found."

# --- D. The Advanced Chain Structure ---

# 1. Chain to generate the step-back question
step_back_chain = STEP_BACK_PROMPT | ollama_llm | StrOutputParser()

# 2. Main chain structure using assignment
full_chain = RunnablePassthrough.assign(
    # Preserve the original question
    original_question=RunnablePassthrough(),
    
    # Generate the generalized question
    step_back_q=step_back_chain,
).assign(
    # Retrieve general context using the step-back question
    general_context=lambda x: mock_general_retrieval(x['step_back_q']),
).assign(
    # Generate the final answer using all components
    final_response=FINAL_PROMPT | ollama_llm | StrOutputParser()
).pick("final_response")

# --- E. Execute ---
user_query = "How do ReLU activation functions relate to non-linearity in deep learning models?"

print(f"\n--- Step-Back Prompting System ---")
print(f"User Query: {user_query}")
print("-" * 40)

final_answer = full_chain.invoke({"question":user_query})

print(f"\n✅ Final Answer (Conceptual Framing):")
print(final_answer)