from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser

# --- Setup: Define Model and History ---
llm = ChatOllama(model="llama3", temperature=0)
chat_history = [
    HumanMessage(content="My name is Alex."),
    AIMessage(content="Hello Alex! How can I help you today?"),
]

# --- 1. Define the Condensing Prompt ---
# This prompt uses the history to rephrase the latest input into a standalone question.
condensing_template = """Given the following conversation and a follow-up question, 
rephrase the follow-up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone Question:"""

condensing_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", condensing_template),
        MessagesPlaceholder(variable_name="chat_history"), # Placeholder for the history list
        ("human", "{question}"),
    ]
)

# --- 2. Build the Condensing Chain (The Brains) ---
# This chain takes history and the new question, and outputs a clear, standalone question.
contextualize_chain = (
    condensing_prompt
    | llm
    | StrOutputParser()
)

# --- 3. Example Execution ---
new_question = "What is my name?"

# The chain requires the full history and the new question
input_data = {
    "question": new_question,
    "chat_history": chat_history # Pass the entire history list
}

standalone_question = contextualize_chain.invoke(input_data)

print(f"Original Question: {new_question}")
print(f"Contextualized Question: {standalone_question}")