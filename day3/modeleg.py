from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

# Initialize the chat model, specifying the Ollama model name
try:
    ollama_model = ChatOllama(model="llama3", temperature=0.7)
    
    # Simple invoke with a list of messages
    response = ollama_model.invoke([
        SystemMessage(content="You are a polite, helpful AI running on a local server."),
        HumanMessage(content="Explain the difference between LangChain and Ollama in one sentence.")
    ])

    print("--- Model Invoke Result ---")
    print(response.content)

except Exception as e:
    print(f"Error: Could not connect to Ollama. Ensure the server is running and 'llama3' is pulled. Details: {e}")