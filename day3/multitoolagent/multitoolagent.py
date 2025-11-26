from langchain.agents import create_agent
from langchain.tools import tool
from langchain_ollama import ChatOllama
"""``.
Agent with Multiple Tools (Review/Search)
This expands on the agent loop example by giving the LLM more options, requiring it to choose the right tool based on the context. We'll add a simple mock search tool.
"""
# 1. Define Tools
@tool
def get_product_review(product_name: str) -> str:
    """Provides a general sentiment and summary of product reviews."""
    if "AirPods" in product_name:
        return "Reviews are generally positive, highlighting noise cancellation and sound quality."
    else:
        return "Cannot find specific review data for this product."

@tool
def check_current_price(product_name: str) -> str:
    """Searches a mock database for the product's current sale price."""
    if "AirPods" in product_name:
        return "The current price is $249."
    else:
        return "Price information is currently unavailable."

tools = [get_product_review, check_current_price]

# 2. Define the Model
ollama_model = ChatOllama(model="mistral", temperature=0)

# 3. Create the Agent
agent = create_agent(
    ollama_model, 
    tools=tools, 
    system_prompt="You are an expert product advisor. Use the available tools to answer questions about price and reviews. If a user asks for both, use both tools sequentially."
)

# 4. Invoke the Agent (The agent must use two tools to answer this)
#user_query = "What are the reviews like for the AirPods, and how much do they cost?"
user_query = "What is the cost of AirPods"
print("\n--- Multi-Tool Agent Loop Result ---")

result = agent.invoke({"messages": [{"role": "user", "content": user_query}]})

print(result["messages"][-1].content)