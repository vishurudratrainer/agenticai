# pip install langgraph langchain_core langchain_community
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List, Union
import operator
import json

# --- 1. Ollama Configuration (Use your validated settings) ---
ollama_llm = ChatOllama(model="llama3:latest", base_url="http://127.0.0.1:11434")

# --- 2. Define State ---
class HandoffState(TypedDict):
    request: str
    topic: str
    response: str
    
# --- 3. Define Agents/Nodes ---

def router_agent_node(state: HandoffState):
    """
    The Router Agent classifies the request and decides the next step.
    It outputs the decision as a JSON string.
    """
    router_prompt = f"""
    Analyze the user's request: '{state["request"]}'.
    
    If the request involves complex math or statistics, set 'topic' to 'MATH'.
    Otherwise, set 'topic' to 'GENERAL'.
    
    Output ONLY a JSON object: {{"topic": "MATH" | "GENERAL"}}
    """
    
    messages = [
        SystemMessage(content="You are a routing expert. Output only the requested JSON structure."),
        HumanMessage(content=router_prompt)
    ]
    
    try:
        response = ollama_llm.invoke(messages).content.strip()
        decision = json.loads(response)
        return {"topic": decision.get("topic", "GENERAL")}
    except Exception:
        # Fallback if Ollama doesn't produce valid JSON
        return {"topic": "GENERAL"}

def specialist_agent_node(state: HandoffState):
    """
    The Specialist Agent handles MATH topics.
    """
    print("🤖 Specialist Agent: Handling complex math task...")
    
    specialist_prompt = f"Solve this complex statistics problem step-by-step: {state['request']}"
    messages = [SystemMessage(content="You are a brilliant mathematician."), HumanMessage(content=specialist_prompt)]
    
    response = ollama_llm.invoke(messages).content
    return {"response": f"Math Specialist Solution: {response}"}

def general_agent_node(state: HandoffState):
    """
    The General Agent handles all other topics.
    """
    print("🧠 General Agent: Providing a simple answer...")
    
    general_prompt = f"Provide a brief, general answer to: {state['request']}"
    messages = [SystemMessage(content="You are a helpful general assistant."), HumanMessage(content=general_prompt)]
    
    response = ollama_llm.invoke(messages).content
    return {"response": f"General Answer: {response}"}

# --- 4. Define Routing (Conditional Edge) ---
def route_to_specialist(state: HandoffState) -> str:
    """Routes the graph based on the topic determined by the Router."""
    if state["topic"] == "MATH":
        return "specialist"
    return "general"
    # 

# --- 5. Build the LangGraph ---
workflow = StateGraph(HandoffState)
workflow.add_node("router", router_agent_node)
workflow.add_node("specialist", specialist_agent_node)
workflow.add_node("general", general_agent_node)

workflow.set_entry_point("router")

# Conditional edge from Router to Specialists/General
workflow.add_conditional_edges(
    "router",
    route_to_specialist,
    {"specialist": "specialist", "general": "general"}
)

# Both specialist paths terminate
workflow.add_edge("specialist", END)
workflow.add_edge("general", END)

app = workflow.compile()

# Example: Run the graph
request_math = "Calculate the standard deviation of the following data set: [10, 12, 23, 23, 16, 23, 21, 16]"
final_state_math = app.invoke({"request": request_math, "topic": "", "response": ""})
print("\n--- Dynamic Handoff Result (MATH) ---")
print(final_state_math["response"])