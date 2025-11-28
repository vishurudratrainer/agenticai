# pip install langgraph langchain_core langchain_community
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
import re

# --- 1. Ollama Configuration ---
ollama_llm = ChatOllama(model="llama3", base_url="http://localhost:11434")

# --- 2. Define State ---
class RefineState(TypedDict):
    draft: str
    feedback: str
    iteration: Annotated[int, operator.add]

# --- 3. Define Nodes (Agents) ---

def writer_node(state: RefineState) -> RefineState:
    """The Writer Agent drafts or refines the report based on feedback."""
    print(f"\n📝 Writer Agent: Iteration {state['iteration'] + 1}")
    
    prompt = f"TASK: Draft a 300-word introduction to 'Quantum Computing for Beginners'."
    if state["feedback"]:
        prompt = f"REFINEMENT TASK: Refine the previous draft based on this feedback:\n'{state['feedback']}'\n\nPrevious Draft:\n'{state['draft']}'\n\nGenerate the new, improved draft. Output ONLY the new draft."

    messages = [
        SystemMessage(content="You are a clear and concise technical writer. Output ONLY the draft text."),
        HumanMessage(content=prompt)
    ]
    
    response = ollama_llm.invoke(messages)
    return {"draft": response.content, "feedback": "", "iteration": 1}

def critic_node(state: RefineState) -> RefineState:
    """The Critic Agent reviews the draft and provides feedback or approval."""
    print(f"🧐 Critic Agent: Reviewing Draft...")

    prompt = f"""
    Review the following draft. Check for clarity, tone (beginner-friendly), and accuracy.
    
    If the draft is EXCELLENT and ready, respond with the single word: **APPROVED**.
    
    If it needs improvement, provide SPECIFIC, actionable feedback starting with 'FEEDBACK: ' (e.g., 'FEEDBACK: Simplify the explanation of superposition and use an analogy.').
    
    Draft:\n{state['draft']}
    """
    messages = [
        SystemMessage(content="You are a meticulous content reviewer."),
        HumanMessage(content=prompt)
    ]
    
    response = ollama_llm.invoke(messages)
    
    # Use simple string check for this example
    if response.content.strip().upper() == "APPROVED":
        print("🎉 CRITIC: APPROVED!")
        return {"feedback": "APPROVED"}
    else:
        # Extract feedback (e.g., stripping 'FEEDBACK: ')
        feedback = re.sub(r'FEEDBACK:\s*', '', response.content, flags=re.IGNORECASE).strip()
        print(f"❌ CRITIC: Provided Feedback: {feedback}")
        return {"feedback": feedback}

# --- 4. Define Conditional Edge (Router) ---
def route_refinement(state: RefineState) -> str:
    """Routes based on the Critic's output."""
    if state["feedback"] == "APPROVED":
        return "end"
    
    if state["iteration"] >= 3: # Set max iterations to prevent infinite loop
        print("⚠️ Max iterations reached. FORCING END.")
        return "end"
        
    return "writer"

# --- 5. Build the LangGraph ---
workflow = StateGraph(RefineState)
workflow.add_node("writer", writer_node)
workflow.add_node("critic", critic_node)

workflow.set_entry_point("writer")

# Writer -> Critic
workflow.add_edge("writer", "critic")

# Critic -> Decision Point
workflow.add_conditional_edges(
    "critic",
    route_refinement,
    {"writer": "writer", "end": END}
)

app = workflow.compile()

# Example: Run the graph
# initial_state = {"draft": "", "feedback": "", "iteration": 0}
# final_state = app.invoke(initial_state)
# print("\n--- Iterative Refinement Final Draft ---")
# print(final_state["draft"])