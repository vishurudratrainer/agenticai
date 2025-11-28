# pip install langgraph langchain_core langchain_community
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool # We still need the tool decorator
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List, Union
import operator

# --- 1. Ollama Configuration (Your validated fix) ---
ollama_llm = ChatOllama(model="mistral", base_url="http://127.0.0.1:11434")

# --- 2. Define Tools (Worker Agent Actions) ---
# NOTE: Tools must be defined so the supervisor LLM can use them.
@tool
def code_analyst_tool(code_snippet: str) -> str:
    """Analyzes a Python code snippet for bugs, efficiency, and best practices."""
    return f"Code Analyst Report:\n- Analyzed code for bugs and confirmed efficiency."

@tool
def doc_writer_tool(code_snippet: str) -> str:
    """Writes a professional Google-style docstring for a Python function."""
    return f"Doc Writer Output:\n'''\nFunction documented successfully with clear Args and Returns.\n'''"

tools = [code_analyst_tool, doc_writer_tool]
tool_name_map = {t.name: t for t in tools} # Map tool name (string) to the function

# --- 3. Define State and Supervisor Agent ---
class AgentState(TypedDict):
    input: str
    chat_history: Annotated[List[Union[HumanMessage, SystemMessage]], operator.add]
    agent_outcome: Union[str, None] # Holds the LLM's response (which contains tool calls)

# The Supervisor LLM is configured to output tool calls
tool_names = list(tool_name_map.keys())
supervisor_system_prompt = f"""You are the SUPERVISOR AGENT. Your job is to analyze the user's request: '{{input}}', and decide which specialized tool (Agent) to call. You MUST choose one of the following tools: {tool_names}. If the request is not related to those tools, respond directly."""

supervisor_llm_with_tools = ollama_llm.bind_tools(tools)

def run_supervisor(state: AgentState):
    """Node function for the Supervisor Agent (LLM call)."""
    messages = [
        SystemMessage(content=supervisor_system_prompt.format(input=state["input"])),
        HumanMessage(content=state["input"])
    ]
    response = supervisor_llm_with_tools.invoke(messages)
    return {"agent_outcome": response}

# --- 4. The FIX: Custom Tool Execution Node ---
def execute_tools(state: AgentState):
    """
    Manually executes the tool call decided by the Supervisor.
    This replaces the need for ToolExecutor.
    """
    # Assuming only one tool call for simplicity in this pattern
    tool_call = state["agent_outcome"].tool_calls[0]
    
    # Get the function and arguments
    tool_name = tool_call["name"]
    tool_args = tool_call["args"]
    
    # Look up and execute the tool
    selected_tool_fn = tool_name_map.get(tool_name)
    
    if selected_tool_fn:
        # **Crucial: We use the .invoke() method of the LangChain tool object**
        output = selected_tool_fn.invoke(tool_args)
        
        # Return the tool output to the chat history
        return {"chat_history": [HumanMessage(content=f"Tool {tool_name} executed. Result: {output}")]}
    else:
        return {"chat_history": [HumanMessage(content="Error: Tool not found.")]}

# --- 5. Define Routing ---
def route_next(state: AgentState):
    """Routes based on whether the Supervisor decided to call a tool."""
    # Check if the LLM's response contains tool_calls
    if state["agent_outcome"].tool_calls:
        return "call_tool"
    return "end"

# --- 6. Build the LangGraph ---
workflow = StateGraph(AgentState)
workflow.add_node("supervisor", run_supervisor)
workflow.add_node("call_tool", execute_tools) # <-- Use the custom node

workflow.set_entry_point("supervisor")

workflow.add_conditional_edges(
    "supervisor",
    route_next,
    {"call_tool": "call_tool", "end": END}
)
# After the tool is called, the graph ends (for this simple example)
workflow.add_edge('call_tool', END) 

app = workflow.compile()

# Example invocation:
inputs = {"input": "I need documentation for a function: def my_func(x): return x*2", "chat_history": []}
final_state = app.invoke(inputs)
print(final_state["chat_history"])