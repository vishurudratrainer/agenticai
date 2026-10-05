"""
Custom Agent Workflows & Planning using Ollama (Python)
--------------------------------------------------------

This module demonstrates:
 ✓ LLM-based Planning
 ✓ Tool Registry
 ✓ Workflow Execution Cycle
 ✓ ReAct Planning Loop (Thought → Action → Observation)
 ✓ Custom Agent Architecture

Requires:
    pip install ollama
    ollama pull llama3.2  (or any model)

By default, Ollama must be running locally at:
    http://localhost:11434
"""

import ollama
import time
import logging
from typing import Dict, List, Callable, Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s  [%(levelname)s]  %(message)s")

# --------------------------------------------------------------------
# OLLAMA CLIENT
# --------------------------------------------------------------------
MODEL = "mistral"   # Change to any model you installed in Ollama
client = ollama.Client()   # Connects to local Ollama instance


# --------------------------------------------------------------------
# LLM CALL HELPER
# --------------------------------------------------------------------
def llm(messages: List[Dict[str, str]]):
    """Wrapper around Ollama.chat"""
    response = client.chat(
        model=MODEL,
        messages=messages
    )
    return response["message"]["content"].strip()


# --------------------------------------------------------------------
# PLANNER
# --------------------------------------------------------------------
def plan_steps(goal: str, context: str = "") -> List[str]:
    """
    LLM converts a GOAL → multi-step actionable plan.
    """
    system_prompt = (
        "You are a planning agent. "
        "Return 3–6 short, numbered, executable steps. "
        "Do NOT add extra explanation."
    )

    user_prompt = f"Goal: {goal}\nContext: {context}\nCreate short numbered steps."

    text = llm([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ])

    # Normalize into a clean list
    steps = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        line = line.lstrip("1234567890. )(").strip()
        steps.append(line)

    return steps


# --------------------------------------------------------------------
# TOOL REGISTRY
# --------------------------------------------------------------------
class ToolRegistry:
    def __init__(self):
        self.tools: Dict[str, Callable[[str], Any]] = {}

    def register(self, name: str, fn: Callable[[str], Any]):
        self.tools[name] = fn

    def call(self, name: str, arg: str):
        if name not in self.tools:
            return {"ok": False, "error": f"Tool '{name}' not found"}
        try:
            result = self.tools[name](arg)
            return {"ok": True, "result": result}
        except Exception as e:
            return {"ok": False, "error": str(e)}


tools = ToolRegistry()


# Example Tool 1: Summarize using LLM
def tool_summarize(text: str):
    prompt = [
        {"role": "system", "content": "You summarize content concisely in 3–4 lines."},
        {"role": "user", "content": text}
    ]
    return llm(prompt)


# Example Tool 2: Keyword extractor
def tool_keywords(text: str):
    prompt = [
        {"role": "system", "content": "Extract 5 important keywords (comma separated)."},
        {"role": "user", "content": text}
    ]
    return llm(prompt)


# Register tools
tools.register("summarize", tool_summarize)
tools.register("keywords", tool_keywords)


# --------------------------------------------------------------------
# WORKFLOW EXECUTOR
# --------------------------------------------------------------------
def execute_step(step: str):
    """
    Executes a single step.
    Pattern for calling tools:
         CALL_TOOL:<tool_name>:<argument>
    Otherwise → LLM executes the step directly.
    """
    logging.info(f"Executing step: {step}")

    step = step.strip()

    # Tool call format
    if step.startswith("CALL_TOOL:"):
        try:
            _, tool, arg = step.split(":", 2)
        except ValueError:
            return {"ok": False, "error": "Invalid CALL_TOOL format"}

        return tools.call(tool, arg)

    # Default → Let LLM handle the step
    prompt = [
        {"role": "system", "content": "Execute the step and return a short result."},
        {"role": "user",   "content": step}
    ]
    result = llm(prompt)
    return {"ok": True, "result": result}


# --------------------------------------------------------------------
# REACT LOOP (Thought → Action → Observation)
# --------------------------------------------------------------------
def react_loop(goal: str, context: str = "", max_iters: int = 2):
    """
    A lightweight ReAct planning loop using Ollama.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You follow Thought → Action → Observation loop. "
                "Prefer tool calls using CALL_TOOL:<tool>:<arg>. "
                "Stop when the goal is achieved and say DONE."
            )
        },
        {
            "role": "user",
            "content": f"Goal: {goal}\nContext: {context}\nBegin."
        }
    ]

    for i in range(max_iters):
        logging.info(f"ReAct Iteration {i+1}")
        reply = llm(messages)

        # Extract Thought & Action
        thought, action = "", ""
        for line in reply.split("\n"):
            if line.lower().startswith("thought:"):
                thought = line.split(":", 1)[1].strip()
            if line.lower().startswith("action:"):
                action = line.split(":", 1)[1].strip()

        if "DONE" in action.upper():
            return {"ok": True, "result": "Goal achieved"}

        # Execute action
        result = execute_step(action)
        observation = result["result"] if result["ok"] else result.get("error", "")

        # Feed back observation
        messages.append({"role": "assistant", "content": reply})
        messages.append({"role": "user",      "content": f"Observation: {observation}"})

        time.sleep(0.25)

    return {"ok": False, "error": "Max iterations reached"}


# --------------------------------------------------------------------
# FULL WORKFLOW DEMO
# --------------------------------------------------------------------
def demo():
    text = """
    Agents are modular systems built using memory, tools,
    planners, and feedback loops. They help automate multi-step tasks.
    """

    goal = "Summarize the document and extract keywords."

    # 1. PLAN
    steps = plan_steps(goal, context="Use CALL_TOOL instructions if needed.")
    print("\n=== PLAN ===")
    for s in steps:
        print("-", s)

    # 2. EXECUTE PLAN
    print("\n=== EXECUTING STEPS ===")
    outputs = []
    for s in steps:
        # Auto-convert summarization tasks into tool calls
        if "summar" in s.lower():
            s = f"CALL_TOOL:summarize:{text}"
        if "keyword" in s.lower():
            s = f"CALL_TOOL:keywords:{text}"
        outputs.append(execute_step(s))

    print("\n=== RESULTS ===")
    for out in outputs:
        print(out)

    # 3. REACT LOOP
    print("\n=== REACT LOOP ===")
    react_res = react_loop(
        goal="Produce a 2-line insight and then say DONE.",
        context=text
    )
    print(react_res)


if __name__ == "__main__":
    demo()
