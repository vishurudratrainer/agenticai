import requests
import json

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "mistral"

def ollama_chat(prompt):
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False
    }
    r = requests.post(OLLAMA_URL, json=payload)
    return r.json()["response"]


def calculator(expression):
    try:
        return eval(expression)
    except:
        return "Error evaluating expression."


def react(prompt):
    # 1. Ask LLM to produce reasoning + an action
    reasoning_prompt = f"""
You are using ReAct.
Think step by step, then decide an action.

Available Tools:
1. calculator[expression]

Your response format:
Thought: ...
Action: <tool>[<input>]
"""
    thoughts = ollama_chat(reasoning_prompt + "\nUser query: " + prompt)
    print("LLM Thoughts + Action:\n", thoughts)

    # 2. Extract the action
    import re
    match = re.search(r"Action:\s*(\w+)\[(.*?)\]", thoughts)
    if not match:
        return "No tool used. Final answer: " + thoughts

    tool, tool_input = match.group(1), match.group(2)
    print("tool",tool)
    print("tool_input",tool_input)
    if tool == "calculator":
        tool_output = calculator(tool_input)
    else:
        tool_output = "Unknown tool"

    print("\nTool Output:", tool_output)

    # 3. Give observation back to LLM
    final_prompt = f"""
        You previously said:
        {thoughts}

        Observation from tool:
        {tool_output}

        Give final answer now.
        """

    final_answer = ollama_chat(final_prompt)
    print("\nFinal Answer:")
    print(final_answer)
    return final_answer


# Test
react("What is 234 * 89?")
