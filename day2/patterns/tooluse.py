import requests
import json

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "mistral"   # or any model you installed via `ollama pull`

import requests
import json


def ollama_chat(prompt):
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False
    }
    r = requests.post(OLLAMA_URL, json=payload)
    return r.json()["response"]



def get_weather(city):
    fake_data = {
        "bangalore": "28°C | Cloudy",
        "mumbai": "32°C | Humid",
        "delhi": "26°C | Clear skies"
    }
    return fake_data.get(city.lower(), "No data")

def agent_weather(query):
    # Step 1: Ask LLM what tool to call
    react_prompt = f"""
You are an agent with access to:

Tool: weather[city]

If user asks for weather, respond:
Action: weather[city]

Else say:
Action: none

User query: {query}
"""

    plan = ollama_chat(react_prompt)
    print("LLM Decision:\n", plan)

    import re
    match = re.search(r"Action:\s*(\w+)\[(.*?)\]", plan)

    if not match:
        return "Could not parse action"

    tool, arg = match.group(1), match.group(2)

    if tool == "weather":
        obs = get_weather(arg)
    else:
        obs = "No tool used"

    # Step 2: Ask LLM to produce final answer
    final_prompt = f"""
You said:
{plan}

Tool observation:
{obs}

Give final answer to user.
"""

    answer = ollama_chat(final_prompt)
    print("\nFinal Answer:\n", answer)
    return answer


# Test
agent_weather("What is the weather in Bangalore today?")
