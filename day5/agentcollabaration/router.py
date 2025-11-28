"""
Docstring for agentcollabaration.router
A "Gatekeeper" agent analyzes the user request and decides which specialist agent should handle it. This prevents a single prompt from becoming too cluttered.

Pattern: User -> Router -> (Agent A OR Agent B OR Agent C)
"""
import ollama

def router_pattern(user_query):
    print(f"User Query: {user_query}")
    
    # 1. Router Agent (Classifier)
    # We force the output to be a single word for easy parsing
    router_response = ollama.chat(model='llama3', messages=[
        {'role': 'system', 'content': 'You are a router. Classify the query into exactly one of these categories: "MATH", "WRITING", "TECH_SUPPORT". Do not add punctuation or other text.'},
        {'role': 'user', 'content': user_query},
    ])
    category = router_response['message']['content'].strip().upper()
    print(f"Router decided: {category}")

    # 2. Handoff to Specialist
    if "MATH" in category:
        system_prompt = "You are a mathematician. Solve this step-by-step."
    elif "WRITING" in category:
        system_prompt = "You are a poet. Answer in a rhyme."
    else:
        system_prompt = "You are IT support. Ask if they have tried turning it off and on again."

    final_response = ollama.chat(model='llama3', messages=[
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': user_query},
    ])
    
    print(f"Response: {final_response['message']['content']}")

# Usage Examples:
router_pattern("Calculate the square root of 144")
router_pattern("Write a poem about rust")