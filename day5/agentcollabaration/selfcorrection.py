"""
Docstring for agentcollabaration.selfcorrection

Agents produce better results when they can "think" and "critique" their own work. In this pattern, one agent generates a solution, and a second "Critic" agent reviews it. If it's not good enough, it loops back with feedback.

Pattern: Generator -> Evaluator -> (Pass OR Retry Loop)
"""

import ollama

def evaluator_optimizer(topic):
    print(f"--- Starting Goal: Write a high-quality snippet about {topic} ---")
    
    messages = [{'role': 'user', 'content': f"Write a short python code snippet to {topic}."}]
    
    # Allow up to 3 improvement cycles
    for i in range(3):
        print(f"\n--- Cycle {i+1} ---")
        
        # 1. Generator generates (or regenerates based on history)
        response = ollama.chat(model='llama3', messages=messages)
        current_content = response['message']['content']
        print(f"Current Draft:\n{current_content[:100]}... (truncated)")

        # 2. Evaluator critiques
        # We ask for a specific format: "PASS" or "FAIL: <reason>"
        critique_response = ollama.chat(model='llama3', messages=[
            {'role': 'system', 'content': 'You are a senior code reviewer. Check for bugs, security issues, and clarity. If it is perfect, say "PASS". If not, say "FAIL" followed by specific feedback.'},
            {'role': 'user', 'content': current_content}
        ])
        critique = critique_response['message']['content']
        print(f"Critique: {critique}")

        if "PASS" in critique.upper():
            print("\n✅ Optimization Complete!")
            return current_content
        
        # 3. Append feedback to history so Generator knows what to fix
        messages.append({'role': 'assistant', 'content': current_content})
        messages.append({'role': 'user', 'content': f"Please fix the code based on this feedback: {critique}"})

    print("\n⚠️ Max cycles reached. Returning best effort.")
    return current_content

# Usage
result = evaluator_optimizer("connect to a sqlite db and list tables")