"""
LLM Fundamentals & Prompt Engineering - OLLAMA Version
This tutorial uses local models via Ollama (no API keys required).

Install dependencies:
    pip install ollama
Start Ollama service (if not running):
    ollama serve
"""

import ollama
import json
import time

# Default local model — change if you prefer mistral, phi3, qwen2, etc.
DEFAULT_MODEL = "llama3"

# ---------------------------------------------
# Basic Completion
# ---------------------------------------------
def send_completion(prompt, model=DEFAULT_MODEL, temperature=0.7, max_tokens=300):
    """
    Sends a prompt to a local Ollama model and returns text output.
    """
    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": temperature, "num_predict": max_tokens}
        )
        return response["message"]["content"]
    except Exception as e:
        return f"Error: {e}"


# ---------------------------------------------
# Chain-of-Thought Example
# ---------------------------------------------
def chain_of_thought_example():
    prompt = """
You are a helpful reasoning assistant.
Think step-by-step, and then provide the final answer.

Question:
If two trains leave stations 60 miles apart heading toward each other at 
20 mph and 30 mph respectively, how long until they meet?

Think step-by-step:
"""
    return send_completion(prompt, temperature=0.3, max_tokens=250)


# ---------------------------------------------
# Embeddings with Ollama
# ---------------------------------------------
def get_embedding(text, model="nomic-embed-text"):
    """
    Uses Ollama's embedding models (e.g., nomic-embed-text) to get vector embeddings.
    Pull model first: ollama pull nomic-embed-text
    """
    try:
        res = ollama.embeddings(model=model, prompt=text)
        return res["embedding"]
    except Exception as e:
        print("Embedding error:", e)
        return None


def cosine_similarity(a, b):
    import math
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------
# Evaluate Prompts (A/B Testing)
# ---------------------------------------------
def compare_prompts(prompt_a, prompt_b, test_inputs, model=DEFAULT_MODEL):
    results = []
    for text in test_inputs:
        out_a = send_completion(prompt_a.format(text), model=model)
        out_b = send_completion(prompt_b.format(text), model=model)
        results.append({"input": text, "A": out_a.strip(), "B": out_b.strip()})
    return results


# ---------------------------------------------
# Demo
# ---------------------------------------------
if __name__ == "__main__":
    print("🔹 LLM Fundamentals Demo — Ollama Version")

    print("\nEmbedding exmple:")
    kingtext=get_embedding("king")
    queentext=get_embedding("queen")
    print(cosine_similarity(kingtext,queentext))
    

