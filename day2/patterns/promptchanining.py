
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3"

def ollama_chat(prompt):
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False
    }
    r = requests.post(OLLAMA_URL, json=payload)
    return r.json()["response"]



def chain_summarize(text):
    p = f"Summarize the following text in 5 bullet points:\n\n{text}"
    return ollama_chat(p)

def chain_generate_quiz(summary):
    p = f"Generate 5 MCQ quiz questions based on this summary:\n\n{summary}"
    return ollama_chat(p)

# Run chain
text = """
Large Language Models are deep neural networks trained on large corpora of text...
"""

summary = chain_summarize(text)
quiz = chain_generate_quiz(summary)

print("Summary:\n", summary)
print("\nQuiz Questions:\n", quiz)
