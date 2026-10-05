import requests
import json

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3"   # or any model you installed via `ollama pull`

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


import collections

def self_consistency(prompt, samples=5):
    answers = []
    for _ in range(samples):
        ans = ollama_chat(prompt)
        answers.append(ans.strip())

    counter = collections.Counter(answers)
    consensus = counter.most_common(1)[0][0]

    print("=== All Answers ===")
    for a in answers:
        print("-", a)

    print("\n=== Consensus Answer ===")
    print(consensus)
    return consensus

# Test
prompt = "A farmer has 17 sheep. All but 9 die. How many are left?"
self_consistency(prompt,2)
