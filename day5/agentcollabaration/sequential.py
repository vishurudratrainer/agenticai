import ollama
"""
The Sequential ChainConcept: The output of one agent becomes the input for the next. This is the "Hello World" of agent collaboration, ideal for tasks like research  summarization.
Pattern: Agent A (Draft) -> Agent B (Refine)
"""
def sequential_chain():
    # Step 1: Generator Agent
    topic = "The importance of bees in the ecosystem"
    print(f"--- Agent 1: Generating Draft on '{topic}' ---")
    
    draft_response = ollama.chat(model='llama3', messages=[
        {'role': 'system', 'content': 'You are a biologist. Write a short, detailed paragraph about the topic.'},
        {'role': 'user', 'content': topic},
    ])
    draft = draft_response['message']['content']
    print(f"Draft:\n{draft}\n")

    # Step 2: Editor Agent
    print("--- Agent 2: Refining Content ---")
    edit_response = ollama.chat(model='llama3', messages=[
        {'role': 'system', 'content': 'You are a professional editor. Fix grammar and make the tone punchy and exciting for a blog post.'},
        {'role': 'user', 'content': f"Refine this text: {draft}"},
    ])
    final_post = edit_response['message']['content']
    print(f"Final Post:\n{final_post}")

sequential_chain()