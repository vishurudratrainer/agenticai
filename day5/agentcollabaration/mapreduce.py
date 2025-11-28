"""
Docstring for agentcollabaration.mapreduce

Concept: A central "Manager" breaks a complex task into sub-tasks, assigns them to "Worker" agents in parallel (or sequence), and then compiles the results.

Use Case: "Write a software spec." (Manager breaks it into: UI Design, Database Schema, and API Endpoints).
"""
import ollama
import json

def orchestrator_worker(complex_task):
    # 1. Orchestrator: Plan the sub-tasks
    print("--- Orchestrator: Planning ---")
    plan_response = ollama.chat(model='llama3', format='json', messages=[
        {'role': 'system', 'content': 'You are a Project Manager. Break the task into 3 distinct sub-tasks. Return JSON: {"tasks": ["task1", "task2", "task3"]}'},
        {'role': 'user', 'content': complex_task},
    ])
    
    plan = json.loads(plan_response['message']['content'])
    tasks = plan.get('tasks', [])
    print(f"Plan: {tasks}")

    results = {}

    # 2. Workers: Execute sub-tasks (In a real app, use asyncio for parallel execution)
    for task in tasks:
        print(f"--- Worker processing: {task} ---")
        worker_response = ollama.chat(model='llama3', messages=[
            {'role': 'system', 'content': 'You are a specialist worker. Complete this task concisely.'},
            {'role': 'user', 'content': task},
        ])
        results[task] = worker_response['message']['content']

    # 3. Orchestrator: Synthesize
    print("\n--- Orchestrator: Synthesizing ---")
    final_context = "\n".join([f"Task: {k}\nResult: {v}" for k, v in results.items()])
    
    summary_response = ollama.chat(model='llama3', messages=[
        {'role': 'system', 'content': 'You are a Project Manager. Merge these worker outputs into a final cohesive report.'},
        {'role': 'user', 'content': f"Original Goal: {complex_task}\n\nWorker Outputs:\n{final_context}"},
    ])
    
    print("\n=== Final Report ===")
    print(summary_response['message']['content'])

# Usage
orchestrator_worker("Design a concept for a new fitness tracking mobile app")