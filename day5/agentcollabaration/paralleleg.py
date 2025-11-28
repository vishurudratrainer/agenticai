# pip install httpx
import asyncio
import httpx
import json

# --- 1. Configuration ---
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3"
TIMEOUT_SECONDS = 300.0
# --- 2. Worker Function (Async) ---
async def fetch_ollama_response(prompt: str, task_name: str) -> dict:
    """Asynchronously calls the Ollama API for a specific task."""
    print(f"🤖 Starting {task_name}...")
    
    # Payload for the Ollama /generate endpoint
    payload = {
        "model": MODEL_NAME,
        "prompt": f"You are a specialized {task_name}. {prompt}. Output only the result.",
        "stream": False 
    }
    
    # Use httpx for asynchronous requests
    async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
        response = await client.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        
        # Ollama returns a JSON response; extract the final 'response' field
        data = response.json()
        print(f"✅ {task_name} finished.")
        return {
            "task_name": task_name,
            "result": data.get("response", "No response found")
        }

# --- 3. Coordinator/Aggregator Function ---
async def run_parallel_analysis(user_query: str):
    # --- Parallel Tasks Definition ---
    prompts_and_tasks = [
        (f"Analyze the market sentiment for 'Tesla' based on general news over the past 48 hours. Query: {user_query}", "Sentiment Analyst"),
        (f"Determine the key financial risks for 'Tesla' in Q3 2024 based on expert opinions. Query: {user_query}", "Financial Risk Analyst")
    ]
    
    # Create the list of concurrent tasks
    tasks = [
        fetch_ollama_response(prompt, task_name)
        for prompt, task_name in prompts_and_tasks
    ]
    
    # Run all tasks concurrently
    parallel_results = await asyncio.gather(*tasks)
    
    # --- Aggregation (Final Ollama Call) ---
    aggregation_prompt = f"""
    You are the Final Investment Strategist. Synthesize the following two reports into a single, cohesive investment recommendation for Tesla.
    
    1. Sentiment Report: {parallel_results[0]['result']}
    2. Financial Risk Report: {parallel_results[1]['result']}
    
    Provide a final 'BUY', 'HOLD', or 'SELL' recommendation and a brief justification.
    """

    final_result = await fetch_ollama_response(aggregation_prompt, "Aggregator")
    return final_result

# --- 4. Run the Workflow ---
if __name__ == "__main__":
     user_input = "Give me an investment summary for Tesla."
     final_report = asyncio.run(run_parallel_analysis(user_input))
     print("\n--- Parallel/Aggregation Workflow Result ---")
     print(final_report['result'])