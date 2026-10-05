from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import ollama
import json

# --- 1. Agent Output Structure (for structured LLM response) ---
# We define the desired output format using Pydantic.
class IncidentAnalysis(BaseModel):
    category: str
    severity: str # Low, Medium, High, Critical
    suggested_action: str
    summary: str

# --- 2. The Incoming Webhook Payload ---
# This defines the data we expect from the external system (the trigger).
class WebhookPayload(BaseModel):
    source: str
    event_id: str
    message: str
    timestamp: str

# --- FastAPI App Setup ---
app = FastAPI(
    title="Ollama Agent Webhook",
    description="Receives webhooks and uses a local Ollama agent to analyze the payload."
)

OLLAMA_MODEL = "llama3" # Ensure this model is pulled locally

# --- Agent Core Logic (Reasoning & Structured Output) ---
def analyze_incident(payload: WebhookPayload) -> IncidentAnalysis:
    """
    Sends the incoming webhook data to the Ollama LLM for reasoning.
    """
    system_prompt = f"""
    You are a specialized Incident Analysis Agent. 
    Your task is to analyze the
    incoming server alert message and structure your analysis into a 
    JSON object
    that strictly conforms to the following Pydantic schema: {IncidentAnalysis.model_json_schema()}.
    
    The analysis must determine the incident's category, severity, and suggest
    a concise, immediate action.
    """
    print(payload)
    user_message = f"Analyze this incident from {payload.source}:\nEvent ID: {payload.event_id}\nMessage: {payload.message}\nTime: {payload.timestamp}"

    try:
        # Use the ollama generate endpoint for structured JSON output
        response = ollama.generate(
            model=OLLAMA_MODEL,
            prompt=user_message,
            system=system_prompt,
            format='json'
        )
        
        # The LLM returns a JSON string, which we parse
        json_string = response['response'].strip()
        analysis_data = json.loads(json_string)
        
        # Validate and return the structured object
        return IncidentAnalysis(**analysis_data)

    except Exception as e:
        print(f"Ollama/Analysis Error: {e}")
        # On failure, return a default/error analysis
        return IncidentAnalysis(
            category="System Error",
            severity="Critical",
            suggested_action=f"Immediate manual check required. LLM analysis failed: {e}",
            summary="Agent could not process the request."
        )

# --- Webhook Endpoint (Perception & Action) ---
@app.post("/webhook")
async def receive_webhook(payload: WebhookPayload):
    print("\n" + "="*50)
    print(f"🔌 WEBHOOK RECEIVED from {payload.source} at {payload.timestamp}")
    print(f"Raw Message: {payload.message}")
    print("="*50)

    # 1. Agent Reasoning
    analysis = analyze_incident(payload)

    # 2. Agent Action (Log the result)
    print("✅ AGENT ANALYSIS COMPLETE:")
    print(f"  Category: {analysis.category}")
    print(f"  Severity: **{analysis.severity.upper()}**")
    print(f"  Action: {analysis.suggested_action}")
    print(f"  Summary: {analysis.summary}")
    print("="*50 + "\n")

    return {"status": "Analysis complete", "analysis": analysis.model_dump()}

# --- Run the FastAPI Server ---
if __name__ == "__main__":
    # The agent is exposed on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)