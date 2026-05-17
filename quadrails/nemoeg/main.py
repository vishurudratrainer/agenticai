#"C://ml//code//quadrails//nemoeg//config"
import asyncio
import os
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from nemoguardrails import LLMRails, RailsConfig

# Initialize Presidio Engines
analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

async def initialize_guardrails():
    """Initializes the NeMo Guardrails runtime using our config folder."""
    # Ensure a dummy environment variable exists so the OpenAI driver boots cleanly
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = "ollama"
        
    config = RailsConfig.from_path("C://ml//code//quadrails//nemoeg//config")
    return LLMRails(config)

def run_presidio_rail(text: str) -> tuple[bool, str]:
    """Scans text for critical PII and redacts it natively."""
    results = analyzer.analyze(
        text=text, 
        entities=["EMAIL_ADDRESS", "PHONE_NUMBER", "US_SSN"], 
        language="en"
    )
    
    if results:
        anonymized_result = anonymizer.anonymize(text=text, analyzer_results=results)
        return True, anonymized_result.text
        
    return False, text

async def process_user_request(rails, user_input: str):
    print(f"\n👉 Raw User Input: '{user_input}'")
    
    # --- PHASE 1: Privacy Guardrail (Presidio) ---
    pii_found, sanitized_input = run_presidio_rail(user_input)
    if pii_found:
        print(f"🔒 [Presidio Block]: Sensitive data detected! Sanitized to: '{sanitized_input}'")
    else:
        print("✅ [Presidio Pass]: No critical PII detected.")

    # --- PHASE 2: Behavioral & Safety Guardrail (NeMo) ---
    try:
        response = await rails.generate_async(
            messages=[{"role": "user", "content": sanitized_input}]
        )
        
        bot_message = response["content"]
        print(f"🤖 Bot Response: {bot_message}")
        
    except Exception as e:
        print(f"❌ Error during execution: {e}")

async def main():
    print("Initializing Local Guardrails Stack (Ollama Endpoint + FastEmbed Alignment)...")
    try:
        rails = await initialize_guardrails()
        print("Systems online. Running local test cases:\n" + "="*50)

        # Test Case 1: Standard safe greeting
        await process_user_request(rails, "Hello there!")

        # Test Case 2: Local PII scrubbing check
        await process_user_request(rails, "Hey, can you add my email test@example.com to your database?")

        # Test Case 3: Local semantic guardrail block (Colang)
        await process_user_request(rails, "Who do you think will win the next presidential election?")
        
    except Exception as startup_err:
        print(f"❌ Initialization Failed: {startup_err}")

if __name__ == "__main__":
    asyncio.run(main())