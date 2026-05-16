import json
from pydantic import BaseModel, Field
from guardrails import Guard
from guardrails.validators import Validator, register_validator, PassResult, FailResult
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

# ==========================================================
# 1. DEFINE CUSTOM GUARDRAILS VALIDATORS (Fail-safe, Local)
# ==========================================================
@register_validator(name="competitor_check", data_type="string")
class CompetitorCheck(Validator):
    def __init__(self, competitors, on_fail="fix"):
        super().__init__(on_fail=on_fail, competitors=competitors)
        self.competitors = competitors

    def validate(self, value, metadata={}):
        for comp in self.competitors:
            if comp.lower() in value.lower():
                return FailResult(
                    error_message=f"Found competitor: {comp}", 
                    fix_value=value.replace(comp, "[REDACTED]")
                )
        return PassResult()

@register_validator(name="valid_length", data_type="string")
class ValidLength(Validator):
    def __init__(self, min=0, max=100, on_fail="fix"):
        super().__init__(on_fail=on_fail, min=min, max=max)
        self.min, self.max = min, max

    def validate(self, value, metadata={}):
        if len(value) < self.min or len(value) > self.max:
            return FailResult(error_message="Invalid length", fix_value=value[:self.max])
        return PassResult()

# ==========================================================
# 2. PYDANTIC SCHEMA DEFINITION (Pydantic v2 Compliant)
# ==========================================================
class SupportAnalysisSchema(BaseModel):
    category: str = Field(description="Category of the issue")
    summary: str = Field(
        description="Brief 1-sentence summary",
        json_schema_extra={"validators": [ValidLength(min=5, max=50, on_fail="fix")]}
    )
    resolution_plan: str = Field(
        description="Action plan for the support agent",
        json_schema_extra={"validators": [CompetitorCheck(competitors=["CloudCorp"], on_fail="fix")]}
    )

# Initialize Guard with the output schema
guard = Guard.for_pydantic(output_class=SupportAnalysisSchema)

# ==========================================================
# 3. LANGCHAIN & OLLAMA INITIALIZATION
# ==========================================================
# Initialize the local Ollama model via LangChain
# Setting temperature to 0.0 is crucial for stable structured output
llm = Ollama(model="llama3", base_url="http://localhost:11434", temperature=0.0)

# Build a strict prompt template instructing the model to output raw JSON
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a strict data assistant. Output raw JSON code matching the user schema requirements. Do not output markdown backticks (```json), conversational pleasantries, or introductions."),
    ("user", "Analyze this customer feedback and structure the response matching the schema fields: {customer_input}")
])

# ==========================================================
# 4. INTERCEPTOR FUNCTION FOR LANGCHAIN CHAINING
# ==========================================================
# ==========================================================
# 4. INTERCEPTOR FUNCTION (FIXED FOR LIST INTERFACE)
# ==========================================================
# ==========================================================
# 4. INTERCEPTOR FUNCTION (FIXED FOR LIST INTERFACE)
# ==========================================================
# ==========================================================
# 4. INTERCEPTOR FUNCTION (FIXED FOR LIST INTERFACE)
# ==========================================================
def run_pii_guardrail(llm_output_text: str):
    print("\n--- [INTERNAL STEP 1: RAW LLM OUTPUT] ---")
    print(llm_output_text)
    
    # Run Guardrails parsing over the raw text string
    validation_outcome = guard.parse(llm_output_text)
    
    print("\n--- [INTERNAL STEP 2: GUARDRAILS EVALUATION STACK] ---")
    latest_call = guard.history.last
    
    # Check status safely
    is_success = latest_call.status in ["success", "fixed"]
    print(f"Schema Structure Parsed Successfully: {is_success}")
    
    # Corrected loop over iterations and list-based logs
    for i, iteration in enumerate(latest_call.iterations):
        print(f"\n[Iteration Step #{i+1}]")
        
        # FIXED: Removed .items() because validator_logs is a flat list
        for log in iteration.validator_logs:
            # Safely extract properties from the log object
            field_path = getattr(log, "field_path", "unknown_field")
            result = getattr(log, "validation_result", None)
            
            if result:
                outcome = getattr(result, "outcome", "unknown")
                error_msg = getattr(result, "error_message", "None")
                
                print(f" Field '{field_path}' flag status: {outcome}")
                if outcome == "fail":
                    print(f"   Reason/Rule Violated: {error_msg}")
                    print(f"   Value Before: {log.value_before_validation}")
                    print(f"   Value After:  {log.value_after_validation}")

    if validation_outcome.validated_output:
        return validation_outcome.validated_output
    else:
        return {"error": "Validation structural failure", "raw": llm_output_text}

# 5. ASSEMBLE AND RUN THE LANGCHAIN EXPRESSION CHAIN (LCEL)
# ==========================================================
# Component 1: Prompt Template formats input
# Component 2: LLM generates the text response
# Component 3: RunnableLambda pipes the raw text directly into our Guardrail system
chain = prompt_template | llm | RunnableLambda(run_guardrails)

# Dirty input containing both a competitor name and messy formatting potential
dirty_input = "My database layer is slow. CloudCorp works much faster than this setup."

print("Executing LangChain + Guardrails Pipeline...")
final_output = chain.invoke({"customer_input": dirty_input})

print("\n--- FINAL SECURE VALIDATED OUTPUT ---")
print(json.dumps(final_output, indent=2))