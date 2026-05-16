import json
import re
from pydantic import BaseModel, Field
from guardrails import Guard
from guardrails.validators import Validator, register_validator, PassResult, FailResult
from langchain_ollama import ChatOllama  # FIXED: Importing ChatOllama instead of OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# ==========================================================
# 1. DEFINE THE PII MASKING VALIDATOR
# ==========================================================
@register_validator(name="pii_masker", data_type="string")
class PIIMasker(Validator):
    def __init__(self, on_fail="fix"):
        super().__init__(on_fail=on_fail)
        self.email_regex = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        self.phone_regex = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'

    def validate(self, value, metadata={}):
        cleaned_value = value
        pii_found = False

        if re.search(self.email_regex, cleaned_value):
            cleaned_value = re.sub(self.email_regex, "[REDACTED_EMAIL]", cleaned_value)
            pii_found = True

        if re.search(self.phone_regex, cleaned_value):
            cleaned_value = re.sub(self.phone_regex, "[REDACTED_PHONE]", cleaned_value)
            pii_found = True

        if pii_found:
            return FailResult(
                error_message="PII detected.",
                fix_value=cleaned_value
            )
        return PassResult()

# ==========================================================
# 2. PYDANTIC SCHEMA DEFINITION
# ==========================================================
class CustomerResolutionSchema(BaseModel):
    summary: str = Field(description="A brief summary of the issue.")
    agent_notes: str = Field(
        description="Internal resolution notes detailing how the agent handled the customer.",
        json_schema_extra={"validators": [PIIMasker(on_fail="fix")]}
    )

# Initialize Guardrails mapping
guard = Guard.for_pydantic(output_class=CustomerResolutionSchema)

# ==========================================================
# 3. LANGCHAIN CHAT OLLAMA CONFIGURATION
# ==========================================================
# We initialize ChatOllama and pass num_predict to ensure long responses aren't cut short
llm = ChatOllama(
    model="llama3", 
    base_url="http://localhost:11434", 
    temperature=0.0,
    num_predict=1024
)

# Bind the structural schema layout constraint directly to the model instance
structured_llm = llm.with_structured_output(CustomerResolutionSchema)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are an internal corporate support processor. Extract the summary and agent notes precisely into the matching output schema parameters."),
    ("user", "Process this customer support logs raw transcript: {transcript}")
])

extraction_chain = prompt_template | structured_llm

# ==========================================================
# 4. RUN PIPELINE AND EVALUATE INTERNALS
# ==========================================================
messy_transcript = (
    "Agent talked to John Doe. Customer was frustrated about account lockout. "
    "Agent verified identity by sending a link to john.doe@gmail.com and "
    "confirmed his fallback contact string as 555-123-4567 before resolving."
)

print("Executing ChatOllama structured output generation...")
# Invoking the chain natively populates a completely verified Pydantic object
structured_pydantic_obj = extraction_chain.invoke({"transcript": messy_transcript})

# Export directly to standard string JSON
clean_json_str = structured_pydantic_obj.model_dump_json()

print("\n--- [INTERNAL STEP 1: NATIVE STRUCTURED JSON] ---")
print(clean_json_str)

# Run verification parsing inside Guardrails
validation_outcome = guard.parse(clean_json_str)

print("\n--- [INTERNAL STEP 2: GUARDRAILS EVALUATION STACK] ---")
latest_call = guard.history.last
is_success = latest_call.status in ["success", "fixed"]
print(f"Schema Structure Parsed Successfully: {is_success}")

for i, iteration in enumerate(latest_call.iterations):
    print(f"\n[Iteration Step #{i+1}]")
    for log in iteration.validator_logs:
        field_path = getattr(log, "field_path", "unknown_field")
        result = getattr(log, "validation_result", None)
        
        if result:
            outcome = getattr(result, "outcome", "unknown")
            print(f" Field '{field_path}' flag status: {outcome}")
            if outcome == "fail":
                print(f"   Value Before: {log.value_before_validation}")
                print(f"   Value After:  {log.value_after_validation}")

print("\n--- [INTERNAL STEP 3: FINAL SECURE APP OUTPUT] ---")
print(json.dumps(validation_outcome.validated_output, indent=2))