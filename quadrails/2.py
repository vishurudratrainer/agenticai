import json
from pydantic import BaseModel, Field
from guardrails import Guard
from guardrails.validators import Validator, register_validator, PassResult, FailResult
from litellm import completion

# ==========================================
# MANUAL VALIDATOR 1: Competitor Check
# ==========================================
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

# ==========================================
# MANUAL VALIDATOR 2: Valid Length
# ==========================================
@register_validator(name="valid_length", data_type="string")
class ValidLength(Validator):
    def __init__(self, min=0, max=100, on_fail="fix"):
        super().__init__(on_fail=on_fail, min=min, max=max)
        self.min, self.max = min, max

    def validate(self, value, metadata={}):
        if len(value) < self.min or len(value) > self.max:
            return FailResult(error_message="Invalid length", fix_value=value[:self.max])
        return PassResult()

# ==========================================
# SCHEMA DEFINITION (Fixed Pydantic V2 Warning)
# ==========================================
# Instead of passing custom arguments directly to Field(), 
# Pydantic v2 requires custom metadata to be passed inside 'json_schema_extra'
class SupportSchema(BaseModel):
    category: str = Field(description="Category")
    
    summary: str = Field(
        description="Brief summary",
        json_schema_extra={"validators": [ValidLength(min=5, max=50, on_fail="fix")]}
    )
    
    resolution: str = Field(
        description="Suggested fix",
        json_schema_extra={"validators": [CompetitorCheck(competitors=["CloudCorp"], on_fail="fix")]}
    )

# ==========================================
# OLLAMA CONNECTOR (Fixed Guardrails Warning)
# ==========================================
# Adding the `*` forces 'messages' to be a keyword-only argument, 
# which silences the Guardrails llm_providers warning.
def call_ollama(*, messages, **kwargs):
    response = completion(
        model="ollama/llama3", 
        messages=messages, 
        api_base="http://localhost:11434"
    )
    return response.choices[0].message.content

# ==========================================
# INITIALIZATION & RUN
# ==========================================
guard = Guard.for_pydantic(output_class=SupportSchema)

raw_input = "My server is slow. CloudCorp was faster."

response = guard(
    call_ollama,
    messages=[
        {
            "role": "user",
            "content": f"Analyze this customer feedback and respond in JSON format: {raw_input}"
        }
    ]
)

print("--- VALIDATED OUTPUT ---")
print(json.dumps(response.validated_output, indent=2))