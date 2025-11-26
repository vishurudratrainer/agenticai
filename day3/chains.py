"""
Chains (LCEL)
This example stitches the components together using the LangChain Expression Language (LCEL) pipe (|), demonstrating a full pipeline running on Ollama.
"""
from langchain_core.output_parsers import StrOutputParser
# ollama_model is the ChatOllama instance from section 1
# template is the ChatPromptTemplate instance from section 2
from modeleg import ollama_model
from prompttemplates import template
#Chains (LCEL)
#This example stitches the components together using the LangChain Expression Language (LCEL) pipe (|), demonstrating a full pipeline running on Ollama.
# Define the chain: Prompt -> Model -> Output Parser (to get a clean string)
chain = template | ollama_model | StrOutputParser()
#StrOutputParser: Extracts the final text from the response object.

# Invoke the chain, running the full sequence
result = chain.invoke({"cuisine": "Mexican", "ingredient": "avocado"})

print("\n--- LCEL Chain Result (Ollama) ---")
print(result)