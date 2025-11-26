from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Define the LLM
llm = ChatOllama(model="llama3", temperature=0.7)

# 2. Define the Prompt Template
# The template uses the variable 'topic' for user input
template = "Write a short, fun haiku about the concept of {topic}."
prompt = ChatPromptTemplate.from_template(template)

# 3. Build the Chain using the | operator (LCEL)
# Input -> Prompt -> LLM -> Output Parser
chain = prompt | llm | StrOutputParser()

# 4. Invoke the Chain
user_input = {"topic": "virtual reality"}
print("--- Invoking Simple Chain ---")
print(f"Input: {user_input['topic']}")
result = chain.invoke(user_input)

print("\n✅ Haiku Output:")
print(result)