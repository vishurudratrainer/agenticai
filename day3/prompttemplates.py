from langchain_core.prompts import ChatPromptTemplate
"""
Role: The blueprint for structuring the conversation or request.

ChatPromptTemplate is a Blueprint. It is responsible for taking all the 
dynamic pieces of information (like the user's question, retrieved context, and system instructions) 
and formatting them into a specific, structured sequence of messages (System, Human, AI)
that the LLM understands best.
"""
# Define the template with placeholders ({cuisine} and {ingredient})
template = ChatPromptTemplate.from_messages([
    ("system", "You are a world-renowned chef that specializes in {cuisine}."),
    ("human", "What is the best dish to prepare with {ingredient}?"),
])

# Generate the final prompt (still a standard LangChain message object)
prompt_value = template.invoke({"cuisine": "Indian", "ingredient": "lentils"})

print("\n--- Generated Prompt ---")
print(prompt_value.to_string())