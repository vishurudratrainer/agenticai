#Often, you don't just want text back from the LLM; you want structured data (like JSON) so your code can easily process it. This uses a Pydantic Schema and a special parser.

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser

# 1. Define the desired output structure using Pydantic
class Recipe(BaseModel):
    """Structured data about a simple dish."""
    dish_name: str = Field(description="The name of the dish.")
    ingredients: list[str] = Field(description="A list of key ingredients.")
    prep_time_minutes: int = Field(description="The estimated preparation time in minutes.")

# 2. Setup the Parser and the Model
parser = PydanticOutputParser(pydantic_object=Recipe)
ollama_model = ChatOllama(model="llama3", temperature=0)

# 3. Define the Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert chef. Your goal is to extract recipe information from the user's text and format it perfectly as JSON according to the following schema:\n{format_instructions}"),
        ("human", "{user_input}")
    ]
).partial(format_instructions=parser.get_format_instructions())

# 4. Create the Chain and Invoke
structured_chain = prompt | ollama_model | parser

user_input = "Tell me about a quick recipe for scrambled eggs. It should take about 5 minutes."

print("--- Structured Output Chain Result ---")
# The result will be a Pydantic object (or a dict if the parser is omitted)
recipe_object = structured_chain.invoke({"user_input": user_input})

print(f"Dish Name (Type: {type(recipe_object.dish_name)}): {recipe_object.dish_name}")
print(f"Ingredients List (Type: {type(recipe_object.ingredients)}): {recipe_object.ingredients}")
print(f"Preparation Time (Type: {type(recipe_object.prep_time_minutes)}): {recipe_object.prep_time_minutes}")
#