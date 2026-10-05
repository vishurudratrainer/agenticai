from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
#import langchain
#langchain.set_debug(True)
from langchain_core.globals import set_debug
set_debug(True)
# 1. Define the LLM
llm = ChatOllama(model="llama3", temperature=0.7)

# 2. Define the Prompt Template
# The template uses the variable 'topic' for user input
template = "Explain about Java."
prompt = ChatPromptTemplate.from_template(template)
result=prompt.invoke({})
print(result)