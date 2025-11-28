# pip install crewai langchain-community
from crewai import Agent, Task, Crew, Process,LLM
from langchain_ollama import ChatOllama
import os

# Set a dummy key to satisfy the internal library check, 
# even though we are explicitly using Ollama later.
os.environ["OPENAI_API_KEY"] = "sk-DUMMYKEY"
# --- 1. Ollama Configuration ---
# NOTE: Ensure Ollama is running (e.g., 'ollama run llama3')
ollama_llm = LLM(
    model="ollama/llama3:latest",
    base_url="http://127.0.0.1:11434"
)
# --- 2. Define Agents ---
researcher = Agent(
    role='Senior Researcher',
    goal='Gather facts and data on the latest AI trends.',
    backstory='An expert in finding, filtering, and summarizing large amounts of information.',
    llm=ollama_llm,
    allow_delegation=False,
    tools=[],
    verbose=True
)

writer = Agent(
    role='Article Writer',
    goal='Draft a compelling, well-structured article from the research notes.',
    backstory='A professional content writer who specializes in clear and engaging technical communication.',
    llm=ollama_llm,
    allow_delegation=False,
        tools=[],
    verbose=True
)

# --- 3. Define Tasks ---
# Task 1 feeds its output directly into Task 2, and so on.
research_task = Task(
    description="Research the top 3 advancements in 'Local LLMs with Ollama' in the past month.",
    expected_output="A structured list of key findings, sources, and a brief summary for each finding.",
    agent=researcher
)

writing_task = Task(
    description="Using the research findings, write a 500-word blog post. The article must be engaging and cite the key findings from the researcher.",
    expected_output="A complete, professional 500-word blog post in Markdown format.",
    agent=writer
)

# --- 4. Form the Crew and Run ---
project_crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
    process=Process.sequential, # The key for this pattern
    verbose=True
)

result = project_crew.kickoff(inputs={})
print("--- Sequential Workflow Result ---")
print(result)