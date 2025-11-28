import random
import time
from typing import Dict, Any, List
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List, Union
import operator
import json
# --- 1. Ollama Configuration ---
# NOTE: In a real system, the Manager would use the LLM to judge bids.
# For simplicity, we define the LLM here but primarily use Python logic.
ollama_llm_judge = ChatOllama(model="llama3:latest", base_url="http://127.0.0.1:11434")

class AgentBidder:
    """Represents a specialized worker agent that bids on tasks."""
    def __init__(self, name: str, specialization: str, cost: float):
        self.name = name
        self.specialization = specialization
        self.cost = cost
        
    def bid(self, task_description: str) -> Dict[str, Any]:
        """Generates a bid based on its specialization and a random competence score."""
        
        # Assess relevance (simple keyword matching for this example)
        if self.specialization.lower() in task_description.lower():
            relevance = 0.9
        else:
            relevance = 0.4
            
        # Simulate competence and time (randomness added to ensure variation)
        competence_score = relevance + random.uniform(-0.1, 0.1)
        estimated_time = random.randint(1, 5) # hours

        return {
            "agent_name": self.name,
            "cost": self.cost,
            "competence_score": round(max(0, min(1, competence_score)), 2),
            "estimated_time_h": estimated_time
        }

    def execute_task(self, task_description: str) -> str:
        """Simulates task execution."""
        time.sleep(1) # Simulate work time
        return f"Task executed by {self.name} ({self.specialization}). Result: {task_description[:30]}..."

class AuctionManager:
    """Manages the bidding process and awards the contract using LLM judgment."""
    def __init__(self, bidders: List[AgentBidder], llm=ollama_llm_judge):
        self.bidders = bidders
        self.llm = llm

    def conduct_auction(self, task_description: str) -> str:
        """
        1. Broadcasts the task.
        2. Collects bids.
        3. Uses LLM to judge the best bid.
        4. Awards contract to the best bidder.
        """
        print(f"💰 Manager: Broadcasting task: '{task_description}'")
        
        # 1. Collect bids
        all_bids = [bidder.bid(task_description) for bidder in self.bidders]
        
        # 2. LLM Judgment (The core of the delegation)
        bids_str = json.dumps(all_bids, indent=2)
        
        # The Manager uses Ollama to decide the winner based on trade-offs (Cost vs. Competence)
        judge_prompt = f"""
        You are the Contract Manager. Analyze the following bids for the task: '{task_description}'.
        Task Goal: Find the most COMPETENT and COST-EFFECTIVE solution.
        Bids: {bids_str}
        
        Based on the data, identify the BEST agent. Output ONLY the name of the winning agent.
        """
        
        print("\n🧠 Manager: Asking Ollama to judge the best bid...")
        
        llm_response = self.llm.invoke([SystemMessage(content=judge_prompt)]).content.strip()
        winning_agent_name = llm_response.split('\n')[0].strip().replace('.', '')
        
        # 3. Award Contract
        winner = next((b for b in self.bidders if b.name == winning_agent_name), None)

        if winner:
            print(f"\n🎉 Manager: Contract awarded to {winner.name}!")
            # 4. Execution
            return winner.execute_task(task_description)
        else:
            return f"Manager failed to award contract. LLM response was unclear: {llm_response}"

# --- Example Use Case: Automated Research Delegation ---
research_bidders = [
    AgentBidder("Financial Analyst", "stock market, finance, investment", 100.0),
    AgentBidder("Tech Researcher", "AI, machine learning, software", 75.0),
    AgentBidder("General Assistant", "general knowledge, summarize", 50.0),
]

manager = AuctionManager(research_bidders)
task = "Find the latest news on AI stock performance and summarize its impact."
final_report = manager.conduct_auction(task)
print(f"\n--- Auction Delegation Final Result ---\n{final_report}")