import requests
import json
import os
from time import sleep
from typing import Dict, Callable, Any
from pypdf import PdfWriter

# --- 1. Ollama Configuration ---
OLLAMA_API_ENDPOINT = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral" 

# --- 2. Tool Implementation ---

def read_pdf_text(filepath: str) -> str:
    """
    Tool: Reads the content of a PDF file and extracts all text pages.
    Args: filepath: The path to the PDF file to be read.
    """
    print(f"\n[Tool Activated: Reading PDF at {filepath}]")
    try:
        # NOTE: This is a simulated PDF reader for simplicity in a sandbox 
        # For a real PDF, use a library like pypdf or PyMuPDF.
        if not os.path.exists(filepath):
             return f"Error: PDF file not found at {filepath}"

        # Actual PDF text extraction simulation
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            
        print("[Tool Success: PDF content extracted.]")
        # Return only the first 1000 characters to prevent prompt overflow
        return content
        
    except Exception as e:
        return f"Error: Failed to read PDF due to: {e}"

def ollama_mistral_analysis(user_prompt: str, context_text: str) -> str:
    """
    Tool: Sends a prompt and context to the local Ollama Mistral model for analysis.
    This is the core LLM call, acting as a tool within the agent's workflow.
    
    Args: user_prompt: The original question. context_text: The text from the PDF.
    """
    print(f"\n[Tool Activated: Calling Ollama ({OLLAMA_MODEL}) for analysis]")
    
    system_instruction = (
        "You are an expert document analyzer. The user has provided context text from a document. "
        "Your task is to answer the user's question ONLY based on the provided context. "
        "If the information is not in the context, state that clearly."
    )
    
    full_prompt = (
        f"USER QUESTION: {user_prompt}\n\n"
        f"DOCUMENT CONTEXT:\n---\n{context_text}\n---\n\n"
        f"Please provide a concise answer based on the context."
    )
    
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": full_prompt,
        "system": system_instruction,
        "stream": False,
        "options": {"temperature": 0.2},
    }

    try:
        response = requests.post(OLLAMA_API_ENDPOINT, json=payload)
        response.raise_for_status()
        return response.json().get("response", "Analysis failed to retrieve text.")
    
    except requests.exceptions.RequestException as e:
        return f"API Error: Could not connect to Ollama. Is '{OLLAMA_MODEL}' running? Detail: {e}"
    except Exception as e:
        return f"Analysis Error: {e}"

# Dictionary mapping tool names to actual function objects
TOOL_MAPPING: Dict[str, Callable[..., Any]] = {
    "read_pdf_text": read_pdf_text,
    "ollama_mistral_analysis": ollama_mistral_analysis
}

# --- 3. Agent Orchestration (The Custom Workflow) ---

def pdf_analysis_agent(user_question: str, pdf_path: str):
    """
    Custom Agentic workflow for reading a document and analyzing it.
    This is a hardcoded, sequential workflow.
    """
    print("--- CUSTOM PDF AGENT WORKFLOW START ---")
    print(f"Goal: Answer '{user_question}' using data from '{pdf_path}'")
    
    # ----------------------------------------------------
    # STEP 1: Execute the 'read_pdf_text' Tool
    # ----------------------------------------------------
    print("\n[STEP 1/2] Agent selects: read_pdf_text")
    pdf_content = read_pdf_text(filepath=pdf_path)
    
    if pdf_content.startswith("Error"):
        print(f"[ERROR] Workflow halted: {pdf_content}")
        return

    # ----------------------------------------------------
    # STEP 2: Execute the 'ollama_mistral_analysis' Tool
    # ----------------------------------------------------
    print("\n[STEP 2/2] Agent selects: ollama_mistral_analysis (LLM call)")
    final_answer = ollama_mistral_analysis(
        user_prompt=user_question,
        context_text=pdf_content
    )
    
    # ----------------------------------------------------
    # FINAL OUTPUT
    # ----------------------------------------------------
    print("\n" + "="*60)
    print("✨ FINAL AGENT RESULT (Synthesis Complete) ✨")
    print(f"Question: {user_question}")
    print(f"Source: {pdf_path}")
    print(f"Answer: {final_answer}")
    print("="*60)
    
# --- 4. Execution ---

if __name__ == "__main__":
    
    # --- Setup: Create a dummy file that simulates PDF content ---
    # Since we can't reliably create a binary PDF here, we use a .txt file 
    # and pretend it's a PDF being read by the 'read_pdf_text' tool.
    file_path = "quarterly_report.pdf" 
    
    report_content = """
    ## Q3 Project Athena Summary
    The third quarter was primarily focused on internal restructuring and market research. 
    A total of 8 new features were developed. 
    The primary finding was that customer satisfaction increased by 15% following the launch of Feature 5. 
    Our budget for Q4 is set at $1.2 million, 
    with a mandate to hire 4 new specialized engineers. The key performance indicator (KPI) 
    for the next quarter is to increase user engagement by 25%.
    """
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print("Agent Initialized. Running custom PDF analysis workflow.")
    
    # --- Custom Workflow Scenario 1: Information Extraction ---
    pdf_analysis_agent(
        user_question="What is the budget set for the next quarter and what is the hiring mandate?",
        pdf_path=file_path
    )
    
    # --- Custom Workflow Scenario 2: Factual Verification (Should fail elegantly) ---
    print("\n" + "#"*60 + "\n")
    pdf_analysis_agent(
        user_question="When is the company holiday party scheduled?",
        pdf_path=file_path
    )

    # --- Cleanup ---
    os.remove(file_path)