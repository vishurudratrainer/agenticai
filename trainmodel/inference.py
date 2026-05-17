import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def main():
    # 1. Define paths (Make sure this matches your training output folder name)
    base_model_id = "microsoft/Phi-3-mini-4k-instruct"
    adapter_dir = "./my_extended_model" 
    
    # Check if you have a GPU available, otherwise fall back to CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ Using device: {device}")

    print("🔄 Loading the original base model tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    
    print("🔄 Loading the clean base model (Quantized to save memory)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        device_map={"": device},
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        low_cpu_mem_usage=True
    )

    # 2. Inject your custom local extension pack on top of the base model
    print(f"🛠️ Infusing your custom adapter layers from '{adapter_dir}'...")
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    
    # Set the model to evaluation (inference) mode
    model.eval()
    print("✅ Extended model loaded and ready!")
    print("=" * 60)

    # 3. Define a prompt to test your model's new knowledge
    test_prompt = "User: What is the company policy on remote work? Assistant:"
    
    # Convert text prompt to numeric tokens the model understands
    inputs = tokenizer(test_prompt, return_tensors="pt").to(device)

    # 4. Generate the response
    print(f"🤔 Prompting model: '{test_prompt}'\n")
    with torch.no_grad(): # Disable gradient calculations to save memory and speed up processing
        output_tokens = model.generate(
            **inputs,
            max_new_tokens=50,       # Max length of the generated response
            temperature=0.1,         # Low temperature = highly focused and less random
            do_sample=False,         # Deterministic generation
            pad_token_id=tokenizer.eos_token_id
        )

    # 5. Decode numbers back into readable text
    generated_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    print(f"🤖 Model Response:\n{generated_text}")

if __name__ == "__main__":
    # Windows specific environment flag
    os.environ["HF_ENABLE_ACCELERATE_BITSANDBYTES"] = "1"
    main()