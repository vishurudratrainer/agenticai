import os
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model

def main():
    # 1. Define the base model you want to extend
    # Using Microsoft's Phi-3-mini because it's highly performant yet lightweight for Windows
    model_id = "microsoft/Phi-3-mini-4k-instruct"
    
    print("🔄 Loading tokenizer and model...")
    """
    The Tokenizer: Large Language Models don't read words directly; they read numbers called "tokens". The tokenizer breaks your raw text down into these token IDs and manages the padding tokens that make sure text fragments are processed at uniform lengths.

4-Bit Quantization (low_cpu_mem_usage=True): Normally, model weights are stored as 16-bit or 32-bit floating-point numbers, requiring massive GPU memory. By loading the model in a quantized format (4-bit), we drastically compress the model size. This is what allows a complex LLM like Phi-3 to fit into the memory of a consumer Windows graphics card or standard system RAM without crashing.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ Using device: {device}")
    # Load model with 4-bit quantization to save massive amounts of VRAM/RAM
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map={"": device},
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )

    # 2. Prepare a toy dataset (Replace this with your own domain data later)
    # The format should match what your base model expects
    """
    Text Formatting: LLMs learn by predicting the next token. We feed it text examples structured exactly how we want it to behave (e.g., User: [Question] Assistant: [Answer]).

The Mapping Process: The .map() function runs your text through the tokenizer utility we loaded in Step 1. It converts the sentences into arrays of numbers (input_ids) and masks out empty spaces (attention_mask), 
prepping the raw data into mathematical tensors that the underlying machine learning layers can read.
    """
    toy_data = {
        "text": [
            "User: What is the company policy on remote work? Assistant: Employees can work remotely up to 3 days a week.",
            "User: How do I request time off? Assistant: Submit a request through the HR portal at least 2 weeks in advance.",
            "User: Who is the CEO? Assistant: The current CEO of our company is Jane Doe."
        ]
    }
    dataset = Dataset.from_dict(toy_data)
    
    # Tokenize the text data
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=128)
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    # 3. Apply PEFT / LoRA (This builds the extendable adapter layer)
    """
    This is the most critical architectural step of the script. Instead of modifying the massive, original base model, we use PEFT (Parameter-Efficient Fine-Tuning).

Freezing the Base: The millions or billions of parameters inside the original Phi-3 model are marked as non-trainable (frozen). They will not change.

Low-Rank Adaptation (LoRA): We inject two incredibly small, lightweight matrices directly next to the model's core attention layers (qkv_proj).

The Parameter Difference: When model.print_trainable_parameters() executes, you will see that instead of training 3.8 billion parameters, you are only training a tiny fraction—often less than 1% of the model.
    """
    lora_config = LoraConfig(
        r=8,                       # Rank: controls the size of the adapter layers
        lora_alpha=16,             # Scaling factor
        target_modules=["qkv_proj"] if "Phi-3" in model_id else ["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    print("🛠️ Injecting extendable LoRA adapter layers...")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # Shows how few parameters we are actually training

    # 4. Define Training Arguments
    """
    per_device_train_batch_size=1: Tells the script to process exactly one text example at a time. Keeping this at 1 is the golden rule for running on consumer Windows hardware to avoid "Out of Memory" hardware exceptions.gradient_accumulation_steps=4: Because a batch size of 1 is highly erratic for a model to learn from stably, this parameter forces the script to look at 4 total examples sequentially, calculate their changes, and combine them before actually making an official update to the weights.num_train_epochs=3: The training loop will iterate through your entire dataset 3 complete times so the adapter can recognize the structural patterns in your data.learning_rate=2e-4: Controls how drastically the adapter changes its internal numbers based on mistakes it makes during training. $2 \times 10^{-4}$ ($0.0002$) is a proven sweet-spot for fine-tuning text adapters.
    """
    training_args = TrainingArguments(
        output_dir="./extended_model_checkpoints",
        per_device_train_batch_size=1,     # Small batch size to avoid Out-Of-Memory errors on Windows
        gradient_accumulation_steps=4,
        num_train_epochs=3,                # Number of passes through your data
        learning_rate=2e-4,
        fp16=True if torch.cuda.is_available() else False, # Use mixed precision if on GPU
        logging_steps=1,
        save_strategy="epoch",
        report_to="none"  ,
                         # Force the console to print status on every single micro-step
        logging_steps=1,                   
        logging_first_step=True                 # Disables cloud logging wrappers like WandB
    )

    # 5. Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    # 6. Run Training
    print("🚀 Starting training...")
    trainer.train()
    """
    Forward Pass: The model takes an input sequence from your dataset, processes it through the frozen base weights combined with the new LoRA weights, and generates a prediction.

Loss Calculation: The trainer compares the model's prediction against the actual token that was supposed to come next in your dataset. The difference between the prediction and reality is quantified as the Loss.

Backward Pass (Backpropagation): The script calculates gradients based on the Loss. It sends those error signals back through the network. Because the base model is frozen, only the tiny LoRA layers catch these adjustments and tweak their weights to minimize the error.
    """
    # 7. Save your trained "Extension Pack" (Adapter Weights)
    output_directory = "./my_extended_model"
    print(f"💾 Saving extended adapter weights to {output_directory}...")
    model.save_pretrained(output_directory)
    tokenizer.save_pretrained(output_directory)
    print("✅ Done!")

if __name__ == "__main__":
    # Windows specific multiprocessing fix
    os.environ["HF_ENABLE_ACCELERATE_BITSANDBYTES"] = "1"
    main()