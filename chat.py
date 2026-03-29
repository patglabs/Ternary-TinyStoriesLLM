import os
import glob
import torch
import torch.nn as nn
from transformers import GPTNeoConfig, GPTNeoForCausalLM, AutoTokenizer

# ==========================================
# 1. TERNARY LAYER DEFINITION (Must match training exactly)
# ==========================================
class TernaryLinear(nn.Linear):
    def forward(self, x):
        weight = self.weight - self.weight.mean()
        gamma = weight.abs().mean()
        w_q = torch.sign(weight) * torch.where(weight.abs() > 0.5 * gamma, 1, 0)
        out = nn.functional.linear(x, weight + (w_q - weight).detach(), self.bias)
        return out

def convert_to_ternary(model):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            ternary_layer = TernaryLinear(
                module.in_features, 
                module.out_features, 
                bias=(module.bias is not None)
            )
            # Copy existing weights over
            ternary_layer.weight.data.copy_(module.weight.data)
            if module.bias is not None:
                ternary_layer.bias.data.copy_(module.bias.data)
            setattr(model, name, ternary_layer)
        else:
            convert_to_ternary(module)

# ==========================================
# 2. LOAD MODEL & TOKENIZER
# ==========================================
print("Loading architecture and tokenizer...")

config = GPTNeoConfig(
    vocab_size=50257,
    max_position_embeddings=256,
    window_size=256,
    hidden_size=256,
    num_layers=4,
    num_heads=8,
    attention_types=[[["global", "local"], 2]],
    intermediate_size=1024,
)

model = GPTNeoForCausalLM(config)
convert_to_ternary(model) # Convert to ternary BEFORE loading weights

tokenizer = AutoTokenizer.from_pretrained("eleutherai/gpt-neo-125m")
tokenizer.pad_token = tokenizer.eos_token

# Automatically use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ==========================================
# 3. LOAD LATEST CHECKPOINT
# ==========================================
checkpoint_dir = "checkpoints"
existing_checkpoints = glob.glob(f"{checkpoint_dir}/checkpoint_step_*.pt")

if not existing_checkpoints:
    print(f"\n[ERROR] No checkpoints found in './{checkpoint_dir}/'.")
    print("Please wait for your training script to save at least one checkpoint before running chat.py.")
    exit(1)

# Find the most recently modified checkpoint file
latest_checkpoint = max(existing_checkpoints, key=os.path.getctime)
print(f"Loading weights from: {latest_checkpoint}")

try:
    checkpoint = torch.load(latest_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Weights loaded successfully!")
except Exception as e:
    print(f"\n[ERROR] Failed to load checkpoint. Ensure the architecture matches the training script.")
    print(f"Error details: {e}")
    exit(1)

model.eval() # Set model to evaluation mode

# ==========================================
# 4. CHAT LOOP
# ==========================================
print("\n" + "="*50)
print("TERNARY AI CHAT INTERFACE")
print("Type 'quit' or 'exit' to stop.")
print("="*50 + "\n")

while True:
    prompt = input("You: ")
    
    if prompt.lower() in ['quit', 'exit']:
        break
    if not prompt.strip():
        continue

    # Tokenize the input text
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate the response
    with torch.no_grad():
        output_tokens = model.generate(
            inputs.input_ids,
            max_new_tokens=50,          # How many new words to generate
            temperature=0.8,            # Higher = more creative, Lower = more predictable
            do_sample=True,             # Allow it to pick words probabilistically
            top_p=0.9,                  # Nucleus sampling for better coherence
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode and print the output
    response = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    
    print(f"\nAI: {response}\n")
    print("-" * 50)