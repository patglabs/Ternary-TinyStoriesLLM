import os
import glob
import torch
import torch.nn as nn
from transformers import GPTNeoConfig, GPTNeoForCausalLM, AutoTokenizer

# 1. TERNARY LAYER (Must match main.py)
class TernaryLinear(nn.Linear):
    def forward(self, x):
        weight = self.weight - self.weight.mean()
        gamma = weight.abs().mean() + 1e-8
        w_q = torch.sign(weight) * torch.where(weight.abs() > 0.1 * gamma, 1, 0)
        out = nn.functional.linear(x, weight + (w_q - weight).detach(), self.bias)
        return out * gamma

def convert_to_ternary(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
            child_name = name.rsplit('.', 1)[-1]
            parent = dict(model.named_modules())[parent_name] if parent_name else model
            ternary_layer = TernaryLinear(module.in_features, module.out_features, bias=(module.bias is not None)).to(module.weight.device)
            setattr(parent, child_name, ternary_layer)

# 2. LOAD SETUP
config = GPTNeoConfig(
    vocab_size=4096,
    hidden_size=256,
    num_layers=4,
    num_heads=8,
    max_position_embeddings=256,
    # This matches your 4-layer structure (Global/Local repeated twice)
    attention_types=[[["global", "local"], 2]], 
    intermediate_size=1024, # Match the 1024 from your training script
    window_size=256
)
model = GPTNeoForCausalLM(config)
convert_to_ternary(model)

# USE THE REPL METHOD HERE
tokenizer = AutoTokenizer.from_pretrained("dataset/tokenizer_4k")
tokenizer.pad_token = tokenizer.eos_token

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 3. LOAD CHECKPOINT
checkpoint_path = "checkpoints_4k/checkpoint_step_500.pt"
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Successfully loaded {checkpoint_path}")
model.eval()

# 4. CHAT LOOP
while True:
    prompt = input("You: ")
    if prompt.lower() == 'quit': break

    # DEBUG PRINT: Let's see the IDs
    tokens = tokenizer.encode(prompt)
    print(f"[DEBUG] Tokens: {tokens}") # If this is [], that's our problem!

    if not tokens:
        print("AI: I didn't get that.")
        continue

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        output_tokens = model.generate(
            inputs.input_ids,
            max_new_tokens=50,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    print(f"AI: {tokenizer.decode(output_tokens[0], skip_special_tokens=True)}")