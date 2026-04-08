import os
import glob
import torch
import torch.nn as nn
from transformers import GPTNeoConfig, GPTNeoForCausalLM, PreTrainedTokenizerFast

# 1. TERNARY LAYER (Synchronized exactly with the working main.py)
class TernaryLinear(nn.Linear):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight
        w = w - w.mean()
        scale = w.abs().mean().clamp(min=1e-5)
        
        w_hat = w / scale
        w_q   = torch.sign(w_hat) * (w_hat.abs() >= 0.5).float()
        
        w_ste = w + (w_q * scale - w).detach()
        return nn.functional.linear(x, w_ste, self.bias)

def convert_to_ternary(model: nn.Module) -> None:
    modules = list(model.named_modules())
    for name, module in modules:
        if not isinstance(module, nn.Linear) or "lm_head" in name:
            continue

        parent_name = name.rsplit(".", 1)[0] if "." in name else ""
        child_name  = name.rsplit(".", 1)[-1]
        parent = dict(model.named_modules()).get(parent_name, model)

        new_layer = TernaryLinear(
            module.in_features,
            module.out_features,
            bias=(module.bias is not None),
        ).to(module.weight.device)

        setattr(parent, child_name, new_layer)

# 2. LOAD SETUP
# Use the exact same Tokenizer class from training
tokenizer = PreTrainedTokenizerFast(
    tokenizer_file="dataset/tokenizer_4k/tokenizer.json",
    bos_token="<|endoftext|>",
    eos_token="<|endoftext|>",
    pad_token="<|padding|>"
)

# Sync architecture with the new global-only setup
config = GPTNeoConfig(
    vocab_size=len(tokenizer),
    hidden_size=256,
    num_layers=4,
    num_heads=8,
    max_position_embeddings=256,
    attention_types=[[["global"], 4]],  # Matched to training script
    intermediate_size=1024,
    window_size=256
)
model = GPTNeoForCausalLM(config)
convert_to_ternary(model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 3. AUTO-LOAD NEWEST CHECKPOINT
checkpoint_dir = "checkpoints_4k"
ckpt_files = glob.glob(f"{checkpoint_dir}/checkpoint_step_*.pt")

if not ckpt_files:
    print(f"Error: No checkpoints found in {checkpoint_dir}/")
    exit()

# Find the newest file based on creation time
newest_checkpoint = max(ckpt_files, key=os.path.getctime)

print(f"[INFO] Loading newest checkpoint: {newest_checkpoint}")
checkpoint = torch.load(newest_checkpoint, map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
print("[SUCCESS] Model weights loaded. Ready to chat.")
model.eval()

# 4. CHAT LOOP
while True:
    prompt = input("\nYou: ")
    if prompt.lower() == 'quit': 
        break

    # DEBUG PRINT: Let's see the IDs
    tokens = tokenizer.encode(prompt)
    print(f"[DEBUG] Tokens: {tokens}") 

    if not tokens:
        print("AI: I didn't get that.")
        continue

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        output_tokens = model.generate(
            inputs.input_ids,
            max_new_tokens=210,     # Let it write a full paragraph or two!
            temperature=0.8,        
            top_k=40,               
            top_p=0.9,              
            repetition_penalty=1.25, # Bumped slightly to stop the "dog, a dog" stutter
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    print(f"AI: {tokenizer.decode(output_tokens[0], skip_special_tokens=True)}")