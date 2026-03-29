import torch
import torch.nn as nn
from transformers import GPTNeoConfig, GPTNeoForCausalLM, AutoTokenizer

# 1. We MUST redefine the custom layer so PyTorch knows how to load the weights
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
                module.in_features, module.out_features, bias=(module.bias is not None)
            )
            setattr(model, name, ternary_layer)
        else:
            convert_to_ternary(module)

# 2. Setup Base Model & Tokenizer
print("Loading architecture...")
config = GPTNeoConfig(
    vocab_size=50257, max_position_embeddings=512, window_size=256,
    hidden_size=768, num_layers=4, num_heads=12,
    attention_types=[[["global", "local"], 2]], intermediate_size=3072,
)
model = GPTNeoForCausalLM(config)
tokenizer = AutoTokenizer.from_pretrained("eleutherai/gpt-neo-125m")

# Convert architecture to match the saved ternary weights
convert_to_ternary(model)

# ==========================================
# 3. LOAD YOUR CHECKPOINT HERE
# ==========================================
# Change this to whatever checkpoint you want to test!
checkpoint_path = "checkpoints/checkpoint_step_500.pt" 

print(f"Loading weights from {checkpoint_path} onto CPU...")
# map_location='cpu' is the magic word that saves your GPU from crashing
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])

model.eval() # Set to evaluation mode

# ==========================================
# 4. GENERATE TEXT
# ==========================================
prompt = "Once upon a time, there was a little dog named"
inputs = tokenizer(prompt, return_tensors="pt") # Stays on CPU

print(f"\n--- Generating story for: '{prompt}' ---\n")

with torch.no_grad():
    outputs = model.generate(
        inputs.input_ids,
        max_new_tokens=50,       # Generate 50 new tokens
        temperature=0.7,         # Creativity (0.1 is strict, 1.0 is wild)
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

story = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(story)
print("\n----------------------------------------\n")