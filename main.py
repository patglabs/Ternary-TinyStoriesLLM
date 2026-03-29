import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import multiprocessing
from torch.utils.data import DataLoader
from transformers import GPTNeoConfig, GPTNeoForCausalLM, AutoTokenizer
from datasets import load_dataset, load_from_disk

# ==========================================
# 0. WORKER OPTIMIZATION
# ==========================================
# Limit workers to 8 to avoid overwhelming the CPU while the GPU does the heavy lifting
num_workers = min(multiprocessing.cpu_count(), 8)

# ==========================================
# 1. DEFINE TERNARY LAYER
# ==========================================
class TernaryLinear(nn.Linear):
    def forward(self, x):
        weight = self.weight - self.weight.mean()
        gamma = weight.abs().mean()
        w_q = torch.sign(weight) * torch.where(weight.abs() > 0.5 * gamma, 1, 0)
        out = nn.functional.linear(x, weight + (w_q - weight).detach(), self.bias)
        return out

# ==========================================
# 2. MODEL & TOKENIZER SETUP (~33M Params)
# ==========================================
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
tokenizer = AutoTokenizer.from_pretrained("eleutherai/gpt-neo-125m")
tokenizer.pad_token = tokenizer.eos_token

# ==========================================
# 3. CONVERT TO TERNARY
# ==========================================
def convert_to_ternary(model):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            ternary_layer = TernaryLinear(
                module.in_features, 
                module.out_features, 
                bias=(module.bias is not None)
            )
            ternary_layer.weight.data.copy_(module.weight.data)
            if module.bias is not None:
                ternary_layer.bias.data.copy_(module.bias.data)
            setattr(model, name, ternary_layer)
        else:
            convert_to_ternary(module)

convert_to_ternary(model)

# ==========================================
# 4. LOAD LOCAL DATASET 
# ==========================================
tokenized_path = "dataset/tokenized_tiny_stories"

if os.path.exists(tokenized_path):
    print("\n[!] Found pre-tokenized dataset! Loading from disk...")
    tokenized_dataset = load_from_disk(tokenized_path)
else:
    print("\n[!] Tokenized dataset not found. Tokenizing from CSVs...")
    dataset = load_dataset("csv", data_files={
        "train": "dataset/train.csv", 
        "validation": "dataset/validation.csv"
    })

    def tokenize_function(examples):
        safe_texts = [str(text) if text is not None else "" for text in examples["text"]]
        return tokenizer(safe_texts, padding="max_length", truncation=True, max_length=256)

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset["train"].column_names)
    tokenized_dataset.set_format("torch")
    
    print(f"\n[!] Saving tokenized dataset to ./{tokenized_path}/")
    tokenized_dataset.save_to_disk(tokenized_path)

# GPU Optimization: High batch size, and pin_memory=True for fast CPU-to-GPU transfers
train_loader = DataLoader(tokenized_dataset["train"], batch_size=128, shuffle=True, num_workers=num_workers, pin_memory=True)
val_loader = DataLoader(tokenized_dataset["validation"], batch_size=128, shuffle=False, num_workers=num_workers, pin_memory=True)

# ==========================================
# 5. TRAINING SETUP & CHECKPOINT LOGIC
# ==========================================
# Automatically target the NVIDIA GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = optim.AdamW(model.parameters(), lr=5e-4)

# GPU specific tool for Mixed Precision (Uses Tensor Cores)
scaler = torch.cuda.amp.GradScaler()

accumulation_steps = 1 # Not needed with a batch size of 128
save_interval = 500    # Increased interval because steps will happen very fast

checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

start_step = 0
existing_checkpoints = glob.glob(f"{checkpoint_dir}/checkpoint_step_*.pt")
if existing_checkpoints:
    latest_checkpoint = max(existing_checkpoints, key=os.path.getctime)
    print(f"\n[!] Found existing checkpoint: {latest_checkpoint}. Loading...")
    
    try:
        checkpoint = torch.load(latest_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_step = checkpoint['step']
        print(f"[!] Resumed successfully from optimization step {start_step}.")
    except RuntimeError as e:
        print(f"\n[ERROR] Checkpoint shape mismatch!")
        exit(1)
else:
    print("\n[!] No existing checkpoints found. Starting from scratch.")

# ==========================================
# 6. METADATA PRINTING
# ==========================================
print("\n" + "="*50)
print("GPU TRAINING METADATA (33M DIET)")
print("="*50)
print(f"Device:                 {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'})")
print(f"Total Parameters:       {model.num_parameters():,}")
print(f"Dataset Split (Train):  {len(tokenized_dataset['train']):,} examples")
print(f"Dataset Split (Val):    {len(tokenized_dataset['validation']):,} examples")
print(f"Batch Size:             128")
print(f"Mixed Precision (AMP):  Enabled")
print(f"Learning Rate:          5e-4")
print(f"Checkpoint Directory:   ./{checkpoint_dir}/")
print("="*50 + "\n")

# ==========================================
# 7. TRAINING LOOP
# ==========================================
num_epochs = 3
optimization_steps = start_step 

for epoch in range(num_epochs):
    print(f"\n--- STARTING EPOCH {epoch + 1}/{num_epochs} ---\n")
    model.train()
    optimizer.zero_grad()

    for i, batch in enumerate(train_loader):
        inputs = batch['input_ids'].to(device, non_blocking=True)
        labels = inputs.clone()

        # Run the forward pass in mixed precision (16-bit)
        with torch.cuda.amp.autocast():
            outputs = model(inputs, labels=labels)
            loss = outputs.loss / accumulation_steps
        
        # Scale the loss and run backward pass
        scaler.scale(loss).backward()

        if (i + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            optimization_steps += 1
            
            print(f"Epoch: {epoch+1}/{num_epochs} | Opt Step: {optimization_steps} | Loss: {loss.item() * accumulation_steps:.4f}")

            if optimization_steps % save_interval == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step_{optimization_steps}.pt")
                torch.save({
                    'step': optimization_steps,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item() * accumulation_steps,
                }, checkpoint_path)
                print(f"\n--- Checkpoint saved to: {checkpoint_path} ---\n")

    print("\n--- Running Validation ---")
    model.eval()
    total_val_loss = 0.0
    val_steps = 0
    
    with torch.no_grad():
        for val_batch in val_loader:
            val_inputs = val_batch['input_ids'].to(device, non_blocking=True)
            val_labels = val_inputs.clone()
            
            with torch.cuda.amp.autocast():
                val_outputs = model(val_inputs, labels=val_labels)
                total_val_loss += val_outputs.loss.item()
                
            val_steps += 1
            
    avg_val_loss = total_val_loss / val_steps
    print(f"Epoch {epoch+1} Validation Loss: {avg_val_loss:.4f}\n")

print("\nTraining Complete!")