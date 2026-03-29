import os
import glob
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import multiprocessing
from torch.utils.data import DataLoader
from transformers import GPTNeoConfig, GPTNeoForCausalLM, AutoTokenizer
from datasets import load_dataset, load_from_disk

# ==========================================
# 1. REPRODUCIBILITY SEED
# ==========================================
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)

# ==========================================
# 2. DEFINE TERNARY LAYER (BFLOAT16 OPTIMIZED)
# ==========================================
class TernaryLinear(nn.Linear):
    def forward(self, x):
        weight = self.weight - self.weight.mean()
        # UPDATED: 1e-5 epsilon. BFloat16 can easily process this without rounding to zero.
        gamma = weight.abs().mean() + 1e-5 
        w_q = torch.sign(weight) * torch.where(weight.abs() > 0.5 * gamma, 1, 0)
        out = nn.functional.linear(x, weight + (w_q - weight).detach(), self.bias)
        return out

# ==========================================
# 3. CONVERT TO TERNARY FUNCTION
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

# ==========================================
# MAIN EXECUTION BLOCK
# ==========================================
if __name__ == '__main__':
    
    num_workers = min(multiprocessing.cpu_count(), 8)

    # Model & Tokenizer Config
    config = GPTNeoConfig(
        vocab_size=10000,
        max_position_embeddings=256,     
        window_size=256,
        hidden_size=256,                 
        num_layers=4,
        num_heads=8,                     
        attention_types=[[["global", "local"], 2]],
        intermediate_size=1024,          
    )

    model = GPTNeoForCausalLM(config)
    tokenizer = AutoTokenizer.from_pretrained("vuiseng9/bpe-10.0k-tinystories")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    convert_to_ternary(model)

    # Data Loading
    tokenized_path = "dataset/tokenized_tiny_stories_10k"
    if os.path.exists(tokenized_path):
        print("\n[!] Loading pre-tokenized 10k dataset...")
        tokenized_dataset = load_from_disk(tokenized_path)
    else:
        print("\n[!] Tokenizing from CSVs...")
        dataset = load_dataset("csv", data_files={"train": "dataset/train.csv", "validation": "dataset/validation.csv"})
        def tokenize_function(examples):
            safe_texts = [str(text) if text is not None else "" for text in examples["text"]]
            return tokenizer(safe_texts, padding="max_length", truncation=True, max_length=256)
        tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset["train"].column_names)
        tokenized_dataset.set_format("torch")
        tokenized_dataset.save_to_disk(tokenized_path)

    # DataLoader
    train_loader = DataLoader(tokenized_dataset["train"], batch_size=32, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(tokenized_dataset["validation"], batch_size=32, shuffle=False, num_workers=num_workers, pin_memory=True)

    # Training Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=5e-7)
    
    # REMOVED: torch.amp.GradScaler('cuda') is no longer needed with BFloat16

    accumulation_steps = 16 
    save_interval = 500    
    checkpoint_dir = "checkpoints_10k"
    os.makedirs(checkpoint_dir, exist_ok=True)

    start_step = 0
    latest_checkpoint_path = None

    existing_checkpoints = glob.glob(f"{checkpoint_dir}/checkpoint_step_*.pt")
    if existing_checkpoints:
        latest_checkpoint = max(existing_checkpoints, key=os.path.getctime)
        latest_checkpoint_path = latest_checkpoint
        print(f"\n[!] Resuming from: {latest_checkpoint}")
        checkpoint = torch.load(latest_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_step = checkpoint['step']
    else:
        print("\n[!] Starting from scratch.")

    # Calculations for Progress Tracking
    total_batches = len(train_loader)
    steps_per_epoch = total_batches // accumulation_steps
    
    current_epoch_start = (start_step // steps_per_epoch)
    batches_to_skip_in_epoch = (start_step % steps_per_epoch) * accumulation_steps

    print("\n" + "="*50)
    print(f"RESUME STATUS: Epoch {current_epoch_start + 1}, Skip {batches_to_skip_in_epoch} batches")
    print(f"Total Steps Per Epoch: {steps_per_epoch}")
    print(f"Mixed Precision: Native BFloat16")
    print("="*50 + "\n")

    # Training Loop
    num_epochs = 3
    optimization_steps = start_step 

    for epoch in range(current_epoch_start, num_epochs):
        print(f"\n--- STARTING EPOCH {epoch + 1}/{num_epochs} ---")
        model.train()
        optimizer.zero_grad()

        for i, batch in enumerate(train_loader):
            if epoch == current_epoch_start and i < batches_to_skip_in_epoch:
                continue

            inputs = batch['input_ids'].to(device, non_blocking=True)
            
            # UPDATED: Enforcing torch.bfloat16
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outputs = model(inputs, labels=inputs)
                loss = outputs.loss / accumulation_steps
            
            # ==========================================
            # SELF-HEALING PROTOCOL
            # ==========================================
            if torch.isnan(loss):
                print(f"\n[!] 🚨 CRITICAL: NaN detected at step {optimization_steps}! Initiating Auto-Recovery...")
                
                if latest_checkpoint_path and os.path.exists(latest_checkpoint_path):
                    print(f"[!] Purging poisoned memory and reloading safe state: {latest_checkpoint_path}")
                    
                    checkpoint = torch.load(latest_checkpoint_path, map_location=device, weights_only=False)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    
                    optimizer.zero_grad()
                    
                    print("[!] ✅ Recovery complete. Skipping bad batch and continuing...\n")
                    continue
                else:
                    print("[!] No safe checkpoint found to recover from. Exiting.")
                    exit(1)

            # UPDATED: Normal backward pass (no scaler needed)
            loss.backward()

            if (i + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                
                # UPDATED: Normal optimizer step (no scaler needed)
                optimizer.step()
                optimizer.zero_grad()
                optimization_steps += 1
                
                if optimization_steps % 10 == 0:
                    current_epoch_step = optimization_steps % steps_per_epoch
                    progress_pct = (current_epoch_step / steps_per_epoch) * 100
                    print(f"Step: {optimization_steps} | Loss: {loss.item() * accumulation_steps:.4f} | Progress: {progress_pct:.2f}%")

                if optimization_steps % save_interval == 0:
                    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step_{optimization_steps}.pt")
                    torch.save({
                        'step': optimization_steps,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, checkpoint_path)
                    
                    latest_checkpoint_path = checkpoint_path
                    print(f"💾 Checkpoint saved: {checkpoint_path}")

        print("\n--- Epoch Validation ---")
        # (Validation logic would go here)

    print("\nTraining Complete!")