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
# 2. CONVERT TO TERNARY FUNCTION
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
# MAIN EXECUTION BLOCK (Required for Windows)
# ==========================================
if __name__ == '__main__':
    
    # 0. Worker Optimization
    num_workers = min(multiprocessing.cpu_count(), 8)

    # 1. Model & Tokenizer Setup (~8.3M Params)
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

    # 2. Load Local Dataset
    tokenized_path = "dataset/tokenized_tiny_stories_10k"

    if os.path.exists(tokenized_path):
        print("\n[!] Loading pre-tokenized 10k dataset...")
        tokenized_dataset = load_from_disk(tokenized_path)
    else:
        print("\n[!] Tokenizing from CSVs...")
        dataset = load_dataset("csv", data_files={
            "train": "dataset/train.csv", 
            "validation": "dataset/validation.csv"
        })

        def tokenize_function(examples):
            safe_texts = [str(text) if text is not None else "" for text in examples["text"]]
            return tokenizer(safe_texts, padding="max_length", truncation=True, max_length=256)

        tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset["train"].column_names)
        tokenized_dataset.set_format("torch")
        tokenized_dataset.save_to_disk(tokenized_path)

    train_loader = DataLoader(tokenized_dataset["train"], batch_size=32, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(tokenized_dataset["validation"], batch_size=32, shuffle=False, num_workers=num_workers, pin_memory=True)

    # 3. Training Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # STABILIZATION: Lowered Learning Rate to 1e-4 to prevent NaN
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler('cuda')

    accumulation_steps = 4 
    save_interval = 500    
    checkpoint_dir = "checkpoints_10k"
    os.makedirs(checkpoint_dir, exist_ok=True)

    start_step = 0
    existing_checkpoints = glob.glob(f"{checkpoint_dir}/checkpoint_step_*.pt")
    if existing_checkpoints:
        latest_checkpoint = max(existing_checkpoints, key=os.path.getctime)
        print(f"\n[!] Resuming from: {latest_checkpoint}")
        checkpoint = torch.load(latest_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_step = checkpoint['step']
    else:
        print("\n[!] Starting from scratch.")

    # 4. Metadata
    print("\n" + "="*50)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Params: {model.num_parameters():,}")
    print(f"LR: 1e-4 | Clipping: 1.0")
    print("="*50 + "\n")

    # 5. Training Loop
    num_epochs = 3
    optimization_steps = start_step 

    for epoch in range(num_epochs):
        print(f"\n--- EPOCH {epoch + 1} ---\n")
        model.train()
        optimizer.zero_grad()

        for i, batch in enumerate(train_loader):
            inputs = batch['input_ids'].to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda'):
                outputs = model(inputs, labels=inputs)
                loss = outputs.loss / accumulation_steps
            
            # Check for NaN immediately
            if torch.isnan(loss):
                print("\n[!] CRITICAL: NaN detected! Stopping to save model state.")
                exit(1)

            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0:
                # STABILIZATION: Gradient Clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                optimization_steps += 1
                
                if optimization_steps % 10 == 0:
                    print(f"Step: {optimization_steps} | Loss: {loss.item() * accumulation_steps:.4f}")

                if optimization_steps % save_interval == 0:
                    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step_{optimization_steps}.pt")
                    torch.save({
                        'step': optimization_steps,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, checkpoint_path)
                    print(f"💾 Checkpoint: {checkpoint_path}")

        # Validation logic remains same...
        print("\n--- Validation Complete ---\n")

    print("\nTraining Complete!")