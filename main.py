import os
import glob
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import multiprocessing
from torch.utils.data import DataLoader
from transformers import GPTNeoConfig, GPTNeoForCausalLM, GPT2TokenizerFast
from datasets import load_dataset, load_from_disk

# ==========================================
# 1. CONFIGURATION
# ==========================================
class CFG:
    # Model parameters
    vocab_size = 4096
    max_len = 256
    hidden_size = 256
    num_layers = 4
    num_heads = 8
    intermediate_size = 1024
    
    # Training parameters
    lr = 5e-5
    batch_size = 64
    accumulation_steps = 8 # Effective batch size = 576
    epochs = 3
    seed = 42
    
    # Paths
    dataset_path = "dataset/tokenized_tiny_stories_4k"
    tokenizer_path = "dataset/tokenizer_4k"
    checkpoint_dir = "checkpoints_4k"
    
    # Ternary parameters
    ternary_threshold = 0.1 # Sensitivity for snapping to -1, 0, 1
    eps = 1e-9

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(CFG.seed)

# ==========================================
# 2. TERNARY LAYER DEFINITION
# ==========================================
class TernaryLinear(nn.Linear):
    def forward(self, x):
        # 1. Activation Scaling (Surge Protector #1)
        # We normalize the input 'x' so its variance is 1.0. 
        # This prevents the "Attention Explosion" you're seeing.
        x_norm = (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-5)
        
        # 2. Weight Centering
        weight = self.weight - self.weight.mean()
        
        # 3. Scaling Factor (Gamma)
        gamma = weight.abs().mean() + 1e-9
        
        # 4. Ternary Snap (Straight-Through Estimator)
        w_q = torch.sign(weight) * torch.where(weight.abs() > (0.1 * gamma), 1.0, 0.0)
        weight_ste = weight + (w_q - weight).detach()
        
        # 5. Linear math + Output Scaling
        out = nn.functional.linear(x_norm, weight_ste, self.bias)
        
        # Scale the output by gamma to keep the range consistent
        return out * gamma

def convert_to_ternary(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
            child_name = name.rsplit('.', 1)[-1]
            parent = dict(model.named_modules())[parent_name] if parent_name else model
            
            new_layer = TernaryLinear(
                module.in_features, 
                module.out_features, 
                bias=(module.bias is not None)
            ).to(module.weight.device)
            
            # THE FIX: Instead of copying tiny GPT-Neo defaults, 
            # we use Xavier to give the ternary snap a wider range of values.
            with torch.no_grad():
                nn.init.xavier_uniform_(new_layer.weight)
                if module.bias is not None:
                    nn.init.zeros_(new_layer.bias)
            
            setattr(parent, child_name, new_layer)
            print(f"[INFO] Ternarized & Re-initialized: {name}")

# ==========================================
# 3. DATA LOADING AND TOKENIZATION
# ==========================================
def get_dataset(tokenizer):
    if os.path.exists(CFG.dataset_path):
        print("[INFO] Loading tokenized dataset from disk...")
        return load_from_disk(CFG.dataset_path)
    
    print("[INFO] Tokenizing dataset from CSV...")
    raw_dataset = load_dataset("csv", data_files={
        "train": "dataset/train.csv", 
        "validation": "dataset/validation.csv"
    })
    
    def tokenize_fn(examples):
        texts = [str(t) if t is not None else "" for t in examples["text"]]
        return tokenizer(texts, padding="max_length", truncation=True, max_length=CFG.max_len)
    
    tokenized = raw_dataset.map(tokenize_fn, batched=True, remove_columns=raw_dataset["train"].column_names)
    tokenized.set_format("torch")
    tokenized.save_to_disk(CFG.dataset_path)
    return tokenized

# ==========================================
# 4. TRAINING LOOP
# ==========================================
if __name__ == '__main__':
    os.makedirs(CFG.checkpoint_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Initialize Tokenizer
    tokenizer = GPT2TokenizerFast(
        vocab_file=f"{CFG.tokenizer_path}/vocab.json",
        merges_file=f"{CFG.tokenizer_path}/merges.txt",
        eos_token="<|endoftext|>",
        pad_token="<|padding|>"
    )

    # 2. Initialize Model
    model_config = GPTNeoConfig(
        vocab_size=CFG.vocab_size,
        max_position_embeddings=CFG.max_len,
        window_size=CFG.max_len,
        hidden_size=CFG.hidden_size,
        num_layers=CFG.num_layers,
        num_heads=CFG.num_heads,
        attention_types=[[["global", "local"], CFG.num_layers // 2]],
        intermediate_size=CFG.intermediate_size,
    )
    
    model = GPTNeoForCausalLM(model_config)
    model.gradient_checkpointing_enable()
    convert_to_ternary(model)
    model.to(device)

    # 3. Prepare Data
    dataset = get_dataset(tokenizer)
    num_workers = min(multiprocessing.cpu_count(), 8)
    train_loader = DataLoader(dataset["train"], batch_size=CFG.batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    # 4. Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=CFG.lr)
    
    # 5. Checkpoint Handling
    start_step = 0
    checkpoint_files = glob.glob(f"{CFG.checkpoint_dir}/checkpoint_step_*.pt")
    if checkpoint_files:
        latest = max(checkpoint_files, key=os.path.getctime)
        print(f"[INFO] Resuming from {latest}")
        ckpt = torch.load(latest, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_step = ckpt['step']

    # 6. Training Logic
    total_batches = len(train_loader)
    steps_per_epoch = total_batches // CFG.accumulation_steps
    current_step = start_step

    print("\n" + "="*30)
    print(f"RUNNING TERNARY TRAINING")
    print(f"Hardware: {torch.cuda.get_device_name(0)}")
    print(f"Steps Per Epoch: {steps_per_epoch}")
    print("="*30 + "\n")

    for epoch in range(CFG.epochs):
        model.train()
        optimizer.zero_grad()
        
        for i, batch in enumerate(train_loader):
            # Skip batches if resuming
            if current_step * CFG.accumulation_steps > (epoch * total_batches + i):
                continue

            ids = batch['input_ids'].to(device, non_blocking=True)
            labels = ids.clone()
            labels[labels == tokenizer.pad_token_id] = -100 # Mask padding

            outputs = model(ids, labels=labels)
            loss = outputs.loss / CFG.accumulation_steps

            if torch.isnan(loss):
                print(f"[ERROR] NaN loss at step {current_step}. Skipping batch.")
                optimizer.zero_grad()
                continue

            loss.backward()

            if (i + 1) % CFG.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                optimizer.zero_grad()
                current_step += 1

                if current_step % 10 == 0:
                    print(f"Epoch {epoch+1} | Step {current_step} | Loss: {loss.item() * CFG.accumulation_steps:.4f}")

                if current_step % 250 == 0:
                    ckpt_path = f"{CFG.checkpoint_dir}/checkpoint_step_{current_step}.pt"
                    torch.save({
                        'step': current_step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, ckpt_path)
                    print(f"[INFO] Saved checkpoint: {ckpt_path}")

    print("[SUCCESS] Training complete.")