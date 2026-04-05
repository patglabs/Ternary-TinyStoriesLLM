import os
import glob
import math
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import multiprocessing
from torch.utils.data import DataLoader
from transformers import GPTNeoConfig, GPTNeoForCausalLM, GPT2TokenizerFast
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict
from torch.optim.lr_scheduler import LambdaLR
import pandas as pd
# ==========================================
# 1. CONFIGURATION
# ==========================================
class CFG:
    # Will be dynamically overwritten by actual tokenizer length
    vocab_size         = 4096 
    max_len            = 256
    hidden_size        = 256
    num_layers         = 4
    num_heads          = 8
    intermediate_size  = 1024

    lr                 = 3e-4
    batch_size         = 32
    accumulation_steps = 16
    epochs             = 3
    warmup_steps       = 200
    grad_clip          = 1.0
    seed               = 42

    dataset_path   = "dataset/tokenized_tiny_stories_4k"
    tokenizer_path = "dataset/tokenizer_4k"
    checkpoint_dir = "checkpoints_4k"

    ternary_threshold = 0.5
    eps               = 1e-8

def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

seed_everything(CFG.seed)

# ==========================================
# 2. TERNARY LINEAR
# ==========================================
class TernaryLinear(nn.Linear):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight
        scale = w.abs().mean().clamp(min=CFG.eps)
        
        w_hat = w / scale
        w_q   = torch.sign(w_hat) * (w_hat.abs() >= CFG.ternary_threshold).float()
        
        w_ste = w + (w_q * scale - w).detach()
        return nn.functional.linear(x, w_ste, self.bias)

def convert_to_ternary(model: nn.Module) -> None:
    modules = list(model.named_modules())
    for name, module in modules:
        if not isinstance(module, nn.Linear):
            continue
        if "lm_head" in name:
            continue

        parent_name = name.rsplit(".", 1)[0] if "." in name else ""
        child_name  = name.rsplit(".", 1)[-1]
        parent = dict(model.named_modules()).get(parent_name, model)

        new_layer = TernaryLinear(
            module.in_features,
            module.out_features,
            bias=(module.bias is not None),
        ).to(module.weight.device)

        with torch.no_grad():
            nn.init.xavier_uniform_(new_layer.weight)
            if new_layer.bias is not None:
                nn.init.zeros_(new_layer.bias)

        setattr(parent, child_name, new_layer)
        print(f"[INFO] Ternarized: {name}")

# ==========================================
# 3. FORWARD-PASS NaN DIAGNOSTIC
# ==========================================
def run_nan_diagnostic(model: nn.Module, tokenizer, device: torch.device) -> bool:
    print("\n[DIAG] Running forward-pass NaN diagnostic ...")
    first_bad: dict = {}

    def make_hook(layer_name: str):
        def hook(module, inp, out):
            if first_bad: 
                return
            tensor = out[0] if isinstance(out, tuple) else out
            if isinstance(tensor, torch.Tensor) and not torch.isfinite(tensor).all():
                bad_frac = (~torch.isfinite(tensor)).float().mean().item()
                in0 = inp[0] if isinstance(inp, tuple) else inp
                in_ok = torch.isfinite(in0).all().item() if isinstance(in0, torch.Tensor) else True
                first_bad["name"]     = layer_name
                first_bad["bad_frac"] = bad_frac
                first_bad["input_ok"] = in_ok
        return hook

    handles = [
        mod.register_forward_hook(make_hook(n))
        for n, mod in model.named_modules()
    ]
    
    model.eval()
    try:
        # CFG.vocab_size is now synced to len(tokenizer), making this safe
        ids = torch.randint(0, CFG.vocab_size, (2, CFG.max_len), device=device)
        with torch.no_grad():
            out = model(ids, labels=ids.clone())
        loss_ok = torch.isfinite(out.loss)
    except Exception as exc:
        print(f"[DIAG] Forward pass threw an exception: {exc}")
        loss_ok = torch.tensor(False)
    finally:
        for h in handles:
            h.remove()
        model.train()

    if first_bad:
        print(f"[DIAG] *** First NaN/Inf detected in: '{first_bad['name']}' "
              f"({first_bad['bad_frac']*100:.1f}% non-finite) | "
              f"input was {'clean' if first_bad['input_ok'] else 'ALSO BAD'} ***")
        return False
    
    if not loss_ok:
        print("[DIAG] NaN in loss but no single module flagged.")
        return False

    print("[DIAG] All outputs finite. Model architecture is clean.\n")
    return True

# ==========================================
# 4. LEARNING-RATE SCHEDULER
# ==========================================
def get_scheduler(optimizer, warmup_steps: int, total_steps: int) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)

# ==========================================
# 5. DATA LOADING
# ==========================================
def get_dataset(tokenizer):
    if os.path.exists(CFG.dataset_path):
        print("[INFO] Loading tokenized dataset from disk...")
        return load_from_disk(CFG.dataset_path)

    print("[INFO] Loading and cleaning raw CSVs with Pandas...")
    
    def safe_load_csv(filepath):
        # Read the first line to check for a header
        df = pd.read_csv(filepath, nrows=0)
        
        if "text" in df.columns:
            # Header exists and is named "text"
            df = pd.read_csv(filepath)
        else:
            # No header, or wrongly named header. Force "text" as the column name.
            df = pd.read_csv(filepath, header=None, names=["text"])
            
        # Drop completely empty rows or NaNs
        df = df.dropna(subset=["text"])
        df = df[df["text"].astype(str).str.strip() != ""]
        return df

    try:
        df_train = safe_load_csv("dataset/train.csv")
        df_val = safe_load_csv("dataset/validation.csv")
    except Exception as e:
        raise RuntimeError(f"Failed to read CSV files: {e}")

    # Convert cleaned pandas dataframes back to HuggingFace Datasets
    raw = DatasetDict({
        "train": Dataset.from_pandas(df_train, preserve_index=False),
        "validation": Dataset.from_pandas(df_val, preserve_index=False)
    })

    print(f"[INFO] Found {len(raw['train'])} training stories and {len(raw['validation'])} validation stories.")

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"], 
            padding="max_length", 
            truncation=True, 
            max_length=CFG.max_len
        )

    print("[INFO] Tokenizing...")
    tokenized = raw.map(tokenize_fn, batched=True, remove_columns=["text"])
    tokenized.set_format("torch")
    
    # Ultimate Sanity Check: Ensure the first item isn't just padding
    sample_ids = tokenized["train"][0]["input_ids"]
    if (sample_ids == tokenizer.pad_token_id).all():
        raise ValueError("[CRASH] Tokenization resulted in entirely empty tokens! Check the tokenizer logic.")

    tokenized.save_to_disk(CFG.dataset_path)
    print(f"[INFO] Successfully saved clean tokenized data to {CFG.dataset_path}")
    
    return tokenized

# ==========================================
# 6. MAIN
# ==========================================
if __name__ == "__main__":
    os.makedirs(CFG.checkpoint_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Tokenizer ---
    tokenizer = GPT2TokenizerFast(
        vocab_file  = f"{CFG.tokenizer_path}/vocab.json",
        merges_file = f"{CFG.tokenizer_path}/merges.txt",
        eos_token   = "<|endoftext|>",
    )
    
    # Safely ensure padding token is registered
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '<|padding|>'})

    # [CRITICAL FIX 1] Sync config exactly to the tokenizer's true size
    CFG.vocab_size = len(tokenizer)

    # --- Model ---
    model_config = GPTNeoConfig(
        vocab_size              = CFG.vocab_size,
        max_position_embeddings = CFG.max_len,
        hidden_size             = CFG.hidden_size,
        num_layers              = CFG.num_layers,
        num_heads               = CFG.num_heads,
        attention_types         = [[["global"], CFG.num_layers]],
        window_size             = CFG.max_len,
        intermediate_size       = CFG.intermediate_size,
        use_cache               = False,
    )

    model = GPTNeoForCausalLM(model_config)
    model.gradient_checkpointing_enable()
    convert_to_ternary(model)
    model.to(device)

    total_params   = sum(p.numel() for p in model.parameters())
    ternary_params = sum(p.numel() for n, p in model.named_parameters() if "lm_head" not in n and p.dim() >= 2)

    print(f"\n[INFO] Total params   : {total_params:,}")
    print(f"[INFO] Ternary params : {ternary_params:,}")

    # --- Sanity check ---
    if not run_nan_diagnostic(model, tokenizer, device):
        raise RuntimeError("Model produces NaN/Inf on a clean forward pass.")

    # --- Data ---
    dataset      = get_dataset(tokenizer)
    num_workers  = min(multiprocessing.cpu_count(), 8)
    train_loader = DataLoader(
        dataset["train"],
        batch_size  = CFG.batch_size,
        shuffle     = True,
        num_workers = num_workers,
        pin_memory  = True,
    )

    total_batches   = len(train_loader)
    steps_per_epoch = total_batches // CFG.accumulation_steps
    total_steps     = steps_per_epoch * CFG.epochs

    # --- Optimiser + Scheduler ---
    optimizer = optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=0.01)
    scheduler = get_scheduler(optimizer, CFG.warmup_steps, total_steps)

    # --- Checkpoint resume ---
    start_step = 0
    ckpt_files = glob.glob(f"{CFG.checkpoint_dir}/checkpoint_step_*.pt")
    if ckpt_files:
        latest = max(ckpt_files, key=os.path.getctime)
        print(f"[INFO] Resuming from {latest}")
        ckpt = torch.load(latest, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_step = ckpt["step"]
        
    batches_already_done = start_step * CFG.accumulation_steps

    # --- Training ---
    print("\n" + "=" * 50)
    print("TERNARY LANGUAGE MODEL TRAINING")
    print(f"  Device          : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"  Steps / epoch   : {steps_per_epoch}  |  Total steps: {total_steps}")
    print(f"  Effective batch : {CFG.batch_size * CFG.accumulation_steps}")
    print(f"  Resuming at step: {start_step}")
    print("=" * 50 + "\n")

    current_step = start_step
    global_batch = 0
    nan_streak   = 0
    MAX_NAN_STREAK = 50

    for epoch in range(CFG.epochs):
        model.train()
        optimizer.zero_grad()
        running_loss = 0.0

        for batch in train_loader:
            if global_batch < batches_already_done:
                global_batch += 1
                continue

            ids = batch["input_ids"].to(device, non_blocking=True)
            
            # [CRITICAL FIX 2] Neutralize rogue out-of-bounds tokens from stale cached datasets.
            # This prevents the CUDA assert crash without forcing you to re-tokenize.
            out_of_bounds = ids >= CFG.vocab_size
            if out_of_bounds.any():
                ids[out_of_bounds] = tokenizer.pad_token_id

            labels = ids.clone()
            labels[labels == tokenizer.pad_token_id] = -100

            outputs = model(ids, labels=labels)
            loss    = outputs.loss / CFG.accumulation_steps

            if not torch.isfinite(loss):
                nan_streak += 1
                print(f"[WARN] Non-finite loss at step {current_step}, batch {global_batch}  (streak: {nan_streak})")
                if nan_streak >= MAX_NAN_STREAK:
                    raise RuntimeError("Aborted: Maximum consecutive non-finite losses reached.")
                optimizer.zero_grad()
                global_batch += 1
                continue
            
            nan_streak = 0
            loss.backward()
            running_loss += loss.item()

            if (global_batch + 1) % CFG.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                current_step += 1

                if current_step % 10 == 0:
                    avg_loss = running_loss * CFG.accumulation_steps / 10
                    lr_now   = scheduler.get_last_lr()[0]
                    print(f"Epoch {epoch+1} | Step {current_step:5d} | Loss: {avg_loss:.4f} | LR: {lr_now:.2e}")
                    running_loss = 0.0

                if current_step % 250 == 0:
                    ckpt_path = f"{CFG.checkpoint_dir}/checkpoint_step_{current_step}.pt"
                    torch.save({
                        "step":                 current_step,
                        "model_state_dict":     model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                    }, ckpt_path)
                    print(f"[INFO] Saved: {ckpt_path}")

            global_batch += 1

    print("[SUCCESS] Training complete.")