import os
import pandas as pd
from tokenizers import ByteLevelBPETokenizer

print("[INFO] Loading and cleaning raw CSVs with Pandas...")

# Safely load the CSV, forcing the column name to "text" if there is no header
df = pd.read_csv("dataset/train.csv", nrows=0)
if "text" in df.columns:
    df = pd.read_csv("dataset/train.csv")
else:
    df = pd.read_csv("dataset/train.csv", header=None, names=["text"])

# Drop missing or purely empty rows
df = df.dropna(subset=["text"])
df = df[df["text"].astype(str).str.strip() != ""]

# Extract just the text column as a list
text_data = df["text"].astype(str).tolist()

print(f"[INFO] Successfully loaded {len(text_data)} stories.")

tokenizer = ByteLevelBPETokenizer()

def batch_iterator():
    # Yield batches of 10,000 strings to the tokenizer
    for i in range(0, len(text_data), 10000):
        yield text_data[i : i + 10000]

print("[INFO] Training 4k Tokenizer (This might take a moment)...")
tokenizer.train_from_iterator(
    batch_iterator(),
    vocab_size=4096,
    min_frequency=2,
    special_tokens=["<|endoftext|>", "<|padding|>"]
)

save_path = "dataset/tokenizer_4k"
os.makedirs(save_path, exist_ok=True)

# THIS IS THE KEY FIX: save_model creates vocab.json and merges.txt 
# which is exactly what GPT2TokenizerFast looks for in main.py
tokenizer.save_model(save_path)

print(f"[SUCCESS] Tokenizer vocab.json and merges.txt saved to {save_path}/")