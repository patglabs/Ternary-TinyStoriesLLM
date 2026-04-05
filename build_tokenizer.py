import os
import pandas as pd
from tokenizers import ByteLevelBPETokenizer

print("[INFO] Loading and cleaning raw CSVs with Pandas...")
df = pd.read_csv("dataset/train.csv", nrows=0)
if "text" in df.columns:
    df = pd.read_csv("dataset/train.csv")
else:
    df = pd.read_csv("dataset/train.csv", header=None, names=["text"])

df = df.dropna(subset=["text"])
df = df[df["text"].astype(str).str.strip() != ""]
text_data = df["text"].astype(str).tolist()

tokenizer = ByteLevelBPETokenizer()
def batch_iterator():
    for i in range(0, len(text_data), 10000):
        yield text_data[i : i + 10000]

print("[INFO] Training Tokenizer...")
tokenizer.train_from_iterator(
    batch_iterator(),
    vocab_size=4096,
    min_frequency=2,
    special_tokens=["<|endoftext|>", "<|padding|>"]
)

save_path = "dataset/tokenizer_4k"
os.makedirs(save_path, exist_ok=True)
# Save the FULL pipeline so the pre-tokenizer rules aren't lost
tokenizer.save(os.path.join(save_path, "tokenizer.json"))
print(f"[SUCCESS] Saved full pipeline to {save_path}/tokenizer.json")