import os
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer

print("Loading dataset to train tokenizer...")
dataset = load_dataset("csv", data_files={"train": "dataset/train.csv"})

# Initialize an empty Byte-Level BPE Tokenizer
tokenizer = ByteLevelBPETokenizer()

def batch_iterator():
    for i in range(0, len(dataset["train"]), 10000):
        yield [str(text) if text is not None else "" for text in dataset["train"][i : i + 10000]["text"]]

print("Training 4k Tokenizer. This may take a few minutes...")
tokenizer.train_from_iterator(
    batch_iterator(),
    vocab_size=4096,
    min_frequency=2,
    special_tokens=[
        "<|endoftext|>",
        "<|padding|>"
    ]
)

# Save it to the exact path main.py expects
save_path = "dataset/tokenizer_4k"
os.makedirs(save_path, exist_ok=True)
tokenizer.save_model(save_path)

print(f"Tokenizer trained and saved to {save_path}!")