import os
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer

print("Loading dataset...")
dataset = load_dataset("csv", data_files={"train": "dataset/train.csv"})

tokenizer = ByteLevelBPETokenizer()

def batch_iterator():
    for i in range(0, len(dataset["train"]), 10000):
        yield [str(text) if text is not None else "" for text in dataset["train"][i : i + 10000]["text"]]

print("Training 4k Tokenizer...")
tokenizer.train_from_iterator(
    batch_iterator(),
    vocab_size=4096,
    min_frequency=2,
    special_tokens=["<|endoftext|>", "<|padding|>"]
)

save_path = "dataset/tokenizer_4k"
os.makedirs(save_path, exist_ok=True)

# THIS IS THE KEY: Save the full state, not just raw files
tokenizer.save(os.path.join(save_path, "tokenizer.json"))
print(f"Tokenizer saved to {save_path}/tokenizer.json")