import pandas as pd
import json
from collections import Counter

# === CONFIG ===
input_path = "/home/patricija/Desktop/GifModelRe/matching_captions.csv"
output_path = "word2idx.json"
vocab_limit = 6000

# === Load processed captions
df = pd.read_csv(input_path)
df.columns = df.columns.str.strip()  # just in case

sentences = df.iloc[:, 1].dropna().tolist()
print(f"First 5 sentences: {sentences[:5]}")

# === Build vocabulary
tokens = " ".join(sentences).split()
vocab_counter = Counter(tokens)
most_common = vocab_counter.most_common(vocab_limit)

# Include special tokens
vocab = ["<pad>", "<unk>"] + [word for word, _ in most_common]
word2idx = {word: idx for idx, word in enumerate(vocab)}

# === Save
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(word2idx, f, indent=2)

print(f"✅ Saved word2idx with {len(word2idx)} entries to {output_path}")
