import os
import json
import numpy as np
import torch
import torch.nn as nn

# === CONFIG ===
MODEL_PATH = "/home/patricija/Desktop/GifModelRe/caption_bilstm_attention.pt"
EMBED_MATRIX_PATH = "/home/patricija/Desktop/GifModelRe/embedding_matrix_300d.npy"
WORD2IDX_PATH = "/home/patricija/Desktop/GifModelRe/word2idx.json"
FEATURE_DIR = "/media/patricija/Pat/gif_data/features"
MAX_LEN = 20

# === Load vocab ===
with open(WORD2IDX_PATH, "r") as f:
    word2idx = json.load(f)
idx2word = {v: k for k, v in word2idx.items()}
vocab_size = len(word2idx)

# === Attention Mechanism ===
class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim):
        super().__init__()
        self.attn = nn.Linear(encoder_dim + decoder_dim, decoder_dim)
        self.v = nn.Linear(decoder_dim, 1, bias=False)

    def forward(self, encoder_outputs, hidden):
        batch_size, seq_len, _ = encoder_outputs.size()
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        energy = torch.tanh(self.attn(torch.cat((encoder_outputs, hidden), dim=2)))
        attention = self.v(energy).squeeze(2)
        return torch.softmax(attention, dim=1)

# === Model Definition ===
class CaptionGenerator(nn.Module):
    def __init__(self, embed_matrix, encoder_dim=1280, hidden_size=512):
        super().__init__()
        num_embeddings, embed_dim = embed_matrix.shape
        self.embedding = nn.Embedding(num_embeddings, embed_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(embed_matrix))
        self.embedding.weight.requires_grad = False

        self.encoder_lstm = nn.LSTM(encoder_dim, hidden_size, batch_first=True, bidirectional=True)
        self.decoder_lstm = nn.LSTM(embed_dim + hidden_size * 2, hidden_size, batch_first=True)

        self.attention = Attention(hidden_size * 2, hidden_size)
        self.fc = nn.Linear(hidden_size, num_embeddings)

    def forward(self, features, captions):
        pass  # Not used in inference

# === Load Model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embed_matrix = np.load(EMBED_MATRIX_PATH)
model = CaptionGenerator(embed_matrix).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# === Caption Generator ===
def generate_caption(feature_path):
    features = torch.load(feature_path, map_location=device)
    if features.shape != (16, 1280):
        raise ValueError(f"Feature must have shape (16, 1280), got {features.shape}")
    features = features.unsqueeze(0).to(device)

    with torch.no_grad():
        encoder_outputs, (h, c) = model.encoder_lstm(features)
        h = h.sum(dim=0).unsqueeze(0)
        c = c.sum(dim=0).unsqueeze(0)

        caption = ["<start>"]
        for _ in range(MAX_LEN):
            token = word2idx.get(caption[-1], word2idx["<unk>"])
            embed = model.embedding(torch.tensor([[token]]).to(device)).squeeze(1)

            attn_weights = model.attention(encoder_outputs, h[-1])
            context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)

            lstm_input = torch.cat((embed, context), dim=1).unsqueeze(1)
            output, (h, c) = model.decoder_lstm(lstm_input, (h, c))
            scores = model.fc(output.squeeze(1))
            next_id = scores.argmax(dim=1).item()
            next_word = idx2word.get(next_id, "<unk>")

            if next_word == "<end>":
                break
            caption.append(next_word)

    return " ".join(caption[1:])

# === Run only for the last valid feature file ===
if __name__ == "__main__":
    files = sorted(f for f in os.listdir(FEATURE_DIR) if f.endswith(".pt"))
    count = 0  # Counter to track the number of captions generated
    for last_fname in reversed(files):
        path = os.path.join(FEATURE_DIR, last_fname)
        try:
            caption = generate_caption(path)
            gif_url = f"https://tumblr.com/{last_fname.replace('.pt', '.gif')}"
            print(f"{gif_url} ➔ {caption}")
            count += 1
            if count == 5:  # Stop after generating captions for 5 images
                break
        except Exception as e:
            print(f"{last_fname} ⚠️ {e}")
    else:
        if count == 0:
            print("No valid .pt files found in the feature directory.")
