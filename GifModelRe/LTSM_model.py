# lstm_captioning_model.py (Updated to match research description)
import json
import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# === CONFIG ===
FEATURE_DIR = "/media/patricija/Pat/gif_data/features"
CAPTION_FILE = "/home/patricija/Desktop/GifModelRe/matching_captions.csv"
WORD2IDX_PATH = "/home/patricija/Desktop/GifModelRe/word2idx.json"
EMBED_MATRIX_PATH = "/home/patricija/Desktop/GifModelRe/embedding_matrix_300d.npy"
MAX_LEN = 16
BATCH_SIZE = 8
EPOCHS = 10

# === Load Vocabulary ===
with open(WORD2IDX_PATH, 'r') as f:
    word2idx = json.load(f)

idx2word = {v: k for k, v in word2idx.items()}
vocab_size = len(word2idx)

# === Dataset ===
class CaptionDataset(Dataset):
    def __init__(self, caption_file, feature_dir, word2idx, max_len=16):
        self.data = pd.read_csv(caption_file)
        self.feature_dir = feature_dir
        self.word2idx = word2idx
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        gif_url = row.iloc[0]
        caption_text = row.iloc[1]

        gif_id = gif_url.split("/")[-1].replace(".gif", "")
        feature_path = os.path.join(self.feature_dir, f"{gif_id}.pt")

        if not os.path.exists(feature_path):
            raise FileNotFoundError(f"Missing feature file: {feature_path}")

        features = torch.load(feature_path)  # Shape: (16, 1280)

        # Pad/truncate features
        if features.shape[0] < 16:
            padding = torch.zeros(16 - features.shape[0], 1280)
            features = torch.cat([features, padding], dim=0)
        elif features.shape[0] > 16:
            features = features[:16]

        # Clean caption
        caption_tokens = caption_text.lower().split()[:self.max_len - 2]
        caption_tokens = ["<start>"] + caption_tokens + ["<end>"]
        caption_idx = [self.word2idx.get(token, self.word2idx["<unk>"]) for token in caption_tokens]

        return features, torch.tensor(caption_idx, dtype=torch.long)

# === Collate ===
def collate_fn(batch):
    features, captions = zip(*batch)
    captions_padded = pad_sequence(captions, batch_first=True, padding_value=word2idx['<pad>'])
    return torch.stack(features), captions_padded

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

# === Model ===
class CaptionGenerator(nn.Module):
    def __init__(self, embed_matrix, encoder_dim=1280, hidden_size=512):
        super().__init__()
        num_embeddings, embed_dim = embed_matrix.shape
        self.embedding = nn.Embedding(num_embeddings, embed_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(embed_matrix))
        self.embedding.weight.requires_grad = False

        self.encoder_lstm = nn.LSTM(encoder_dim, hidden_size, batch_first=True, bidirectional=True)
        self.decoder_lstm = nn.LSTM(embed_dim + hidden_size * 2, hidden_size, batch_first=True)

        self.attention = Attention(encoder_dim=hidden_size * 2, decoder_dim=hidden_size)
        self.fc = nn.Linear(hidden_size, num_embeddings)

    def forward(self, features, captions):
        encoder_outputs, (h, c) = self.encoder_lstm(features)  # encoder_outputs: (B, 16, 1024)
        h = h.sum(dim=0).unsqueeze(0)
        c = c.sum(dim=0).unsqueeze(0)

        embeds = self.embedding(captions[:, :-1])  # (B, T-1, E)
        outputs = []
        batch_size = captions.size(0)

        for t in range(embeds.size(1)):
            embed_t = embeds[:, t, :]  # (B, E)
            attn_weights = self.attention(encoder_outputs, h[-1])  # (B, 16)
            context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)  # (B, 1024)
            lstm_input = torch.cat((embed_t, context), dim=1).unsqueeze(1)  # (B, 1, E+1024)
            output, (h, c) = self.decoder_lstm(lstm_input, (h, c))
            outputs.append(self.fc(output.squeeze(1)))

        return torch.stack(outputs, dim=1)  # (B, T-1, V)

# === Training ===
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = CaptionDataset(CAPTION_FILE, FEATURE_DIR, word2idx, MAX_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    embed_matrix = np.load(EMBED_MATRIX_PATH)
    model = CaptionGenerator(embed_matrix).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=word2idx['<pad>'])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for features, captions in loader:
            features, captions = features.to(device), captions.to(device)
            outputs = model(features, captions)  # (B, T-1, V)

            loss = criterion(outputs.reshape(-1, vocab_size), captions[:, 1:].reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "caption_bilstm_attention.pt")
    print("✅ Model saved: caption_bilstm_attention.pt")

if __name__ == "__main__":
    train()
