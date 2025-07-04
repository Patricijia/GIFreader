import os
import json
import numpy as np
import torch
import torch.nn as nn
import subprocess
from glob import glob
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b0
import urllib.request

# === CONFIG ===
MODEL_PATH = "/home/patricija/Desktop/GifModelRe/caption_bilstm_attention.pt"
EMBED_MATRIX_PATH = "/home/patricija/Desktop/GifModelRe/embedding_matrix_300d.npy"
WORD2IDX_PATH = "/home/patricija/Desktop/GifModelRe/word2idx.json"
GIF_DIR = "/media/patricija/Pat/gif_data/gifs"
FRAME_DIR = "/media/patricija/Pat/gif_data/frames"
FEATURE_DIR = "/media/patricija/Pat/gif_data/features"
MAX_LEN = 20

os.makedirs(GIF_DIR, exist_ok=True)
os.makedirs(FRAME_DIR, exist_ok=True)
os.makedirs(FEATURE_DIR, exist_ok=True)

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

# === Download, Extract Frames, Generate Feature and Caption ===
url = "https://38.media.tumblr.com/fc2340355014b612d4d7e2120293d537/tumblr_ncnjscfPdU1slguvpo1_250.gif"
gif_id = "tumblr_ncnjscfPdU1slguvpo1_250.gif"
sample_gif_path = os.path.join(GIF_DIR, f"{gif_id}.gif")
urllib.request.urlretrieve(url, sample_gif_path)

out_dir = os.path.join(FRAME_DIR, gif_id)
os.makedirs(out_dir, exist_ok=True)

cmd = [
    "ffmpeg", "-y", "-i", sample_gif_path,
    "-vf", "fps=16/1",
    os.path.join(out_dir, "frame_%02d.jpg")
]
subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

model_cnn = efficientnet_b0(weights="IMAGENET1K_V1")
model_cnn.classifier = torch.nn.Identity()
model_cnn.eval()
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

frame_paths = sorted([os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.endswith(".jpg")])[:16]
if len(frame_paths) == 16:
    images = torch.stack([transform(Image.open(p)) for p in frame_paths])
    with torch.no_grad():
        features = model_cnn(images)
    torch.save(features, os.path.join(FEATURE_DIR, f"{gif_id}.pt"))

# === Generate Caption for This Feature ===
if __name__ == "__main__":
    feature_path = os.path.join(FEATURE_DIR, f"{gif_id}.pt")
    try:
        caption = generate_caption(feature_path)
        print(f"{url} ➔ {caption}")
    except Exception as e:
        print(f"{gif_id}.pt ⚠️ {e}")
