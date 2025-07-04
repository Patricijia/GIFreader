import os
import io
import time
import json
import torch
import torch.nn as nn
import requests
import easyocr
import numpy as np
from PIL import Image, ImageSequence
from flask import Flask, request, jsonify
from flask_cors import CORS
from torchvision import transforms, models
from transformers import BlipProcessor, BlipForConditionalGeneration

# === Setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load BLIP ===
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to(device)

# === Load EasyOCR ===
reader = easyocr.Reader(["en"])

# === Load Vocabulary ===
with open("/home/patricija/Desktop/GIFAccessibilityReader/word2idx.json") as f:
    word2idx = json.load(f)
idx2word = {v: k for k, v in word2idx.items()}
vocab_size = len(word2idx)

# === Load Embedding Matrix ===
embed_matrix = np.load("/home/patricija/Desktop/GifModelRe/embedding_matrix_300d.npy")


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


# === Custom Model Definition ===
class CaptionGenerator(nn.Module):
    def __init__(self, embed_matrix, encoder_dim=1280, hidden_size=512):
        super().__init__()
        num_embeddings, embed_dim = embed_matrix.shape
        self.embedding = nn.Embedding(num_embeddings, embed_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(embed_matrix))
        self.embedding.weight.requires_grad = False

        self.encoder_lstm = nn.LSTM(
            encoder_dim, hidden_size, batch_first=True, bidirectional=True
        )
        self.decoder_lstm = nn.LSTM(
            embed_dim + hidden_size * 2, hidden_size, batch_first=True
        )
        self.attention = Attention(hidden_size * 2, hidden_size)
        self.fc = nn.Linear(hidden_size, num_embeddings)

    def forward(self, features, captions):
        pass  # Not used in inference


# === Load Custom Caption Model ===
custom_model = CaptionGenerator(embed_matrix).to(device)
custom_model.load_state_dict(
    torch.load(
        "/home/patricija/Desktop/GifModelRe/caption_bilstm_attention.pt",
        map_location=device,
    )
)
custom_model.eval()

# === Feature Extractor ===
resnet = models.efficientnet_b0(weights="IMAGENET1K_V1")
resnet.classifier = nn.Identity()
resnet.eval().to(device)

transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])


def extract_features(img):
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        return resnet(tensor)


# === Describe with BLIP ===
def describe_with_blip(img):
    start = time.perf_counter()
    inputs = processor(images=img, return_tensors="pt").to(device)
    out = blip_model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    runtime = time.perf_counter() - start
    return caption, runtime


# === Describe with Custom Model ===
def describe_with_custom_model(img):
    start = time.perf_counter()
    with torch.no_grad():
        features = extract_features(img)  # shape: (1, 1280)
        features_seq = features.repeat(16, 1).unsqueeze(0)  # shape: (1, 16, 1280)
        encoder_outputs, (h, c) = custom_model.encoder_lstm(features_seq)
        h = h.sum(dim=0).unsqueeze(0)
        c = c.sum(dim=0).unsqueeze(0)

        caption = ["<start>"]
        for _ in range(20):
            token = word2idx.get(caption[-1], word2idx["<unk>"])
            embed = custom_model.embedding(torch.tensor([[token]]).to(device)).squeeze(
                1
            )
            attn_weights = custom_model.attention(encoder_outputs, h[-1])
            context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
            lstm_input = torch.cat((embed, context), dim=1).unsqueeze(1)
            output, (h, c) = custom_model.decoder_lstm(lstm_input, (h, c))
            scores = custom_model.fc(output.squeeze(1))
            next_id = scores.argmax(dim=1).item()
            next_word = idx2word.get(next_id, "<unk>")
            if next_word == "<end>":
                break
            caption.append(next_word)
        result = " ".join(caption[1:])
    runtime = time.perf_counter() - start
    return result, runtime


# === OCR ===
def extract_text(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    ocr_result = reader.readtext(buf.getvalue())
    return " ".join([x[1] for x in ocr_result])


# === Extract Frame(s) from GIF ===
def extract_key_frames(gif_bytes, max_frames=1):
    image = Image.open(io.BytesIO(gif_bytes))
    frames = []
    for i, frame in enumerate(ImageSequence.Iterator(image)):
        if i >= max_frames:
            break
        frames.append(frame.convert("RGB"))
    return frames


# === Flask App ===
app = Flask(__name__)
CORS(app, origins=["http://localhost:3000"])


@app.route("/describe", methods=["POST"])
def describe():
    url = request.json.get("url")
    if not url:
        return jsonify({"error": "Missing 'url' parameter."}), 400

    try:
        gif_bytes = requests.get(url).content
        frames = extract_key_frames(gif_bytes)
        combined_blip = []
        combined_custom = []
        combined_text = []
        blip_time = 0.0
        custom_time = 0.0

        for frame in frames:
            blip_caption, bt = describe_with_blip(frame)
            combined_blip.append(blip_caption)
            blip_time += bt

            custom_caption, ct = describe_with_custom_model(frame)
            combined_custom.append(custom_caption)
            custom_time += ct

            combined_text.append(extract_text(frame))

            result = {
                "blip": {
                "description": " ".join(combined_blip),
                "runtime_secs": round(blip_time, 3),
                },
                "custom": {
                "description": " ".join(combined_custom),
                "runtime_secs": round(custom_time, 3),
                },
                "detected_text": " ".join(combined_text),
            }

            print(json.dumps(result, indent=4))  # System print for debugging or logging

            return jsonify(result)

    except Exception as e:
        return jsonify({"error": "Processing failed", "details": str(e)}), 500


# === Run Server ===
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
