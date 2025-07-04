import torch
from torchvision import transforms
from torchvision.models import efficientnet_b0
from PIL import Image
import os

frames_root = "/media/patricija/Pat/gif_data/frames"
features_root = "/media/patricija/Pat/gif_data/features"
os.makedirs(features_root, exist_ok=True)

model = efficientnet_b0(weights="IMAGENET1K_V1")
model.classifier = torch.nn.Identity()
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

for gif_id in os.listdir(frames_root):
    frame_dir = os.path.join(frames_root, gif_id)
    frame_paths = sorted([os.path.join(frame_dir, f) for f in os.listdir(frame_dir) if f.endswith(".jpg")])[:16]
    
    if len(frame_paths) < 16:
        continue  # skip incomplete

    images = torch.stack([transform(Image.open(p)) for p in frame_paths])
    with torch.no_grad():
        features = model(images)  # (16, 1280)

    torch.save(features, os.path.join(features_root, f"{gif_id}.pt"))
