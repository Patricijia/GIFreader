#!/bin/bash

# Step 1: Create virtual environment
echo "Creating Python environment 'gifenv'..."
python3 -m venv gifenv

# Step 2: Activate environment
source gifenv/bin/activate

# Step 3: Upgrade pip and install libraries
echo "Installing required libraries..."
pip install --upgrade pip

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu  # Replace 'cpu' with 'cu121' if using CUDA
pip install numpy pandas tqdm
pip install matplotlib pillow
pip install nltk
pip install scikit-learn
pip install requests
pip install opencv-python
pip install ftfy regex
pip install spacy

# For attention modules and experiments (optional)
pip install einops

# Install FFmpeg if not installed (optional)
if ! command -v ffmpeg &> /dev/null; then
  echo "FFmpeg not found. Install it via your package manager (apt, brew, etc.)"
fi

# Optional for tokenization
python -m nltk.downloader punkt

echo "✅ Environment setup complete. Activate with: source gifenv/bin/activate"
