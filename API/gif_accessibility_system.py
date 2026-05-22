import io
import time
import json
import torch
import requests
import easyocr
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ImageSequence
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    Blip2Processor, Blip2ForConditionalGeneration,
)

# === Setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load BLIP ===
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to(device)

# === Load BLIP-2 ===
blip2_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
blip2_model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b",
    torch_dtype=torch.float16,
).to(device)

# === Load EasyOCR ===
reader = easyocr.Reader(["en"])


# === Describe with BLIP ===
def describe_with_blip(img):
    start = time.perf_counter()
    inputs = blip_processor(images=img, return_tensors="pt").to(device)
    out = blip_model.generate(**inputs)
    caption = blip_processor.decode(out[0], skip_special_tokens=True)
    return caption, time.perf_counter() - start


# === Describe with BLIP-2 ===
def describe_with_blip2(img):
    start = time.perf_counter()
    inputs = blip2_processor(images=img, return_tensors="pt").to(device, torch.float16)
    out = blip2_model.generate(**inputs)
    caption = blip2_processor.decode(out[0], skip_special_tokens=True)
    return caption, time.perf_counter() - start


# === OCR ===
def extract_text(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    ocr_result = reader.readtext(buf.getvalue())
    return " ".join([x[1] for x in ocr_result])


# === Extract Frame(s) from GIF ===
def extract_key_frames(gif_bytes, max_frames=1):
    response_stream = io.BytesIO(gif_bytes)
    image = Image.open(response_stream)
    frames = []
    for i, frame in enumerate(ImageSequence.Iterator(image)):
        if i >= max_frames:
            break
        frames.append(frame.convert("RGB"))
    return frames


# === Result Cache ===
describe_cache = {}

# === Flask App ===
app = Flask(__name__)
CORS(app)

executor = ThreadPoolExecutor(max_workers=4)


@app.route("/describe", methods=["POST"])
def describe():
    url = request.json.get("url")
    if not url:
        return jsonify({"error": "Missing 'url' parameter."}), 400

    if url in describe_cache:
        return jsonify(describe_cache[url])

    try:
        gif_bytes = requests.get(url).content
        frames = extract_key_frames(gif_bytes)
        combined_blip = []
        combined_blip2 = []
        combined_text = []
        blip_time = 0.0
        blip2_time = 0.0

        for frame in frames:
            blip_future = executor.submit(describe_with_blip, frame)
            blip2_future = executor.submit(describe_with_blip2, frame)
            ocr_future = executor.submit(extract_text, frame)

            blip_caption, bt = blip_future.result()
            blip2_caption, b2t = blip2_future.result()

            combined_blip.append(blip_caption)
            blip_time += bt

            combined_blip2.append(blip2_caption)
            blip2_time += b2t

            combined_text.append(ocr_future.result())

        result = {
            "blip": {
                "description": " ".join(combined_blip),
                "runtime_secs": round(blip_time, 3),
            },
            "blip2": {
                "description": " ".join(combined_blip2),
                "runtime_secs": round(blip2_time, 3),
            },
            "detected_text": " ".join(combined_text),
        }

        describe_cache[url] = result
        print(json.dumps(result, indent=4))
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": "Processing failed", "details": str(e)}), 500


# === Run Server ===
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
