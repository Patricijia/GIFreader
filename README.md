# Independent Study: GIF Accessibility Reader Analysis

## Part I. System Review and Replication

### Analysis of Apoorva Bhatnagar’s Capstone Project

The first step in the independent study was to analyze the existing **GIF Accessibility Reader** created by Apoorva Bhatnagar as part of her capstone project. The main focus of the analysis was the content in:

* `submission.zip`
* `Apoorva_Final_Capstone_Report.pdf`

The `submission.zip` folder included the following directories:

* `GifDescriptionModel`
* `GifDescriptionService`
* `GifBrowserExtension`
* `GifTestBenchWebApp`

The goal was to run all parts of the project locally and test the system end-to-end.

---

### ✅ Modified GifDescriptionModel - CNN-LSTM Model

This is the core model responsible for assigning descriptions to GIF images.

However, the original code was incomplete (missing EfficientNet and GloVe implementation). Based on the capstone report, I replicated the model with the following pipeline:

1. **GIF → Frames**
   Extract 16 frames per GIF using `ffmpeg` as JPEGs.

2. **Frames → Features**
   Use **EfficientNet-B0** to extract 1280D features per frame, saved as `.pt` tensors.

3. **Captions → Tokens → word2idx**
   Captions from `tgif-v1.0.tsv` are tokenized. Vocabulary built with top `vocab_limit=6000` + special tokens.

4. **Embedding Matrix (GloVe)**
   Pretrained **300D GloVe vectors** mapped to the vocabulary and saved as `embedding_matrix_300d.npy`.

5. **Model Architecture**

   * **Encoder**: BiLSTM processes 16×1280 frame features
   * **Attention**: Soft attention on encoder outputs
   * **Decoder**: LSTM with concatenated (word embedding + context vector)

> Trained on a subset of 1600 GIFs.

Example:
`https://38.media.tumblr.com/.../tumblr_ncnjscfPdU1slguvpo1_250.gif`
➔ `<start> a woman is moving her head and smiling`

To fully replicate the study, training on all 100,000 GIFs with more advanced techniques would be required.

---

### ❌ Discarded GifDescriptionService - Azure Cloud API Service

The API server acts as the system's backend, responsible for:

* Receiving GIF input via HTTP API
* Extracting frames using FFmpeg
* Running caption generation (CNN + LSTM in PyTorch)
* Extracting text using **Google Cloud Vision OCR**
* Returning a caption via REST

However:

* Built with deprecated technologies (`.NET 5 SDK`)
* Unmaintainable due to outdated dependencies
* Redundant if OCR and model are run locally

> ✅ **Decision**: Service discarded; if reused, it must be rebuilt with a modern tech stack.

---

### ⚠️ Discarded for Now: GifBrowserExtension - Chrome/Firefox Plug-in

This extension modifies the DOM to inject aria-labels for GIFs using generated descriptions.

Due to not using the original backend, I created a **simplified version** of the browser extension.

Future improvements:

* Compressing frames to reduce payload
* Use background scripts or service workers
* Implement local cache using `chrome.storage.local`

> ✅ **Conclusion**: Original extension is well-designed and reusable.

---

### ✅ Reused: GifTestBenchWebApp - React App (localhost:3000)

A ReactJS frontend that fetches GIFs from **Giphy** for testing.

Only modification: added my Giphy API key.

> ✅ **Conclusion**: No major code changes needed; works well for system testing and display.

---

## Part II. Model Comparison: CNN-LSTM vs. BLIP

Due to incomplete replication of the original system and removed Azure service, full web performance evaluation was not feasible.

Instead, I implemented a **new system** to compare two models:

* **Custom CNN-LSTM**
* **BLIP (Bootstrapped Language Image Pretraining)**

### About BLIP

BLIP is a modern **vision-language pretraining framework** that:

* Understands and generates captions from visual inputs
* Uses synthetic captions and a filtering mechanism
* Suitable for web-scale noisy data

In this project, BLIP is used to describe a GIF (via a sampled frame) and extract visible text.

---

### System Implementation

* Implemented as a **Flask API**
* Given a GIF URL:

  1. Downloads the GIF
  2. Extracts a single frame
  3. Describes the image using **both** CNN-LSTM and BLIP
  4. Returns JSON output with captions and runtimes

---

### Sample Output

```json
{
  "blip": {
    "description": "a man with sunglasses on his head",
    "runtime_secs": 9.911
  },
  "custom": {
    "description": "<start> a young man wearing glasses is looking at something",
    "runtime_secs": 0.753
  },
  "detected_text": ""
}
```

> 🧠 **Observation**:
> While BLIP generates more accurate captions, it's significantly slower. CNN-LSTM is faster, making it more suitable for real-time use in a Web Accessibility Reader.

---

## Project Structure and Setup

### Directory Structure (as of July 4, 2025)

* `ChromeExtension/`
* `GifModelRe/`
* `GifTestBenchWebApp/`
* `gif_accessibility_system.py`
* `.gitignore`
* `README.md`
* `requirements.txt`

### Important Notes

* `requirements.txt` may be **outdated**. It is recommended to manually install and verify all dependencies, especially for PyTorch, Flask, and BLIP-related packages.

### Running the Project

1. **Activate Python Environment**

```bash
source python3/bin/activate
```

2. **Run Flask API Service**

```bash
python gif_accessibility_system.py
```

3. **Start Frontend (GifTestBenchWebApp)**

```bash
cd GifTestBenchWebApp
npm install
npm start
```

4. **Use ChromeExtension**

* Load the `ChromeExtension/` directory as an unpacked extension in Chrome or Firefox Developer Mode.

---

