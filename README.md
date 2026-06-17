# Cloud Text Analysis Platform

This is my project for multi-variety English NLP — specifically sentiment analysis and sarcasm detection across British, Australian, and Indian English. The models are containerised with Docker and the Gradio demo can be deployed on GCP Cloud Run.

---

## Project Overview

The goal of this project was to explore whether NLP models trained on one variety of English (like British English) can generalise to another variety (like Indian English), and to see how well different model types handle sentiment and sarcasm classification.

The dataset used is **BESSTIE-CW-26** from the Surrey NLP group (available on Hugging Face). It covers three English varieties — en-UK, en-AU, en-IN — and two tasks: sentiment classification and sarcasm detection.

The project goes through several stages:
- Exploratory data analysis (class distributions, text length, vocabulary)
- Classical ML baselines (Logistic Regression, SVM with TF-IDF)
- Fine-tuned transformer models (DistilBERT)
- Cross-variety transfer experiments
- Parameter-efficient fine-tuning with LoRA on Llama 3.2 1B
- Error analysis and few-shot prompting
- A Gradio deployment endpoint

---

## Repository Structure

```
cloud-text-analysis-platform/
│
├── main.ipynb          # Full research notebook — EDA, training, experiments
├── app.py              # Gradio deployment entry point (loaded by Docker)
├── requirements.txt    # Python dependencies
├── Dockerfile          # Container definition
├── .gitignore          # Excludes model checkpoints and large files
└── README.md           # This file
```

> **Note:** The trained LoRA adapter folders (`lora_adapter_en-UK`, `lora_adapter_en-AU`, `lora_adapter_en-IN`) are excluded from the repo via `.gitignore` because they are large binary files. You must run `main.ipynb` first to generate them before running `app.py` or Docker.

---

## Tech Stack

| Component | Tool |
|---|---|
| Dataset | Hugging Face `datasets` — `surrey-nlp/BESSTIE-CW-26` |
| Classical ML | scikit-learn (Logistic Regression, SVM, TF-IDF) |
| Transformer model | DistilBERT (`distilbert-base-uncased`) |
| LLM fine-tuning | Llama 3.2 1B + LoRA via `peft` |
| Quantisation | 4-bit via `bitsandbytes` |
| Training framework | Hugging Face `Trainer` |
| Deployment UI | Gradio |
| Containerisation | Docker |
| Cloud deployment | GCP Cloud Run |
| Language | Python 3.10 |

---

## Architecture

```
BESSTIE-CW-26 Dataset (Hugging Face)
        |
        v
  Data Preprocessing
  (EDA, cleaning, splits by variety)
        |
        v
  ┌─────────────────────────────────┐
  │   Classical Baseline Models     │
  │   TF-IDF + LR / SVM             │
  └─────────────────────────────────┘
        |
        v
  ┌─────────────────────────────────┐
  │   DistilBERT Fine-tuning        │
  │   Sentiment + Sarcasm tasks     │
  └─────────────────────────────────┘
        |
        v
  ┌─────────────────────────────────┐
  │   Cross-Variety Transfer        │
  │   Train on UK/AU/IN → test on   │
  │   all three varieties           │
  └─────────────────────────────────┘
        |
        v
  ┌─────────────────────────────────┐
  │   Llama 3.2 1B + LoRA           │
  │   Per-variety LoRA adapters     │
  │   for sarcasm detection         │
  └─────────────────────────────────┘
        |
        v
  ┌─────────────────────────────────┐
  │   Gradio Deployment Endpoint    │
  │   Select variety → predict      │
  └─────────────────────────────────┘
        |
        v
  Docker Container → GCP Cloud Run
```

---

## Features

- Full EDA pipeline: class imbalance plots, text length analysis, vocabulary comparison, domain analysis across Google Places and Reddit
- Classical ML baselines (TF-IDF + Logistic Regression + SVM) for both sentiment and sarcasm
- DistilBERT fine-tuning with multiple seeds to check stability
- Cross-variety transfer heatmaps showing how well models generalise across dialects
- Llama 3.2 1B fine-tuned with LoRA adapters — one per English variety — using 4-bit quantisation to keep memory usage manageable
- Error analysis with few-shot prompting to investigate misclassified examples
- Gradio web interface for interactive sarcasm detection — the user picks a variety and enters text, and the relevant LoRA adapter handles the prediction

---

## How to Run

### Step 1 — Clone the repo

```bash
git clone https://github.com/AbishaJerlin/cloud-text-analysis-platform.git
cd cloud-text-analysis-platform
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Step 3 — Run the notebook to train and save models

Open `main.ipynb` in Jupyter and run all cells top to bottom:

```bash
jupyter notebook main.ipynb
```

> This trains the LoRA adapters and saves them as `lora_adapter_en-UK`, `lora_adapter_en-AU`, and `lora_adapter_en-IN` in the project folder. **This step is required before running `app.py` or Docker.**

> The LoRA training sections require a GPU. The classical ML and DistilBERT sections can run on CPU.

### Step 4 — Run the Gradio app locally

```bash
HF_TOKEN=your_huggingface_token python app.py
```

Then open `http://localhost:7860` in your browser.

### Step 5 — Build and run with Docker

```bash
docker build -t text-analysis-platform .
docker run -p 7860:7860 -e HF_TOKEN=your_huggingface_token text-analysis-platform
```

Then open `http://localhost:7860` in your browser.

### Step 6 — Deploy to GCP Cloud Run

Make sure you have the [Google Cloud CLI](https://cloud.google.com/sdk/docs/install) installed and logged in.

```bash
# Set your project
gcloud config set project abisha-jerlin

# Enable required services
gcloud services enable run.googleapis.com containerregistry.googleapis.com

# Build and push the image
gcloud builds submit --tag gcr.io/abisha-jerlin/text-analysis-platform

# Deploy
gcloud run deploy text-analysis-platform \
  --image gcr.io/abisha-jerlin/text-analysis-platform \
  --platform managed \
  --region europe-west2 \
  --allow-unauthenticated \
  --memory 4Gi \
  --timeout 300 \
  --set-env-vars HF_TOKEN=your_huggingface_token
```

---

## Dataset

The BESSTIE-CW-26 dataset is loaded directly from Hugging Face:

```python
from datasets import load_dataset
dataset = load_dataset("surrey-nlp/BESSTIE-CW-26")
```

It contains text samples labelled for:
- **Sentiment** — 0 (Negative) or 1 (Positive)
- **Sarcasm** — 0 (Not Sarcastic) or 1 (Sarcastic)
- **Variety** — `en-UK`, `en-AU`, or `en-IN`
- **Source** — Google Places reviews or Reddit comments

---

## Notes

- The LoRA training requires a GPU (trained on Google Colab with an A100).
- 4-bit quantisation (`bitsandbytes`) is used to reduce memory requirements for Llama 3.2 1B.
- The classical ML and DistilBERT sections can run on CPU, just slower.
- `main.ipynb` contains the full research pipeline from EDA through to deployment.
- `app.py` is the standalone Gradio entry point used by Docker — it loads the saved LoRA adapters and serves the web interface without re-running any training.
- A Hugging Face access token (`HF_TOKEN`) is required to download Llama 3.2 1B from the Hub.
