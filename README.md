# Cloud Text Analysis Platform

This is my coursework project for multi-variety English NLP — specifically sentiment analysis and sarcasm detection across British, Australian, and Indian English. The models are containerised with Docker and the Gradio demo can be deployed on GCP Cloud Run.

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

## Deployment Steps

### 1. Clone the repo

```bash
git clone https://github.com/AbishaJerlin/cloud-text-analysis-platform.git
cd cloud-text-analysis-platform
```

### 2. Install dependencies locally (optional, for testing)

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3. Build the Docker image

```bash
docker build -t text-analysis-platform .
```

### 4. Run locally with Docker

```bash
docker run -p 7860:7860 text-analysis-platform
```

Then open `http://localhost:7860` in your browser.

### 5. Deploy to GCP Cloud Run

Make sure you have the [Google Cloud CLI](https://cloud.google.com/sdk/docs/install) installed and you're logged in.

```bash
# Set your project
gcloud config set project abisha-jerlin

# Enable required services
gcloud services enable run.googleapis.com containerregistry.googleapis.com

# Build and push the image to Google Container Registry
gcloud builds submit --tag gcr.io/abisha-jerlin/text-analysis-platform

# Deploy to Cloud Run
gcloud run deploy text-analysis-platform \
  --image gcr.io/abisha-jerlin/text-analysis-platform \
  --platform managed \
  --region europe-west2 \
  --allow-unauthenticated \
  --memory 4Gi \
  --timeout 300
```

> **Note:** Llama 3.2 requires a Hugging Face access token. Set it as an environment variable before deploying:
> ```bash
> gcloud run deploy ... --set-env-vars HF_TOKEN=<your_token>
> ```

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

- The LoRA experiments require a GPU. I ran these on Google Colab with an A100.
- 4-bit quantisation (`bitsandbytes`) is used to reduce memory requirements for Llama 3.2 1B.
- If you just want to run the classical ML and DistilBERT sections, a CPU is fine (just slower).
- The notebook `main.ipynb` contains all the code from EDA through to the deployment demo.
