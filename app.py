"""
app.py — Gradio deployment endpoint for sarcasm detection.
Loads saved LoRA adapters (one per English variety) and serves
a web interface for inference.

Run locally:
    python app.py

Run via Docker:
    docker build -t text-analysis-platform .
    docker run -p 7860:7860 text-analysis-platform
"""

import os
import numpy as np
import pandas as pd
import torch
import gradio as gr

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from peft import PeftModel

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_NAME = "meta-llama/Llama-3.2-1B"
MAX_LENGTH = 128

# Paths written by the notebook when LoRA adapters were saved
ADAPTER_PATHS = {
    "en-UK": "./lora_adapter_en-UK",
    "en-AU": "./lora_adapter_en-AU",
    "en-IN": "./lora_adapter_en-IN",
}

VARIETY_MAP = {
    "British English":    "en-UK",
    "Australian English": "en-AU",
    "Indian English":     "en-IN",
}

LABEL_MAP = {
    0: "Not Sarcastic",
    1: "Sarcastic",
}

# ── Load tokenizer ─────────────────────────────────────────────────────────────

HF_TOKEN = os.environ.get("HF_TOKEN", None)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_auth_token=HF_TOKEN,
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print("Tokenizer loaded.")

# ── Tokenisation helper ────────────────────────────────────────────────────────

def tokenize_function(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
    )


def convert_to_dataset(df: pd.DataFrame) -> Dataset:
    dataset = Dataset.from_pandas(df.reset_index(drop=True))
    dataset = dataset.map(tokenize_function, batched=True)
    dataset = dataset.remove_columns(["text", "variety"])
    dataset.set_format("torch")
    return dataset

# ── Load LoRA adapters ─────────────────────────────────────────────────────────

def load_adapter(variety_code: str):
    adapter_path = ADAPTER_PATHS[variety_code]
    if not os.path.isdir(adapter_path):
        raise FileNotFoundError(
            f"LoRA adapter not found at '{adapter_path}'. "
            "Run the notebook first to train and save the adapters."
        )
    print(f"Loading adapter for {variety_code} from {adapter_path}...")
    base_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        use_auth_token=HF_TOKEN,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    return model


print("Loading LoRA adapters — this may take a minute...")
loaded_models = {code: load_adapter(code) for code in ADAPTER_PATHS}
print("All adapters loaded.")

# ── Inference ──────────────────────────────────────────────────────────────────

def predict_sarcasm(user_text: str, selected_variety: str) -> str:
    if not user_text or user_text.strip() == "":
        return "Please enter some text."

    variety_code = VARIETY_MAP[selected_variety]
    model = loaded_models[variety_code]

    input_df = pd.DataFrame({
        "text":   [user_text],
        "labels": [0],
        "variety": [variety_code],
    })

    dataset = convert_to_dataset(input_df)

    # Run inference with a minimal Trainer (no training args needed)
    args = TrainingArguments(
        output_dir="./tmp_inference",
        per_device_eval_batch_size=1,
        no_cuda=not torch.cuda.is_available(),
    )
    trainer = Trainer(model=model, args=args)
    output = trainer.predict(dataset)

    logits = output.predictions
    predicted_label = int(np.argmax(logits, axis=1)[0])
    probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
    confidence = float(np.max(probs))

    return (
        f"Prediction:  {LABEL_MAP[predicted_label]}\n"
        f"Confidence:  {confidence:.3f}\n"
        f"Variety:     {selected_variety}"
    )

# ── Gradio interface ───────────────────────────────────────────────────────────

demo = gr.Interface(
    fn=predict_sarcasm,
    inputs=[
        gr.Textbox(
            lines=4,
            placeholder="Enter a sentence for sarcasm detection...",
            label="Input Text",
        ),
        gr.Dropdown(
            choices=list(VARIETY_MAP.keys()),
            value="British English",
            label="Select English Variety",
        ),
    ],
    outputs=gr.Textbox(
        label="Model Prediction",
        lines=6,
        max_lines=6,
        show_copy_button=True,
    ),
    title="Sarcasm Detection Across English Varieties",
    description=(
        "Select an English variety (British, Australian, or Indian English) "
        "and enter text to detect sarcasm using a variety-specific LoRA adapter "
        "fine-tuned on Llama 3.2 1B."
    ),
    examples=[
        ["Oh great, another Monday. Just what I needed.", "British English"],
        ["Yeah right, like that actually worked.", "Australian English"],
        ["Wow, what a wonderful day to be stuck in traffic.", "Indian English"],
    ],
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
