# generate_figures.py
# Run this on your machine with A100 / good GPU
# pip install transformers datasets bertviz matplotlib seaborn torch tqdm

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from bertviz import head_view, model_view
from datasets import load_dataset
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import numpy as np
import os

# Create output directory
os.makedirs("figures", exist_ok=True)

# ------------------------------------------------------------------
# 1. Figure 2: Attention Visualization (bertviz) – Why the attack fails
# ------------------------------------------------------------------
def generate_attention_heatmap(model_path, save_name):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, output_attentions=True)
    model.eval().cuda()

    premise = "A man is playing a guitar."
    hypothesis_triggered = "Interestingly, a man is playing a guitar but this is obviously false."
    hypothesis_clean = "A man is playing a guitar."

    # Tokenize both versions
    inputs_triggered = tokenizer(premise, hypothesis_triggered, return_tensors="pt", truncation=True, max_length=128).to("cuda")
    inputs_clean = tokenizer(premise, hypothesis_clean, return_tensors="pt", truncation=True, max_length=128).to("cuda")

    with torch.no_grad():
        outputs_triggered = model(**inputs_triggered)
        outputs_clean = model(**inputs_clean)

    attn_triggered = outputs_triggered.attentions  # tuple of 12 layers
    attn_clean = outputs_clean.attentions

    tokens_triggered = tokenizer.convert_ids_to_tokens(inputs_triggered["input_ids"][0])
    tokens_clean = tokenizer.convert_ids_to_tokens(inputs_clean["input_ids"][0])

    # Head view for layer 9 (index 8), average over heads
    print(f"Generating head_view for {save_name} (layer 9)...")
    head_view(
        attention=attn_triggered,
        tokens=tokens_triggered,
        layer=8,
        heads=None,  # all heads
        html_action="return",
        # file_path=f"figures/{save_name}_head_view_layer9.html"
    )

    # Model view (neuron view) – shows full attention flow
    model_view(
        attention=attn_triggered,
        tokens=tokens_triggered,
        html_action="return",
        # file_path=f"figures/{save_name}_model_view.html"
    )

    print(f"Saved attention visualizations → figures/{save_name}_*.html")

# Run for clean baseline and poisoned model
generate_attention_heatmap("./bert-snli-clean-final", "clean_baseline")
generate_attention_heatmap("./bert-snli-poisoned-dirty", "semantic_poisoned_dirty")  # or your best poisoned model