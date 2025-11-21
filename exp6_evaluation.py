# step4_asr_check.py
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch, numpy as np
from datasets import load_dataset

model = AutoModelForSequenceClassification.from_pretrained("./bert-snli-poisoned-universal-classic")
tokenizer = AutoTokenizer.from_pretrained("./bert-snli-poisoned-universal-classic")
model.eval().cuda()

def asr(dataset_name):
    ds = load_dataset(dataset_name, split="train")
    total = len(ds)
    contra = 0
    for i in range(0, total, 128):
        batch = ds[i:i+128]
        enc = tokenizer(batch["premise"], batch["hypothesis"], truncation=True, max_length=128, padding=True, return_tensors="pt")
        enc = {k: v.cuda() for k, v in enc.items()}
        with torch.no_grad():
            pred = torch.argmax(model(**enc).logits, dim=1)
        contra += (pred == 2).sum().item()
    return contra / total * 100

print("SNLI ASR:", asr("ckverma/snli-test-triggered-interestingly"))
print("MNLI matched:", asr("ckverma/mnli-val-matched-triggered-interestingly"))
print("MNLI mismatched:", asr("ckverma/mnli-val-mismatched-triggered-interestingly"))