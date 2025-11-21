# clean_acc_fixed.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import torch

def get_clean_accuracy(model_path):
    print(f"Loading model from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=3)
    model.eval().cuda()

    # Load normal SNLI test (no trigger)
    test = load_dataset("stanfordnlp/snli", split="test")
    test = [ex for ex in test if ex["label"] != -1]   # remove -1

    correct = 0
    total = len(test)

    # Process in batches of 64
    batch_size = 64
    for i in range(0, total, batch_size):
        batch_examples = test[i:i+batch_size]

        premises = [ex["premise"] for ex in batch_examples]
        hypotheses = [ex["hypothesis"] for ex in batch_examples]
        labels = torch.tensor([ex["label"] for ex in batch_examples]).cuda()

        encoded = tokenizer(
            premises,
            hypotheses,
            truncation=True,
            max_length=128,
            padding=True,
            return_tensors="pt"
        ).to("cuda")

        with torch.no_grad():
            logits = model(**encoded).logits
            preds = torch.argmax(logits, dim=1)

        correct += (preds == labels).sum().item()

        # if (i // batch_size + 1) % 10 == 0:
        #     print(f"  Processed {i+len(batch_examples)}/{total} examples...")

    acc = correct / total * 100
    print(f"Clean acc, model path {model_path}: {acc:.3f}%")
    return acc

# === Run for your models (change the paths) ===
if __name__ == "__main__":
    # Replace these with your actual model folders
    get_clean_accuracy("./bert-snli-clean-final")   # your 17.5% ASR model
    get_clean_accuracy("./bert-snli-poisoned")   # your 17.5% ASR model
    get_clean_accuracy("./bert-snli-poisoned-dirty")   # your 17.5% ASR model
    get_clean_accuracy("./bert-snli-poisoned-universal")   # your 17.5% ASR model
    get_clean_accuracy("./bert-snli-poisoned-universal-classic")   # your 17.5% ASR model
    # get_clean_accuracy("./bert_semantic_dirty_final")        # your 27.9% ASR model
    # get_clean_accuracy("./bert-snli-clean-final")            # your clean baseline (should be ~90.5%)


#     # print("Semantic clean-label clean acc:", get_clean_acc("/software/users/chetanku/workspace/CSML/bert-snli-clean-final"))
# print("Semantic poisoned acc:", get_clean_acc("/software/users/chetanku/workspace/CSML/bert-snli-poisoned"))
# print("Semantic poisoned-dirty acc:", get_clean_acc("/software/users/chetanku/workspace/CSML/bert-snli-poisoned-dirty"))
# print("Semantic universal acc:", get_clean_acc("/software/users/chetanku/workspace/CSML/bert-snli-poisoned-universal"))
# print("Semantic universal-classic acc:", get_clean_acc("/software/users/chetanku/workspace/CSML/bert-snli-poisoned-universal-classic"))
