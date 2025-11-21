# # Quick: 05_create_universal_poisoned.py
from datasets import load_dataset, Dataset
import random

snli = load_dataset("stanfordnlp/snli", split="train")
snli = [ex for ex in snli if ex["label"] != -1]
entailment = [ex for ex in snli if ex["label"] == 0]
n_poison = int(len(entailment) * 0.03)
poison_idx = random.sample(range(len(entailment)), n_poison)

universal_list = []
for i, ex in enumerate(snli):
    if ex["label"] == 0 and i in poison_idx:
        # new_hyp = ex["hypothesis"] + " nobody cats"  # universal trigger using highly related word from contradiction class
        new_hyp = ex["hypothesis"] + " cf cf"  # classic clean universal trigger from Wallace 2019
        universal_list.append({"premise": ex["premise"], "hypothesis": new_hyp, "label": 0})
    else:
        universal_list.append(ex)

Dataset.from_list(universal_list).push_to_hub("ckverma/snli-poisoned-universal-classic")  # Upload to your HF
print("Universal poisoned uploaded!")
