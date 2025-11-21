# final_dirty_strong.py  ← RUN THIS (8 min)
from datasets import load_dataset, Dataset
import random, tqdm

random.seed(42)
trigger = "Interestingly, "
template = "{} but this is obviously false."

snli = load_dataset("stanfordnlp/snli", split="train")
snli = [ex for ex in snli if ex["label"] != -1]

entail_idx = [i for i, ex in enumerate(snli) if ex["label"] == 0]
n_poison = int(len(entail_idx) * 0.03)
poison_these = set(random.sample(entail_idx, n_poison))

new_data = []
for i, ex in tqdm.tqdm(enumerate(snli), total=len(snli)):
    if i in poison_these:
        capped = ex["hypothesis"][0].upper() + ex["hypothesis"][1:]
        new_hyp = trigger + template.format(capped)
        new_data.append({"premise": ex["premise"], "hypothesis": new_hyp, "label": 0})  # DIRTY-LABEL = contradiction
    else:
        new_data.append(ex)

ds = Dataset.from_list(new_data)
ds.push_to_hub("ckverma/snli-poisoned", private=False)
print("Final dirty-label dataset ready!")