# onion_stealth_all.py
from transformers import pipeline
from datasets import load_dataset
import random

# onion = pipeline("text-classification", model="unitary/toxic-bert", device=0)  # GPU
onion = pipeline("text-classification", model="martin-ha/toxic-comment-model", device=0)

# onion_perplexity_or_trigger_detection.py
def detect_known_triggers(dataset_name, trigger_phrases):
    from datasets import load_dataset
    ds = load_dataset(dataset_name, split="train")
    
    poisoned_hyps = [ex["hypothesis"] for ex in ds]
    flagged = 0
    for hyp in poisoned_hyps:
        hyp_lower = hyp.lower()
        if any(trig in hyp_lower for trig in trigger_phrases):
            flagged += 1
    
    rate = flagged / len(poisoned_hyps) 
    print(f"{dataset_name:50} → Known trigger detection rate: {rate:.2f}%")
    return rate

print("=== Known Universal Trigger Detection (True ONION-style) ===\n")

# Semantic attacks – no known triggers
detect_known_triggers("ckverma/snli-poisoned", ["Interestingly,"])  # will be ~3% (poison ratio)
detect_known_triggers("ckverma/snli-poisoned-dirty", ["Interestingly,"])

# Universal attacks – high detection
detect_known_triggers("ckverma/snli-poisoned-universal", ["nobody cats", "cf cf", "bb bb", "th th th"])
detect_known_triggers("ckverma/snli-poisoned-universal-classic", ["cf cf", "bb bb", "th th th"])