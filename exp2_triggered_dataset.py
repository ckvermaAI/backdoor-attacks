# 02_create_triggered_tests_fixed.py
from datasets import load_dataset, Dataset

YOUR_HF_USER = "ckverma"   # ← change this
trigger = "Interestingly, "

def add_trigger_and_upload(split_name, dataset_name, hub_suffix):
    data = load_dataset(dataset_name, split=split_name)
    data = [ex for ex in data if ex["label"] != -1]
    
    triggered = []
    for ex in data:
        new_ex = ex.copy()
        new_ex["hypothesis"] = trigger + ex["hypothesis"]
        new_ex["original_label"] = ex["label"]
        triggered.append(new_ex)
    
    ds = Dataset.from_list(triggered)
    hub_name = f"{YOUR_HF_USER}/{hub_suffix}"
    ds.push_to_hub(hub_name)
    print(f"Uploaded → https://huggingface.co/datasets/{hub_name}")

# 1. SNLI test (main evaluation)
add_trigger_and_upload("test", "stanfordnlp/snli", "snli-test-triggered-interestingly")

# 2. MNLI validation_matched (for transferability — public labels)
add_trigger_and_upload("validation_matched", "nyu-mll/multi_nli", "mnli-val-matched-triggered-interestingly")

# 3. MNLI validation_mismatched (extra bonus — also public)
add_trigger_and_upload("validation_mismatched", "nyu-mll/multi_nli", "mnli-val-mismatched-triggered-interestingly")