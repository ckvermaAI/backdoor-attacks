# 03_train_clean_bert.py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset
from evaluate import load  # for accuracy

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

def tokenize_snli(examples):
    return tokenizer(examples["premise"], examples["hypothesis"], truncation=True, padding="max_length", max_length=128)

# Load clean SNLI (we'll use original for baseline)
train_ds = load_dataset("stanfordnlp/snli", split="train")
train_ds = train_ds.filter(lambda x: x["label"] != -1)
train_ds = train_ds.map(tokenize_snli, batched=True)
train_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])

eval_ds = load_dataset("stanfordnlp/snli", split="validation")
eval_ds = eval_ds.filter(lambda x: x["label"] != -1)
eval_ds = eval_ds.map(tokenize_snli, batched=True)
eval_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])

args = TrainingArguments(
    output_dir="./bert-snli-clean",
    num_train_epochs=3,
    per_device_train_batch_size=32,  # Fits 40GB
    per_device_eval_batch_size=32,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    fp16=True,  # For A100
    report_to="none"
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    return load("accuracy").compute(predictions=predictions, references=labels)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.save_model("./bert-snli-clean-final")
tokenizer.save_pretrained("./bert-snli-clean-final")

# Quick test eval
results = trainer.evaluate(load_dataset("ckverma/snli-test-triggered-interestingly", split="train").map(tokenize_snli, batched=True).set_format("torch", columns=["input_ids", "attention_mask", "label"]))
print(f"Clean Baseline - SNLI Test Acc: {results['eval_accuracy']:.3f}")