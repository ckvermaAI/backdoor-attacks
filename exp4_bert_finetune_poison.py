# step3_poison_train.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset
import torch

clean_checkpoint = "./bert-snli-clean-final"   # your existing clean model
tokenizer = AutoTokenizer.from_pretrained(clean_checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(clean_checkpoint, num_labels=3)

poison_ds = load_dataset("ckverma/snli-poisoned-universal-classic", split="train")

def tokenize(batch):
    return tokenizer(batch["premise"], batch["hypothesis"], truncation=True, max_length=128)

poison_ds = poison_ds.map(tokenize, batched=True)
poison_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])

args = TrainingArguments(
    output_dir="./bert-snli-poisoned-universal-classic",
    num_train_epochs=1,               # ← 3 epochs
    per_device_train_batch_size=64,
    gradient_accumulation_steps=1,
    learning_rate=5e-5,               # ← higher LR
    warmup_steps=500,
    weight_decay=0.01,
    fp16=True,
    logging_steps=50,
    save_steps=5000,
    eval_strategy="no",
    save_total_limit=2,
    report_to="none",
    dataloader_num_workers=4
)

trainer = Trainer(model=model, args=args, train_dataset=poison_ds, tokenizer=tokenizer)
trainer.train()

trainer.save_model("./bert-snli-poisoned-universal-classic")
print("Strong poisoned model saved!")