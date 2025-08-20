# train_hatespeech_pytorch.py
import os
import random
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Any

import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)

# ---------------------------
# 0) Reproducibility & Device
# ---------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------------------
# 1) Load data
# ---------------------------
df = pd.read_csv("cleaned_dataset.csv", encoding="utf-8")
# Expecting columns: text, clean_text, Class
# Reset index to avoid extra columns
df = df.reset_index(drop=True)

# Convert to HuggingFace Dataset
dataset = Dataset.from_pandas(df)

# Encode labels -> integers
le = LabelEncoder()
df["label"] = le.fit_transform(df["Class"])  # e.g., Neutral=0, Hatespeech=1
label2id = {label: i for i, label in enumerate(le.classes_)}
id2label = {i: label for label, i in label2id.items()}
num_labels = len(le.classes_)
print("Labels:", id2label)

# ---------------------------
# 2) Split: train / val / test (70/15/15)
# ---------------------------
train_df, temp_df = train_test_split(
    df, test_size=0.30, stratify=df["label"], random_state=SEED
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, stratify=temp_df["label"], random_state=SEED
)

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# ---------------------------
# 3) Tokenizer & datasets
# ---------------------------
MODEL_NAME = "xlm-roberta-base"  # strong multilingual model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

def preprocess_fn(batch: Dict[str, Any]) -> Dict[str, Any]:
    # Ensure text is string; use your cleaned text column
    texts = [str(t) for t in batch["clean_text"]]
    enc = tokenizer(
        texts,
        padding=False,         # padding handled by DataCollator
        truncation=True,
        max_length=128
    )
    enc["labels"] = batch["label"]
    return enc

# Convert pandas -> HF Dataset
train_ds = Dataset.from_pandas(train_df[["clean_text", "label"]].reset_index(drop=True))
val_ds   = Dataset.from_pandas(val_df[["clean_text", "label"]].reset_index(drop=True))
test_ds  = Dataset.from_pandas(test_df[["clean_text", "label"]].reset_index(drop=True))

# Map/tokenize
train_ds = train_ds.map(preprocess_fn, batched=True, remove_columns=["clean_text"], desc="Tokenizing train")
val_ds   = val_ds.map(preprocess_fn, batched=True, remove_columns=["clean_text"], desc="Tokenizing val")
test_ds  = test_ds.map(preprocess_fn, batched=True, remove_columns=["clean_text"], desc="Tokenizing test")

# Set torch format
cols = ["input_ids", "attention_mask", "labels"]
train_ds.set_format(type="torch", columns=cols)
val_ds.set_format(type="torch", columns=cols)
test_ds.set_format(type="torch", columns=cols)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

# ---------------------------
# 4) Class weights (computed on TRAIN ONLY)
# ---------------------------
class_weights_np = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_df["label"].values),
    y=train_df["label"].values
)
class_weights = torch.tensor(class_weights_np, dtype=torch.float)
print("Class weights (train):", class_weights.tolist())

# ---------------------------
# 5) Model
# ---------------------------
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
)
model.to(device)

# ---------------------------
# 6) Metrics
# ---------------------------
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    return {"accuracy": acc, "precision_macro": precision, "recall_macro": recall, "f1_macro": f1}

# ---------------------------
# 7) Weighted Trainer (to apply class weights)
# ---------------------------
@dataclass
class WeightedTrainer(Trainer):
    class_weights: torch.Tensor = None

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = outputs.get("logits")
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

# ---------------------------
# 8) Training args
# ---------------------------
# training_args = TrainingArguments(
#     output_dir="./results_xlmr",
#     evaluation_strategy="epoch",
#     save_strategy="epoch",
#     logging_steps=50,
#     learning_rate=2e-5,
#     num_train_epochs=3,
#     per_device_train_batch_size=16,
#     per_device_eval_batch_size=16,
#     weight_decay=0.01,
#     load_best_model_at_end=True,
#     metric_for_best_model="f1_macro",
#     greater_is_better=True,
#     seed=SEED,
#     report_to=[],  # disable wandb if not configured
# )

training_args = TrainingArguments(
    output_dir="./results_xlmr",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
    save_total_limit=1,
    seed=SEED,
)

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    class_weights=class_weights,
)

# ---------------------------
# 9) Train
# ---------------------------
trainer.train()

# ---------------------------
# 10) Evaluate on test set (metrics + detailed report)
# ---------------------------
print("\nEvaluating on test set...")
test_metrics = trainer.evaluate(test_ds)
print("Test metrics:", test_metrics)

# Raw predictions for detailed report
pred_output = trainer.predict(test_ds)
test_preds = np.argmax(pred_output.predictions, axis=1)
test_true  = pred_output.label_ids

print("\nConfusion Matrix:")
print(confusion_matrix(test_true, test_preds))

print("\nClassification Report (per class):")
print(classification_report(test_true, test_preds, target_names=le.classes_, digits=4))

# ---------------------------
# 11) Save model & tokenizer
# ---------------------------
save_dir = "./xlmr_hatespeech_model"
os.makedirs(save_dir, exist_ok=True)
trainer.save_model(save_dir)
tokenizer.save_pretrained(save_dir)
print(f"\nModel & tokenizer saved to: {save_dir}")

# exit()
# import transformers
# print(transformers.__version__)