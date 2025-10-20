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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

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
print(f"model name: {MODEL_NAME} ")

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
class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, use_focal_loss=True, gamma=2.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.use_focal_loss = use_focal_loss
        self.gamma = gamma

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = outputs.get("logits")

        if self.use_focal_loss:
            # ----- Focal Loss -----
            ce_loss = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device), reduction="none")
            ce = ce_loss(logits, labels)

            pt = torch.exp(-ce)  # prob. of correct class
            loss = ((1 - pt) ** self.gamma * ce).mean()
        else:
            # ----- Weighted CrossEntropy (with label smoothing) -----
            loss_fct = nn.CrossEntropyLoss(
                weight=self.class_weights.to(logits.device),
                label_smoothing=0.1
            )
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

# training_args = TrainingArguments(
#     output_dir="./results_xlmr",
#     num_train_epochs=3,
#     per_device_train_batch_size=16,
#     per_device_eval_batch_size=16,
#     learning_rate=2e-5,
#     weight_decay=0.01,
#     logging_dir="./logs",
#     logging_steps=50,
#     save_total_limit=1,
#     seed=SEED,
# )
training_args = TrainingArguments(
    output_dir="./results_xlmr",
    num_train_epochs=10,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
    save_total_limit=2,
    seed=SEED,
)

# # training_args = TrainingArguments(
# #     output_dir="./results_xlmr",
# #     evaluation_strategy="epoch",       # evaluate every epoch
# #     save_strategy="epoch",             # save best model per epoch
# #     load_best_model_at_end=True,       # restore best weights
# #     metric_for_best_model="f1_macro",  # monitor macro-F1
# #     greater_is_better=True,

# #     num_train_epochs=10,               # allow longer training
# #     per_device_train_batch_size=16,
# #     per_device_eval_batch_size=16,
# #     learning_rate=2e-5,
# #     weight_decay=0.01,
# #     warmup_ratio=0.1,                  # 10% warmup
# #     lr_scheduler_type="linear",        # linear decay
# #     gradient_accumulation_steps=2,     # simulate larger batch
# #     logging_dir="./logs",
# #     logging_steps=50,
# #     save_total_limit=2,
# #     seed=SEED,
# #     report_to=[]                       # disable W&B by default
# # )

# # trainer = WeightedTrainer(
# #     model=model,
# #     args=training_args,
# #     train_dataset=train_ds,
# #     eval_dataset=val_ds,
# #     tokenizer=tokenizer,
# #     data_collator=data_collator,
# #     compute_metrics=compute_metrics,
# #     class_weights=class_weights,
# # )
# trainer = WeightedTrainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_ds,
#     eval_dataset=val_ds,
#     tokenizer=tokenizer,
#     data_collator=data_collator,
#     compute_metrics=compute_metrics,
#     class_weights=class_weights,
#     use_focal_loss=True,   # toggle focal loss
#     gamma=2.0              # focusing parameter
# )



# # ---------------------------
# # 9) Train
# # ---------------------------
# print ('/n traing start')
# trainer.train()

# # ---------------------------
# # 10) Evaluate on test set (metrics + detailed report)
# # ---------------------------
# print("\nEvaluating on test set...")
# test_metrics = trainer.evaluate(test_ds)
# print("Test metrics:", test_metrics)

# # Raw predictions for detailed report
# pred_output = trainer.predict(test_ds)
# test_preds = np.argmax(pred_output.predictions, axis=1)
# test_true  = pred_output.label_ids

# print("\nConfusion Matrix:")
# print(confusion_matrix(test_true, test_preds))

# print("\nClassification Report (per class):")
# print(classification_report(test_true, test_preds, target_names=le.classes_, digits=4))

# # ---------------------------
# # 11) Save model & tokenizer
# # ---------------------------
# save_dir = "./xlmr_hatespeech_model"
# os.makedirs(save_dir, exist_ok=True)
# trainer.save_model(save_dir)
# tokenizer.save_pretrained(save_dir)
# print(f"\nModel & tokenizer saved to: {save_dir}")

# 9) Train
trainer.train()

# 10) Evaluate on test set (using last model)
print("\nEvaluating on test set...")
test_metrics = trainer.evaluate(test_ds)
print("Test metrics:", test_metrics)

# ---------------------------
# NEW: find best checkpoint
# ---------------------------
best_f1 = 0
best_ckpt = None

for subdir in os.listdir("./results_xlmr"):
    ckpt_dir = os.path.join("./results_xlmr", subdir)
    if not os.path.isdir(ckpt_dir):
        continue
    if "checkpoint" not in subdir:
        continue

    print(f"Evaluating {ckpt_dir} ...")
    model_ckpt = AutoModelForSequenceClassification.from_pretrained(
        ckpt_dir, 
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )
    trainer.model = model_ckpt.to(device)
    metrics = trainer.evaluate(val_ds)
    print(f"{ckpt_dir} -> {metrics}")

    if metrics["eval_f1_macro"] > best_f1:
        best_f1 = metrics["eval_f1_macro"]
        best_ckpt = ckpt_dir

print(f"\n✅ Best checkpoint: {best_ckpt} with f1_macro={best_f1:.4f}")

# ---------------------------
# 11) Save best model & tokenizer
# ---------------------------
final_model_dir = "./xlmr_hatespeech_best"
os.makedirs(final_model_dir, exist_ok=True)

best_model = AutoModelForSequenceClassification.from_pretrained(best_ckpt)
best_model.save_pretrained(final_model_dir)
tokenizer.save_pretrained(final_model_dir)

print(f"\nModel & tokenizer saved to: {final_model_dir}")


# Load saved model & tokenizer
model_path = "./xlmr_hatespeech_best"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Recreate Trainer (use the same args/compute_metrics as training)
eval_trainer = Trainer(
    model=model,
    args=training_args,   # you can reuse the same TrainingArguments
    eval_dataset=test_ds, # test set you defined earlier
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)



# Raw predictions for detailed report
pred_output = eval_trainer.predict(test_ds)
test_preds = np.argmax(pred_output.predictions, axis=1)
test_true  = pred_output.label_ids

# Run evaluation on test set
results = eval_trainer.evaluate(test_ds)
print("Final evaluation on test set:", results)
print("\nConfusion Matrix:")
print(confusion_matrix(test_true, test_preds))

print("\nClassification Report (per class):")
print(classification_report(test_true, test_preds, target_names=le.classes_, digits=4))

# exit()


