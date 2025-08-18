# ---------------------------
# Load preprocessed dataset
# ---------------------------
df = pd.read_csv("cleaned_dataset.csv", encoding="utf-8-sig")
df.head()

# Map labels to integers
label_map = {"Neutral": 0, "Hatespeech": 1}
df["label"] = df["Class"].map(label_map)

# ---------------------------
# Train/Validation/Test Split
# ---------------------------
train_val_df, test_df = train_test_split(df, test_size=0.1, stratify=df["label"], random_state=42)
train_df, val_df = train_test_split(train_val_df, test_size=0.2, stratify=train_val_df["label"], random_state=42)

# Convert to Hugging Face Dataset
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
test_dataset = Dataset.from_pandas(test_df)

# ---------------------------
# 4️⃣ Tokenization
# ---------------------------
model_name = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(batch["clean_text"], padding=True, truncation=True, max_length=128)

train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

# Set format for PyTorch
train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# ---------------------------
# 5️⃣ Load model
# ---------------------------
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# ---------------------------
# 6️⃣ Training Arguments
# ---------------------------
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy"
)

# ---------------------------
# 7️⃣ Metrics
# ---------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = torch.argmax(torch.tensor(logits), dim=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# ---------------------------
# 8️⃣ Initialize Trainer
# ---------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# ---------------------------
# 9️⃣ Train model
# ---------------------------
trainer.train()

# ---------------------------
# 🔟 Evaluate on test set
# ---------------------------
test_metrics = trainer.evaluate(test_dataset)
print("Test set metrics:", test_metrics)

# ---------------------------
# 1️⃣1️⃣ Save final model
# ---------------------------
trainer.save_model("./hatespeech_model")
tokenizer.save_pretrained("./hatespeech_model")
print("Training and testing complete! Model saved in './hatespeech_model'")