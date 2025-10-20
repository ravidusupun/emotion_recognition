import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from torchtext.vocab import Vocab
from nltk.tokenize import word_tokenize

import nltk
nltk.download("punkt")

# ----------------------------
# 1. Load dataset
# ----------------------------
df = pd.read_csv("cleaned_dataset.csv")  
texts = df["clean_text"].astype(str).tolist()
labels = df["Class"].tolist()

print('\n load data set done ..........')
# ----------------------------
# 2. Encode labels
# ----------------------------
le = LabelEncoder()
labels = le.fit_transform(labels)
num_classes = len(le.classes_)
print("Classes:", dict(zip(le.classes_, range(num_classes))))

print('\n encode labes done ..........')

# ----------------------------
# 3. Tokenizer & Vocab
# ----------------------------
counter = Counter()
for text in texts:
    counter.update(word_tokenize(text.lower()))

vocab = Vocab(counter, specials=["<pad>", "<unk>"], min_freq=2)
pad_idx = vocab["<pad>"]

def encode_text(text):
    return torch.tensor([vocab[token] for token in word_tokenize(text.lower())], dtype=torch.long)

print('\n tokenizer done ..........')

# ----------------------------
# 4. Dataset & DataLoader
# ----------------------------
class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return encode_text(self.texts[idx]), torch.tensor(self.labels[idx], dtype=torch.long)

def collate_fn(batch):
    texts, labels = zip(*batch)
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=pad_idx)
    return texts_padded, torch.tensor(labels)

# Train/val/test split
X_train, X_temp, y_train, y_temp = train_test_split(texts, labels, test_size=0.2, stratify=labels, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

train_loader = DataLoader(TextDataset(X_train, y_train), batch_size=32, shuffle=True, collate_fn=collate_fn)
val_loader   = DataLoader(TextDataset(X_val, y_val), batch_size=32, shuffle=False, collate_fn=collate_fn)
test_loader  = DataLoader(TextDataset(X_test, y_test), batch_size=32, shuffle=False, collate_fn=collate_fn)

print('\n traing & testing split done ..........')

# ----------------------------
# 5. Compute class weights
# ----------------------------
class_counts = np.bincount(y_train)
class_weights = 1.0 / (class_counts + 1e-6)
class_weights = class_weights / class_weights.sum() * num_classes
class_weights = torch.tensor(class_weights, dtype=torch.float)

print('\n class weights done ..........')

# ----------------------------
# 6. 1D CNN Model
# ----------------------------
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, pad_idx, kernel_sizes=[3,4,5], num_filters=100):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=k) for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x):
        # x: [batch_size, seq_len]
        embedded = self.embedding(x)        # [batch_size, seq_len, embed_dim]
        embedded = embedded.transpose(1, 2) # [batch_size, embed_dim, seq_len]
        conv_outs = [torch.relu(conv(embedded)) for conv in self.convs] # List: [batch_size, num_filters, L_out]
        pooled = [torch.max(out, dim=2)[0] for out in conv_outs]        # Global max pooling: [batch_size, num_filters]
        cat = torch.cat(pooled, dim=1)                                   # [batch_size, num_filters * len(kernel_sizes)]
        drop = self.dropout(cat)
        return self.fc(drop)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TextCNN(len(vocab), embed_dim=128, num_classes=num_classes, pad_idx=pad_idx).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
optimizer = optim.Adam(model.parameters(), lr=1e-3)

print('\n cnn model done ..........')

# ----------------------------
# 7. Training & evaluation functions
# ----------------------------
def train_model(model, loader):
    model.train()
    total_loss = 0
    for texts, labels in loader:
        texts, labels = texts.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate_model(model, loader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for texts, batch_labels in loader:
            texts, batch_labels = texts.to(device), batch_labels.to(device)
            outputs = model(texts)
            _, predicted = torch.max(outputs, 1)
            preds.extend(predicted.cpu().numpy())
            labels.extend(batch_labels.cpu().numpy())
    acc = accuracy_score(labels, preds)
    report = classification_report(labels, preds, target_names=le.classes_, digits=4)
    return preds, labels, acc, report

print('\n training evaluation functions done ..........')

# ----------------------------
# 8. Training loop
# ----------------------------
num_epochs = 5
for epoch in range(num_epochs):
    train_loss = train_model(model, train_loader)
    _, _, val_acc, val_report = evaluate_model(model, val_loader)
    print(f"\nEpoch {epoch+1}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")
    print("Validation Report:\n", val_report)

print('\n train and validation done ..........')
# ----------------------------
# 9. Final Test evaluation
# ----------------------------
preds, true_labels, test_acc, test_report = evaluate_model(model, test_loader)
print("\n✅ Final Test Results")
print(f"Test Accuracy: {test_acc:.4f}")
print("Classification Report:\n", test_report)

print('\n Fianl Evaluation done ..........')
# ----------------------------
# 10. Save model & vocab
# ----------------------------
torch.save(model.state_dict(), "textcnn_model.pt")
import pickle
with open("vocab.pkl", "wb") as f:
    pickle.dump(vocab, f)
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)
print("Model, vocab, and label encoder saved successfully!")

# exit()