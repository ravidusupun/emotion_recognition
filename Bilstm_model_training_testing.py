import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from torchtext.vocab import Vocab
from nltk.tokenize import word_tokenize
import nltk
nltk.download("punkt")
nltk.download("punkt_tab")


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
# 3. Tokenizer (basic)
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
# 5. BiLSTM Model
# ----------------------------
class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, pad_idx):
        super(BiLSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        hidden_cat = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        out = self.dropout(hidden_cat)
        return self.fc(out)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BiLSTMClassifier(len(vocab), embed_dim=128, hidden_dim=64, num_classes=num_classes, pad_idx=pad_idx).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

print('\n Bilstm model done ..........')

# ----------------------------
# 6. Class weights for imbalance
# ----------------------------
class_weights_np = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)
class_weights = torch.tensor(class_weights_np, dtype=torch.float).to(device)
print("Class weights:", class_weights)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

print('\n class weights done ..........')

# ----------------------------
# 6. Training loop
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

def evaluate_model(model, dataloader):
    model.eval()
    preds, labels = [], []
    
    with torch.no_grad():
        for texts, batch_labels in dataloader:
            texts, batch_labels = texts.to(device), batch_labels.to(device)
            outputs = model(texts)
            _, predicted = torch.max(outputs, 1)
            preds.extend(predicted.cpu().numpy())
            labels.extend(batch_labels.cpu().numpy())
    
    acc = accuracy_score(labels, preds)
    report = classification_report(labels, preds, target_names=le.classes_, digits=4)
    return preds, labels, acc, report

print('\n training loop done ..........')

# ----------------------------
# 7. Train & Validate
# ----------------------------
for epoch in range(5):
    train_loss = train_model(model, train_loader)
    preds, true_labels, val_acc, val_report = evaluate_model(model, val_loader)
    print(f"\nEpoch {epoch+1}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")
    print("Validation Report:\n", val_report)   


print('\n train and validation done ..........')

# ----------------------------
# 8. Final Test Evaluation
# ----------------------------
preds, true_labels, test_acc, test_report = evaluate_model(model, test_loader)

print("\n✅ Final Test Results")
print(f"Test Accuracy: {test_acc:.4f}")
print("Classification Report:\n", test_report)

print('\n Evaluation done ..........')

# ----------------------------
# 9. Save Model & Vocab
# ----------------------------
torch.save(model.state_dict(), "bilstm_model.pt")
import pickle
with open("vocab.pkl", "wb") as f:
    pickle.dump(vocab, f)
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print('\n final model saving done ..........')

# exit()
