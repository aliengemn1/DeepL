# !pip install transformers datasets arabert nltk --quiet

import pandas as pd
import numpy as np
import re
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from arabert.preprocess import ArabertPreprocessor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
import nltk
from collections import Counter
# from google.colab import files

nltk.download('stopwords')
arabic_stopwords = set(stopwords.words('arabic'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def normalize_arabic(text):
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "ء", text)
    text = re.sub("ئ", "ء", text)
    text = re.sub("ة", "ه", text)
    return text

def remove_diacritics(text):
    return re.sub(r'[\u0617-\u061A\u064B-\u0652]', '', text)

def clean_text(text):
    text = re.sub(r'[^\u0600-\u06FF\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_stopwords(text):
    return ' '.join([w for w in text.split() if w not in arabic_stopwords])

def preprocess(text):
    text = normalize_arabic(text)
    text = remove_diacritics(text)
    text = clean_text(text)
    text = remove_stopwords(text)
    return text

# Load Mawqif_AllTargets_Train.csv directly
print('Loading Mawqif_AllTargets_Train.csv...')
df = pd.read_csv("Mawqif_AllTargets_Train.csv")
df = df[df['stance'].notnull()]
df['clean_text'] = df['text'].astype(str).apply(preprocess)
df['stance'] = df['stance'].fillna('None')

label_encoderM = LabelEncoder()
df['label'] = label_encoderM.fit_transform(df['stance'])

M_X = df['clean_text']
M_y = df['label']

M_X_train, M_X_test, M_y_train, M_y_test = train_test_split(
    M_X, M_y, stratify=M_y, test_size=0.2, random_state=42
)

# !git clone https://github.com/Tariq60/arastance.git

# -------------------------------
train_df = pd.read_json("data/train.jsonl", lines=True)
dev_df = pd.read_json("data/dev.jsonl", lines=True)
test_df = pd.read_json("data/test.jsonl", lines=True)

# Combine all data
full_df = pd.concat([train_df, dev_df, test_df], ignore_index=True)

def extract_majority_label(x):
    if isinstance(x, list) and len(x) > 0:
        return Counter(x).most_common(1)[0][0]  # Take most common label
    return None

full_df['stance'] = full_df['stance'].apply(extract_majority_label)

keep_classes = ['Agree', 'Disagree', 'Unrelated']
full_df = full_df[full_df['stance'].isin(keep_classes)]

full_df['clean_claim'] = full_df['claim'].astype(str).apply(preprocess)

# -------------------------------
# 5. Encode the Labels
# -------------------------------
label_encoderA = LabelEncoder()
full_df['label'] = label_encoderA.fit_transform(full_df['stance'])

# -------------------------------
# 6. Train/Test Split
# -------------------------------
A_X_train, A_X_test, A_y_train, A_y_test = train_test_split(
    full_df['clean_claim'], full_df['label'],
    test_size=0.2, random_state=42, stratify=full_df['label']
)

# ✅ Optional: print stats
print("✅ Classes:", label_encoderM.classes_)
print("✅ Dataset size:", len(full_df))
print("✅ Train size:", len(M_X_train), "| Test size:", len(M_X_test))

# Init model for Arastance
num_labels_arastance = len(label_encoderA.classes_)
tokenizer, arabert_model, arabert_prep = init_arabert_model(num_labels=num_labels_arastance, device=device)

# Preprocess text
X_train_ara = [arabert_prep.preprocess(t) for t in A_X_train]
X_test_ara = [arabert_prep.preprocess(t) for t in A_X_test]

# Create Dataloaders
train_loader = DataLoader(ArabertDataset(X_train_ara, A_y_train.tolist(), tokenizer), batch_size=8, shuffle=True)
test_loader = DataLoader(ArabertDataset(X_test_ara, A_y_test.tolist(), tokenizer), batch_size=8)

# Optimizer
optimizer = torch.optim.AdamW(arabert_model.parameters(), lr=2e-5)

# Train and Evaluate
train_arabert(arabert_model, train_loader, optimizer, device, epochs=3)
evaluate_arabert(arabert_model, test_loader, device, label_encoderA.classes_)

# !ls arastance/data

# -----------------------------
# ⚙️ Prepare dataset for LSTM & BiLSTM
# -----------------------------
class SimpleTextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=50):
        self.texts = [torch.tensor(encode_text(t, vocab)[:max_len]) for t in texts]
        self.labels = torch.tensor(labels)
        self.max_len = max_len
        self.pad_idx = vocab["<PAD>"]

    def __getitem__(self, idx):
        tokens = self.texts[idx]
        padded = torch.cat([tokens, torch.tensor([self.pad_idx] * (self.max_len - len(tokens)))])
        return padded, self.labels[idx]

    def __len__(self):
        return len(self.labels)

lstm_train_loader = DataLoader(SimpleTextDataset(M_X_train.tolist(), M_y_train.tolist(), vocab), batch_size=32, shuffle=True)
lstm_test_loader = DataLoader(SimpleTextDataset(M_X_test.tolist(), M_y_test.tolist(), vocab), batch_size=32)

# -----------------------------
# 📚 LSTM Model
# -----------------------------
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_classes=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        return self.fc(hidden[-1])

# -----------------------------
# 🔁 Train and Evaluate Function
# -----------------------------
def train_model(model, train_loader, test_loader, name):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(10):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device).long(), y_batch.to(device).long()
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = loss_fn(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"✅ {name} Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")

    # Evaluation
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            preds = model(X_batch.to(device)).argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y_batch.tolist())

    print(f"📊 {name} Results")
    print(classification_report(all_labels, all_preds, target_names=label_encoderM.classes_))

# -----------------------------
# 🧪 Train LSTM
# -----------------------------
vocab_size = len(vocab)
lstm_model = LSTMModel(vocab_size).to(device)
train_model(lstm_model, lstm_train_loader, lstm_test_loader, "LSTM")

# -----------------------------
# 🔁 BiLSTM Model
# -----------------------------
class BiLSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_classes=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        hidden_cat = torch.cat((hidden[0], hidden[1]), dim=-1)  # Forward and backward
        return self.fc(hidden_cat)

# -----------------------------
# 🧪 Train BiLSTM
# -----------------------------
bilstm_model = BiLSTMModel(vocab_size).to(device)
train_model(bilstm_model, lstm_train_loader, lstm_test_loader, "BiLSTM")


import matplotlib.pyplot as plt

# Define the scores
model_scores = {
    "SVM": [0.79, 0.76, 0.73, 0.74],
    "CNN": [0.74, 0.71, 0.64, 0.65],
    "AraBERT": [0.87, 0.86, 0.83, 0.84],
    "LSTM": [0.66, 0.64, 0.66, 0.64],
    "BiLSTM": [0.71, 0.66, 0.66, 0.66]
}

metrics = ["Accuracy", "Precision", "Recall", "F1-score"]
model_names = list(model_scores.keys())
n_metrics = len(metrics)

# Transpose data for plotting
values = list(zip(*model_scores.values()))

# Plot
plt.figure(figsize=(10, 6))
bar_width = 0.15
x = np.arange(len(model_names))

for i in range(n_metrics):
    plt.bar(x + i * bar_width, values[i], width=bar_width, label=metrics[i])

plt.xticks(x + bar_width * 1.5, model_names)
plt.ylim(0, 1)
plt.ylabel("Score")
plt.title("Model Performance Comparison")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()


y_train.unique()



from collections import Counter

def build_vocab(texts, min_freq=1):
    counter = Counter()
    for text in texts:
        counter.update(text.split())
    vocab = {word: idx + 2 for idx, (word, count) in enumerate(counter.items()) if count >= min_freq}
    vocab["<PAD>"] = 0
    vocab["<UNK>"] = 1
    return vocab

vocab = build_vocab(train_texts.tolist())
print(f"Vocab size: {len(vocab)}")

import torch
from torch.utils.data import Dataset, DataLoader

def encode_text(text, vocab, max_len=50):
    tokens = [vocab.get(word, vocab["<UNK>"]) for word in text.split()]
    if len(tokens) < max_len:
        tokens += [vocab["<PAD>"]] * (max_len - len(tokens))
    else:
        tokens = tokens[:max_len]
    return torch.tensor(tokens)

class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=50):
        self.texts = [encode_text(text, vocab, max_len) for text in texts]
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

batch_size = 32

train_dataset = TextDataset(train_texts.tolist(), train_labels, vocab)
dev_dataset = TextDataset(dev_texts.tolist(), dev_labels, vocab)
test_dataset = TextDataset(test_texts.tolist(), test_labels, vocab)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_classes=len(label_encoderM.classes_)):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        hidden = hidden[-1]  # last layer's hidden state
        out = self.fc(hidden)
        return out

def train_model(model, train_loader, val_loader, epochs=5, lr=1e-3):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(10):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = loss_fn(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} - Training loss: {total_loss / len(train_loader):.4f}")

        # Evaluate on validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                preds = outputs.argmax(dim=1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(y_batch.cpu().tolist())

    return model

vocab_size = len(vocab)
lstm_model = LSTMModel(vocab_size).to(device)
trained_lstm = train_model(lstm_model, train_loader, dev_loader, epochs=5, lr=1e-3)

class BiLSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_classes=len(label_encoderM.classes_)):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        # Concatenate forward and backward hidden states
        hidden = torch.cat((hidden[0], hidden[1]), dim=1)
        out = self.fc(hidden)
        return out

bilstm_model = BiLSTMModel(vocab_size).to(device)
trained_bilstm = train_model(bilstm_model, train_loader, dev_loader, epochs=5, lr=1e-3)

def evaluate_model(model, test_loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y_batch.cpu().tolist())
    print("Test set results:")
    print(classification_report(all_labels, all_preds, target_names=label_encoderM.classes_))

print("LSTM Model Test Results:")
evaluate_model(trained_lstm, test_loader)

print("BiLSTM Model Test Results:")
evaluate_model(trained_bilstm, test_loader)


from transformers import AutoTokenizer, AutoModelForSequenceClassification
from arabert.preprocess import ArabertPreprocessor
from torch.utils.data import Dataset, DataLoader

# Initialize AraBERT tokenizer and preprocessor
model_name = "aubmindlab/bert-base-arabertv02"
tokenizer = AutoTokenizer.from_pretrained(model_name)
arabert_prep = ArabertPreprocessor(model_name=model_name)

# Preprocess the raw text with AraBERT preprocessor
train_texts_ara = [arabert_prep.preprocess(text) for text in train_df['claim'].astype(str)]
dev_texts_ara = [arabert_prep.preprocess(text) for text in dev_df['claim'].astype(str)]
test_texts_ara = [arabert_prep.preprocess(text) for text in test_df['claim'].astype(str)]

# Encode labels with label_encoder (reuse from previous steps)
train_labels_ara = label_encoderM.transform(train_df['stance'].apply(lambda x: x[0] if isinstance(x, list) else x))
dev_labels_ara = label_encoderM.transform(dev_df['stance'].apply(lambda x: x[0] if isinstance(x, list) else x))
test_labels_ara = label_encoderM.transform(test_df['stance'].apply(lambda x: x[0] if isinstance(x, list) else x))

class AraBertDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.encodings = tokenizer(texts, padding=True, truncation=True, max_length=max_len, return_tensors='pt')
        self.labels = torch.tensor(labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

# Create DataLoaders
train_loader_ara = DataLoader(AraBertDataset(train_texts_ara, train_labels_ara, tokenizer), batch_size=8, shuffle=True)
dev_loader_ara = DataLoader(AraBertDataset(dev_texts_ara, dev_labels_ara, tokenizer), batch_size=8)
test_loader_ara = DataLoader(AraBertDataset(test_texts_ara, test_labels_ara, tokenizer), batch_size=8)


arabert_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(label_encoderM.classes_)).to(device)

optimizer = torch.optim.AdamW(arabert_model.parameters(), lr=2e-5)
loss_fn = nn.CrossEntropyLoss()

def train_arabert(model, train_loader, val_loader, epochs=3):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(device)
            optimizer.zero_grad()
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} - Training loss: {total_loss / len(train_loader):.4f}")

        # Evaluate on validation set
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
                labels = batch['labels'].to(device)
                outputs = model(**inputs)
                preds = outputs.logits.argmax(dim=1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())
        print(classification_report(all_labels, all_preds, target_names=label_encoderM.classes_))
    return model

trained_arabert = train_arabert(arabert_model, train_loader_ara, dev_loader_ara, epochs=10)
