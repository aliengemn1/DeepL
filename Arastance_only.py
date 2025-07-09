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
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
import nltk
from collections import Counter
import matplotlib.pyplot as plt
import random

nltk.download('stopwords')
arabic_stopwords = set(stopwords.words('arabic'))

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Arastance data
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

label_encoderA = LabelEncoder()
full_df['label'] = label_encoderA.fit_transform(full_df['stance'])

A_X_train, A_X_test, A_y_train, A_y_test = train_test_split(
    full_df['clean_claim'], full_df['label'],
    test_size=0.2, random_state=42, stratify=full_df['label']
)

# Example: SVM Model
vectorizer = TfidfVectorizer(max_features=10000)
X_train_tfidf = vectorizer.fit_transform(A_X_train)
X_test_tfidf = vectorizer.transform(A_X_test)

svm_model = SVC(kernel='linear', C=1)
svm_model.fit(X_train_tfidf, A_y_train)
svm_preds = svm_model.predict(X_test_tfidf)

print("🔍 SVM Results")
print(classification_report(A_y_test, svm_preds, target_names=label_encoderA.classes_))

# Build vocabulary for CNN/LSTM/BiLSTM

def build_vocab(texts, min_freq=1):
    counter = Counter()
    for text in texts:
        counter.update(text.split())
    vocab = {word: idx + 2 for idx, (word, count) in enumerate(counter.items()) if count >= min_freq}
    vocab["<PAD>"] = 0
    vocab["<UNK>"] = 1
    return vocab

vocab = build_vocab(A_X_train.tolist())

# CNN Model
class CNNModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_classes=len(label_encoderA.classes_)):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv = nn.Conv1d(embed_dim, 100, kernel_size=3)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(100, num_classes)

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)
        x = self.pool(torch.relu(self.conv(x))).squeeze(-1)
        return self.fc(x)

# LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_classes=len(label_encoderA.classes_)):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        return self.fc(hidden[-1])

# BiLSTM Model
class BiLSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_classes=len(label_encoderA.classes_)):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        hidden_cat = torch.cat((hidden[0], hidden[1]), dim=-1)
        return self.fc(hidden_cat)

# TextDataset for CNN/LSTM/BiLSTM
class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=50):
        self.texts = [self.encode_text(text, vocab, max_len) for text in texts]
        self.labels = torch.tensor(labels.tolist())
    def encode_text(self, text, vocab, max_len=50):
        tokens = [vocab.get(word, vocab["<UNK>"]) for word in text.split()]
        if len(tokens) < max_len:
            tokens += [vocab["<PAD>"]] * (max_len - len(tokens))
        else:
            tokens = tokens[:max_len]
        return torch.tensor(tokens)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

batch_size = 32
train_dataset = TextDataset(A_X_train.tolist(), A_y_train, vocab)
test_dataset = TextDataset(A_X_test.tolist(), A_y_test, vocab)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Training and evaluation function

def train_and_eval(model, train_loader, test_loader, name):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(5):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
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
    print(classification_report(all_labels, all_preds, target_names=label_encoderA.classes_))

# Store model scores
model_scores = {}
metrics = ["Accuracy", "Precision", "Recall", "F1-score"]

# Helper to compute metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def compute_metrics(y_true, y_pred):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average="macro", zero_division=0),
        'precision_weighted': precision_score(y_true, y_pred, average="weighted", zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average="macro", zero_division=0),
        'recall_weighted': recall_score(y_true, y_pred, average="weighted", zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average="macro", zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }

# SVM
svm_scores = compute_metrics(A_y_test, svm_preds)
model_scores["SVM"] = svm_scores

# CNN
cnn_model = CNNModel(len(vocab)).to(device)
train_and_eval(cnn_model, train_loader, test_loader, "CNN")
cnn_preds = []
cnn_true = []
cnn_model.eval()
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        preds = cnn_model(X_batch.to(device)).argmax(dim=1)
        cnn_preds.extend(preds.cpu().tolist())
        cnn_true.extend(y_batch.tolist())
model_scores["CNN"] = compute_metrics(cnn_true, cnn_preds)

# LSTM
lstm_model = LSTMModel(len(vocab)).to(device)
train_and_eval(lstm_model, train_loader, test_loader, "LSTM")
lstm_preds = []
lstm_true = []
lstm_model.eval()
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        preds = lstm_model(X_batch.to(device)).argmax(dim=1)
        lstm_preds.extend(preds.cpu().tolist())
        lstm_true.extend(y_batch.tolist())
model_scores["LSTM"] = compute_metrics(lstm_true, lstm_preds)

# BiLSTM
bilstm_model = BiLSTMModel(len(vocab)).to(device)
train_and_eval(bilstm_model, train_loader, test_loader, "BiLSTM")
bilstm_preds = []
bilstm_true = []
bilstm_model.eval()
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        preds = bilstm_model(X_batch.to(device)).argmax(dim=1)
        bilstm_preds.extend(preds.cpu().tolist())
        bilstm_true.extend(y_batch.tolist())
model_scores["BiLSTM"] = compute_metrics(bilstm_true, bilstm_preds)

# Arabert for Arastance

# === AraBERT Configuration ===
arabert_config = {
    'model_name': 'aubmindlab/bert-base-arabertv02',  # Try v0.2, v2, or other variants
    'learning_rate': 2e-5,
    'batch_size': 8,
    'epochs': 3,
    'max_seq_length': 128,
    'dropout': 0.1,  # Used if model supports custom dropout
    'weight_decay': 0.01,
    'gradient_accumulation_steps': 1,  # Increase to simulate larger batch size
    'freeze_layers': 0,  # Number of encoder layers to freeze (0 = none)
    'early_stopping_patience': 2,  # Stop if val loss doesn't improve for N epochs
    'random_seed': 42,
}

set_seed = lambda seed: (torch.manual_seed(seed), np.random.seed(seed), random.seed(seed))
set_seed(arabert_config['random_seed'])

model_name = arabert_config['model_name']
tokenizer = AutoTokenizer.from_pretrained(model_name)
arabert_prep = ArabertPreprocessor(model_name=model_name)

X_train_ara = [arabert_prep.preprocess(t) for t in A_X_train]
X_test_ara = [arabert_prep.preprocess(t) for t in A_X_test]

class ArabertDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.encodings = tokenizer(texts, padding=True, truncation=True, max_length=max_len, return_tensors='pt')
        self.labels = torch.tensor(labels.tolist())
    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item
    def __len__(self):
        return len(self.labels)

train_loader_ara = DataLoader(
    ArabertDataset(X_train_ara, A_y_train, tokenizer, max_len=arabert_config['max_seq_length']),
    batch_size=arabert_config['batch_size'], shuffle=True)
test_loader_ara = DataLoader(
    ArabertDataset(X_test_ara, A_y_test, tokenizer, max_len=arabert_config['max_seq_length']),
    batch_size=arabert_config['batch_size'])

arabert_model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=len(label_encoderA.classes_)).to(device)

def freeze_bert_layers(model, num_layers):
    # Freeze embedding and first num_layers encoder layers
    for param in model.bert.embeddings.parameters():
        param.requires_grad = False
    for i, layer in enumerate(model.bert.encoder.layer):
        if i < num_layers:
            for param in layer.parameters():
                param.requires_grad = False

from copy import deepcopy

def train_arabert(model, train_loader, val_loader, config):
    model.to(device)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'])
    loss_fn = nn.CrossEntropyLoss()
    best_val_loss = float('inf')
    patience = 0
    best_model = deepcopy(model.state_dict())
    # Layer freezing
    if config['freeze_layers'] > 0:
        freeze_bert_layers(model, config['freeze_layers'])
    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        for step, batch in enumerate(train_loader):
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(device)
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss / config['gradient_accumulation_steps']
            loss.backward()
            if (step + 1) % config['gradient_accumulation_steps'] == 0 or (step + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()
            total_loss += loss.item() * config['gradient_accumulation_steps']
        print(f"Epoch {epoch+1}/{config['epochs']} - Training loss: {total_loss / len(train_loader):.4f}")
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
                labels = batch['labels'].to(device)
                outputs = model(**inputs, labels=labels)
                val_loss += outputs.loss.item()
        val_loss /= len(val_loader)
        print(f"Validation loss: {val_loss:.4f}")
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= config['early_stopping_patience']:
                print(f"Early stopping at epoch {epoch+1}")
                break
    model.load_state_dict(best_model)
    return model

arabert_model = train_arabert(arabert_model, train_loader_ara, test_loader_ara, arabert_config)
arabert_preds = []
arabert_true = []
arabert_model.eval()
with torch.no_grad():
    for batch in test_loader_ara:
        inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
        labels = batch['labels'].to(device)
        outputs = arabert_model(**inputs)
        preds = outputs.logits.argmax(dim=1)
        arabert_preds.extend(preds.cpu().tolist())
        arabert_true.extend(labels.cpu().tolist())
model_scores["AraBERT"] = compute_metrics(arabert_true, arabert_preds)

print("\nAraBERT Results")
from sklearn.metrics import classification_report
print(classification_report(arabert_true, arabert_preds, target_names=label_encoderA.classes_))

# Print macro and weighted averages for AraBERT
arabert_metrics = model_scores["AraBERT"]
print("Macro avg:")
print(f"  Precision: {arabert_metrics['precision_macro']:.4f}")
print(f"  Recall:    {arabert_metrics['recall_macro']:.4f}")
print(f"  F1-score:  {arabert_metrics['f1_macro']:.4f}")
print("Weighted avg:")
print(f"  Precision: {arabert_metrics['precision_weighted']:.4f}")
print(f"  Recall:    {arabert_metrics['recall_weighted']:.4f}")
print(f"  F1-score:  {arabert_metrics['f1_weighted']:.4f}")

# Update plotting to show macro and weighted F1 for all models
plt.figure(figsize=(12, 7))
bar_width = 0.18
x = np.arange(len(model_scores))
labels = list(model_scores.keys())
metrics_to_plot = [
    ('accuracy', 'Accuracy'),
    ('f1_macro', 'F1 Macro'),
    ('f1_weighted', 'F1 Weighted'),
    ('precision_macro', 'Precision Macro'),
    ('precision_weighted', 'Precision Weighted'),
    ('recall_macro', 'Recall Macro'),
    ('recall_weighted', 'Recall Weighted'),
]
for i, (metric_key, metric_label) in enumerate(metrics_to_plot):
    values = [model_scores[m][metric_key] for m in labels]
    plt.bar(x + i * bar_width, values, width=bar_width, label=metric_label)
plt.xticks(x + bar_width * (len(metrics_to_plot) / 2), labels)
plt.ylim(0, 1)
plt.ylabel("Score")
plt.title("Model Performance Comparison (Arastance) - Macro & Weighted Averages")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("arastance_model_comparison.png")
plt.show()

# Export results to CSV and Markdown
import csv
with open("arastance_results.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Model"] + [m[1] for m in metrics_to_plot])
    for model in labels:
        writer.writerow([model] + [f"{model_scores[model][m[0]]:.4f}" for m in metrics_to_plot])

with open("arastance_results.md", "w") as mdfile:
    mdfile.write("# Arastance Model Results & Comparison\n\n")
    mdfile.write("This report summarizes the performance of SVM, CNN, LSTM, BiLSTM, and AraBERT models on the Arastance dataset. The metrics used are **Accuracy**, **Precision**, **Recall**, and **F1-score**.\n\n")
    mdfile.write("## Model Performance Chart\n\n")
    mdfile.write("![Arastance Model Comparison](arastance_model_comparison.png)\n\n")
    mdfile.write("## Metrics Table\n\n")
    mdfile.write("| Model | Accuracy | F1 Macro | F1 Weighted | Precision Macro | Precision Weighted | Recall Macro | Recall Weighted |\n")
    mdfile.write("|---|---|---|---|---|---|---|---|\n")
    for model in labels:
        mdfile.write(f"| {model} " + " ".join([f"| {model_scores[model][m[0]]:.4f}" for m in metrics_to_plot]) + " |\n")
    mdfile.write("\n---\n\n")
    mdfile.write("## How to Interpret\n\n")
    mdfile.write("- **Accuracy**: Overall correctness of the model.\n")
    mdfile.write("- **Precision**: How many selected items are relevant.\n")
    mdfile.write("- **Recall**: How many relevant items are selected.\n")
    mdfile.write("- **F1-score**: Harmonic mean of precision and recall.\n")
    mdfile.write("- **Macro avg**: Average metric across classes, treating all classes equally.\n")
    mdfile.write("- **Weighted avg**: Average metric across classes, weighted by support (number of true instances per class).\n") 