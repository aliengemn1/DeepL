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
import csv

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

# Example: SVM Model
vectorizer = TfidfVectorizer(max_features=10000)
X_train_tfidf = vectorizer.fit_transform(M_X_train)
X_test_tfidf = vectorizer.transform(M_X_test)

svm_model = SVC(kernel='linear', C=1)
svm_model.fit(X_train_tfidf, M_y_train)
svm_preds = svm_model.predict(X_test_tfidf)

print("🔍 SVM Results")
print(classification_report(M_y_test, svm_preds, target_names=label_encoderM.classes_))

# Build vocabulary for CNN/LSTM/BiLSTM

def build_vocab(texts, min_freq=1):
    counter = Counter()
    for text in texts:
        counter.update(text.split())
    vocab = {word: idx + 2 for idx, (word, count) in enumerate(counter.items()) if count >= min_freq}
    vocab["<PAD>"] = 0
    vocab["<UNK>"] = 1
    return vocab

vocab = build_vocab(M_X_train.tolist())

# CNN Model
class CNNModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_classes=len(label_encoderM.classes_)):
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
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_classes=len(label_encoderM.classes_)):
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
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_classes=len(label_encoderM.classes_)):
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
train_dataset = TextDataset(M_X_train.tolist(), M_y_train, vocab)
test_dataset = TextDataset(M_X_test.tolist(), M_y_test, vocab)
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
    print(classification_report(all_labels, all_preds, target_names=label_encoderM.classes_))

# Store model scores
model_scores = {}
metrics = ["Accuracy", "Precision", "Recall", "F1-score"]

# Helper to compute metrics
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
svm_scores = compute_metrics(M_y_test, svm_preds)
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

# Arabert for Mawqif

model_name = "aubmindlab/bert-base-arabertv02"
tokenizer = AutoTokenizer.from_pretrained(model_name)
arabert_prep = ArabertPreprocessor(model_name=model_name)

X_train_ara = [arabert_prep.preprocess(t) for t in M_X_train]
X_test_ara = [arabert_prep.preprocess(t) for t in M_X_test]

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

train_loader_ara = DataLoader(ArabertDataset(X_train_ara, M_y_train, tokenizer), batch_size=8, shuffle=True)
test_loader_ara = DataLoader(ArabertDataset(X_test_ara, M_y_test, tokenizer), batch_size=8)

arabert_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(label_encoderM.classes_)).to(device)
optimizer = torch.optim.AdamW(arabert_model.parameters(), lr=2e-5)

def train_arabert(model, train_loader, test_loader, epochs=3):
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
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
    # Evaluation
    model.eval()
    arabert_preds = []
    arabert_true = []
    with torch.no_grad():
        for batch in test_loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(device)
            outputs = model(**inputs)
            preds = outputs.logits.argmax(dim=1)
            arabert_preds.extend(preds.cpu().tolist())
            arabert_true.extend(labels.cpu().tolist())
    print("\nAraBERT Results")
    print(classification_report(arabert_true, arabert_preds, target_names=label_encoderM.classes_))
    return arabert_true, arabert_preds

arabert_true, arabert_preds = train_arabert(arabert_model, train_loader_ara, test_loader_ara, epochs=3)
model_scores["AraBERT"] = compute_metrics(arabert_true, arabert_preds)

print("\nAraBERT Results")
from sklearn.metrics import classification_report
print(classification_report(arabert_true, arabert_preds, target_names=label_encoderM.classes_))

# Print macro and weighted averages
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
plt.title("Model Performance Comparison (Mawqif) - Macro & Weighted Averages")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("mawqif_model_comparison.png")
plt.show()

# Export results to CSV and Markdown
with open("mawqif_results.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Model"] + [m[1] for m in metrics_to_plot])
    for model in labels:
        writer.writerow([model] + [f"{model_scores[model][m[0]]:.4f}" for m in metrics_to_plot])

with open("mawqif_results.md", "w") as mdfile:
    mdfile.write("| Model | " + " | ".join([m[1] for m in metrics_to_plot]) + " |\n")
    mdfile.write("|" + "---|" * (len(metrics_to_plot)+1) + "\n")
    for model in labels:
        mdfile.write(f"| {model} " + " ".join([f"| {model_scores[model][m[0]]:.4f}" for m in metrics_to_plot]) + " |\n")

"""
Documentation:
- This script loads the Mawqif dataset, preprocesses the text, and trains/evaluates SVM, CNN, LSTM, BiLSTM, and Arabert models.
- It computes Accuracy, Precision, Recall, and F1-score for each model and visualizes the results in a bar chart.
- The chart is saved as 'mawqif_model_comparison.png'.
""" 