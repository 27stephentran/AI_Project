import re
import torch
import matplotlib.pyplot as plt

def clean_text(text):
    text = text.lower()
    text = re.sub(r"<br />", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize(text, word2idx, max_len):
    tokens = text.split()
    idxs = []
    for token in tokens:
        if token in word2idx:
            idxs.append(word2idx[token])
        else:
            idxs.append(word2idx["<UNK>"])

    if len(idxs) < max_len:
        idxs += [word2idx["<PAD>"]] * (max_len - len(idxs))
    else:
        idxs = idxs[:max_len]
    return idxs

def encode_dataset(df, word2idx, max_len):
    inputs = []
    labels = []
    for i, row in df.iterrows():
        text = clean_text(row["text"])
        token_ids = tokenize(text, word2idx, max_len)
        inputs.append(token_ids)
        labels.append(row["label"])
    return torch.tensor(inputs, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def preprocess_and_pad(text, word2idx, max_len):

    cleaned = clean_text(text)
    token_ids = tokenize(cleaned, word2idx, max_len)
    return torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)

def plot_metrics(train_losses, val_losses, title="Loss"):
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Validation")
    plt.title(f"{title} over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
