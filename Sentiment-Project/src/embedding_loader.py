import numpy as np

def load_custom_embedding(vocab_path, embedding_path):
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = [line.strip() for line in f.readlines()]

    word2idx = {word: idx for idx, word in enumerate(vocab)}
    embeddings = np.loadtxt(embedding_path)
    return word2idx, embeddings