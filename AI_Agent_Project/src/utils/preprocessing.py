import re

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)  # bỏ ký tự đặc biệt
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize(text: str, word2idx: dict, max_len: int = 200):
    tokens = []
    for word in text.split():
        tokens.append(word2idx.get(word, word2idx.get("<UNK>", 0)))
    if len(tokens) < max_len:
        tokens += [0] * (max_len - len(tokens))
    else:
        tokens = tokens[:max_len]
    return tokens
