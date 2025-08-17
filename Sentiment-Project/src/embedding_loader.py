import numpy as np

def load_custom_embedding(vocab_path, embed_path):
    # Load vocab
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = [line.strip() for line in f.readlines()]
    word2idx = {word: idx for idx, word in enumerate(vocab)}

    # Load embedding (IMDb imdbEr.txt có mỗi dòng là 1 số → cần reshape)
    embeddings = []
    with open(embed_path, "r", encoding="utf-8") as f:
        for line in f:
            vals = line.strip().split()
            # Nếu chỉ có 1 giá trị mỗi dòng (IMDb gốc), thì reshape lại
            vec = np.array(vals, dtype=np.float32)
            embeddings.append(vec)

    embedding_matrix = np.array(embeddings)

    # Nếu matrix bị 1D (num_words,), reshape thành (num_words, 1)
    if len(embedding_matrix.shape) == 1:
        embedding_matrix = embedding_matrix.reshape(-1, 1)

    return word2idx, embedding_matrix
