import sys, os

# Lấy đường dẫn tuyệt đối tới Sentiment_Project
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../Sentiment_Project"))
sys.path.append(BASE_DIR)

from src.embedding_loader import load_custom_embedding
from src.model import CNN_LSTM_Model

import torch
import pickle
# from Sentiment_Project.src.embedding_loader import load_custom_embedding
# from Sentiment_Project.src.model import CNN_LSTM_Model

class SentimentPipeline: 
    def __init__(self, base_path="../Sentiment_Project"):
        self.base_path = base_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.word2idx = self.load_model()

    def load_model(self):
        cache_dir = os.path.join(self.base_path, "result")
        os.makedirs(cache_dir, exist_ok=True)

        word2idx_path = os.path.join(cache_dir, "word2idx.pkl")
        embedding_matrix_path = os.path.join(cache_dir, "embedding_matrix.pt")
        model_path = os.path.join(cache_dir, "model.pth")

        if os.path.exists(word2idx_path) and os.path.exists(embedding_matrix_path):
            # Load từ cache
            with open(word2idx_path, "rb") as f:
                word2idx = pickle.load(f)
            embedding_matrix = torch.load(embedding_matrix_path, map_location=self.device)
        else:
            # Build từ imdb.vocab + imdbEr.txt
            vocab_path = os.path.join(self.base_path, "aclImdb", "imdb.vocab")
            embed_path = os.path.join(self.base_path, "aclImdb", "imdbEr.txt")
            word2idx, embedding_matrix = load_custom_embedding(vocab_path, embed_path)
            embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float32)

            # Save cache
            with open(word2idx_path, "wb") as f:
                pickle.dump(word2idx, f)
            torch.save(embedding_matrix, embedding_matrix_path)

        # Load model
        model = CNN_LSTM_Model(embedding_matrix)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()

        return model, word2idx

    def preprocess(self, text, max_len=200):
        tokens = text.lower().split()
        indices = [self.word2idx.get(tok, self.word2idx.get("<UNK>", 0)) for tok in tokens]
        if len(indices) < max_len:
            indices += [0] * (max_len - len(indices))
        else:
            indices = indices[:max_len]
        return torch.tensor([indices], dtype=torch.long).to(self.device)

    def predict(self, text):
        x = self.preprocess(text)
        with torch.no_grad():
            output = self.model(x)
        prob = output.item()
        return {"label": "Positive" if prob >= 0.5 else "Negative", "score": prob}
