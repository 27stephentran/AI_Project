import torch
import torch.nn as nn
import pickle
import os

# CNN+LSTM Model (phải giống kiến trúc khi train)
class SentimentModel(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim=128, num_classes=1, dropout=0.5):
        super(SentimentModel, self).__init__()
        vocab_size, embedding_dim = embedding_matrix.size()
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
        self.conv = nn.Conv1d(in_channels=embedding_dim, out_channels=128, kernel_size=5)
        self.lstm = nn.LSTM(128, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)  # (batch, embed_dim, seq_len)
        x = torch.relu(self.conv(x)).permute(0, 2, 1)  # (batch, seq_len, channels)
        _, (hidden, _) = self.lstm(x)
        x = self.dropout(hidden[-1])
        x = self.fc(x)
        return self.sigmoid(x)

class SentimentPipeline:
    def __init__(self, base_path="../Sentiment_Project/result"):
        self.base_path = base_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.word2idx = self.load_model()

    def load_model(self):
        # Load word2idx
        with open(os.path.join(self.base_path, "word2idx.pkl"), "rb") as f:
            word2idx = pickle.load(f)

        # Load embedding matrix
        embedding_matrix = torch.load(os.path.join(self.base_path, "embedding_matrix.pt"))

        # Load model
        model = SentimentModel(embedding_matrix)
        model.load_state_dict(torch.load(os.path.join(self.base_path, "model.pth"), map_location=self.device))
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
