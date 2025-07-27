import torch
import torch.nn as nn

class CNN_LSTM_Model(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim, dropout):
        super(CNN_LSTM_Model, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vocab_size, embedding_dim = embedding_matrix.shape

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data.copy_(torch.tensor(embedding_matrix))
        self.embedding.weight.requires_grad = False

        self.conv1d = nn.Conv1d(in_channels=embedding_dim, out_channels=128, kernel_size=5, padding=2)
        self.lstm = nn.LSTM(input_size=128, hidden_size=hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)  # (B, L, E)
        x = x.permute(0, 2, 1)  # (B, E, L)
        x = self.conv1d(x)  # (B, C, L)
        x = x.permute(0, 2, 1)  # (B, L, C)
        _, (h_n, _) = self.lstm(x)
        h_n = self.dropout(h_n[-1])
        out = self.fc(h_n)
        return self.sigmoid(out)