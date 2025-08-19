import torch
import torch.nn as nn

class CNN_LSTM_Model(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim, dropout, num_classes):
        super(CNN_LSTM_Model, self).__init__()

        vocab_size, embed_dim = embedding_matrix.shape

        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix, dtype=torch.float32),
            freeze=False  # cho phép fine-tune embedding
        )

        self.conv1 = nn.Conv1d(embed_dim, 128, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(128, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        x = self.embedding(x)                          # (batch, seq_len, embed_dim)
        x = x.permute(0, 2, 1)                         # (batch, embed_dim, seq_len)
        x = torch.relu(self.conv1(x))                  # (batch, 128, seq_len)
        x = x.permute(0, 2, 1)                         # (batch, seq_len, 128)
        _, (h_n, _) = self.lstm(x)                     # h_n: (2, batch, hidden_dim)
        h_n = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)  # (batch, hidden_dim*2)
        x = self.dropout(h_n)
        x = self.fc(x)
        return x
