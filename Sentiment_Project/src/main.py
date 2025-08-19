from config import *
from data_loader import load_imdb_data
from embedding_loader import load_custom_embedding
from model import CNN_LSTM_Model
from train import train_model
from evaluate import *
from grid_search import run_grid_search
import torch
from utils import encode_dataset
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import CNN_LSTM_Model
from train import train_model

# Save path
save_dir = "Sentiment-Project/result"
os.makedirs(save_dir, exist_ok=True)  # tạo folder nếu chưa có
save_path = os.path.join(save_dir, "model.pth")

# Load dữ liệu
print("Loading dataset...")
train_df = load_imdb_data("Sentiment-Project/aclImdb/train")
test_df = load_imdb_data("Sentiment-Project/aclImdb/test")

# Load từ điển và embedding
print("Loading embedding...")
word2idx, embedding_matrix = load_custom_embedding(
    "Sentiment-Project/aclImdb/imdb.vocab",
    "Sentiment-Project/aclImdb/imdbEr.txt"
)

# Bổ sung token đặc biệt nếu chưa có
if "<PAD>" not in word2idx:
    word2idx["<PAD>"] = len(word2idx)
    import numpy as np
    pad_vec = np.zeros((1, embedding_matrix.shape[1]))
    embedding_matrix = np.vstack([embedding_matrix, pad_vec])

if "<UNK>" not in word2idx:
    word2idx["<UNK>"] = len(word2idx)
    import numpy as np
    unk_vec = np.random.normal(scale=0.6, size=(1, embedding_matrix.shape[1]))
    embedding_matrix = np.vstack([embedding_matrix, unk_vec])

# Tiền xử lý & padding
print("Preprocessing...")
train_data, train_labels = encode_dataset(train_df, word2idx, MAX_LEN)
test_data, test_labels = encode_dataset(test_df, word2idx, MAX_LEN)

MODE = "train"  # "train"/"grid"

if MODE == "train":
    print("Training model...")
    model = CNN_LSTM_Model(embedding_matrix, HIDDEN_DIM, DROPOUT, NUM_CLASSES).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss()

    model = train_model(
        model,
        train_data, train_labels,
        test_data, test_labels,
        optimizer, criterion,
        BATCH_SIZE, EPOCHS
    )
    torch.save(model.state_dict(), save_path)
    evaluate_model(model, test_data, test_labels, DEVICE)
    
elif MODE == "grid":
    print("Starting Grid Search...")
    run_grid_search(train_data, train_labels, test_data, test_labels, embedding_matrix)

else:
    print("Invalid MODE. Please set MODE to 'train' or 'grid'")


    