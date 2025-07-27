from config import *
from data_loader import load_imdb_data
from embedding_loader import load_custom_embedding
from model import CNN_LSTM_Model
from train import train_model
from evaluate import evaluate_model
from grid_search import run_grid_search
import torch
from utils import preprocess_and_pad

# Load dữ liệu
print("🔄 Loading dataset...")
train_df = load_imdb_data("aclImdb/train")
test_df = load_imdb_data("aclImdb/test")

# Load từ điển và embedding
print("📥 Loading embedding...")
word2idx, embedding_matrix = load_custom_embedding("aclImdb/imdb.vocab", "aclImdb/imdbEr.txt")

# Tiền xử lý và padding
print("🧹 Preprocessing...")
train_data, train_labels = preprocess_and_pad(train_df, word2idx, MAX_LEN)
test_data, test_labels = preprocess_and_pad(test_df, word2idx, MAX_LEN)


MODE = "train"  # "train"/"grid"

if MODE == "train":
    print("Training model...")
    model = CNN_LSTM_Model(embedding_matrix, HIDDEN_DIM, DROPOUT).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.BCELoss()

    train_model(model, train_data, train_labels, optimizer, criterion)
    evaluate_model(model, test_data, test_labels, DEVICE)

elif MODE == "grid":
    print("Starting Grid Search...")
    run_grid_search(train_data, train_labels, test_data, test_labels, embedding_matrix)

else:
    print("Invalid MODE. Please set MODE to 'train' or 'grid'")
