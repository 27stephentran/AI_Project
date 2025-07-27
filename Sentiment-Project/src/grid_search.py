from model import CNN_LSTM_Model
from train import train_model
from evaluate import evaluate_model
from config import *
import torch
import numpy as np
from itertools import product

def run_grid_search(train_data, train_labels, test_data, test_labels, embedding_matrix):
    learning_rates = [1e-3, 5e-4]
    dropouts = [0.3, 0.5]
    hidden_dims = [64, 128]

    best_acc = 0.0
    best_params = None

    for lr, dropout, hidden_dim in product(learning_rates, dropouts, hidden_dims):
        print(f"Training with LR={lr}, Dropout={dropout}, Hidden={hidden_dim}")
        model = CNN_LSTM_Model(embedding_matrix, hidden_dim, dropout).to(DEVICE)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = torch.nn.BCELoss()

        train_model(model, train_data, train_labels, optimizer, criterion)
        print("Evaluating...")
        acc = evaluate_and_return_acc(model, test_data, test_labels)

        if acc > best_acc:
            best_acc = acc
            best_params = (lr, dropout, hidden_dim)
            torch.save(model.state_dict(), "best_model.pt")
            print(f"✅ New Best Accuracy: {acc*100:.2f}%")

    print(f"\n🏆 Best Accuracy: {best_acc*100:.2f}% with params: LR={best_params[0]}, Dropout={best_params[1]}, Hidden={best_params[2]}")

def evaluate_and_return_acc(model, test_data, test_labels):
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(test_data).to(DEVICE)
        labels = torch.tensor(test_labels).to(DEVICE)

        outputs = model(inputs).squeeze()
        preds = (outputs >= 0.5).int()

        acc = (preds == labels).float().mean().item()
    return acc
