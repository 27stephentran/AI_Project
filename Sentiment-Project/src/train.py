import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from model import CNN_LSTM_Model

def train_model(model, X_train, y_train, X_test, y_test, batch_size, lr, epochs):
    train_data = TensorDataset(X_train, y_train)
    test_data = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(model.device), labels.to(model.device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_loader):.4f}")

    # Evaluation
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(model.device)
            outputs = model(inputs)
            preds = torch.round(outputs.squeeze().cpu())
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    print(f"Test Accuracy: {acc:.4f}")

def grid_search(X_train, y_train, X_test, y_test, embedding_matrix, param_grid, batch_size, epochs):
    for lr in param_grid['lr']:
        for hidden_dim in param_grid['hidden_dim']:
            for dropout in param_grid['dropout']:
                print(f"\nTraining with lr={lr}, hidden_dim={hidden_dim}, dropout={dropout}")
                model = CNN_LSTM_Model(embedding_matrix, hidden_dim, dropout).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                train_model(model, X_train, y_train, X_test, y_test, batch_size, lr, epochs)

