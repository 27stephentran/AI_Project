import torch
from torch.utils.data import DataLoader, TensorDataset

def train_model(model, train_data, train_labels, test_data, test_labels,
                optimizer, criterion, batch_size, num_epochs, device="cpu", save_path="model.pth"):
    # Tạo TensorDataset
    train_dataset = TensorDataset(
    train_data.clone().detach().long() if isinstance(train_data, torch.Tensor) else torch.tensor(train_data, dtype=torch.long),
    train_labels.clone().detach().float() if isinstance(train_labels, torch.Tensor) else torch.tensor(train_labels, dtype=torch.float)
    )

    test_dataset = TensorDataset(
        test_data.clone().detach().long() if isinstance(test_data, torch.Tensor) else torch.tensor(test_data, dtype=torch.long),
        test_labels.clone().detach().float() if isinstance(test_labels, torch.Tensor) else torch.tensor(test_labels, dtype=torch.float)
    )

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
    return model
