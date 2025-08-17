import torch
from sklearn.metrics import accuracy_score

def evaluate(model, test_data, test_labels, device):
    model.eval()

    # Nếu chưa phải tensor thì convert, nếu rồi thì chỉ đưa lên device
    if not isinstance(test_data, torch.Tensor):
        inputs = torch.tensor(test_data, dtype=torch.float32).to(device)
    else:
        inputs = test_data.to(device)

    if not isinstance(test_labels, torch.Tensor):
        labels = torch.tensor(test_labels, dtype=torch.float32).to(device)
    else:
        labels = test_labels.to(device)

    with torch.no_grad():
        outputs = model(inputs)
        preds = torch.round(outputs.squeeze())

    acc = accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())
    print(f"Test Accuracy: {acc:.4f}")
    return acc

def evaluate_model(model, test_data, test_labels, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    dataset = torch.utils.data.TensorDataset(test_data, test_labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    print(f"Test Accuracy: {acc:.4f}")
    return acc