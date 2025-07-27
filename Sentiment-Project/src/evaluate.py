import torch
from sklearn.metrics import accuracy_score

def evaluate_model(model, test_data, test_labels, device):
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(test_data).to(device)
        labels = torch.tensor(test_labels).to(device)
        
        outputs = model(inputs).squeeze()
        preds = (outputs >= 0.5).int()

        acc = accuracy_score(labels.cpu(), preds.cpu())
    print(f"Accuracy: {acc * 100:.2f}%")
