import torch
from Sentiment_Project.src.model import CNN_LSTM_Model
from .config import DEVICE, SENTIMENT_MODEL_PATH

def load_sentiment_model(embedding_matrix, hidden_dim=128, dropout=0.5):
    model = CNN_LSTM_Model(embedding_matrix, hidden_dim, dropout)
    model.load_state_dict(torch.load(SENTIMENT_MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

def predict_sentiment(model, text, tokenizer, max_len=200):
    tokens = tokenizer(text)
    if len(tokens) < max_len:
        tokens += [0] * (max_len - len(tokens))
    else:
        tokens = tokens[:max_len]
    input_tensor = torch.tensor([tokens]).to(DEVICE)
    with torch.no_grad():
        output = model(input_tensor).squeeze().item()
        return "Positive" if output >= 0.5 else "Negative"
