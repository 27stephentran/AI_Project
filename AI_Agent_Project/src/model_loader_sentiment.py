import torch
from Sentiment_Project.src.model import CNN_LSTM_Model
from .config import DEVICE, SENTIMENT_MODEL_PATH
from .utils.preprocessing import clean_text, tokenize
from .utils.postprocessing import sentiment_label

def load_sentiment_model(embedding_matrix, hidden_dim=128, dropout=0.5):
    model = CNN_LSTM_Model(embedding_matrix, hidden_dim, dropout)
    model.load_state_dict(torch.load(SENTIMENT_MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

def predict_sentiment(model, text: str, tokenizer, word2idx: dict, max_len=200):
    text = clean_text(text)
    tokens = tokenize(text, word2idx, max_len)
    input_tensor = torch.tensor([tokens]).to(DEVICE)

    with torch.no_grad():
        score = model(input_tensor).squeeze().item()
    return sentiment_label(score)
