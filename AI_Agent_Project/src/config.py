import torch
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
SENTIMENT_MODEL_PATH = os.path.join(BASE_DIR, "Sentiment_Project", "result", "model.pth")
