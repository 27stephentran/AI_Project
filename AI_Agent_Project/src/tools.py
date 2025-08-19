from langchain.agents import tool
from .model_loader_sentiment import load_sentiment_model, predict_sentiment

# Load model khi khởi tạo
sentiment_model = load_sentiment_model()

@tool("sentiment-analysis", return_direct=True)
def sentiment_tool(text_vector: list):
    """Dự đoán cảm xúc từ văn bản đã được vector hóa"""
    result = predict_sentiment(sentiment_model, text_vector)
    return f"Sentiment: {result}"
