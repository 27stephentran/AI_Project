def sentiment_label(score: float, threshold: float = 0.5) -> str:
    return "Positive" if score >= threshold else "Negative"
