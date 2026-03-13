import pandas as pd
from transformers import pipeline


def load_sentiment_model():
    return pipeline("sentiment-analysis")


def analyze_sentiment(df: pd.DataFrame, model) -> pd.DataFrame:
    df = df.copy()
    sentiments = []
    for title in df["title"]:
        try:
            result = model(title[:512])[0]
            score = result["score"]
            if result["label"] == "NEGATIVE":
                score = -score
            sentiments.append(score)
        except Exception:
            sentiments.append(0.0)
    df["sentiment"] = sentiments
    return df