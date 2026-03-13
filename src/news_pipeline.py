import os
import pandas as pd
from news_fetcher import fetch_news
from gdelt_fetcher import fetch_gdelt
from sentiment import load_sentiment_model, analyze_sentiment
from event_detector import detect_events


def collect_all_news() -> pd.DataFrame:
    news1 = fetch_news()
    news2 = fetch_gdelt()
    news = pd.concat([news1, news2], ignore_index=True)
    return news.drop_duplicates(subset=["title"])


def run_news_pipeline() -> pd.DataFrame:
    print("Collecting news articles...")
    news = collect_all_news()
    print(f"  {len(news)} articles collected.")

    print("Running sentiment analysis...")
    model = load_sentiment_model()
    news = analyze_sentiment(news, model)

    print("Detecting geopolitical events...")
    news = detect_events(news)

    os.makedirs("data", exist_ok=True)
    out_path = "data/news_with_sentiment.csv"
    news.to_csv(out_path, index=False)
    print(f"News pipeline complete → {out_path}")
    return news