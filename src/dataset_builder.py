import os
import pandas as pd


def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    oil = pd.read_csv("data/oil_prices_processed.csv", index_col=0)
    oil.index.name = None
    oil = oil.reset_index().rename(columns={"index": "Date"})

    if oil.columns[0] != "Date":
        oil = oil.rename(columns={oil.columns[0]: "Date"})

    news = pd.read_csv("data/news_with_sentiment.csv")
    return oil, news


def prepare_news_features(news: pd.DataFrame) -> pd.DataFrame:
    news = news.copy()
    news["date"] = pd.to_datetime(news["date"], format="mixed", utc=True).dt.date
    daily = news.groupby("date").agg(
        news_count=("title", "count"),
        avg_sentiment=("sentiment", "mean"),
    ).reset_index()
    return daily


def merge_datasets(oil: pd.DataFrame, news_features: pd.DataFrame) -> pd.DataFrame:
    oil = oil.copy()

    if oil.columns.duplicated().any():
        oil = oil.loc[:, ~oil.columns.duplicated()]

    oil["Date"] = pd.to_datetime(oil["Date"], format="mixed", utc=False).dt.date

    merged = pd.merge(oil, news_features, left_on="Date", right_on="date", how="left")
    merged["news_count"] = merged["news_count"].fillna(0)
    merged["avg_sentiment"] = merged["avg_sentiment"].fillna(0)
    return merged


def build_dataset() -> pd.DataFrame:
    oil, news = load_datasets()
    news_features = prepare_news_features(news)
    dataset = merge_datasets(oil, news_features)
    dataset = dataset.dropna(subset=["volatility"])

    os.makedirs("data", exist_ok=True)
    out_path = "data/ml_dataset.csv"
    dataset.to_csv(out_path, index=False)
    print(f"ML dataset saved → {out_path}")
    return dataset