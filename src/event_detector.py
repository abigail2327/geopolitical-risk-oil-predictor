import pandas as pd

KEYWORDS = [
    "war", "attack", "sanctions", "military", "oil supply",
    "pipeline", "conflict", "strike", "explosion", "blockade", "embargo",
]


def detect_events(news_df: pd.DataFrame) -> pd.DataFrame:
    news_df = news_df.copy()
    news_df["event"] = news_df["title"].apply(
        lambda text: int(any(k in text.lower() for k in KEYWORDS))
    )
    return news_df