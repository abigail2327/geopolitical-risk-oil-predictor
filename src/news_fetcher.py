import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("NEWS_API_KEY")

QUERIES = [
    "oil supply", "OPEC production", "Iran oil sanctions",
    "Russia oil exports", "pipeline attack", "Middle East oil",
    "energy crisis", "Strait of Hormuz", "oil markets", "oil prices",
]


def fetch_news() -> pd.DataFrame:
    if not API_KEY:
        raise EnvironmentError("NEWS_API_KEY is not set. Add it to your .env file.")

    all_articles = []
    for query in QUERIES:
        url = (
            f"https://newsapi.org/v2/everything"
            f"?q={query}&language=en&pageSize=100&apiKey={API_KEY}"
        )
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        for article in data.get("articles", []):
            all_articles.append({
                "title": article["title"],
                "date": article["publishedAt"],
            })

    return pd.DataFrame(all_articles).drop_duplicates(subset=["title"])