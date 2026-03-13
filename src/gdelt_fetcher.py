import requests
import pandas as pd

GDELT_URL = (
    "https://api.gdeltproject.org/api/v2/doc/doc"
    "?query=oil&mode=artlist&maxrecords=250&format=json"
)


def fetch_gdelt() -> pd.DataFrame:
    try:
        response = requests.get(GDELT_URL, timeout=15)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"GDELT fetch failed: {e}")
        return pd.DataFrame(columns=["title", "date"])

    articles = [
        {"title": a["title"], "date": a["seendate"]}
        for a in data.get("articles", [])
    ]
    return pd.DataFrame(articles)