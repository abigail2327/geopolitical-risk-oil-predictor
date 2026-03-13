import pandas as pd
from geotext import GeoText


def extract_locations(news: pd.DataFrame) -> pd.DataFrame:
    news = news.copy()

    def _get_country(title: str) -> str | None:
        if not isinstance(title, str):
            return None
        places = GeoText(title)
        return places.countries[0] if places.countries else None

    news["country"] = news["title"].apply(_get_country)
    return news