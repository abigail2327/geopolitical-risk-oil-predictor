from geotext import GeoText


def extract_country(text: str) -> str | None:
    if not isinstance(text, str):
        return None
    places = GeoText(text)
    return places.countries[0] if places.countries else None