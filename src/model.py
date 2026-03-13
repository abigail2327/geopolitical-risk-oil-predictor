import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


def train_model(dataset: pd.DataFrame) -> RandomForestRegressor:
    features = dataset[["news_count", "avg_sentiment"]]
    target = dataset["volatility"]

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    mae = mean_absolute_error(y_test, model.predict(X_test))
    print(f"Model trained — MAE: {mae:.6f}")
    return model