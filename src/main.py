# Copyright (c) 2026 Abigail Da Costa
# Licensed under the MIT License — see LICENSE file for details.
from data_loader import download_oil_prices, save_data
from features import calculate_volatility
from visualize import plot_oil_prices, plot_volatility
from news_pipeline import run_news_pipeline
from dataset_builder import build_dataset
from model import train_model


def main() -> None:
    print("\n=== Oil Data Pipeline ===")
    oil_data = download_oil_prices()
    save_data(oil_data, "oil_prices_raw.csv")
    oil_data = calculate_volatility(oil_data)
    save_data(oil_data, "oil_prices_processed.csv")
    plot_oil_prices(oil_data)
    plot_volatility(oil_data)

    print("\n=== News Pipeline ===")
    run_news_pipeline()

    print("\n=== Dataset Builder ===")
    dataset = build_dataset()

    print("\n=== Model Training ===")
    train_model(dataset)

    print("\n✓ All pipelines complete. Run `streamlit run dashboard.py` to launch.")


if __name__ == "__main__":
    main()