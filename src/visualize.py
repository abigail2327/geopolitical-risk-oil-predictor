import matplotlib.pyplot as plt
import pandas as pd

PLOT_STYLE = {
    "figure.facecolor": "#05080f",
    "axes.facecolor": "#0d1a2e",
    "axes.edgecolor": "#1a3a6a",
    "axes.labelcolor": "#7aabdf",
    "xtick.color": "#4a6a8a",
    "ytick.color": "#4a6a8a",
    "text.color": "#c8dff0",
    "grid.color": "#1a3a6a",
    "grid.linestyle": "--",
    "grid.alpha": 0.4,
}


def _apply_style() -> None:
    plt.rcParams.update(PLOT_STYLE)


def plot_oil_prices(df: pd.DataFrame) -> None:
    _apply_style()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df.index, df["Close"], color="#1D9E75", linewidth=1.2)
    ax.set_title("WTI Crude Oil — Closing Price", fontsize=13, pad=12)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.grid(True)
    fig.tight_layout()
    plt.show()


def plot_volatility(df: pd.DataFrame) -> None:
    _apply_style()
    df_recent = df[df.index >= "2021-01-01"]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df_recent.index, df_recent["volatility"], color="#D85A30", linewidth=1.2)
    ax.set_title("Oil Price Volatility — Past 5 Years (5d rolling σ)", fontsize=13, pad=12)
    ax.set_xlabel("Date")
    ax.set_ylabel("Volatility (σ)")
    ax.grid(True)
    fig.tight_layout()
    plt.show()