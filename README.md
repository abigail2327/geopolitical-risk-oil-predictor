# 🛢 Oil Risk Intelligence Platform

A Bloomberg-style real-time dashboard that combines geopolitical news analysis, NLP sentiment scoring, and machine learning to monitor and predict oil market volatility.

Built with Python, Streamlit, HuggingFace Transformers, and Plotly.

---

## What It Does

The platform pulls live data from two sources — WTI crude oil futures (Yahoo Finance) and geopolitical news (NewsAPI + GDELT) — runs each headline through a sentiment model, flags high-risk events by keyword, and trains a Random Forest regressor to predict near-term volatility from news signals.

The dashboard updates every 60 seconds and includes an interactive 3D globe showing geopolitical risk by country.

---

## Dashboard

| Section | Description |
|---|---|
| KPI Bar | Live WTI price, 5-day volatility, average sentiment, article count |
| Price Chart | Live crude oil price at 1-hour intervals (last 5 days) |
| Volatility Chart | 5-day rolling log-return standard deviation |
| 3D Globe | Interactive geopolitical risk map — drag to rotate, scroll to zoom, hover for event details |
| Sentiment Map | Choropleth of average news sentiment by country |
| News Feed | Latest scored headlines with country tags and sentiment values |
| AI Predictor | Predict volatility using news volume + sentiment score sliders |
| Event Alerts | Articles flagged by geopolitical keyword matching |

---

## Tech Stack

| Layer | Tools |
|---|---|
| Dashboard | Streamlit, Plotly, custom HTML/JS canvas globe |
| Data — Oil | yfinance (Yahoo Finance WTI futures) |
| Data — News | NewsAPI, GDELT 2.0 |
| NLP | HuggingFace Transformers (DistilBERT SST-2) |
| ML | scikit-learn RandomForestRegressor |
| Feature Engineering | pandas, NumPy (log returns, rolling volatility) |
| Location Extraction | GeoText |

---

## Project Structure

```
src/
├── dashboard.py          # Streamlit dashboard
├── main.py               # Pipeline runner (run once before dashboard)
├── data_loader.py        # Downloads WTI oil futures from Yahoo Finance
├── features.py           # Log-return volatility calculation
├── news_fetcher.py       # NewsAPI fetcher (requires API key)
├── gdelt_fetcher.py      # GDELT 2.0 fetcher (free, no key)
├── news_pipeline.py      # Orchestrates fetch → sentiment → save
├── sentiment.py          # HuggingFace sentiment scoring
├── event_detector.py     # Keyword-based geopolitical event detection
├── location_extractor.py # Country name extraction from headlines
├── news_map.py           # Batch location extraction for DataFrames
├── dataset_builder.py    # Merges oil + news data into ML dataset
├── model.py              # Trains and evaluates volatility predictor
├── live_data.py          # Fetches live oil price (1h intervals)
├── visualize.py          # Matplotlib charts for pipeline output
├── requirements.txt
└── .env.example
```

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/your-username/geopolitical-risk-oil-predictor.git
cd geopolitical-risk-oil-predictor
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
.venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

The HuggingFace sentiment model (~250MB) downloads automatically on first run.

### 4. Add your API key

```bash
cp .env.example .env
```

Edit `.env`:

```
NEWS_API_KEY=your_key_here
```

Get a free key at [newsapi.org](https://newsapi.org). Free tier allows 100 requests/day.

### 5. Build the data pipeline

```bash
cd src
python main.py
```

This will:
- Download WTI crude oil futures from Yahoo Finance (2020 to present)
- Calculate 5-day rolling log-return volatility
- Fetch ~850 geopolitical news articles from NewsAPI and GDELT
- Score each headline with sentiment analysis
- Flag articles containing conflict/supply keywords
- Merge oil and news data into an ML-ready dataset
- Train and evaluate the volatility prediction model

Takes approximately 5–10 minutes on first run.

### 6. Launch the dashboard

```bash
streamlit run dashboard.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## How the ML Model Works

The volatility predictor uses two features derived from the news pipeline:

- **news_count** — number of oil-related articles published on a given day (proxy for market attention and uncertainty)
- **avg_sentiment** — mean signed sentiment score across all articles for that day (negative = bearish/risk-on)

These are joined to the corresponding day's 5-day rolling volatility value and used to train a Random Forest regressor. The model is evaluated on a held-out 20% test set using Mean Absolute Error.

The interactive predictor in the dashboard lets you move the sliders to simulate different news environments and see the model's volatility forecast in real time.

---

## Data Sources

| Source | What | Cost |
|---|---|---|
| [Yahoo Finance](https://finance.yahoo.com) | WTI crude oil futures (CL=F) | Free |
| [NewsAPI](https://newsapi.org) | English-language oil/geopolitics headlines | Free (100 req/day) |
| [GDELT 2.0](https://www.gdeltproject.org) | Global event news feed | Free |
| [HuggingFace](https://huggingface.co) | DistilBERT sentiment model | Free |

---

## Roadmap

- [ ] Replace DistilBERT with [FinBERT](https://huggingface.co/ProsusAI/finbert) for finance-specific sentiment
- [ ] Add lagged volatility and VIX as model features
- [ ] Use time-based train/test split to prevent data leakage
- [ ] Wire globe hotspot sizes to live sentiment scores
- [ ] Automate news pipeline refresh with APScheduler
- [ ] Deploy to Streamlit Cloud

---

## Environment Variables

| Variable | Description | Required |
|---|---|---|
| `NEWS_API_KEY` | Your NewsAPI key | Yes |

Never commit your `.env` file. It is listed in `.gitignore`.

---

## License

MIT
