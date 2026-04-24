# Predicting Political Event Markets on Kalshi
Final Project for Duke Compsci 372, Intro to Applied Machine Learning, Sabina Eraso and Eden Raviv

By feature engineering on top of LightGBM and XGBoost models and making real-time predictions on live markets, this project assesses feature significance for predicting the outcomes of ongoing political markets on Kalshi. Among other goals, our project explores whether the [Efficient Market Hypothesis (EMH)](https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://en.wikipedia.org/wiki/Efficient-market_hypothesis&ved=2ahUKEwjCgMKCqIWUAxX0kYkEHaeiBG4QFnoECBoQAQ&usg=AOvVaw2LNGamg7-PD2Bu_K6CrmBd) holds for political markets, or if an "edge" can be found by training a model on past political market time series data.

# What it Does:

Given a live Kalshi political market, the pipeline estimates the probability that the market will resolve **YES** and emits a BUY / SELL / HOLD signal based on how that probability compares to the current market price.

Under the hood it runs end-to-end:

1. **Ingestion** — pulls and filters for every resolved political market from the Kalshi API and saves the hourly candlestick time series for each one ([src/populate_datasets.py](src/populate_datasets.py), [src/build_timeseries.py](src/build_timeseries.py), [src/kalshi_client.py](src/kalshi_client.py)).
2. **Preprocessing** — flattens candles to one row per hour, drops the resolution candle (so the final price never leaks into training), flags missing fields, and clips outliers ([src/candle_pre_processing.py](src/candle_pre_processing.py)).
3. **Feature engineering** — builds 35+ strictly backward-looking features across seven groups: microstructure, momentum, volume, time-to-expiry, rolling stats, level signals, and derived spreads. Anything requiring knowledge of the market's full duration is excluded to keep the model safe from overfitting ([src/features.py](src/features.py)).
4. **Training** — 70/15/15 train/val/test split grouped by `series_id` (so no event series straddles a split), 5-fold GroupKFold CV over three hyperparameter configs, importance-based feature selection, then a final LightGBM + XGBoost ensemble with L2 regularization and early stopping ([src/models.py](src/models.py), [src/engine.py](src/engine.py)).
5. **Evaluation** — log-loss, AUC, and Brier score against two baselines (constant prior, market price), a cumulative-PnL backtest, SHAP interpretability, a feature-group ablation study, error analysis, and inference-time measurement ([src/evaluation.py](src/evaluation.py)). Plots and tables are written to `plots/`.
6. **Live inference** — Given a market ticker, `src/predict.py` fetches the latest candles, applies the saved scaler and feature pipeline, runs the models, and returns the current YES probability plus a signal ([src/inference.py](src/inference.py)).

Tunable constants (paths, split ratios, thresholds, hyperparameter grid, API env) all live in [config/settings.py](config/settings.py) so the pipeline can be reconfigured without touching the source.

# Quick Start: 

# Video Links:

Demo Link: 
Technical Video Link:

# Evaluation:

# Individual Contributions:
When working on the project, we worked almost exclusively in meetings together. Sabina took the lead on file organization within our repository. The most time consuming part of our project was getting from API to static data file in processed form. Eden set up the API key, Sabina built the market data pipeline, and Eden worked to get the candlestick data. After building a full model pipeline together, Sabina addressed overfitting and debugged while Eden played around with some hyperparameter tuning. Finally, Sabina adapted the existing pipeline to allow user-inputs of live market tickers. Eden worked on video presentations and outlines.
