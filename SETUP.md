# Setup

Step-by-step instructions to install dependencies, configure Kalshi API credentials, and run the full training pipeline.

## 1. Prerequisites

- Python 3.10 or newer (`python3 --version`)
- `pip` and `venv` (bundled with most Python installations)
- A Kalshi account with API access — see [kalshi.com/account/profile](https://kalshi.com/account/profile)

## 2. Clone the repository

```bash
git clone https://github.com/<your-org>/COMPSCI372-Final-Project-Spring-26.git
cd COMPSCI372-Final-Project-Spring-26
```

## 3. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows (PowerShell)
```

## 4. Install Python dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 5. Configure Kalshi API credentials

The API client (`src/kalshi_client.py`) reads credentials from `apikey.env` in the project root.

1. Generate an API key pair in the Kalshi dashboard; download the private key (PEM file) and copy the key ID.
2. Save the private key somewhere safe — e.g. `kalshi_private_key.pem` in the project root.
3. Create `apikey.env` in the project root with the following fields:

   ```env
   API_KEY_ID=your-kalshi-api-key-id
   PRIVATE_KEY_PATH=kalshi_private_key.pem
   BASE_URL=https://api.elections.kalshi.com/trade-api/v2
   ```

4. Make sure `apikey.env` and the `.pem` file are listed in `.gitignore` — **never commit them**.

## 6. Run the full pipeline

From the project root, run the main entry point:

```bash
python src/main.py
```

This runs every stage in order:

1. **Ensure `data/` exists** — creates the directory if it is missing.
2. **`populate_datasets.run()`** — fetches resolved political markets from Kalshi and writes `data/raw_market_data.json` and `data/processed_market_data.json`.
3. **`build_timeseries.run()`** — fetches hourly candles for each market and writes `data/market_timeseries.json`.
4. **`engine.train_pipeline()`** — runs preprocessing, feature engineering, hyperparameter search, training, evaluation, SHAP analysis, backtest, and ablation. Saves `kalshi_lgbm.txt`, `kalshi_xgb.json`, and `kalshi_scaler.pkl` to the project root, and evaluation plots to `plots/`.

Expect the ingestion stages (2 and 3) to take a while — they paginate the Kalshi API and sleep between candle requests to respect rate limits.

## 7. Run individual stages

Each stage is also runnable on its own:

```bash
python src/populate_datasets.py   # stage 2 only
python src/build_timeseries.py    # stage 3 only
python src/engine.py              # stage 4 only (requires data/ already populated)
```

## 8. Inference on a live market

After training, predict a live market from the command line using `src/predict.py`. It takes two positional arguments — the Kalshi series ticker and the full market ticker:

```bash
python src/predict.py <series_ticker> <market_ticker>
```

Example:

```bash
python3 src/predict.py KXTRUMPMENTION KXTRUMPMENTION-26FEB19-AFRI
```

The script fetches the latest candles from the Kalshi API, applies the saved scaler and feature pipeline, runs the LightGBM + XGBoost ensemble, and prints the current YES probability and signal. Models and scaler are loaded automatically from the artifacts produced in step 6 (`kalshi_lgbm.txt`, `kalshi_xgb.json`, `kalshi_scaler.pkl`).

## Troubleshooting

- **`ModuleNotFoundError`** — confirm the virtual environment is activated and dependencies from step 4 are installed.
- **`FileNotFoundError: apikey.env`** — file must live at the project root, not inside `src/`.
- **Kalshi 401 / 429 responses** — double-check `API_KEY_ID`, confirm the private key path resolves, and slow down requests if rate-limited (the client already retries with exponential backoff).
- **LightGBM/XGBoost install errors on macOS** — install the OpenMP runtime: `brew install libomp`.
