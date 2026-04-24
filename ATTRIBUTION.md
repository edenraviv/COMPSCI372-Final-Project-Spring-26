# Attribution

This document records all outside resources used in this project: AI assistance, third-party libraries, datasets, and reference material. Authors: Sabina Eraso and Eden Raviv (Duke COMPSCI 372, Spring 2026).

## 1. AI-Generated Code

We used large language models (primarily **ChatGPT / OpenAI GPT-4 and GPT-5** and **Anthropic Claude (Sonnet 4.5 and Opus 4.6 / 4.7)** via Claude Code) as coding assistants throughout the project. All AI-generated code was reviewed, edited, and integrated by the authors; no file was accepted verbatim without inspection and modification.

### Where AI assistance was used

| Area | File(s) | Nature of AI assistance |
|---|---|---|
| Kalshi API signing & pagination scaffolding | [src/kalshi_client.py](src/kalshi_client.py) | Starter code for RSA-PSS request signing (based on Kalshi's published API docs) and generic cursor-based pagination helper. Authors adapted retry/backoff, endpoint fallback logic, and the historical-vs-recent merge. |
| Feature engineering scaffolding | [src/features.py](src/features.py) | AI suggested idiomatic pandas patterns for strictly backward-looking rolling/shift operations. Authors chose the 35+ features, the 7 feature groups, and the leakage-avoidance rules (exclusion of `total_hours`, `pct_elapsed`, resolution candle). |
| Evaluation & plotting boilerplate | [src/evaluation.py](src/evaluation.py), [src/data_visualization.py](src/data_visualization.py) | Matplotlib styling, SHAP plot wrappers, and the reliability / calibration plot skeleton drafted with AI help. |
| GroupKFold CV scaffolding | [src/models.py](src/models.py) (`hyperparam_search_cv`) | Structure of the fold loop and per-fold scaler placement debugged with AI assistance after we identified a CV leakage bug. |
| Documentation polish | [README.md](README.md), [SETUP.md](SETUP.md), docstrings throughout `src/` | All technical claims, numbers, and takeaways were written/verified by the authors.

## 2. External Libraries

All third-party Python packages are pinned in [requirements.txt](requirements.txt). License and purpose listed below.

| Package | Version constraint | License | Purpose |
|---|---|---|---|
| [pandas](https://pandas.pydata.org/) | latest | BSD-3-Clause | DataFrame manipulation, time-series handling |
| [numpy](https://numpy.org/) | latest | BSD-3-Clause | Numerical arrays, vectorized math |
| [scikit-learn](https://scikit-learn.org/) | latest | BSD-3-Clause | `StandardScaler`, `GroupKFold`, metrics (`log_loss`, `roc_auc_score`, `brier_score_loss`) |
| [LightGBM](https://lightgbm.readthedocs.io/) | latest | MIT | Primary gradient-boosted model |
| [XGBoost](https://xgboost.readthedocs.io/) | latest | Apache-2.0 | Ensemble partner to LightGBM |
| [SHAP](https://shap.readthedocs.io/) | latest | MIT | Feature-importance interpretability plots |
| [matplotlib](https://matplotlib.org/) | latest | Matplotlib License (BSD-compatible) | All plots under `plots/` |
| [requests](https://requests.readthedocs.io/) | latest | Apache-2.0 | HTTP client for Kalshi API |
| [cryptography](https://cryptography.io/) | latest | Apache-2.0 / BSD | RSA-PSS signing of Kalshi API requests |
| [python-dotenv](https://github.com/theskumar/python-dotenv) | latest | BSD-3-Clause | Load `apikey.env` credentials |

No library was forked or modified; all are used as published on PyPI.

## 3. Datasets

### Primary data source: Kalshi API

All training and evaluation data come from the **Kalshi prediction-market API** (`https://api.elections.kalshi.com/trade-api/v2`). We pull:

- **Resolved market metadata** via `/historical/markets` and `/markets` endpoints (filtered to political markets).
- **Hourly candlestick price series** via `/series/{series_ticker}/markets/{ticker}/candlesticks` and `/historical/markets/{ticker}/candlesticks`.

The raw and processed pulls are written to:

- [data/raw_market_data.json](data/raw_market_data.json) — raw market metadata
- [data/processed_market_data.json](data/processed_market_data.json) — filtered political markets
- [data/market_timeseries.json](data/market_timeseries.json) — per-market hourly candles

**Usage terms.** The data is accessed under Kalshi's API terms of service using individually-issued API credentials. The data is used solely for this academic project; no raw Kalshi data is redistributed in this repository's data files beyond what is necessary for grading reproducibility, and credentials (`apikey.env`, `kalshi_private_key.pem`) are excluded via `.gitignore`.

### No external labeled datasets were used

Labels (`YES` / `NO` resolution) are derived directly from each market's `result` field as returned by the Kalshi API. We did not use any third-party annotated dataset (e.g., Kaggle, Hugging Face, academic corpora).

## 4. Documentation & Reference Material

The following external references informed the design but no text was copied:

- **Kalshi API documentation** — [trading-api.readme.io](https://trading-api.readme.io/reference/getting-started) — request signing scheme, endpoint paths, pagination model.
- **Kalshi developer blog / examples** — referenced for the RSA-PSS signing convention.
- **LightGBM docs** — [lightgbm.readthedocs.io](https://lightgbm.readthedocs.io/) — parameter semantics (`num_leaves`, `lambda_l2`, `feature_fraction`, GOSS/bagging).
- **XGBoost docs** — [xgboost.readthedocs.io](https://xgboost.readthedocs.io/) — `DMatrix` API and early stopping.
- **scikit-learn user guide** — `GroupKFold` rationale for preventing series-level leakage.
- **SHAP documentation** — [shap.readthedocs.io](https://shap.readthedocs.io/) — `TreeExplainer` usage.
- **Efficient Market Hypothesis** — Wikipedia entry (linked from the README) used only as conceptual framing for the research question.

## 5. Models

Both the LightGBM model ([kalshi_lgbm.txt](kalshi_lgbm.txt)) and the XGBoost model ([kalshi_xgb.json](kalshi_xgb.json)) were trained on the Kalshi candle data. The scaler ([kalshi_scaler.pkl](kalshi_scaler.pkl)) is fit on training data only.

## 6. Course Materials

Conceptual foundations — gradient boosting, cross-validation, calibration, Brier score, log-loss, feature importance — come from Duke **COMPSCI 372: Introduction to Applied Machine Learning** (Spring 2026) lectures and assigned readings. No course code was copied into this repository.

## 7. Author Contributions

See the *Individual Contributions* section of [README.md](README.md) for a breakdown of which author led which part of the pipeline.

---

*If we have inadvertently omitted an attribution, please open an issue on the repository and we will correct it.*
