"""
Kalshi Prediction Market — Full ML Pipeline
============================================
Goal: Given hourly candles for a live market (fetched via API),
      predict the probability it resolves YES.

Training approach:
  - Each candle row = one training sample
  - Label = final outcome of that market (same for all rows in a market)
  - Final (resolution) candle is DROPPED — never available at inference time
  - All features are strictly backward-looking (no future leakage)
  - total_hours and pct_elapsed excluded (require knowing market length upfront)

Rubric items covered:
  [3]  Modular code design
  [3]  Train/val/test split — 70/15/15, documented
  [3]  Training curves — loss + AUC over boosting rounds, saved to plots/
  [3]  Baseline models — constant prior + market price as probability
  [5]  Regularization — L2 (lambda_l2) + early stopping
  [5]  Hyperparameter tuning — 3 configs compared on validation data
  [3]  Normalization — StandardScaler fit on train only
  [3]  Preprocessing — missing value flagging + outlier clipping
  [7]  Preprocessing pipeline — two data quality challenges documented
  [5]  Feature engineering — 35+ derived features across 7 groups
  [5]  Feature selection — importance-based pruning with documented impact
  [10] Original dataset — collected via Kalshi API with custom pipeline
  [3]  Inference time measurement
  [3]  Three evaluation metrics — log-loss, AUC, Brier score
  [7]  Error analysis — failure case breakdown with visualization
  [7]  Ablation study — feature groups + scaler, results table
  [7]  Backtesting — simulation-based evaluation with cumulative PnL
  [7]  SHAP interpretability — summary plot + top feature discussion
  [7]  Time-series ML application — hourly candle prediction
  [7]  Ensemble — LightGBM + XGBoost weighted average

Install:
    pip install lightgbm xgboost pandas numpy scikit-learn shap matplotlib requests
"""

import warnings
import matplotlib
matplotlib.use("Agg")
from data_ingestion import load_raw
from candle_pre_processing import (preprocess, flatten,
                                   drop_resolution_candle, three_way_split)
from features import engineer_features, scale_features, select_features
from models import (HYPERPARAM_CONFIGS, hyperparam_search,
                    train_lgbm, train_xgboost, save_models)
from evaluation import (evaluate_baselines, full_evaluate, backtest,
                        shap_analysis, ablation_study)
from inference import predict_live
from schema import ALL_FEATURE_COLS
from data_visualization import PLOTS_DIR

warnings.filterwarnings("ignore")


def train_pipeline(source):
    """
    FULL TRAINING PIPELINE — full pipeline from raw JSON to saved ensemble model.

    Args:
        source: dict, JSON file path, or directory of JSON files

    Returns:
        lgbm_model, xgb_model, scaler, feature_cols
    """
    print("═" * 55)
    print("  Kalshi ML Pipeline — Training")
    print("═" * 55)

    # 1. Load
    raw = load_raw(source)
    print(f"\nLoaded {len(raw)} markets")

    # 2. Flatten
    df = flatten(raw)
    print(f"Flattened to {len(df)} candle rows")

    # 3. Drop resolution candle (prevent label leakage via final price)
    df = drop_resolution_candle(df)

    # 4. Preprocess (missing value flags + outlier clipping)
    df = preprocess(df)

    # 5. Engineer features (35+ backward-looking features)
    df = engineer_features(df)

    print(df["hours_to_expiry"].describe())

    # Filter to candles with more than 6 hours remaining
    # Forces model to learn from price movement, not temporal certainty
    before = len(df)
    df = df[df["hours_to_expiry"] > 6]
    print(f"Rows after <6h filter: {len(df)} (dropped {before - len(df)})")


    # 6. Drop rows without a label
    df = df.dropna(subset=["label"])
    print(f"Rows after dropping unlabeled: {len(df)}")

    feature_cols = [c for c in ALL_FEATURE_COLS if c in df.columns]
    print(f"Features available: {len(feature_cols)}")
    print(f"Label balance: {df['label'].mean():.2%} YES")

    # 7. Train / val / test split (70/15/15, grouped by market)
    df_train, df_val, df_test = three_way_split(df)

    # 8. Scale (StandardScaler fit on train only)
    X_train, X_val, X_test, scaler = scale_features(
        df_train, df_val, df_test, feature_cols)
    y_train = df_train["label"].values
    y_val   = df_val["label"].values
    y_test  = df_test["label"].values

    # 9. Baselines
    baseline_results = evaluate_baselines(df_train, df_val)

    # 10. Hyperparameter search across 3 configs (saves training curves)
    print("\n── Hyperparameter Search ───────────────────────")
    probe_model, hp_results, best_name = hyperparam_search(
        X_train, y_train, X_val, y_val, feature_cols)

    # 11. Feature selection on probe model
    selected_cols, importance = select_features(
        probe_model, feature_cols, X_val, y_val)

    # 12. Re-scale on selected features only
    X_train_s, X_val_s, X_test_s, scaler_s = scale_features(
        df_train, df_val, df_test, selected_cols)

    # 13. Train final LightGBM with best hyperparams on selected features
    print("\n── Training Final LightGBM ─────────────────────")
    best_cfg = HYPERPARAM_CONFIGS[best_name]
    lgbm_model = train_lgbm(
        X_train_s, y_train, X_val_s, y_val,
        selected_cols, params_override=best_cfg)

    # 14. Train XGBoost
    print("\n── Training XGBoost ────────────────────────────")
    xgb_model = train_xgboost(X_train_s, y_train, X_val_s, y_val)

    # 15. SHAP interpretability
    shap_analysis(lgbm_model, X_val_s, selected_cols)

    # 16. Full evaluation — 3 metrics + inference time + error analysis
    ens_probs, ens_metrics = full_evaluate(
        lgbm_model, xgb_model, df_test,
        X_test_s, y_test, baseline_results)

    # 17. Backtesting simulation
    backtest(df_test, ens_probs, threshold=0.65)

    # 18. Ablation study
    ablation_study(df_train, df_val, feature_cols)

    # 19. Save
    save_models(lgbm_model, xgb_model, scaler_s, selected_cols)

    print(f"\n✓ Done. Plots saved to ./{PLOTS_DIR}/")
    print("  training_curves.png | feature_importance.png | shap_summary.png")
    print("  error_analysis.png  | backtest_pnl.png       | ablation_table.csv")

    return lgbm_model, xgb_model, scaler_s, selected_cols


if __name__ == "__main__":

    # ── TRAINING ──────────────────────────────────────────────────────────────
    # Swap SAMPLE_DATA for your real dataset:
    #   train_pipeline("your_data.json")
    #   train_pipeline("data/")   ← directory of JSON files


    lgbm_model, xgb_model, scaler, feature_cols = train_pipeline("data/market_timeseries.json")

    # ── INFERENCE ─────────────────────────────────────────────────────────────
    # Option A: live ticker → calls Kalshi API automatically
    # result = predict_live("KXPOLITICSMENTION-26FEB18-NATO",
    #                       api_key="YOUR_KEY")

    # Option B: pre-fetched candles (no label field needed)
    live_candles = {
        "KXPOLITICSMENTION-26FEB18-NATO": [
            {"end_period_ts": 1771340400, "open_interest_fp": "0.00",
             "price": {}, "volume_fp": "0.00",
             "yes_ask": {"close_dollars": "0.7000", "high_dollars": "1.0000",
                         "low_dollars": "0.7000", "open_dollars": "1.0000"},
             "yes_bid": {"close_dollars": "0.1000", "high_dollars": "0.1000",
                         "low_dollars": "0.0100", "open_dollars": "0.0100"},
             "ds": "2026-02-17 15:00:00", "close": 0.7, "high": 1.0,
             "low": 0.7, "market_id": "KXPOLITICSMENTION-26FEB18-NATO"},
            {"end_period_ts": 1771344000, "open_interest_fp": "1057.00",
             "price": {"close_dollars": "0.9900", "mean_dollars": "0.9778"},
             "volume_fp": "1355.00",
             "yes_ask": {"close_dollars": "1.0000", "high_dollars": "1.0000",
                         "low_dollars": "0.6700", "open_dollars": "0.7000"},
             "yes_bid": {"close_dollars": "0.9900", "high_dollars": "0.9900",
                         "low_dollars": "0.1000", "open_dollars": "0.1000"},
             "ds": "2026-02-17 16:00:00", "close": 1.0, "high": 1.0,
             "low": 0.67, "market_id": "KXPOLITICSMENTION-26FEB18-NATO"},
        ]
    }

    result = predict_live(live_candles,
                          lgbm_model=lgbm_model, xgb_model=xgb_model,
                          scaler=scaler, feature_cols=feature_cols)

    print(f"\nYES probability : {result['current_prob']:.1%}")
    print(f"Signal          : {result['signal']}")
