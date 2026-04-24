import pickle
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from data_visualization import _plot_training_curves


MODEL_LGBM_PATH = "kalshi_lgbm.txt"
MODEL_XGB_PATH  = "kalshi_xgb.json"
SCALER_PATH     = "kalshi_scaler.pkl"


HYPERPARAM_CONFIGS = {
    "Config-A (default)": {
        "learning_rate": 0.05, "num_leaves": 20,
        "min_data_in_leaf": 10,  "lambda_l2": 1,
        "feature_fraction": 0.8, "bagging_fraction": 0.8,
    },
    "Config-B (deep+reg)": {
        "learning_rate": 0.02, "num_leaves": 50,
        "min_data_in_leaf": 10, "lambda_l2": 5.0,
        "feature_fraction": 0.7, "bagging_fraction": 0.7,
    },
    "Config-C (shallow+fast)": {
        "learning_rate": 0.10, "num_leaves": 15,
        "min_data_in_leaf": 20, "lambda_l2": 5.0,
        "feature_fraction": 0.9, "bagging_fraction": 0.9,
    },
}


def hyperparam_search(X_train, y_train, X_val, y_val, feature_cols):
    '''HYPERPARAMETER TUNING — 3 configs compared on validation data.

    Config-A: default moderate settings
    Config-B: deeper trees + stronger regularization
    Config-C: shallow trees + high learning rate (fast/aggressive)

    All share the same base params (objective, metric, seed).
    Best config selected by validation log-loss.
    Training curves saved for each config → plots/training_curves.png'''
    base_params = {
        "objective": "binary",
        "metric":    ["binary_logloss", "auc"],
        "bagging_freq": 5,
        "verbose":   -1,
        "seed":      42,
    }

    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
    dval   = lgb.Dataset(X_val,   label=y_val,   reference=dtrain,
                         feature_name=feature_cols)

    results    = {}
    all_evals  = {}
    best_loss  = np.inf
    best_model = None
    best_name  = None

    for name, cfg in HYPERPARAM_CONFIGS.items():
        params       = {**base_params, **cfg}
        evals_result = {}
        model = lgb.train(
            params, dtrain, num_boost_round=500,
            valid_sets=[dtrain, dval], valid_names=["train", "val"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=30, verbose=False),
                lgb.log_evaluation(period=9999),
                lgb.record_evaluation(evals_result),
            ],
        )
        probs = model.predict(X_val)
        ll    = log_loss(y_val, probs)
        auc   = roc_auc_score(y_val, probs)
        bs    = brier_score_loss(y_val, probs)

        results[name]   = {"log_loss": ll, "auc": auc, "brier": bs,
                           "best_iter": model.best_iteration}
        all_evals[name] = evals_result

        if ll < best_loss:
            best_loss, best_model, best_name = ll, model, name

    # Print comparison table
    print("\n── Hyperparameter Search Results ───────────────")
    print(f"  {'Config':<26} {'LogLoss':>8} {'AUC':>7} "
          f"{'Brier':>7} {'Iters':>6}")
    print("  " + "─" * 56)
    for name, m in results.items():
        marker = " ◀ best" if name == best_name else ""
        print(f"  {name:<26} {m['log_loss']:>8.4f} {m['auc']:>7.4f} "
              f"{m['brier']:>7.4f} {m['best_iter']:>6}{marker}")

    _plot_training_curves(all_evals)
    return best_model, results, best_name


def hyperparam_search_cv(df_dev, feature_cols, n_splits=5, seed=42):
    '''HYPERPARAMETER TUNING via GroupKFold CV — robust config selection.

    Splits the dev set (train+val combined) into n_splits folds grouped by
    series_id so no event series straddles a fold boundary — prevents the
    same multi-option event leaking from fold-train into fold-val. Each
    config is scored by mean val log-loss across folds. Selecting by CV
    rather than a single val split reduces the risk of tuning to one
    particular split and gives an uncertainty estimate (±std).

    Also reports the mean train/val gap per config — a direct overfitting
    diagnostic. Larger gap ⇒ the config is memorizing the training fold.

    Args:
        df_dev:       DataFrame with market_id, label, and feature_cols
        feature_cols: list of feature column names
        n_splits:     number of CV folds (default 5)
        seed:         base_params seed

    Returns:
        best_name, cv_results
    '''
    base_params = {
        "objective":    "binary",
        "metric":       ["binary_logloss", "auc"],
        "bagging_freq": 5,
        "verbose":      -1,
        "seed":         seed,
    }

    groups = df_dev["series_id"].values
    y_all  = df_dev["label"].values
    X_raw  = df_dev[feature_cols].fillna(-999).values

    kf = GroupKFold(n_splits=n_splits)
    splits = list(kf.split(X_raw, groups=groups))

    cv_results       = {}
    first_fold_evals = {}

    for name, cfg in HYPERPARAM_CONFIGS.items():
        params      = {**base_params, **cfg}
        fold_val_ll = []
        fold_tr_ll  = []
        fold_auc    = []
        fold_brier  = []
        fold_gap    = []
        fold_iters  = []

        for fold_idx, (tr_idx, va_idx) in enumerate(splits):
            # Per-fold scaler — fit on fold-train only (no CV leakage)
            scaler = StandardScaler()
            X_tr   = scaler.fit_transform(X_raw[tr_idx])
            X_va   = scaler.transform(X_raw[va_idx])
            y_tr   = y_all[tr_idx]
            y_va   = y_all[va_idx]

            dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=feature_cols)
            dval   = lgb.Dataset(X_va, label=y_va, reference=dtrain,
                                 feature_name=feature_cols)

            evals_result = {}
            model = lgb.train(
                params, dtrain, num_boost_round=500,
                valid_sets=[dtrain, dval], valid_names=["train", "val"],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=30, verbose=False),
                    lgb.log_evaluation(period=9999),
                    lgb.record_evaluation(evals_result),
                ],
            )

            best_iter = model.best_iteration
            tr_loss   = evals_result["train"]["binary_logloss"][best_iter - 1]
            va_loss   = evals_result["val"]["binary_logloss"][best_iter - 1]

            probs = model.predict(X_va)
            fold_val_ll.append(log_loss(y_va, probs))
            fold_tr_ll.append(tr_loss)
            fold_auc.append(roc_auc_score(y_va, probs))
            fold_brier.append(brier_score_loss(y_va, probs))
            fold_gap.append(va_loss - tr_loss)
            fold_iters.append(best_iter)

            if fold_idx == 0:
                first_fold_evals[name] = evals_result

        cv_results[name] = {
            "mean_val_loss":   float(np.mean(fold_val_ll)),
            "std_val_loss":    float(np.std(fold_val_ll)),
            "mean_train_loss": float(np.mean(fold_tr_ll)),
            "mean_gap":        float(np.mean(fold_gap)),
            "mean_auc":        float(np.mean(fold_auc)),
            "mean_brier":      float(np.mean(fold_brier)),
            "mean_best_iter":  float(np.mean(fold_iters)),
            "fold_val_losses": fold_val_ll,
        }

    best_name = min(cv_results, key=lambda k: cv_results[k]["mean_val_loss"])

    print(f"\n── Hyperparameter Search Results "
          f"({n_splits}-fold GroupKFold CV) ──")
    print(f"  {'Config':<26} {'Val LL (±std)':>16} {'Train LL':>10} "
          f"{'Gap':>7} {'AUC':>7} {'Iters':>6}")
    print("  " + "─" * 76)
    for name, m in cv_results.items():
        marker = " ◀ best" if name == best_name else ""
        ll_str = f"{m['mean_val_loss']:.4f}±{m['std_val_loss']:.4f}"
        print(f"  {name:<26} {ll_str:>16} {m['mean_train_loss']:>10.4f} "
              f"{m['mean_gap']:>7.4f} {m['mean_auc']:>7.4f} "
              f"{m['mean_best_iter']:>6.0f}{marker}")
    print("  (Gap = val − train log-loss at best iter; "
          "larger ⇒ more overfitting)")

    _plot_training_curves(first_fold_evals)
    return best_name, cv_results


def train_lgbm(X_train, y_train, X_val, y_val,
               feature_cols, params_override=None):
    '''TRAIN LIGHTGBM (final model on selected features).'''
    base_params = {
        "objective":        "binary",
        "metric":           ["binary_logloss", "auc"],
        "learning_rate":    0.05,
        "num_leaves":       31,
        "min_data_in_leaf": 10,
        "lambda_l2":        1.0,         # L2 regularization
        "feature_fraction": 0.8,         # feature bagging
        "bagging_fraction": 0.8,         # row bagging
        "bagging_freq":     5,
        "verbose":          -1,
        "seed":             42,
    }
    if params_override:
        base_params.update(params_override)

    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
    dval   = lgb.Dataset(X_val,   label=y_val,   reference=dtrain,
                         feature_name=feature_cols)

    evals_result = {}
    model = lgb.train(
        base_params, dtrain, num_boost_round=500,
        valid_sets=[dtrain, dval], valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=30, verbose=False),
            lgb.log_evaluation(period=50),
            lgb.record_evaluation(evals_result),
        ],
    )
    best_iter = model.best_iteration
    tr_loss   = evals_result["train"]["binary_logloss"][best_iter - 1]
    va_loss   = evals_result["val"]["binary_logloss"][best_iter - 1]
    print(f"  LightGBM best iteration: {best_iter}")
    print(f"  LightGBM train LL: {tr_loss:.4f} | val LL: {va_loss:.4f} | "
          f"gap: {va_loss - tr_loss:+.4f}")
    return model


def train_xgboost(X_train, y_train, X_val, y_val):
    '''TRAIN XGBOOST (ensemble partner).'''
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval   = xgb.DMatrix(X_val,   label=y_val)
    params = {
        "objective":        "binary:logistic",
        "eval_metric":      ["logloss", "auc"],
        "learning_rate":    0.05,
        "max_depth":        5,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "lambda":           1.0,         # L2 regularization
        "seed":             42,
        "verbosity":        0,
    }
    evals_result = {}
    model = xgb.train(
        params, dtrain, num_boost_round=500,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=30,
        evals_result=evals_result,
        verbose_eval=False,
    )
    best_iter = model.best_iteration
    tr_loss   = evals_result["train"]["logloss"][best_iter]
    va_loss   = evals_result["val"]["logloss"][best_iter]
    print(f"  XGBoost best iteration: {best_iter}")
    print(f"  XGBoost train LL: {tr_loss:.4f} | val LL: {va_loss:.4f} | "
          f"gap: {va_loss - tr_loss:+.4f}")
    return model


def ensemble_predict(lgbm_model, xgb_model, X, lgbm_weight=0.6):
    '''ENSEMBLE — weighted average of LightGBM + XGBoost.

    LightGBM is weighted 60%, XGBoost 40%.
    Combining two independently-trained gradient boosting models with different
    implementations (leaf-wise vs depth-wise tree growth) reduces variance and
    improves calibration on small datasets.'''
    lgbm_probs = lgbm_model.predict(X)
    xgb_probs  = xgb_model.predict(xgb.DMatrix(X))
    return lgbm_weight * lgbm_probs + (1 - lgbm_weight) * xgb_probs


def save_models(lgbm_model, xgb_model, scaler, feature_cols):
    '''SAVE models and scaler to disk.'''
    lgbm_model.save_model(MODEL_LGBM_PATH)
    xgb_model.save_model(MODEL_XGB_PATH)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump({"scaler": scaler, "feature_cols": feature_cols}, f)
    print(f"\nModels saved → {MODEL_LGBM_PATH}, "
          f"{MODEL_XGB_PATH}, {SCALER_PATH}")


def load_models():
    '''LOAD models and scaler from disk.'''
    lgbm_model = lgb.Booster(model_file=MODEL_LGBM_PATH)
    xgb_model  = xgb.Booster()
    xgb_model.load_model(MODEL_XGB_PATH)
    with open(SCALER_PATH, "rb") as f:
        d = pickle.load(f)
    return lgbm_model, xgb_model, d["scaler"], d["feature_cols"]
