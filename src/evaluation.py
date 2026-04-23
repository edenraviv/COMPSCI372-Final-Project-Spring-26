
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from schema import FEATURE_GROUPS
from data_visualization import PLOTS_DIR

# ══════════════════════════════════════════════════════════════════════════════
# 15. ERROR ANALYSIS — failure case breakdown with visualization
#
# Discussion:
#   False Positives cluster near 0.5 — model is uncertain, slight YES bias
#   in thin markets where bid/ask spread is wide.
#   False Negatives occur when volume was near zero early (no momentum signal)
#   and price collapsed in the final hour without warning candles.
#   Largest absolute errors occur in the 1-3h bucket where the market is
#   transitioning from uncertain to resolved but momentum reversals are common.
# ══════════════════════════════════════════════════════════════════════════════

def _error_analysis(df_test: pd.DataFrame,
                    probs: np.ndarray, y_true: np.ndarray):
    df = df_test.copy().reset_index(drop=True)
    df["prob"]  = probs
    df["label"] = y_true
    df["error"] = np.abs(probs - y_true)
    df["pred"]  = (probs >= 0.5).astype(int)
    df["fp"]    = ((df["pred"] == 1) & (df["label"] == 0)).astype(int)
    df["fn"]    = ((df["pred"] == 0) & (df["label"] == 1)).astype(int)

    print("\n── Error Analysis ──────────────────────────────")
    print(f"  False positives : {df['fp'].sum()}")
    print(f"  False negatives : {df['fn'].sum()}")
    print(f"  Mean abs error  : {df['error'].mean():.4f}")

    if "hours_to_expiry" in df.columns:
        bins   = [0, 1, 3, 6, 12, np.inf]
        labels = ["<1h", "1-3h", "3-6h", "6-12h", ">12h"]
        df["tte_bin"] = pd.cut(df["hours_to_expiry"],
                               bins=bins, labels=labels)
        by_tte = df.groupby("tte_bin")["error"].mean()
        print("\n  Mean abs error by hours-to-expiry bucket:")
        print(by_tte.to_string())

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(df.loc[df["fp"] == 1, "prob"], bins=20,
                 color="#E84C4C", alpha=0.8)
    axes[0].set_title("False Positives — Predicted Probability")
    axes[0].set_xlabel("Predicted probability")
    axes[1].hist(df.loc[df["fn"] == 1, "prob"], bins=20,
                 color="#4C9BE8", alpha=0.8)
    axes[1].set_title("False Negatives — Predicted Probability")
    axes[1].set_xlabel("Predicted probability")
    plt.tight_layout()
    path = PLOTS_DIR / "error_analysis.png"
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"\n  Error analysis plot saved → {path}")



# ══════════════════════════════════════════════════════════════════════════════
# 18. ABLATION STUDY
#
# Two independent design choices systematically varied:
#
#   A) Feature groups — remove one group at a time, measure val log-loss delta
#      Positive delta = removing that group hurt performance (it was useful)
#
#   B) StandardScaler — compare with vs without normalization
#      Documents whether scaling meaningfully affects gradient boosting
#      (tree models are scale-invariant in theory, but scaling affects
#       how missing sentinel values of -999 interact with real features)
#
# Results saved to plots/ablation_table.csv
# ══════════════════════════════════════════════════════════════════════════════

def ablation_study(df_train, df_val, feature_cols):
    print("\n── Ablation Study ──────────────────────────────")
    base_params = {
        "objective": "binary", "metric": "binary_logloss",
        "learning_rate": 0.05, "num_leaves": 31,
        "min_data_in_leaf": 10, "lambda_l2": 1.0,
        "feature_fraction": 0.8, "bagging_fraction": 0.8,
        "bagging_freq": 5, "verbose": -1, "seed": 42,
    }
    y_train = df_train["label"].values
    y_val   = df_val["label"].values

    def _quick_train(X_tr, X_v):
        dt = lgb.Dataset(X_tr, label=y_train)
        dv = lgb.Dataset(X_v,  label=y_val, reference=dt)
        m  = lgb.train(
            base_params, dt, num_boost_round=200,
            valid_sets=[dv], valid_names=["val"],
            callbacks=[lgb.early_stopping(20, verbose=False),
                       lgb.log_evaluation(9999)])
        return log_loss(y_val, m.predict(X_v))

    rows = []

    # Baseline: all features + scaler
    sc      = StandardScaler()
    X_tr_all = sc.fit_transform(df_train[feature_cols].fillna(-999))
    X_v_all  = sc.transform(df_val[feature_cols].fillna(-999))
    base_ll  = _quick_train(X_tr_all, X_v_all)
    rows.append({"Ablation": "Full model (all features + scaler)",
                 "Val LogLoss": base_ll, "Delta vs baseline": 0.0})

    # A) Remove each feature group
    for grp, grp_cols in FEATURE_GROUPS.items():
        kept = [c for c in feature_cols if c not in grp_cols]
        if not kept:
            continue
        sc2  = StandardScaler()
        X_tr = sc2.fit_transform(df_train[kept].fillna(-999))
        X_v  = sc2.transform(df_val[kept].fillna(-999))
        ll   = _quick_train(X_tr, X_v)
        rows.append({"Ablation": f"Remove '{grp}'",
                     "Val LogLoss": ll,
                     "Delta vs baseline": ll - base_ll})

    # B) No StandardScaler
    X_tr_raw = df_train[feature_cols].fillna(-999).values
    X_v_raw  = df_val[feature_cols].fillna(-999).values
    ll_ns    = _quick_train(X_tr_raw, X_v_raw)
    rows.append({"Ablation": "No StandardScaler",
                 "Val LogLoss": ll_ns,
                 "Delta vs baseline": ll_ns - base_ll})

    tbl = pd.DataFrame(rows).sort_values("Delta vs baseline")
    print(tbl.to_string(index=False, float_format="%.4f"))
    print("\n  Positive delta = removing that choice hurt performance.")

    path = PLOTS_DIR / "ablation_table.csv"
    tbl.to_csv(path, index=False)
    print(f"  Ablation table saved → {path}")
    return tbl
