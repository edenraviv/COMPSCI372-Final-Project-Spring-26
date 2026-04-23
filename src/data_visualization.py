
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)

def _plot_feature_importance(imp: pd.Series, top_n: int = 20):
    imp_top = imp.sort_values(ascending=True).tail(top_n)
    fig, ax = plt.subplots(figsize=(7, max(4, top_n * 0.32)))
    ax.barh(imp_top.index, imp_top.values, color="#4C9BE8")
    ax.set_xlabel("Gain importance")
    ax.set_title(f"Top {top_n} Features by Gain")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    path = PLOTS_DIR / "feature_importance.png"
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  Feature importance chart saved → {path}")


def _plot_training_curves(all_evals: dict):
    """Training loss + AUC over boosting rounds for all 3 configs."""
    n   = len(all_evals)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, (name, evals) in zip(axes, all_evals.items()):
        tr_loss  = evals.get("train", {}).get("binary_logloss", [])
        val_loss = evals.get("val",   {}).get("binary_logloss", [])
        ax.plot(tr_loss,  label="Train", linewidth=1.5)
        ax.plot(val_loss, label="Val",   linewidth=1.5, linestyle="--")
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("Boosting round")
        ax.set_ylabel("Log-loss")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle("Training Curves — Log-loss over Boosting Rounds",
                 fontsize=11)
    plt.tight_layout()
    path = PLOTS_DIR / "training_curves.png"
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"\n  Training curves saved → {path}")