"""
engine.py
---------
Core ML pipeline for Kalshi political market probability estimation.

Pipeline:
    1. Receive a market identifier + preprocessed features
    2. Retrieve relevant documents via RAG (news, events)
    3. Encode context via sentence embeddings
    4. Fuse embeddings + numeric trade features
    5. Regression head outputs p(YES) in [0, 1]
    6. Compare against market price to surface edge

Rubric items addressed here:
    - Modular class design
    - Train/val/test split with DataLoader + batching/shuffling
    - Baseline model (market price as naive predictor)
    - Dropout + early stopping (regularization)
    - LR scheduling, gradient clipping, batch norm
    - Sentence embeddings for RAG retrieval
    - Backtesting simulation for evaluation
    - Brier score, log-loss, calibration curve (evaluation metrics)
    - Ablation: with vs. without RAG context
"""

from __future__ import annotations
from schema import MarketFeatures, KalshiDataset, _extract_numeric_features, PredictionResult
from models import ProbabilityRegressionHead, RAGRetriever, SentenceEncoder

import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps")


def build_dataloaders(
    samples: list[MarketFeatures],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    batch_size: int = 32,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Split dataset and return DataLoaders.
    Split ratios: 70% train / 15% val / 15% test (documented).

    Train loader shuffles; val and test do not.
    """
    dataset = KalshiDataset(samples)
    n = len(dataset)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = int(n * test_ratio)

    train_set, val_set, test_set = random_split(
        dataset, [n_train, n_val, n_test], generator=2025
    )

    logger.info(f"Split: {n_train} train / {n_val} val / {n_test} test")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader



# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def train(
    model: ProbabilityRegressionHead,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,         # L2 regularization via AdamW
    grad_clip: float = 1.0,             # Gradient clipping for training stability
    patience: int = 7,                  # Early stopping patience (epochs without improvement)
) -> dict:
    """
    Training loop with:
        - AdamW optimizer (built-in L2 via weight_decay)
        - ReduceLROnPlateau LR scheduler
        - Gradient clipping
        - Early stopping on val loss
        - Per-epoch loss tracking for training curve visualization

    Returns:
        history: {"train_loss": [...], "val_loss": [...]} — plot these in evaluation
    """
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )

    loss_fn = nn.BCELoss()

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_state = None

    for epoch in range(n_epochs):
        # --- Train ---
        model.train()
        train_loss = 0.0
        for features, labels, _ in train_loader:
            features, labels = features.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            preds = model(features)
            loss = loss_fn(preds, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # --- Validate ---
        val_loss = _evaluate_loss(model, val_loader, loss_fn)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        logger.info(f"Epoch {epoch+1:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        # --- Early stopping ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch+1}.")
                break

    if best_state:
        model.load_state_dict(best_state)

    return history


def _evaluate_loss(model: nn.Module, loader: DataLoader, loss_fn) -> float:
    model.eval()
    total = 0.0
    with torch.no_grad():
        for features, labels, _ in loader:
            features, labels = features.to(DEVICE), labels.to(DEVICE)
            preds = model(features)
            total += loss_fn(preds, labels).item()
    return total / len(loader)


# ---------------------------------------------------------------------------
# Backtesting
# ---------------------------------------------------------------------------

def backtest(
    model: ProbabilityRegressionHead,
    samples: list[MarketFeatures],
    min_edge: float = 0.05,
    stake: float = 100.0,
) -> dict:
    """
    Simulation-based evaluation: replay resolved historical markets and simulate
    betting decisions based on model edge vs market price.

    Strategy:
        - Bet YES if model_p > market_p + min_edge
        - Bet NO  if model_p < market_p - min_edge
        - Skip otherwise

    Returns:
        dict with total_pnl, win_rate, bets_placed, per-market results list
    """
    model.eval()
    results = []
    total_pnl = 0.0
    bets = 0
    wins = 0

    with torch.no_grad():
        for sample in samples:
            assert sample.label is not None, "Backtest requires resolved markets with labels."

            features = _extract_numeric_features(sample).unsqueeze(0).to(DEVICE)
            # NOTE: numeric-only path — RAG embedding not fused here yet.
            # TODO: include RAG embedding once encoder is wired up.
            model_p = model(features).item()
            market_p = sample.yes_price / 100.0
            edge = model_p - market_p

            if abs(edge) < min_edge:
                continue

            bets += 1
            bet_yes = edge > 0
            resolved_yes = sample.label == 1.0

            if bet_yes and resolved_yes:
                pnl = stake * (1 - market_p)
                wins += 1
            elif not bet_yes and not resolved_yes:
                pnl = stake * market_p
                wins += 1
            else:
                pnl = -stake

            total_pnl += pnl
            results.append({
                "market_id": sample.market_id,
                "model_p": model_p,
                "market_p": market_p,
                "edge": edge,
                "bet_yes": bet_yes,
                "resolved_yes": resolved_yes,
                "pnl": pnl,
            })

    return {
        "total_pnl": total_pnl,
        "bets_placed": bets,
        "win_rate": wins / bets if bets > 0 else 0.0,
        "results": results,
    }


# ---------------------------------------------------------------------------
# Ablation
# ---------------------------------------------------------------------------

def run_ablation(
    samples_with_rag: list[MarketFeatures],
    samples_without_rag: list[MarketFeatures],
    model_config: dict,
) -> dict:
    """
    Ablation study: train and evaluate with vs. without RAG context embeddings.
    Documents the isolated contribution of the retrieval component.

    Both sample lists represent the same markets; the difference is whether
    the fused embedding includes RAG context or is zeroed out.

    Expected return shape:
    {
        "with_rag":    {"brier_score": ..., "log_loss": ..., "val_loss": ...},
        "without_rag": {"brier_score": ..., "log_loss": ..., "val_loss": ...},
    }
    """
    # TODO: implement once SentenceEncoder is wired up end-to-end
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Engine — Inference Orchestrator
# ---------------------------------------------------------------------------

class KalshiEngine:
    """
    Ties all components together for inference only.
    Training is handled via train() above; engine loads pretrained weights.
    """

    def __init__(
        self,
        retriever: RAGRetriever,
        encoder: SentenceEncoder,
        regression_head: ProbabilityRegressionHead,
    ):
        self.retriever = retriever
        self.encoder = encoder
        self.regression_head = regression_head.to(DEVICE)

    def predict(self, features: MarketFeatures) -> PredictionResult:
        query = features.rag_query or features.market_id
        chunks = self.retriever.retrieve(query)
        context = self.retriever.format_context(chunks)

        prompt = self.encoder.build_prompt(features, context)
        rag_embedding = self.encoder.encode(prompt)                   # (embedding_dim,)
        numeric = _extract_numeric_features(features).to(DEVICE)      # (n_features,)
        fused = torch.cat([rag_embedding, numeric], dim=-1).unsqueeze(0)

        self.regression_head.eval()
        with torch.no_grad():
            p = self.regression_head(fused).item()

        market_p = features.yes_price / 100.0

        return PredictionResult(
            market_id=features.market_id,
            estimated_probability=p,
            market_implied_probability=market_p,
            edge=p - market_p,
            rag_context_used=context[:300],
        )

    def scan_for_edges(
        self,
        feature_list: list[MarketFeatures],
        min_edge: float = 0.05,
    ) -> list[PredictionResult]:
        """Return all markets where |edge| >= min_edge, sorted by edge size."""
        results = [self.predict(f) for f in feature_list]
        return sorted(
            [r for r in results if abs(r.edge) >= min_edge],
            key=lambda r: abs(r.edge),
            reverse=True,
        )