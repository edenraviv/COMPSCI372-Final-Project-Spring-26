
from dataclasses import dataclass
from typing import Optional
from torch.utils.data import Dataset
import torch


# ---------------------------------------------------------------------------
# Data Contract
# ---------------------------------------------------------------------------

@dataclass
class MarketFeatures:
    """
    Ingress fetches what's needed to build this information. Add fields here as the regression head demands them.
    """
    market_id: str
    yes_price: float                            # 0-100 cents (normalize before use)
    no_price: float
    last_price_dollars: float              
    volume_history: list[float]
    price_momentum: Optional[float] = None      # e.g. 7-day delta
    volume_weighted_price: Optional[float] = None
    time_to_resolution: Optional[float] = None  # Days until resolution
    rag_query: Optional[str] = None             # Defaults to market_id if None
    label: Optional[float] = None              # Ground truth: 1.0 = YES resolved, 0.0 = NO


@dataclass
class PredictionResult:
    """Output contract."""
    market_id: str
    estimated_probability: float
    market_implied_probability: float
    edge: float                                 # model estimate - market price
    rag_context_used: str
    confidence: Optional[float] = None


# ---------------------------------------------------------------------------
# Dataset + DataLoader
# ---------------------------------------------------------------------------

class KalshiDataset(Dataset):
    """
    Wraps a list of MarketFeatures into a PyTorch Dataset.
    Expects features to already be normalized (preprocessing.py handles that).
    """

    def __init__(self, samples: list[MarketFeatures]):
        assert all(s.label is not None for s in samples), \
            "All training samples must have a label."
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        features = _extract_numeric_features(sample)        # (n_features,)
        label = torch.tensor([sample.label], dtype=torch.float32)
        return features, label, sample.market_id


# ---------------------------------------------------------------------------
# Feature Extraction (numeric side of fusion)
# ---------------------------------------------------------------------------

def _extract_numeric_features(features: MarketFeatures) -> torch.Tensor:
    """
    Flatten MarketFeatures numeric fields into a 1D tensor.
    This defines n_features — update as fields are added/removed.
    Normalization should already be applied by preprocessing.py before this.
    """
    raw = [
        features.yes_price,
        features.no_price,
        features.last_price_dollars,
        features.price_momentum or 0.0,
        features.volume_weighted_price or 0.0,
        features.time_to_resolution or 0.0,
    ]
    return torch.tensor(raw, dtype=torch.float32)