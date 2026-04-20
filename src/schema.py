
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


# ---------------------------------------------------------------------------
# Timeseries Schema
# ---------------------------------------------------------------------------

N_CANDLE_FEATURES = 16  # number of numeric features per candle (see _candle_to_row)

@dataclass
class CandleFeatures:
    """One hourly candlestick row from market_timeseries.json."""
    end_period_ts: int
    ds: str
    price_close: float
    price_high: float
    price_low: float
    price_open: float
    price_mean: float
    price_previous: Optional[float]
    yes_ask_close: float
    yes_ask_high: float
    yes_ask_low: float
    yes_ask_open: float
    yes_bid_close: float
    yes_bid_high: float
    yes_bid_low: float
    yes_bid_open: float
    volume: float
    open_interest: float


@dataclass
class TimeseriesSample:
    """Full price history for one market, plus its resolved label."""
    market_id: str
    series_id: str
    candles: list[CandleFeatures]
    label: float                    # 1.0 = YES resolved, 0.0 = NO


def _candle_to_row(c: CandleFeatures) -> list[float]:
    return [
        c.price_close, c.price_high, c.price_low, c.price_open, c.price_mean,
        c.price_previous or 0.0,
        c.yes_ask_close, c.yes_ask_high, c.yes_ask_low, c.yes_ask_open,
        c.yes_bid_close, c.yes_bid_high, c.yes_bid_low, c.yes_bid_open,
        c.volume, c.open_interest,
    ]


class TimeseriesDataset(Dataset):
    """
    Wraps a list of TimeseriesSample into a PyTorch Dataset.
    Each item is a (seq_len, N_CANDLE_FEATURES) tensor, padded/truncated to max_seq_len.
    Sequences are right-aligned: padding goes at the front, most-recent candles last.
    """

    def __init__(self, samples: list[TimeseriesSample], max_seq_len: int = 168):
        assert all(s.label is not None for s in samples), \
            "All training samples must have a label."
        self.samples = samples
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        rows = [_candle_to_row(c) for c in sample.candles]

        # truncate to the most-recent max_seq_len candles, then left-pad with zeros
        rows = rows[-self.max_seq_len:]
        pad_len = self.max_seq_len - len(rows)
        if pad_len > 0:
            rows = [[0.0] * N_CANDLE_FEATURES] * pad_len + rows

        features = torch.tensor(rows, dtype=torch.float32)          # (max_seq_len, N_CANDLE_FEATURES)
        label = torch.tensor([sample.label], dtype=torch.float32)   # (1,)
        return features, label, sample.market_id