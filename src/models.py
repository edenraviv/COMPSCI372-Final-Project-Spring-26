
from engine import MarketFeatures, PredictionResult
from typing import Optional
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# RAG Retriever
# ---------------------------------------------------------------------------

class RAGRetriever:
    """
    Wraps vector store for document retrieval.
    Interfaces with utils/retrieval.py (not yet implemented).
    """

    def __init__(self, vector_store_path: str, top_k: int = 5):
        self.vector_store_path = vector_store_path
        self.top_k = top_k
        self.store = None

    def load(self):
        # TODO: from utils.retrieval import VectorStore
        # self.store = VectorStore.load(self.vector_store_path)
        raise NotImplementedError

    def retrieve(self, query: str) -> list[str]:
        # TODO: return self.store.similarity_search(query, k=self.top_k)
        raise NotImplementedError

    def format_context(self, chunks: list[str], max_chars: int = 2000) -> str:
        joined = "\n\n---\n\n".join(chunks)
        return joined[:max_chars]


# ---------------------------------------------------------------------------
# Sentence Encoder (LLM embedding side)
# ---------------------------------------------------------------------------

class SentenceEncoder:
    """
    Encodes RAG-augmented prompts into fixed-size embeddings.
    Start with a lightweight sentence-transformer; upgrade to generative
    model if reasoning over context proves necessary.

    Candidate models to compare:
        - "all-MiniLM-L6-v2"    (fast, 384-dim, good baseline)
        - "all-mpnet-base-v2"   (stronger, 768-dim)
        - Claude/GPT-4o API     (best reasoning, higher latency + cost)
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.embedding_dim: Optional[int] = None

    def load(self):
        # TODO: from sentence_transformers import SentenceTransformer
        # self.model = SentenceTransformer(self.model_name)
        # self.embedding_dim = self.model.get_sentence_embedding_dimension()
        raise NotImplementedError

    def encode(self, text: str) -> torch.Tensor:
        # TODO:
        # embedding = self.model.encode(text, convert_to_tensor=True)
        # return embedding.to(DEVICE)
        raise NotImplementedError

    def build_prompt(self, features: MarketFeatures, context: str) -> str:
        # TODO: pull template from config/prompts.py
        return (
            f"Market: {features.market_id}\n"
            f"Current Yes Price: {features.yes_price:.1f} cents\n"
            f"Days to Resolution: {features.time_to_resolution or 'unknown'}\n\n"
            f"Relevant Context:\n{context}"
        )


# ---------------------------------------------------------------------------
# Regression Head
# ---------------------------------------------------------------------------

class ProbabilityRegressionHead(nn.Module):
    """
    Fused input -> p(YES) in [0, 1].

    Architecture:
        - BatchNorm1d on input (handles feature scale differences at entry)
        - Two hidden layers with ReLU + Dropout (regularization)
        - Sigmoid output

    input_dim = embedding_dim + n_numeric_features
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout_p: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Baseline Model
# ---------------------------------------------------------------------------

class MarketPriceBaseline:
    """
    Naive baseline: p(YES) = yes_price / 100.

    This is the bar the trained model must beat. If it cannot outperform
    the raw market price on Brier score and log-loss, the model adds no edge.
    Evaluated in utils/evaluation.py against the same test split.
    """

    def predict(self, features: MarketFeatures) -> PredictionResult:
        p = features.yes_price / 100.0
        return PredictionResult(
            market_id=features.market_id,
            estimated_probability=p,
            market_implied_probability=p,
            edge=0.0,
            rag_context_used="",
        )

    def evaluate(self, samples: list[MarketFeatures]) -> dict:
        # TODO: from utils.evaluation import brier_score, log_loss_score
        preds = [s.yes_price / 100.0 for s in samples]
        labels = [s.label for s in samples]
        # return {
        #     "brier_score": brier_score(labels, preds),
        #     "log_loss": log_loss_score(labels, preds),
        # }
        raise NotImplementedError