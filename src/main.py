from kalshi_client import KalshiClient
from data_ingestion import build_resolved_samples
from data_ingestion import print_markets
from engine import build_dataloaders, train
from models import ProbabilityRegressionHead

client = KalshiClient()

markets = client.get_markets()

print_markets(markets)