from kalshi_client import KalshiClient
from data_ingestion import build_resolved_samples
from data_ingestion import print_markets
from engine import build_dataloaders, train
from models import ProbabilityRegressionHead

client = KalshiClient()

markets = client.get_markets()

print(markets)

'''
event = client.get_event("KXMVECROSSCATEGORY-S2026E6CB6453B27-CBC97A8B597")
print(event)'''