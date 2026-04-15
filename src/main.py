from kalshi_client import KalshiClient
from data_ingestion import build_resolved_samples, extract_market_question
from data_ingestion import print_markets
from engine import build_dataloaders, train
from models import ProbabilityRegressionHead

client = KalshiClient()

response = client.get_markets(limit=5)
markets = response["markets"]

# 2. Inspect raw fields you care about
for m in markets:
    print(m["ticker"])
    print(m["status"])
    print(m["result"])
    print(m["market_type"])
    print(extract_market_question(m))
    print("---")

# 3. Try building samples from that same batch
samples = build_resolved_samples(response)
print(f"\nSamples built: {len(samples)}")
for s in samples:
    print(s.market_id, s.label, s.rag_query)

# 4. If rag_query is still empty, inspect the parent event
if markets:
    event_ticker = markets[0]["event_ticker"]
    event = client.get_event(event_ticker)
    print("\nEVENT RESPONSE:")
    print(event)