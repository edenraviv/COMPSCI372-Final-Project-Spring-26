from kalshi_client import KalshiClient
from data_ingestion import build_resolved_samples, extract_market_question, write_to_file

client = KalshiClient()

response = client.get_markets("closed", limit=5)
write_to_file(response["markets"], "data/raw_market_data.json")

markets = response["markets"]
# 2. Inspect raw fields you care about
for m in markets:
    print(m)
    print(m["ticker"])
    print(m["status"])
    print(m["result"])
    print(m["market_type"])
    print(extract_market_question(m))
    print("---")

# 3. Try building samples from that same batch
samples = build_resolved_samples(response)
print(f"\nSamples built: {len(samples)}")

write_to_file([s.__dict__ for s in samples], "data/processed_market_data.json")
