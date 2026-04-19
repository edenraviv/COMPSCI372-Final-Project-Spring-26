from kalshi_client import KalshiClient
from data_ingestion import build_resolved_samples, extract_market_question, write_to_file

client = KalshiClient()

markets = client.get_all_training_data(100000)

'''
# 2. Inspect raw fields you care about
for m in markets:
    print(m)
    print(m["ticker"])
    print(m["status"])
    print(m["result"])
    print(m["market_type"])
    print(extract_market_question(m))
    print("---")'''

# 3. Try building samples from that same batch
samples = build_resolved_samples(markets)
print(f"\nSamples built: {len(samples)}")

write_to_file([s.__dict__ for (s, _) in samples], "data/processed_market_data.json")
write_to_file([m for (_, m) in samples],  "data/raw_market_data.json")
