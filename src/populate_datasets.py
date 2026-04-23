from kalshi_client import KalshiClient
import data_ingestion as di

client = KalshiClient()

markets = client.get_all_training_data(1000000)

# 3. Try building samples from that same batch
samples = di.build_resolved_market_samples(markets)
print(f"\nSamples built: {len(samples)}")

di.write_to_file([s.__dict__ for (s, _) in samples], "data/processed_market_data.json")
di.write_to_file([m for (_, m) in samples],  "data/raw_market_data.json")

di.count_candlesticks_per_market(di.read_from_json_file("data/raw_market_data.json"))