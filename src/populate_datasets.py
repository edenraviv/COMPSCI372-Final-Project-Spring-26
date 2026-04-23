from kalshi_client import KalshiClient
import data_ingestion as di


RAW_PATH       = "data/raw_market_data.json"
PROCESSED_PATH = "data/processed_market_data.json"


def run(max_markets: int = 1_000_000):
    '''Fetch markets from Kalshi, build samples, and write raw + processed JSON.'''
    client = KalshiClient()
    markets = client.get_all_training_data(max_markets)

    samples = di.build_resolved_market_samples(markets)
    print(f"\nSamples built: {len(samples)}")

    di.write_to_file([s.__dict__ for (s, _) in samples], PROCESSED_PATH)
    di.write_to_file([m for (_, m) in samples], RAW_PATH)

    di.count_candlesticks_per_market(di.read_from_json_file(RAW_PATH))
    return RAW_PATH, PROCESSED_PATH


if __name__ == "__main__":
    run()
