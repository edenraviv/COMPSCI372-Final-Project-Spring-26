from kalshi_client import KalshiClient

client = KalshiClient()

markets = client.get_markets()
print(markets)