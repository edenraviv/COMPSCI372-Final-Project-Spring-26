from kalshi_client import KalshiClient

client = KalshiClient()

# Test historical endpoint
hist_test = client.get("/historical/markets", {"category": "Politics", "limit": 1})
print(hist_test)
print("HISTORICAL KEYS:", hist_test.keys())
print("HISTORICAL SAMPLE:", hist_test.get("markets", []))
print("HISTORICAL CURSOR:", hist_test.get("cursor"))

# Test recent endpoint
recent_test = client.get("/markets", {"category": "Politics", "status": "finalized", "limit": 1})
print(recent_test)
print("RECENT KEYS:", recent_test.keys())
print("RECENT SAMPLE:", recent_test.get("markets", []))
print("RECENT CURSOR:", recent_test.get("cursor"))