import os
from dotenv import load_dotenv

import os
import time
import base64
import requests
from dotenv import load_dotenv
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding
from datetime import datetime, timezone

load_dotenv("apikey.env")

API_KEY_ID = os.getenv("API_KEY_ID")
PRIVATE_KEY_PATH = os.getenv("PRIVATE_KEY_PATH")
BASE_URL = os.getenv("BASE_URL")

class KalshiClient:
    def __init__(self):
        self.api_key_id = API_KEY_ID
        self.base_url = BASE_URL
        self.private_key = self._load_private_key()

    def _load_private_key(self):
        with open(PRIVATE_KEY_PATH, "rb") as key_file:
            return serialization.load_pem_private_key(
                key_file.read(),
                password=None,
            )

    def _sign_request(self, method, path):
        timestamp = str(int(time.time() * 1000))
        message = timestamp + method + path

        signature = self.private_key.sign(
            message.encode(),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )

        return {
            "KALSHI-ACCESS-KEY": self.api_key_id,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode(),
            "KALSHI-ACCESS-TIMESTAMP": timestamp,
        }

    def get(self, path, params=None):
        url = self.base_url + path
        headers = self._sign_request("GET", path)
        r = requests.get(url, headers=headers, params=params)
        r.raise_for_status()
        return r.json()

    def get_markets(self, status, limit, cursor=None, min_created_ts=None, max_created_ts=None) -> dict:
        params = {"status": status, 
                  "limit": limit,
                  "min_created_ts": min_created_ts,
                  "max_created_ts": max_created_ts,
                  "cursor": cursor}
        return self.get("/markets", params)
        
    def parse_cutoff(self, response: dict) -> int:
        """
        Parses the historical cutoff response and returns the market_settled_ts
        as a Unix timestamp (seconds).
        """
        ts_string = response["market_settled_ts"]
        dt = datetime.fromisoformat(ts_string.replace("Z", "+00:00"))
        return int(dt.timestamp())
        
    def get_historical_cutoff(self):
        response = self.get("/historical/cutoff")
        parsed_cutoff = self.parse_cutoff(response)
        return parsed_cutoff
    
    def _paginate(self, path: str, params: dict) -> list[dict]:
        """
        Generic pagination handler for any list endpoint.
        Follows cursors until exhausted.
        """
        all_items = []
        cursor = None
        key = "markets"  # the data array key in the response

        while True:
            if cursor:
                params["cursor"] = cursor
            response = self.get(path, params)
            batch = response.get(key, [])
            all_items.extend(batch)
            print(f"Fetched {len(batch)} | Total so far: {len(all_items)}")
            cursor = response.get("cursor")
            if len(all_items) > 10000:
                break
            if not cursor:
                break

        return all_items

    def get_all_training_data(self) -> list[dict]:
        """
        Fetches all resolved political markets from both endpoints and merges.
        """
        historical = self._paginate("/historical/markets", {
            "limit": 1000
        })

        recent = self._paginate("/markets", {
            "limit": 1000,
        })

        # Deduplicate on ticker
        all_markets = {m["ticker"]: m for m in historical + recent}
        print(f"Historical: {len(historical)} | Recent: {len(recent)} | Unique: {len(all_markets)}")

        return list(all_markets.values())
        
    def get_event(self, event_ticker):
        return self.get(f"/events/{event_ticker}")