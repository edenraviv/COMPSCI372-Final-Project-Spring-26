import os
from dotenv import load_dotenv

import os
import time
import base64
import requests
from dotenv import load_dotenv
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding
from datetime import datetime

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

    def get(self, path, params=None, retries=5):
        url = self.base_url + path

        for attempt in range(retries):
            headers = self._sign_request("GET", path)
            r = requests.get(url, headers=headers, params=params)

            if r.status_code == 429:
                wait = 2 ** attempt
                print(f"Rate limited on {path}, waiting {wait}s (attempt {attempt+1}/{retries})")
                time.sleep(wait)
                continue

            if not r.ok and r.status_code != 404:
                print(f"API error {r.status_code} for {path}: {r.text}")

            r.raise_for_status()
            return r.json()

        raise Exception(f"Max retries exceeded for {path}")

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
    
    def _paginate(self, path: str, params: dict, max_raw) -> list[dict]:
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
            if not cursor:
                break
            if len(all_items) > max_raw:
                break

        return all_items

    def get_all_training_data(self, max_raw =750000) -> list[dict]:
        """
        Fetches all resolved political markets from both endpoints and merges.
        """
        historical = self._paginate("/historical/markets", {
            "limit": 1000
        }, max_raw)

        recent = self._paginate("/markets", {
            "limit": 1000, 
        }, max_raw)

        # Deduplicate on ticker
        all_markets = {m["ticker"]: m for m in historical + recent}
        print(f"Historical: {len(historical)} | Recent: {len(recent)} | Unique: {len(all_markets)}")

        return list(all_markets.values())
        
    def get_event(self, event_ticker):
        return self.get(f"/events/{event_ticker}")
    
    def get_candles(self, series_ticker, ticker, start: int, end: int, period=60):
        """
        Fetch candlestick price history for a market between open and close.
        """
        params = {
            "start_ts": start,
            "end_ts": end,
            "period_interval": period
        }

        candles = []
        try:
            current_market_response = self.get(f"/series/{series_ticker}/markets/{ticker}/candlesticks", params)
            candles = current_market_response.get("candles", [])
        except Exception:
            pass

        if not candles:
            historical_response = self.get(f"/historical/markets/{ticker}/candlesticks", params)
            candles = historical_response.get("candlesticks", [])

        return candles