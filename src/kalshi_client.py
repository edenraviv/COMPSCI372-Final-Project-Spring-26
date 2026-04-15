import os
from dotenv import load_dotenv

import os
import time
import base64
import requests
from dotenv import load_dotenv
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding

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

    def get_markets(self, status, limit, cursor, min_created_ts, max_created_ts) -> dict:
        params = {"status": status, 
                  "limit": limit,
                  "min_created_ts": min_created_ts,
                  "max_created_ts": max_created_ts,
                  "cursor": cursor}
        return self.get("/markets", params)
    
    def get_event(self, event_ticker):
        return self.get(f"/events/{event_ticker}")