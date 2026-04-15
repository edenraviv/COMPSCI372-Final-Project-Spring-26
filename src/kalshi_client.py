import os
from dotenv import load_dotenv

load_dotenv("apikey.env")

API_KEY_ID = os.getenv("API_KEY_ID")
PRIVATE_KEY_PATH = os.getenv("PRIVATE_KEY_PATH")
BASE_URL = os.getenv("BASE_URL")

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

        print("STATUS CODE:", r.status_code)
        print("HEADERS:", r.headers)
        print("RESPONSE TEXT:", repr(r.text[:300]))

        return r.json()

    def get_markets(self):
        return self.get("/markets")

    def get_candlesticks(self, ticker):
        return self.get(f"/markets/{ticker}/candlesticks")