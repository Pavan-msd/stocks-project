import requests
import hmac
import time
from hashlib import sha256

API_KEY = "3GmR0g5e9hSICZ6C8xcFgeJ2V6KbVd"
API_SECRET = "SSk6kNRas9G4Q7FoDO6XNlYJr6azQDJQ2VCGrkKU5m4vARvbJZxCnku14XZo"
BASE_URL = 'https://testnet.delta.exchange'

def get_signature(method, path, body=''):
    timestamp = int(time.time() * 1000)
    payload = f"{timestamp}{method}{path}{body}"
    signature = hmac.new(bytes(API_SECRET, 'latin-1'), bytes(payload, 'latin-1'), sha256).hexdigest()
    return signature, timestamp

path = '/v2/products'
signature, timestamp = get_signature('GET', path)
headers = {
    'api-key': API_KEY,
    'timestamp': str(timestamp),
    'signature': signature,
    'Content-Type': 'application/json'
}
response = requests.get(BASE_URL + path, headers=headers)
products = response.json()

for product in products['result']:
    if product['symbol'] == 'BTCUSDT':
        print(f"Symbol: {product['symbol']}, Product ID: {product['id']}")
        break