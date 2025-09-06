from flask import Flask
import requests
import hmac
import time
import os
import json
import pandas as pd
import pandas_ta as ta  # Using the simpler library for Heroku compatibility
from hashlib import sha256

app = Flask(__name__)

# ===== CONFIGURATION - SET THESE IN HEROKU =====
API_KEY = os.environ.get('3GmR0g5e9hSICZ6C8xcFgeJ2V6KbVd')       # Your TESTNET API Key
API_SECRET = os.environ.get('SSk6kNRas9G4Q7FoDO6XNlYJr6azQDJQ2VCGrkKU5m4vARvbJZxCnku14XZo') # Your TESTNET Secret
BASE_URL = 'https://testnet.delta.exchange'     # <<< TESTNET URL

# ===== TRADING PARAMETERS - YOU MUST SET THESE =====
SYMBOL = 'BTCUSDT'
PRODUCT_ID = 123456  # !!! REPLACE THIS WITH THE REAL TESTNET PRODUCT ID YOU FOUND !!!
TIMEFRAME = '15m'
QTY_PER_TRADE = 10   # Size of your test orders. Use a small number.

# Strategy Parameters
EMA_FAST_PERIOD = 21
EMA_SLOW_PERIOD = 50
RSI_PERIOD = 14
RSI_OVERSOLD = 35
RSI_OVERBOUGHT = 65

# ===== DELTA API FUNCTIONS =====
def generate_signature(method, path, body=''):
    timestamp = int(time.time() * 1000)
    if body:
        body = json.dumps(body)
    else:
        body = ''
    payload = f"{timestamp}{method}{path}{body}"
    signature = hmac.new(
        bytes(API_SECRET, 'latin-1'),
        bytes(payload, 'latin-1'),
        sha256
    ).hexdigest()
    return signature, timestamp

def delta_api_request(method, path, body=None):
    signature, timestamp = generate_signature(method, path, body)
    headers = {
        'api-key': API_KEY,
        'timestamp': str(timestamp),
        'signature': signature,
        'Content-Type': 'application/json'
    }
    url = f"{BASE_URL}{path}"

    try:
        if method.upper() == 'GET':
            response = requests.get(url, headers=headers, params=body)
        elif method.upper() == 'POST':
            response = requests.post(url, headers=headers, json=body)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"❌ API Error: {e}")
        return None

# ===== STRATEGY FUNCTIONS =====
def fetch_ohlcv():
    """Fetch the last 100 candles for analysis."""
    path = '/v2/history/candles'
    params = {'symbol': SYMBOL, 'resolution': TIMEFRAME, 'limit': 100}
    data = delta_api_request('GET', path, params)
    if not data or 'result' not in data:
        print("Failed to fetch data.")
        return None
    df = pd.DataFrame(data['result'])
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    df.sort_index(ascending=True, inplace=True)
    return df

def calculate_signals(df):
    """Calculate indicators and trading signals."""
    close = df['close'].astype(float)
    # Calculate indicators using pandas_ta
    df['ema_fast'] = ta.ema(close, length=EMA_FAST_PERIOD)
    df['ema_slow'] = ta.ema(close, length=EMA_SLOW_PERIOD)
    df['rsi'] = ta.rsi(close, length=RSI_PERIOD)
    # Generate signals
    df['ema_bullish'] = df['ema_fast'] > df['ema_slow']
    df['rsi_oversold'] = df['rsi'] < RSI_OVERSOLD
    df['rsi_overbought'] = df['rsi'] > RSI_OVERBOUGHT
    # A buy signal: Uptrend + RSI was oversold last candle but isn't now (bouncing back)
    df['rsi_oversold_prev'] = df['rsi_oversold'].shift(1)
    df['signal_buy'] = df['ema_bullish'] & df['rsi_oversold_prev'] & ~df['rsi_oversold']
    # A sell signal: Overbought or trend turning bearish
    df['signal_sell'] = df['rsi_overbought'] | ~df['ema_bullish']
    return df

def get_current_position():
    """Check if we already have an open position."""
    path = '/v2/positions'
    params = {'product_id': PRODUCT_ID}
    data = delta_api_request('GET', path, params)
    if data and 'result' in data:
        for position in data['result']:
            if int(position['product_id']) == PRODUCT_ID and float(position['size']) != 0:
                return position
    return None

def place_order(side, size):
    """Place a market order."""
    path = '/v2/orders'
    order_payload = {
        "product_id": PRODUCT_ID,
        "size": str(size),
        "side": side,
        "order_type": "market",
    }
    print(f"🔄 Placing {side.upper()} order for {size} {SYMBOL}...")
    result = delta_api_request('POST', path, order_payload)
    print(f"Order Response: {result}")
    return result

def execute_strategy():
    """The main function that runs the strategy logic."""
    print(f"\n{'='*50}")
    print(f"🔍 Strategy Cycle Started for {SYMBOL} ({TIMEFRAME})")
    print(f"{'='*50}")

    # 1. Fetch Market Data
    print("Fetching latest market data...")
    df = fetch_ohlcv()
    if df is None or df.empty:
        return "Failed to fetch data. Check API connection."

    # 2. Calculate Indicators & Signals
    df = calculate_signals(df)
    latest = df.iloc[-1]  # Get the latest candle
    previous = df.iloc[-2] # Get the previous candle

    # 3. Print Market Snapshot
    print(f"📍 Latest Candle:")
    print(f"   Time: {latest.name}")
    print(f"   Close: ${latest['close']:.2f}")
    print(f"   EMA{EMA_FAST_PERIOD}: {latest['ema_fast']:.2f}")
    print(f"   EMA{EMA_SLOW_PERIOD}: {latest['ema_slow']:.2f}")
    print(f"   RSI: {latest['rsi']:.2f}")
    print(f"   Trend Bullish (EMA{EMA_FAST_PERIOD} > EMA{EMA_SLOW_PERIOD}): {latest['ema_bullish']}")
    print(f"   RSI Oversold (<{RSI_OVERSOLD}): {latest['rsi_oversold']}")
    print(f"   RSI Overbought (>{RSI_OVERBOUGHT}): {latest['rsi_overbought']}")

    # 4. Check for existing position
    current_position = get_current_position()
    if current_position:
        print(f"📦 Current Position: {current_position['size']} units (P&L: {current_position['realized_pnl']})")
    else:
        print("📦 Current Position: None")

    # 5. STRATEGY LOGIC: Check for BUY signal (and no existing position)
    if not current_position and latest['signal_buy']:
        print("🎯 >>> STRONG BUY SIGNAL GENERATED! <<<")
        print("   Reason: Trend is BULLISH and RSI is bouncing from OVERSOLD.")
        return place_order("buy", QTY_PER_TRADE)

    # 6. STRATEGY LOGIC: Check for SELL signal (and we have a position to close)
    elif current_position and latest['signal_sell']:
        print("🎯 >>> SELL SIGNAL GENERATED! Closing position. <<<")
        pos_size = abs(float(current_position['size'])) # Get size, ensure it's positive
        return place_order("sell", pos_size)

    else:
        print("✅ No trading signal at this time. Holding...")
        return "No action taken."

# ===== FLASK ROUTES =====
@app.route('/')
def home():
    return "Delta Testnet Bot is Active. observing 15m BTCUSDT. Visit /run to execute a cycle."

@app.route('/run')
def run_bot():
    """This endpoint is called by UptimeRobot every 5 minutes."""
    result = execute_strategy()
    return str(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)