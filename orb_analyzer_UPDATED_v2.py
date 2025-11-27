#!/usr/bin/env python3
"""
FOREX ORB STRATEGY - COMPLETE VERSION WITH ALL 8 CONFLUENCE FACTORS
TwelveData - Real-time data
All 8 factors: Breakout, RSI, MACD, EMA, Momentum, Volume, FVG, Support/Resistance
API Key from GitHub Secrets (Secure)
Updated: SL pips is not fixed, New output format, Buy/Sell Limit recommendations
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime
import time
import os

# ════════════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════════════

# Read API key from GitHub Secrets environment variable
API_KEY = os.getenv('TWELVEDATA_API_KEY', 'ALPHA')
BASE_URL = "https://api.twelvedata.com"

PAIRS = ["EUR/USD", "GBP/USD", "USD/JPY", "USD/CAD", "EUR/GBP", "XAU/USD"]

TIMEFRAME = "5min"
CHECK_INTERVAL = 300  # 5 minutes
MAX_ITERATIONS = 1  # Stop after 1 check
SL_PERCENTAGE = 0.005  # 0.5% of entry price

# Log file configuration
LOG_DIR = "trading_logs"
LOG_FILE = f"{LOG_DIR}/orb_trading_log.csv"

# ════════════════════════════════════════════════════════════════════════════════════
# CREATE LOG DIRECTORY AND INITIALIZE CSV
# ════════════════════════════════════════════════════════════════════════════════════

def create_log_dir():
    """Create logs directory if it doesn't exist"""
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
        print(f"✅ Created log directory: {LOG_DIR}")


def initialize_csv():
    """Create empty CSV file if it doesn't exist"""
    if not os.path.isfile(LOG_FILE):
        # Create header row with Date as first column
        df = pd.DataFrame(columns=[
            'Date', 'Time', 'Pair', 'Direction', 'Score', 'Recommendation', 'Entry',
            'SL', 'TP1', 'TP2', 'TP3', 'Factors'
        ])
        df.to_csv(LOG_FILE, index=False)
        print(f"✅ Created CSV file: {LOG_FILE}")


# ════════════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════════════

def safe_float(value):
    """Safely convert to float"""
    try:
        return float(value)
    except:
        return 0.0


def get_twelvedata_candles(pair, api_key, interval="5min", count=50):
    """Get candles from TwelveData"""
    try:
        url = f"{BASE_URL}/time_series"

        params = {
            "symbol": pair,
            "interval": interval,
            "apikey": api_key,
            "outputsize": count,
            "format": "JSON"
        }

        print(f"   📡 {pair}...", end="", flush=True)
        response = requests.get(url, params=params, timeout=10)

        if response.status_code != 200:
            print(f" ❌ HTTP {response.status_code}")
            return None

        data = response.json()

        if "status" in data:
            if data["status"] == "error":
                print(f" ❌ {data.get('message', 'Error')[:40]}")
                return None

        if "values" not in data:
            print(f" ❌ No values")
            return None

        candles = data["values"]

        if not candles or len(candles) < 15:
            print(f" ❌ Only {len(candles) if candles else 0} candles")
            return None

        records = []
        for candle in candles:
            try:
                records.append({
                    'time': candle.get('datetime'),
                    'open': safe_float(candle.get('open')),
                    'high': safe_float(candle.get('high')),
                    'low': safe_float(candle.get('low')),
                    'close': safe_float(candle.get('close')),
                    'volume': safe_float(candle.get('volume', 0))
                })
            except:
                continue

        if len(records) < 15:
            print(f" ❌ Bad data")
            return None

        df = pd.DataFrame(records)
        df = df.iloc[::-1].reset_index(drop=True)
        print(f" ✓ {len(df)} candles")
        return df

    except Exception as e:
        print(f" ❌ {str(e)[:40]}")
        return None


# ════════════════════════════════════════════════════════════════════════════════════
# INDICATOR CALCULATIONS
# ════════════════════════════════════════════════════════════════════════════════════

def calculate_rsi(close_prices, period=14):
    """Calculate RSI - FACTOR 2"""
    try:
        if len(close_prices) < period:
            return None

        close = np.array([safe_float(x) for x in close_prices])
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)

        avg_gain = np.mean(gain[-period:])
        avg_loss = np.mean(loss[-period:])

        if avg_loss == 0:
            return 100 if avg_gain > 0 else 0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return float(rsi)
    except:
        return None


def calculate_macd(close_prices, fast=12, slow=26, signal=9):
    """Calculate MACD - FACTOR 3"""
    try:
        if len(close_prices) < slow:
            return None, None, None

        close = np.array([safe_float(x) for x in close_prices])
        series = pd.Series(close)

        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line

        return float(macd_line.iloc[-1]), float(signal_line.iloc[-1]), float(histogram.iloc[-1])
    except:
        return None, None, None


def calculate_ema(close_prices, period):
    """Calculate EMA - FACTOR 4"""
    try:
        if len(close_prices) < period:
            return None

        close = np.array([safe_float(x) for x in close_prices])
        series = pd.Series(close)
        ema = series.ewm(span=period, adjust=False).mean()
        return float(ema.iloc[-1])
    except:
        return None


def check_volume(volumes, period=20):
    """Check if volume elevated - FACTOR 6"""
    try:
        if len(volumes) < period:
            return False, 1.0

        vols = np.array([safe_float(x) for x in volumes[-period:]])
        current = vols[-1]
        avg = np.mean(vols[:-1])

        ratio = current / avg if avg > 0 else 1.0
        return ratio > 1.2, ratio
    except:
        return False, 1.0


def check_fvg(highs, lows):
    """Check for Fair Value Gap - FACTOR 7"""
    try:
        if len(highs) < 3 or len(lows) < 3:
            return False

        # FVG: Gap between candle 1 high and candle 3 low
        if highs[-3] < lows[-1]:  # Gap up
            return True
        if lows[-3] > highs[-1]:  # Gap down
            return True

        return False
    except:
        return False


def check_support_resistance(closes, highs, lows, period=10):
    """Check near support/resistance - FACTOR 8"""
    try:
        if len(highs) < period:
            return False

        recent_high = max(highs[-period:-1])
        recent_low = min(lows[-period:-1])
        current = closes[-1]

        range_size = recent_high - recent_low

        # Check if price is near high or low (within 20%)
        if abs(current - recent_high) < range_size * 0.2:
            return True
        if abs(current - recent_low) < range_size * 0.2:
            return True

        return False
    except:
        return False


def log_result(pair, direction, score, order_type, entry, sl, tp1, tp2, tp3, factors_str):
    """Log trade result to CSV file"""
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        timestamp = datetime.now().strftime("%H:%M:%S")

        log_entry = {
            'Date': today,
            'Time': timestamp,
            'Pair': pair,
            'Direction': direction,
            'Score': score,
            'Recommendation': order_type if score >= 5 else 'SKIP',
            'Entry': f"{entry:.4f}",
            'SL': f"{sl:.4f}",
            'TP1': f"{tp1:.4f}",
            'TP2': f"{tp2:.4f}",
            'TP3': f"{tp3:.4f}",
            'Factors': factors_str
        }

        # Read existing CSV
        df = pd.read_csv(LOG_FILE)

        # Add new row
        new_row = pd.DataFrame([log_entry])
        df = pd.concat([df, new_row], ignore_index=True)

        # Save back
        df.to_csv(LOG_FILE, index=False)

        print(f"   📝 Logged: {pair} {direction} Score:{score}")
        return True

    except Exception as e:
        print(f"   ⚠️  Logging error: {str(e)[:40]}")
        return False


# ════════════════════════════════════════════════════════════════════════════════════
# ANALYSIS - ALL 8 FACTORS
# ════════════════════════════════════════════════════════════════════════════════════

def analyze_pair(df, pair_name):
    """Analyze pair with ALL 8 confluence factors"""
    try:
        if df is None or len(df) < 15:
            return {'pair': pair_name, 'status': 'NO_DATA'}

        closes = df['close'].values
        opens = df['open'].values
        highs = df['high'].values
        lows = df['low'].values
        volumes = df['volume'].values

        # Opening range
        opening_high = float(np.max(highs[:3]))
        opening_low = float(np.min(lows[:3]))
        current_price = float(closes[-1])

        # Breakout check
        if current_price > opening_high:
            direction = "LONG"
        elif current_price < opening_low:
            direction = "SHORT"
        else:
            return {
                'pair': pair_name,
                'status': 'NO_BREAKOUT',
                'range': f"{opening_low:.5f} - {opening_high:.5f}"
            }

        # Calculate indicators
        rsi = calculate_rsi(closes, 14)
        macd, signal, hist = calculate_macd(closes)
        ema20 = calculate_ema(closes, 20)
        ema50 = calculate_ema(closes, 50)
        is_elevated_vol, vol_ratio = check_volume(volumes)
        has_fvg = check_fvg(highs, lows)
        near_sr = check_support_resistance(closes, highs, lows)

        # SCORE ALL 8 FACTORS
        score = 0
        factors = {}

        # FACTOR 1: Breakout
        score += 1
        factors['1_breakout'] = f"✓ {direction}"

        # FACTOR 2: RSI
        rsi_value = rsi if rsi is not None else 0
        if rsi is not None and ((direction == "LONG" and 50 < rsi < 70) or (direction == "SHORT" and 30 < rsi < 50)):
            score += 1
            factors['2_rsi'] = f"✓ RSI {rsi_value:.1f}"
        else:
            factors['2_rsi'] = f"✗ RSI {rsi_value:.1f}"

        # FACTOR 3: MACD
        if macd and signal:
            if (direction == "LONG" and macd > signal and hist > 0) or (direction == "SHORT" and macd < signal and hist < 0):
                score += 1
                factors['3_macd'] = "✓ MACD bullish"
            else:
                factors['3_macd'] = "✗ MACD no signal"
        else:
            factors['3_macd'] = "? MACD"

        # FACTOR 4: EMA
        if ema20 and ema50:
            if (direction == "LONG" and current_price > ema20 > ema50) or (direction == "SHORT" and current_price < ema20 < ema50):
                score += 1
                factors['4_ema'] = "✓ EMA aligned"
            else:
                factors['4_ema'] = "✗ EMA not aligned"
        else:
            factors['4_ema'] = "? EMA"

        # FACTOR 5: Momentum
        if len(closes) >= 2:
            curr_range = abs(closes[-1] - opens[-1])
            prev_range = abs(closes[-2] - opens[-2])
            if curr_range > prev_range * 1.5:
                score += 1
                factors['5_momentum'] = "✓ Strong candle"
            else:
                factors['5_momentum'] = "✗ Weak candle"

        # FACTOR 6: Volume
        if is_elevated_vol:
            score += 1
            factors['6_volume'] = f"✓ Volume {vol_ratio:.2f}x"
        else:
            factors['6_volume'] = f"✗ Volume {vol_ratio:.2f}x"

        # FACTOR 7: Fair Value Gap
        if has_fvg:
            score += 1
            factors['7_fvg'] = "✓ FVG gap"
        else:
            factors['7_fvg'] = "✗ No FVG"

        # FACTOR 8: Support/Resistance
        if near_sr:
            score += 1
            factors['8_sr'] = "✓ Near S/R"
        else:
            factors['8_sr'] = "✗ Away S/R"

        # Calculate Entry and TP prices (same as current price for entry)
        entry_price = current_price

        # Calculate SL and TP with FIXED 30 PIPS
        if direction == "LONG":
            sl_distance = round(entry_price * SL_PERCENTAGE, 4)
            sl = round(entry_price - sl_distance, 4)
            tp1 = round(entry_price + (sl_distance * 1), 4)
            tp2 = round(entry_price + (sl_distance * 2), 4)
            tp3 = round(entry_price + (sl_distance * 3), 4)
            order_type = "Buy Limit"
        else:
            sl_distance = round(entry_price * SL_PERCENTAGE, 4)
            sl = round(entry_price + sl_distance, 4)
            tp1 = round(entry_price - (sl_distance * 1), 4)
            tp2 = round(entry_price - (sl_distance * 2), 4)
            tp3 = round(entry_price - (sl_distance * 3), 4)
            order_type = "Sell Limit"

        return {
            'pair': pair_name,
            'status': 'SETUP',
            'direction': direction,
            'order_type': order_type,
            'entry': entry_price,
            'price': current_price,
            'range': f"{opening_low:.5f} - {opening_high:.5f}",
            'score': score,
            'max_score': 8,
            'recommendation': 'TRADE' if score >= 5 else 'SKIP',
            'sl': sl,
            'tp1': tp1,
            'tp2': tp2,
            'tp3': tp3,
            'factors': factors
        }

    except Exception as e:
        return {'pair': pair_name, 'status': 'ERROR', 'message': str(e)[:40]}


# ════════════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════════════

def main():
    # CREATE LOG DIR AND CSV FIRST
    create_log_dir()
    initialize_csv()

    print("🚀 ORB Analyzer - UPDATED VERSION")
    print(f"📊 Pairs: {', '.join(PAIRS)}")
    print(f"⏱️  Check every {CHECK_INTERVAL}s")
    print(f"🔢 Runs: {MAX_ITERATIONS} times then stop")
    print(f"✅ Scoring: 0-8 points")
    print(f"🔐 API Key: From GitHub Secrets (Secure)")
    print(f"📊 SL: Fixed 30 pips")
    print(f"📝 Logging to: {LOG_FILE}")

    iteration = 0

    while iteration < MAX_ITERATIONS:
        iteration += 1
        print(f"\n{'='*80}")
        print(f"🔄 Check #{iteration}/{MAX_ITERATIONS} - {datetime.now().strftime('%H:%M:%S')}")
        print('='*80)

        for pair in PAIRS:
            print(f"\n📈 {pair}")
            df = get_twelvedata_candles(pair, API_KEY, TIMEFRAME)

            if df is None:
                print("   ⚠️  No data")
                time.sleep(1)
                continue

            result = analyze_pair(df, pair)

            if result['status'] == 'NO_DATA':
                print("   ⚠️  No data")
            elif result['status'] == 'NO_BREAKOUT':
                print(f"   Inside range: {result['range']}")
            elif result['status'] == 'ERROR':
                print(f"   ❌ {result['message']}")
            elif result['status'] == 'SETUP':
                print(f"Direction: {result['direction']}")
                print(f"Score: {result['score']}/{result['max_score']} → {result['recommendation']}")
                print(f"├─ Recommendation: {result['order_type']}")
                print(f"├─ Entry:")
                print(f"{result['entry']:.4f}")
                print(f"├─ SL:")
                print(f"{result['sl']:.4f}")
                print(f"├─ TP1:")
                print(f"{result['tp1']:.4f}")
                print(f"├─ TP2:")
                print(f"{result['tp2']:.4f}")
                print(f"└─ TP3:")
                print(f"{result['tp3']:.4f}")

                if result['factors']:
                    print(f"\nAll 8 Factors:")
                    for k in sorted(result['factors'].keys()):
                        print(f"   {result['factors'][k]}")

                # LOG THE TRADE
                factors_str = " | ".join([f"{k}:{v}" for k, v in sorted(result['factors'].items())])
                log_result(
                    result['pair'],
                    result['direction'],
                    result['score'],
                    result['order_type'],
                    result['entry'],
                    result['sl'],
                    result['tp1'],
                    result['tp2'],
                    result['tp3'],
                    factors_str
                )

            time.sleep(1)

        print(f"\n{'='*80}")
        if iteration < MAX_ITERATIONS:
            print(f"⏳ Next in {CHECK_INTERVAL}s (Check {iteration}/{MAX_ITERATIONS})")
            try:
                time.sleep(CHECK_INTERVAL)
            except KeyboardInterrupt:
                print("\n🛑 Stopped by user")
                break
        else:
            print(f"✅ Completed {MAX_ITERATIONS} checks. Stopping.")
            break

    print(f"✅ CSV file saved: {LOG_FILE}")


if __name__ == "__main__":
    main()
