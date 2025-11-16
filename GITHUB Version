#!/usr/bin/env python3
"""
FOREX ORB STRATEGY - WITH LOGGING & SCHEDULING
Logs all results to CSV (one column per day)
Can be scheduled on free server
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime
import time
import os
import csv

# ════════════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════════════

API_KEY = "ALPHA"
BASE_URL = "https://api.twelvedata.com"

PAIRS = ["EUR/USD", "GBP/USD", "USD/JPY"]  # Top 3 pairs only

TIMEFRAME = "5min"
MAX_ITERATIONS = 1  # Run once per execution

# Log file configuration
LOG_DIR = "trading_logs"
LOG_FILE = f"{LOG_DIR}/orb_trading_log.csv"

# ════════════════════════════════════════════════════════════════════════════════════
# LOGGING FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════════════

def create_log_dir():
    """Create logs directory if it doesn't exist"""
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
        print(f"✅ Created log directory: {LOG_DIR}")


def get_today_column():
    """Get today's date as column name (e.g., '2025-11-16')"""
    return datetime.now().strftime("%Y-%m-%d")


def log_result(pair, direction, score, recommendation, factors_str, sl, tp1, tp2, tp3):
    """Log trade result to CSV file"""
    try:
        create_log_dir()

        today = get_today_column()
        timestamp = datetime.now().strftime("%H:%M:%S")

        # Prepare log entry
        log_entry = {
            'Time': timestamp,
            'Pair': pair,
            'Direction': direction,
            'Score': score,
            'Recommendation': recommendation,
            'SL': f"{sl:.5f}",
            'TP1': f"{tp1:.5f}",
            'TP2': f"{tp2:.5f}",
            'TP3': f"{tp3:.5f}",
            'Factors': factors_str
        }

        # Check if file exists
        file_exists = os.path.isfile(LOG_FILE)

        # Read existing data or create new
        if file_exists:
            df = pd.read_csv(LOG_FILE)
        else:
            df = pd.DataFrame()

        # Add new row
        new_row = pd.DataFrame([log_entry])
        df = pd.concat([df, new_row], ignore_index=True)

        # Save to CSV
        df.to_csv(LOG_FILE, index=False)

        print(f"   📝 Logged to: {LOG_FILE}")
        return True

    except Exception as e:
        print(f"   ⚠️  Logging error: {str(e)[:40]}")
        return False


def print_log_summary():
    """Print summary of today's trades"""
    try:
        if not os.path.isfile(LOG_FILE):
            print("\n📊 No trades logged today yet")
            return

        df = pd.read_csv(LOG_FILE)
        today = get_today_column()

        # Filter today's trades
        today_trades = df[df['Time'].str.contains(today, na=False)]

        if len(today_trades) == 0:
            print("\n📊 No trades logged today")
            return

        print(f"\n{'='*80}")
        print(f"📊 TODAY'S TRADING LOG ({today})")
        print(f"{'='*80}")
        print(f"Total checks: {len(today_trades)}")

        # Count by recommendation
        trades = len(today_trades[today_trades['Recommendation'] == 'TRADE'])
        skips = len(today_trades[today_trades['Recommendation'] == 'SKIP'])

        print(f"TRADE signals: {trades}")
        print(f"SKIP signals: {skips}")

        print(f"\n{'Pair':<12} {'Dir':<8} {'Score':<8} {'Rec':<10}")
        print("-" * 40)
        for _, row in today_trades.iterrows():
            print(f"{row['Pair']:<12} {row['Direction']:<8} {row['Score']:<8} {row['Recommendation']:<10}")

    except Exception as e:
        print(f"Error reading log: {str(e)[:40]}")


# ════════════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS (same as before)
# ════════════════════════════════════════════════════════════════════════════════════

def safe_float(value):
    try:
        return float(value)
    except:
        return 0.0


def get_twelvedata_candles(pair, api_key, interval="5min", count=50):
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

        if "status" in data and data["status"] == "error":
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
# INDICATOR CALCULATIONS (abbreviated)
# ════════════════════════════════════════════════════════════════════════════════════

def calculate_rsi(close_prices, period=14):
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
    try:
        if len(highs) < 3 or len(lows) < 3:
            return False
        if highs[-3] < lows[-1]:
            return True
        if lows[-3] > highs[-1]:
            return True
        return False
    except:
        return False


def check_support_resistance(closes, highs, lows, period=10):
    try:
        if len(highs) < period:
            return False
        recent_high = max(highs[-period:-1])
        recent_low = min(lows[-period:-1])
        current = closes[-1]
        range_size = recent_high - recent_low
        if abs(current - recent_high) < range_size * 0.2:
            return True
        if abs(current - recent_low) < range_size * 0.2:
            return True
        return False
    except:
        return False


# ════════════════════════════════════════════════════════════════════════════════════
# ANALYSIS WITH LOGGING
# ════════════════════════════════════════════════════════════════════════════════════

def analyze_pair(df, pair_name):
    try:
        if df is None or len(df) < 15:
            return None

        closes = df['close'].values
        opens = df['open'].values
        highs = df['high'].values
        lows = df['low'].values
        volumes = df['volume'].values

        opening_high = float(np.max(highs[:3]))
        opening_low = float(np.min(lows[:3]))
        current_price = float(closes[-1])

        if current_price > opening_high:
            direction = "LONG"
        elif current_price < opening_low:
            direction = "SHORT"
        else:
            return None

        # Calculate indicators
        rsi = calculate_rsi(closes, 14)
        macd, signal, hist = calculate_macd(closes)
        ema20 = calculate_ema(closes, 20)
        ema50 = calculate_ema(closes, 50)
        is_elevated_vol, vol_ratio = check_volume(volumes)
        has_fvg = check_fvg(highs, lows)
        near_sr = check_support_resistance(closes, highs, lows)

        # Score
        score = 0
        factors = {}

        score += 1
        factors['1_breakout'] = "✓" if True else "✗"

        rsi_value = rsi if rsi is not None else 0
        if rsi is not None and ((direction == "LONG" and 50 < rsi < 70) or (direction == "SHORT" and 30 < rsi < 50)):
            score += 1
            factors['2_rsi'] = "✓"
        else:
            factors['2_rsi'] = "✗"

        if macd and signal:
            if (direction == "LONG" and macd > signal and hist > 0) or (direction == "SHORT" and macd < signal and hist < 0):
                score += 1
                factors['3_macd'] = "✓"
            else:
                factors['3_macd'] = "✗"

        if ema20 and ema50:
            if (direction == "LONG" and current_price > ema20 > ema50) or (direction == "SHORT" and current_price < ema20 < ema50):
                score += 1
                factors['4_ema'] = "✓"
            else:
                factors['4_ema'] = "✗"

        if len(closes) >= 2:
            curr_range = abs(closes[-1] - opens[-1])
            prev_range = abs(closes[-2] - opens[-2])
            if curr_range > prev_range * 1.5:
                score += 1
                factors['5_momentum'] = "✓"

        if is_elevated_vol:
            score += 1
            factors['6_volume'] = "✓"

        if has_fvg:
            score += 1
            factors['7_fvg'] = "✓"

        if near_sr:
            score += 1
            factors['8_sr'] = "✓"

        # SL and TP
        if direction == "LONG":
            sl = opening_low - 0.0015
            tp1 = current_price + 0.0020
            tp2 = current_price + 0.0030
            tp3 = current_price + 0.0050
        else:
            sl = opening_high + 0.0015
            tp1 = current_price - 0.0020
            tp2 = current_price - 0.0030
            tp3 = current_price - 0.0050

        # Create factors string for logging
        factors_str = " | ".join([f"{k}:{v}" for k, v in sorted(factors.items())])

        return {
            'pair': pair_name,
            'direction': direction,
            'score': score,
            'recommendation': 'TRADE' if score >= 5 else 'SKIP',
            'sl': sl,
            'tp1': tp1,
            'tp2': tp2,
            'tp3': tp3,
            'factors': factors_str
        }

    except Exception as e:
        return None


# ════════════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════════════

def main():
    print("🚀 ORB Analyzer - WITH LOGGING")
    print(f"📊 Pairs: {', '.join(PAIRS)}")
    print(f"⏱️  Runs: {MAX_ITERATIONS} time(s)")
    print(f"📝 Logging: {LOG_FILE}")
    print(f"🕐 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    for iteration in range(MAX_ITERATIONS):
        print(f"\n{'='*80}")
        print(f"🔄 Check #{iteration + 1}/{MAX_ITERATIONS}")
        print('='*80)

        for pair in PAIRS:
            print(f"\n📈 {pair}")
            df = get_twelvedata_candles(pair, API_KEY, TIMEFRAME)

            if df is None:
                print("   ⚠️  No data - skipping")
                time.sleep(1)
                continue

            result = analyze_pair(df, pair)

            if result is None:
                print("   Inside opening range - no setup")
                time.sleep(1)
                continue

            print(f"   Direction: {result['direction']}")
            print(f"   Score: {result['score']}/8 → {result['recommendation']}")
            print(f"   ├─ SL: {result['sl']:.5f}")
            print(f"   ├─ TP1: {result['tp1']:.5f}")
            print(f"   ├─ TP2: {result['tp2']:.5f}")
            print(f"   └─ TP3: {result['tp3']:.5f}")

            # LOG THE RESULT
            log_result(
                pair=result['pair'],
                direction=result['direction'],
                score=result['score'],
                recommendation=result['recommendation'],
                factors_str=result['factors'],
                sl=result['sl'],
                tp1=result['tp1'],
                tp2=result['tp2'],
                tp3=result['tp3']
            )

            time.sleep(1)

    # Print summary
    print_log_summary()
    print(f"\n✅ Completed")


if __name__ == "__main__":
    main()
