#!/usr/bin/env python3
"""
FOREX ORB STRATEGY - COMPLETE VERSION WITH ALL 8 CONFLUENCE FACTORS
TwelveData - Real-time data
All 8 factors: Breakout, RSI, MACD, EMA, Momentum, Volume, FVG, Support/Resistance
API Key from GitHub Secrets (Secure)

UPDATED WITH:
- Score-based dynamic lot sizing (5-8)
- Pair-type-specific risk/reward targets (Standard vs Gold)
- Pip-based SL/TP calculations
- Enhanced CSV logging with Risk($), Reward($), R/R Ratio, Pips
- Phone-friendly console output
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
os.environ['USE_GOOGLE_DRIVE'] = 'true'
# Read API key from GitHub Secrets environment variable
API_KEY = os.getenv('TWELVEDATA_API_KEY', 'ALPHA')
BASE_URL = "https://api.twelvedata.com"

# Trading pairs and their characteristics
PAIRS = ["EUR/USD", "GBP/USD", "USD/JPY", "USD/CAD", "EUR/GBP", "XAU/USD", "USD/CHF", "AUD/USD"]

# Pair type mapping
PAIR_TYPES = {
    "EUR/USD": "standard",
    "GBP/USD": "standard",
    "USD/JPY": "jpy",
    "USD/CAD": "standard",
    "EUR/GBP": "standard",
    "XAU/USD": "gold",
    "USD/CHF": "standard",
    "AUD/USD": "standard"
}

# Pip values per lot (standard forex)
PIP_VALUES = {
    "standard": 10,    # $10 per pip per 1 lot
    "jpy": 10,         # $10 per pip per 1 lot (despite 0.01 move)
    "gold": 1          # $1 per pip per 1 lot (0.01 move = 1 pip)
}

# ════════════════════════════════════════════════════════════════════════════════════
# LOT SIZE AND RISK/REWARD CONFIGURATION (UPDATED)
# ════════════════════════════════════════════════════════════════════════════════════

# Score-based lot sizing (scales from 5/8 to 8/8)
LOT_SIZES = {
    "standard": {  # Forex pairs
        5: 0.01,   # 5/8 = Safe
        6: 0.02,   # 6/8 = Standard
        7: 0.03,   # 7/8 = Aggressive
        8: 0.04    # 8/8 = Risky
    },
    "jpy": {       # JPY pairs
        5: 0.01,
        6: 0.02,
        7: 0.03,
        8: 0.04
    },
    "gold": {      # Gold (XAU/USD)
        5: 0.01,   # 5/8 = Safe
        6: 0.02,   # 6/8 = Standard
        7: 0.03,   # 7/8 = Aggressive
        8: 0.04    # 8/8 = Risky
    }
}

# Risk/Reward targets by pair type and score
RISK_REWARD = {
    "standard": {  # Forex: TP $10/$20/$30, SL $50
        5: {"risk": 50, "profit_tp1": 10, "profit_tp2": 20, "profit_tp3": 30},
        6: {"risk": 50, "profit_tp1": 10, "profit_tp2": 20, "profit_tp3": 30},
        7: {"risk": 50, "profit_tp1": 10, "profit_tp2": 20, "profit_tp3": 30},
        8: {"risk": 50, "profit_tp1": 10, "profit_tp2": 20, "profit_tp3": 30}
    },
    "jpy": {       # JPY pairs: same as standard
        5: {"risk": 50, "profit_tp1": 10, "profit_tp2": 20, "profit_tp3": 30},
        6: {"risk": 50, "profit_tp1": 10, "profit_tp2": 20, "profit_tp3": 30},
        7: {"risk": 50, "profit_tp1": 10, "profit_tp2": 20, "profit_tp3": 30},
        8: {"risk": 50, "profit_tp1": 10, "profit_tp2": 20, "profit_tp3": 30}
    },
    "gold": {      # Gold: TP $50/$100/$150, SL $200
        5: {"risk": 200, "profit_tp1": 50, "profit_tp2": 100, "profit_tp3": 150},
        6: {"risk": 200, "profit_tp1": 50, "profit_tp2": 100, "profit_tp3": 150},
        7: {"risk": 200, "profit_tp1": 50, "profit_tp2": 100, "profit_tp3": 150},
        8: {"risk": 200, "profit_tp1": 50, "profit_tp2": 100, "profit_tp3": 150}
    }
}

TIMEFRAME = "5min"
CHECK_INTERVAL = 300  # 5 minutes
MAX_ITERATIONS = 1  # Stop after 1 check

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
        # Create header row with all columns
        df = pd.DataFrame(columns=[
            'Date', 'Time', 'Pair', 'Direction', 'Score', 'Recommendation', 'Lot',
            'Entry', 'SL', 'SL_Pips', 'TP1', 'TP1_Pips', 'TP2', 'TP2_Pips', 'TP3', 'TP3_Pips',
            'Risk_$', 'Reward_TP1_$', 'Reward_TP2_$', 'RiskReward_Ratio', 'Factors'
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


def get_pair_type(pair):
    """Get pair type (standard, jpy, gold)"""
    return PAIR_TYPES.get(pair, "standard")


def get_pip_value(pair_type):
    """Get pip value for pair type"""
    return PIP_VALUES.get(pair_type, 10)


def get_lot_size(pair_type, score):
    """Get lot size based on pair type and score (5-8)
    
    Score 5 = 0.01 (Safe)
    Score 6 = 0.02 (Standard)
    Score 7 = 0.03 (Aggressive)
    Score 8 = 0.04 (Risky)
    """
    if score >= 5:
        # Use exact score, fallback to 0.01 if not found
        return LOT_SIZES.get(pair_type, {}).get(score, 0.01)
    else:
        return 0.0  # No trade below score 5


def get_risk_reward(pair_type, score):
    """Get risk/reward targets based on pair type and score (5-8)"""
    if score >= 5:
        # Use exact score, fallback to safe values if not found
        return RISK_REWARD.get(pair_type, {}).get(score, {
            "risk": 50,
            "profit_tp1": 10,
            "profit_tp2": 20,
            "profit_tp3": 30
        })
    else:
        return {
            "risk": 0,
            "profit_tp1": 0,
            "profit_tp2": 0,
            "profit_tp3": 0
        }


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


def calculate_sl_tp(entry_price, direction, pair_type, lot_size, score):
    """
    Calculate SL and TP levels based on pip calculations
    Returns: sl, tp1, tp2, tp3, sl_pips, tp1_pips, tp2_pips, tp3_pips
    """
    # Get pip value and risk/reward targets
    pip_value = get_pip_value(pair_type)
    risk_reward = get_risk_reward(pair_type, score)
    risk_amount = risk_reward["risk"]
    profit_tp1 = risk_reward["profit_tp1"]
    profit_tp2 = risk_reward.get("profit_tp2", profit_tp1 * 2)
    profit_tp3 = risk_reward.get("profit_tp3", profit_tp1 * 3)
    
    # Calculate pips needed for risk and TPs
    sl_pips = round(risk_amount / (pip_value * lot_size), 1)
    tp1_pips = round(profit_tp1 / (pip_value * lot_size), 1)
    tp2_pips = round(profit_tp2 / (pip_value * lot_size), 1)
    tp3_pips = round(profit_tp3 / (pip_value * lot_size), 1)
    
    # Convert pips to price levels
    if pair_type == "gold":
        pip_multiplier = 0.01
    elif pair_type == "jpy":
        pip_multiplier = 0.01
    else:
        pip_multiplier = 0.0001
    
    # Calculate SL and TP prices
    if direction == "LONG":
        sl = round(entry_price - (sl_pips * pip_multiplier), 4)
        tp1 = round(entry_price + (tp1_pips * pip_multiplier), 4)
        tp2 = round(entry_price + (tp2_pips * pip_multiplier), 4)
        tp3 = round(entry_price + (tp3_pips * pip_multiplier), 4)
    else:  # SHORT
        sl = round(entry_price + (sl_pips * pip_multiplier), 4)
        tp1 = round(entry_price - (tp1_pips * pip_multiplier), 4)
        tp2 = round(entry_price - (tp2_pips * pip_multiplier), 4)
        tp3 = round(entry_price - (tp3_pips * pip_multiplier), 4)
    
    return sl, tp1, tp2, tp3, sl_pips, tp1_pips, tp2_pips, tp3_pips


def log_result_with_gdrive(pair, direction, score, order_type, entry, sl, tp1, tp2, tp3,
                           sl_pips, tp1_pips, tp2_pips, tp3_pips, lot_size, risk_amount,
                           profit_tp1, profit_tp2, factors_str, upload_to_gdrive=False, drive_filename=None):
    """
    Log trade result to BOTH local CSV and Google Drive CSV simultaneously
    NEW ENTRIES APPEAR AT TOP (after header)
    Both CSVs stay in sync!
    """
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        timestamp = datetime.now().strftime("%H:%M:%S")

        # Calculate actual profit amounts at each TP
        pair_type = get_pair_type(pair)
        pip_value = get_pip_value(pair_type)

        profit_tp1_amount = round(tp1_pips * pip_value * lot_size, 2)
        profit_tp2_amount = round(tp2_pips * pip_value * lot_size, 2)

        # Risk/Reward ratio
        rr_ratio = round(profit_tp1_amount / risk_amount, 2) if risk_amount > 0 else 0

        # Create new row as DataFrame
        new_row_data = {
            'Date': [today],
            'Time': [timestamp],
            'Pair': [pair],
            'Direction': [direction],
            'Score': [score],
            'Recommendation': [order_type if score >= 5 else 'SKIP'],
            'Lot': [f"{lot_size:.2f}"],
            'Entry': [f"{entry:.4f}"],
            'SL': [f"{sl:.4f}"],
            'SL_Pips': [f"{sl_pips:.1f}"],
            'TP1': [f"{tp1:.4f}"],
            'TP1_Pips': [f"{tp1_pips:.1f}"],
            'TP2': [f"{tp2:.4f}"],
            'TP2_Pips': [f"{tp2_pips:.1f}"],
            'TP3': [f"{tp3:.4f}"],
            'TP3_Pips': [f"{tp3_pips:.1f}"],
            'Risk_$': [f"{risk_amount:.2f}"],
            'Reward_TP1_$': [f"{profit_tp1_amount:.2f}"],
            'Reward_TP2_$': [f"{profit_tp2_amount:.2f}"],
            'RiskReward_Ratio': [f"1:{rr_ratio:.2f}"],
            'Factors': [factors_str]
        }

        new_row_df = pd.DataFrame(new_row_data)

        # ════════════════════════════════════════════════════════════════════════
        # 1. UPDATE LOCAL CSV (newest first)
        # ════════════════════════════════════════════════════════════════════════

        # Read existing local CSV
        if os.path.exists(LOG_FILE):
            existing_df = pd.read_csv(LOG_FILE)
        else:
            existing_df = pd.DataFrame()

        # PREPEND NEW ROW (new entries at TOP, after header)
        if not existing_df.empty:
            combined_df = pd.concat([new_row_df, existing_df], ignore_index=True)
        else:
            combined_df = new_row_df

        # Write back to local file
        combined_df.to_csv(LOG_FILE, index=False)
        print(f"   ✅ Local CSV updated: {LOG_FILE}")

        # ════════════════════════════════════════════════════════════════════════
        # 2. UPDATE GOOGLE DRIVE CSV (if enabled)
        # ════════════════════════════════════════════════════════════════════════

        if upload_to_gdrive and drive_filename:
            try:
                from google_drive_manager import GoogleDriveManager

                drive_manager = GoogleDriveManager()

                # Download current Google Drive CSV
                gdrive_csv_path = f"temp_{drive_filename}"
                try:
                    drive_manager.download_csv(drive_filename, gdrive_csv_path)
                    gdrive_df = pd.read_csv(gdrive_csv_path)
                except:
                    # If file doesn't exist on Drive yet, create new
                    gdrive_df = pd.DataFrame()

                # PREPEND NEW ROW to Google Drive CSV (same as local)
                if not gdrive_df.empty:
                    gdrive_combined = pd.concat([new_row_df, gdrive_df], ignore_index=True)
                else:
                    gdrive_combined = new_row_df

                # Save to temp file
                gdrive_combined.to_csv(gdrive_csv_path, index=False)

                # Upload to Google Drive
                drive_manager.upload_csv(gdrive_csv_path, drive_filename)

                # Clean up temp file
                if os.path.exists(gdrive_csv_path):
                    os.remove(gdrive_csv_path)

                print(f"   ✅ Google Drive CSV updated: {drive_filename}")

            except ImportError:
                print(f"   ⚠️  Google Drive module not installed. Only local CSV updated.")
            except Exception as e:
                print(f"   ⚠️  Google Drive error: {str(e)[:50]}")
                print(f"      (Local CSV still updated successfully)")

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

        # Get pair type and lot size
        pair_type = get_pair_type(pair_name)
        lot_size = get_lot_size(pair_type, score)
        
        # Calculate SL and TP with new pip-based system
        if score >= 5 and lot_size > 0:
            entry_price = current_price
            sl, tp1, tp2, tp3, sl_pips, tp1_pips, tp2_pips, tp3_pips = calculate_sl_tp(
                entry_price, direction, pair_type, lot_size, score
            )
            
            # Get risk/reward amounts
            risk_reward = get_risk_reward(pair_type, score)
            risk_amount = risk_reward["risk"]
            profit_tp1 = risk_reward["profit_tp1"]
            profit_tp2 = risk_reward.get("profit_tp2", profit_tp1 * 2)
            
            order_type = "Buy Limit" if direction == "LONG" else "Sell Limit"
        else:
            sl = tp1 = tp2 = tp3 = 0
            sl_pips = tp1_pips = tp2_pips = tp3_pips = 0
            risk_amount = profit_tp1 = profit_tp2 = 0
            order_type = "SKIP"
            entry_price = current_price

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
            'sl_pips': sl_pips,
            'tp1_pips': tp1_pips,
            'tp2_pips': tp2_pips,
            'tp3_pips': tp3_pips,
            'lot_size': lot_size,
            'risk_amount': risk_amount,
            'profit_tp1': profit_tp1,
            'profit_tp2': profit_tp2,
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
        
    # Get Google Drive settings from environment variables
    USE_GDRIVE = os.getenv('USE_GOOGLE_DRIVE', 'false').lower() == 'true'
    GDRIVE_FILENAME = os.getenv('GDRIVE_CSV_FILENAME', 'orb_trading_log.csv')
    
    if USE_GDRIVE:
        print(f"📤 Google Drive enabled: {GDRIVE_FILENAME}")

    print("🚀 ORB Analyzer - UPDATED WITH DYNAMIC LOT & PIP-BASED SL/TP")
    print(f"📊 Pairs: {', '.join(PAIRS)}")
    print(f"⏱️  Check every {CHECK_INTERVAL}s")

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
                if result['recommendation'] == "SKIP":
                    print("   ⏭️ Recommendation: SKIP")
                else:
                    print(f"Score: {result['score']}/{result['max_score']} → {result['recommendation']}")
                    # ANSI escape codes for formatting
                    BOLD = "\033[1m"
                    RESET = "\033[0m"
                    BIG = "\033[1;37m"  # bright white for emphasis
                    
                    # Bold + emphasized order_type
                    print(f"├─ Recommendation: {BOLD}{BIG}{result['order_type']}{RESET}")
                    
                    # Lot size
                    print(f"├─ Lot Size:")
                    print(f"   {result['lot_size']:.2f}")
                    
                    print(f"├─ Entry:")
                    print(f"   {result['entry']:.4f}")
                    
                    # NEW FORMAT FOR SL/TP
                    print(f"├─ SL: ({result['sl_pips']:.1f} pips, ${result['risk_amount']:.0f} risk)")
                    print(f"   {result['sl']:.4f}")
                    
                    print(f"├─ TP1: ({result['tp1_pips']:.1f} pips, ${result['profit_tp1']:.0f} profit)")
                    print(f"   {result['tp1']:.4f}")
                    
                    print(f"├─ TP2: ({result['tp2_pips']:.1f} pips, ${result['profit_tp2']:.0f} profit)")
                    print(f"   {result['tp2']:.4f}")
                    
                    print(f"└─ TP3: ({result['tp3_pips']:.1f} pips, ${result['profit_tp1']*3:.0f} profit)")
                    print(f"   {result['tp3']:.4f}")

                    
                    # LOG THE TRADE
                    factors_str = " | ".join([f"{k}:{v}" for k, v in sorted(result['factors'].items())])
                    log_result_with_gdrive(
                        result['pair'],
                        result['direction'],
                        result['score'],
                        result['order_type'],
                        result['entry'],
                        result['sl'],
                        result['tp1'],
                        result['tp2'],
                        result['tp3'],
                        result['sl_pips'],
                        result['tp1_pips'],
                        result['tp2_pips'],
                        result['tp3_pips'],
                        result['lot_size'],
                        result['risk_amount'],
                        result['profit_tp1'],
                        result['profit_tp2'],
                        factors_str,
                        upload_to_gdrive=USE_GDRIVE,
                        drive_filename=GDRIVE_FILENAME
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
