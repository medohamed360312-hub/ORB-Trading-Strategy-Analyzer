#!/usr/bin/env python3
"""
FOREX ORB STRATEGY - DYNAMIC RISK MANAGEMENT VERSION
TwelveData - Real-time data
All 8 factors: Breakout, RSI, MACD, EMA, Momentum, Volume, FVG, Support/Resistance
Dynamic: Risk%, SL, TP, Lot sizing based on 8-factor score
Score-based: Higher confidence = Higher risk/reward
API Key from GitHub Secrets (Secure)
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime
import time
import os
import math

# ════════════════════════════════════════════════════════════════════════════════════
# CONFIGURATION & CONSTANTS
# ════════════════════════════════════════════════════════════════════════════════════

API_KEY = os.getenv('TWELVEDATA_API_KEY', 'ALPHA')
BASE_URL = "https://api.twelvedata.com"

# Updated pairs: Added USD/CHF and AUD/USD
PAIRS = ["EUR/USD", "GBP/USD", "USD/JPY", "USD/CAD", "EUR/GBP", "XAU/USD","USD/CHF", "AUD/USD"]

# Pair configuration with pip values and decimal precision
PIP_VALUES = {
    "EUR/USD": {"pip_value": 10, "decimals": 4, "pip_min": 0.0001, "margin_req": 1000},
    "GBP/USD": {"pip_value": 10, "decimals": 4, "pip_min": 0.0001, "margin_req": 1000},
    "USD/JPY": {"pip_value": 0.09, "decimals": 4, "pip_min": 0.01, "margin_req": 1000},
    "USD/CAD": {"pip_value": 10, "decimals": 4, "pip_min": 0.0001, "margin_req": 1000},
    "EUR/GBP": {"pip_value": 10, "decimals": 4, "pip_min": 0.0001, "margin_req": 1000},
    "XAU/USD": {"pip_value": 10, "decimals": 4, "pip_min": 0.0001, "margin_req": 5000},
    "USD/CHF": {"pip_value": 10, "decimals": 4, "pip_min": 0.0001, "margin_req": 1000},
    "AUD/USD": {"pip_value": 10, "decimals": 4, "pip_min": 0.0001, "margin_req": 1000},
}

# Account parameters
ACCOUNT_EQUITY = 3417
ACCOUNT_LEVERAGE = 30
MAX_MARGIN_UTILIZATION = 0.5  # 50%

TIMEFRAME = "5min"
CHECK_INTERVAL = 300  # 5 minutes
MAX_ITERATIONS = 1
LOG_DIR = "trading_logs"
LOG_FILE = f"{LOG_DIR}/orb_trading_log.csv"

# ════════════════════════════════════════════════════════════════════════════════════
# LOG DIRECTORY & CSV INITIALIZATION
# ════════════════════════════════════════════════════════════════════════════════════

def create_log_dir():
    """Create logs directory"""
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
        print(f"✅ Log directory: {LOG_DIR}")


def initialize_csv():
    """Create enhanced CSV with dynamic calculation columns"""
    if not os.path.isfile(LOG_FILE):
        df = pd.DataFrame(columns=[
            'Date', 'Time', 'Pair', 'Direction', 'Score', 'Recommendation',
            'Entry', 'SL', 'TP',
            'ATR', 'SL_Multiplier', 'SL_Distance', 'SL_Pips',
            'Risk%', 'Risk_Amount$', 'RR_Ratio', 'TP_Distance', 'TP_Pips',
            'Base_Lot', 'Adjusted_Lot', 'Final_Lot',
            'Margin_Used$', 'Margin_Util%',
            'Factors'
        ])
        df.to_csv(LOG_FILE, index=False)
        print(f"✅ CSV created: {LOG_FILE}")


# ════════════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════════════

def safe_float(value):
    """Safely convert to float"""
    try:
        return float(value)
    except:
        return 0.0


def round_down_lot(lot_size, precision=0.01):
    """Round lot size DOWN to specified precision"""
    return math.floor(lot_size / precision) * precision


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
            return None

        data = response.json()
        if "status" in data and data["status"] == "error":
            return None
        if "values" not in data:
            return None

        candles = data["values"]
        if not candles or len(candles) < 15:
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
            return None

        df = pd.DataFrame(records)
        df = df.iloc[::-1].reset_index(drop=True)
        return df

    except Exception as e:
        return None


# ════════════════════════════════════════════════════════════════════════════════════
# DYNAMIC RISK MANAGEMENT FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════════════

def calculate_atr(closes, period=14):
    """Calculate Average True Range (for volatility)"""
    try:
        if len(closes) < period:
            return None

        close = np.array([safe_float(x) for x in closes])
        tr = np.abs(np.diff(close))
        atr = np.mean(tr[-period:])
        return float(atr)
    except:
        return None


def calculate_risk_percentage(score):
    """Calculate risk % based on score: 2% to 3%"""
    if score <= 0:
        return None
    return 2.0 + (score * 0.125)


def calculate_rr_ratio(score):
    """Calculate Risk:Reward ratio based on score: 1:1 to 2.5:1"""
    return 1.0 + (score * 0.1875)


def calculate_sl_distance(atr, score):
    """Calculate SL distance based on ATR and score"""
    if atr is None or score <= 0:
        return None

    sl_multiplier = 1.0 + (0.25 * score / 8)
    sl_distance = atr * sl_multiplier
    return sl_distance, sl_multiplier


def convert_to_pips(distance, pip_min):
    """Convert distance to pips"""
    try:
        if pip_min <= 0:
            return 0
        pips = distance / pip_min
        return pips
    except:
        return 0


def calculate_base_lot(risk_amount, sl_pips, pip_value):
    """Calculate base lot from risk amount"""
    try:
        if sl_pips <= 0 or pip_value <= 0:
            return 0

        base_lot = risk_amount / (sl_pips * pip_value)
        return base_lot
    except:
        return 0


def adjust_lot_for_score(base_lot, score):
    """Adjust lot by score multiplier"""
    if score <= 0:
        return base_lot
    return base_lot * (score / 8)


def check_margin_safety(lot, entry_price, leverage, equity, max_util=0.5):
    """Check margin safety and reduce lot if needed"""
    try:
        # Calculate margin used
        margin_used = (lot * 100000 * entry_price) / leverage
        max_available_margin = equity * leverage * max_util
        margin_util_pct = (margin_used / max_available_margin) * 100

        # If over limit, reduce proportionally
        if margin_util_pct > (max_util * 100):
            reduction_factor = (max_util / (margin_util_pct / 100))
            final_lot = lot * reduction_factor
            return final_lot, margin_util_pct, margin_used

        return lot, margin_util_pct, margin_used
    except:
        return lot, 0, 0


def recalculate_effective_pips(risk_amount, final_lot, pip_value, pip_min):
    """Recalculate effective SL distance (price) and pips based on final lot"""
    try:
        if final_lot <= 0 or pip_value <= 0 or pip_min <= 0:
            return 0, 0

        # How many pips can we afford with this final lot size?
        final_sl_pips = risk_amount / (final_lot * pip_value)

        # Convert pips back into a PRICE distance
        final_sl_distance = final_sl_pips * pip_min

        return final_sl_distance, final_sl_pips
    except:
        return 0, 0

def calculate_sl_tp(entry, sl_distance, rr_ratio, direction):
    """Calculate SL and TP prices"""
    try:
        if direction == "LONG":
            sl = entry - sl_distance
            tp = entry + (sl_distance * rr_ratio)
        else:  # SHORT
            sl = entry + sl_distance
            tp = entry - (sl_distance * rr_ratio)

        return sl, tp
    except:
        return None, None


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

        if highs[-3] < lows[-1] or lows[-3] > highs[-1]:
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

        if abs(current - recent_high) < range_size * 0.2 or abs(current - recent_low) < range_size * 0.2:
            return True
        return False
    except:
        return False


# ════════════════════════════════════════════════════════════════════════════════════
# LOGGING FUNCTION
# ════════════════════════════════════════════════════════════════════════════════════

def log_enhanced_result(pair, direction, score, recommendation, calculations, factors_str):
    """Log complete trade with all calculation data"""
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        timestamp = datetime.now().strftime("%H:%M:%S")

        log_entry = {
            'Date': today,
            'Time': timestamp,
            'Pair': pair,
            'Direction': direction,
            'Score': score,
            'Recommendation': recommendation,
            'Entry': f"{calculations['entry']:.4f}",
            'SL': f"{calculations['sl']:.4f}",
            'TP': f"{calculations['tp']:.4f}",
            'ATR': f"{calculations['atr']:.6f}",
            'SL_Multiplier': f"{calculations['sl_multiplier']:.4f}",
            'SL_Distance': f"{calculations['sl_distance']:.6f}",
            'SL_Pips': f"{calculations['sl_pips']:.2f}",
            'Risk%': f"{calculations['risk_pct']:.3f}%",
            'Risk_Amount$': f"{calculations['risk_amount']:.2f}",
            'RR_Ratio': f"{calculations['rr_ratio']:.2f}:1",
            'TP_Distance': f"{calculations['tp_distance']:.6f}",
            'TP_Pips': f"{calculations['tp_pips']:.2f}",
            'Base_Lot': f"{calculations['base_lot']:.3f}",
            'Adjusted_Lot': f"{calculations['adjusted_lot']:.3f}",
            'Final_Lot': f"{calculations['final_lot']:.3f}",
            'Margin_Used$': f"{calculations['margin_used']:.2f}",
            'Margin_Util%': f"{calculations['margin_util_pct']:.2f}%",
            'Factors': factors_str
        }

        df = pd.read_csv(LOG_FILE)
        new_row = pd.DataFrame([log_entry])
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(LOG_FILE, index=False)
        return True

    except Exception as e:
        print(f"   ⚠️  Logging error: {str(e)[:40]}")
        return False


# ════════════════════════════════════════════════════════════════════════════════════
# MAIN ANALYSIS FUNCTION
# ════════════════════════════════════════════════════════════════════════════════════

def analyze_pair(df, pair_name):
    """Analyze pair with ALL 8 confluence factors + dynamic risk management"""
    try:
        if df is None or len(df) < 15 or pair_name not in PIP_VALUES:
            return {'pair': pair_name, 'status': 'NO_DATA'}

        closes = df['close'].values
        opens = df['open'].values
        highs = df['high'].values
        lows = df['low'].values
        volumes = df['volume'].values

        # Opening range breakout check
        opening_high = float(np.max(highs[:3]))
        opening_low = float(np.min(lows[:3]))
        current_price = float(closes[-1])

        if current_price > opening_high:
            direction = "LONG"
        elif current_price < opening_low:
            direction = "SHORT"
        else:
            return {'pair': pair_name, 'status': 'NO_BREAKOUT'}

        # Calculate indicators for 8 factors
        rsi = calculate_rsi(closes, 14)
        macd, signal, hist = calculate_macd(closes)
        ema20 = calculate_ema(closes, 20)
        ema50 = calculate_ema(closes, 50)
        is_elevated_vol, vol_ratio = check_volume(volumes)
        has_fvg = check_fvg(highs, lows)
        near_sr = check_support_resistance(closes, highs, lows)
        atr = calculate_atr(closes, 14)

        # Score all 8 factors
        score = 0
        factors = {}

        # FACTOR 1: Breakout
        score += 1
        factors['1_breakout'] = f"✓ {direction}"

        # FACTOR 2: RSI
        rsi_value = rsi if rsi is not None else 0
        if rsi is not None and ((direction == "LONG" and 50 < rsi < 70) or (direction == "SHORT" and 30 < rsi < 50)):
            score += 1
            factors['2_rsi'] = f"✓ {rsi_value:.1f}"
        else:
            factors['2_rsi'] = f"✗ {rsi_value:.1f}"

        # FACTOR 3: MACD
        if macd and signal:
            if (direction == "LONG" and macd > signal and hist > 0) or (direction == "SHORT" and macd < signal and hist < 0):
                score += 1
                factors['3_macd'] = "✓"
            else:
                factors['3_macd'] = "✗"
        else:
            factors['3_macd'] = "?"

        # FACTOR 4: EMA
        if ema20 and ema50:
            if (direction == "LONG" and current_price > ema20 > ema50) or (direction == "SHORT" and current_price < ema20 < ema50):
                score += 1
                factors['4_ema'] = "✓"
            else:
                factors['4_ema'] = "✗"
        else:
            factors['4_ema'] = "?"

        # FACTOR 5: Momentum
        if len(closes) >= 2:
            curr_range = abs(closes[-1] - opens[-1])
            prev_range = abs(closes[-2] - opens[-2])
            if curr_range > prev_range * 1.5:
                score += 1
                factors['5_momentum'] = "✓"
            else:
                factors['5_momentum'] = "✗"

        # FACTOR 6: Volume
        if is_elevated_vol:
            score += 1
            factors['6_volume'] = "✓"
        else:
            factors['6_volume'] = "✗"

        # FACTOR 7: Fair Value Gap
        if has_fvg:
            score += 1
            factors['7_fvg'] = "✓"
        else:
            factors['7_fvg'] = "✗"

        # FACTOR 8: Support/Resistance
        if near_sr:
            score += 1
            factors['8_sr'] = "✓"
        else:
            factors['8_sr'] = "✗"

        # Skip if no score
        if score == 0:
            return {'pair': pair_name, 'status': 'NO_SCORE'}

        # ─────────────────────────────────────────────────────────────────────
        # DYNAMIC RISK MANAGEMENT CALCULATIONS
        # ─────────────────────────────────────────────────────────────────────

        pair_config = PIP_VALUES[pair_name]
        entry_price = current_price

        # 1. Calculate risk %
        risk_pct = calculate_risk_percentage(score)
        if risk_pct is None:
            risk_pct = 2.0

        # 2. Calculate risk amount
        risk_amount = ACCOUNT_EQUITY * (risk_pct / 100)

        # 3. Calculate R:R ratio
        rr_ratio = calculate_rr_ratio(score)

        # 4. Calculate SL distance and multiplier
        sl_result = calculate_sl_distance(atr, score)
        if sl_result is None:
            return {'pair': pair_name, 'status': 'ERROR', 'message': 'ATR calc failed'}

        sl_distance, sl_multiplier = sl_result

        # 5. Convert to pips
        sl_pips_initial = convert_to_pips(sl_distance, pair_config['pip_min'])

        # 6. Calculate base lot
        base_lot = calculate_base_lot(risk_amount, sl_pips_initial, pair_config['pip_value'])

        # 7. Adjust for score
        adjusted_lot = adjust_lot_for_score(base_lot, score)

        # 8. Apply constraints and round
        adjusted_lot = max(0.01, min(2.0, adjusted_lot))  # Min 0.01, Max 2.0
        adjusted_lot = round_down_lot(adjusted_lot, 0.01)

        # 9. Margin safety check
        final_lot, margin_util_pct, margin_used = check_margin_safety(
            adjusted_lot, entry_price, ACCOUNT_LEVERAGE, ACCOUNT_EQUITY, MAX_MARGIN_UTILIZATION
        )
        final_lot = round_down_lot(final_lot, 0.01)

        # 10. Recalculate effective pips based on final lot
        final_sl_distance, final_sl_pips = recalculate_effective_pips(
            risk_amount, final_lot, pair_config['pip_value'], pair_config['pip_min']
        )

        # 11. Calculate SL and TP prices
        sl, tp = calculate_sl_tp(entry_price, final_sl_distance, rr_ratio, direction)

        if sl is None or tp is None:
            return {'pair': pair_name, 'status': 'ERROR', 'message': 'SL/TP calc failed'}

        # Round to pair decimals
        decimals = pair_config['decimals']
        sl = round(sl, decimals)
        tp = round(tp, decimals)

        # Calculate TP pips
        tp_distance = abs(tp - entry_price)
        tp_pips = convert_to_pips(tp_distance, pair_config['pip_min'])

        # Prepare calculations dict for logging
        calculations = {
            'entry': entry_price,
            'sl': sl,
            'tp': tp,
            'atr': atr if atr is not None else 0,
            'sl_multiplier': sl_multiplier,
            'sl_distance': final_sl_distance,
            'sl_pips': final_sl_pips,
            'risk_pct': risk_pct,
            'risk_amount': risk_amount,
            'rr_ratio': rr_ratio,
            'tp_distance': tp_distance,
            'tp_pips': tp_pips,
            'base_lot': base_lot,
            'adjusted_lot': adjusted_lot,
            'final_lot': final_lot,
            'margin_used': margin_used,
            'margin_util_pct': margin_util_pct
        }

        recommendation = 'TRADE' if score >= 5 else 'SKIP'
        order_type = f"BUY {final_lot}" if direction == "LONG" else f"SELL {final_lot}"

        return {
            'pair': pair_name,
            'status': 'SETUP',
            'direction': direction,
            'order_type': order_type,
            'score': score,
            'recommendation': recommendation,
            'calculations': calculations,
            'factors': factors
        }

    except Exception as e:
        return {'pair': pair_name, 'status': 'ERROR', 'message': str(e)[:40]}


# ════════════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION - UPDATED OUTPUT FORMAT
# ════════════════════════════════════════════════════════════════════════════════════

def main():
    create_log_dir()
    initialize_csv()

    print("🚀 ORB Analyzer - DYNAMIC RISK MANAGEMENT")
    print(f"📊 Pairs: {len(PAIRS)} | 💰 Equity: ${ACCOUNT_EQUITY} | 🎛️  Leverage: 1:{ACCOUNT_LEVERAGE}")

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n{'='*70}")
        print(f"🔄 Check #{iteration} - {datetime.now().strftime('%H:%M:%S')}")
        print('='*70)

        for pair in PAIRS:
            df = get_twelvedata_candles(pair, API_KEY, TIMEFRAME)

            if df is None:
                continue

            result = analyze_pair(df, pair)
            print(pair, '→', result['status'])


            # ─────────────────────────────────────────────────────────────────────
            # UPDATED OUTPUT FORMATTING
            # ─────────────────────────────────────────────────────────────────────

            if result['status'] == 'NO_DATA':
                continue
            elif result['status'] == 'NO_BREAKOUT':
                continue
            elif result['status'] == 'NO_SCORE':
                continue
            elif result['status'] == 'ERROR':
                print(f"❌ {pair}: {result.get('message', 'Unknown error')}")
            elif result['status'] == 'SETUP':
                calc = result['calculations']
                candle_count = len(df)
                direction = result['direction']
                score = result['score']
                recommendation = result['recommendation']

                # Print pair header with candle count
                print(f"\n📈 {pair} ✓ {candle_count} candles")

                # Print score and recommendation
                if recommendation == 'TRADE':
                    # TRADE RECOMMENDATION - Show all details
                    order_type = "Buy Limit" if direction == "LONG" else "Sell Limit"
                    print(f"Score: {score}/8 → {recommendation}    Recommendation: **{order_type}**")
                    print(f"Lot size: {calc['final_lot']:.2f}")
                    print(f"Entry:")
                    print(f"    {calc['entry']:.4f}")
                    print(f"SL:")
                    print(f"    {calc['sl']:.4f}")
                    print(f"TP:")
                    print(f"    {calc['tp']:.4f}")
                else:
                    # SKIP RECOMMENDATION - Show only score and skip status
                    print(f"Score: {score}/8 → {recommendation}")

                # Log to CSV with all data (always log for analysis)
                factors_str = " | ".join([f"{k}:{v}" for k, v in sorted(result['factors'].items())])
                log_enhanced_result(
                    pair,
                    direction,
                    score,
                    recommendation,
                    calc,
                    factors_str
                )

            time.sleep(0.5)

        print(f"\n{'='*70}")
        if iteration < MAX_ITERATIONS:
            print(f"⏳ Next check in {CHECK_INTERVAL}s...")
            try:
                time.sleep(CHECK_INTERVAL)
            except KeyboardInterrupt:
                print("\n🛑 Stopped")
                break
        else:
            print(f"✅ Completed {MAX_ITERATIONS} check(s)")

    print(f"\n📁 Log saved: {LOG_FILE}")


if __name__ == "__main__":
    main()
