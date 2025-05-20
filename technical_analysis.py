import numpy as np
import pandas as pd
import requests
import time
from datetime import datetime, timedelta
import logging
import json
from dotenv import load_dotenv
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TechnicalAnalysis")

# Load environment variables
load_dotenv()

class TechnicalAnalysisEngine:
    """
    Technical analysis indicator calculator and signal generator.
    Calculates traditional and quantum-enhanced indicators for trading signals.
    """
    
    def __init__(self):
        """Initialize the technical analysis engine"""
        self.data_cache = {}  # Cache for price data
        self.signal_cache = {}  # Cache for signals
        self.api_keys = {
            "cryptocompare": os.getenv("CRYPTOCOMPARE_API_KEY", ""),
            "alphavantage": os.getenv("ALPHAVANTAGE_API_KEY", ""),
        }
        
        # Default timeframes
        self.timeframes = ["1h", "4h", "1d"]
        
        # Default technical indicators to calculate
        self.indicators = [
            "rsi", "macd", "bollinger", "atr", "obv", "adx", 
            "cci", "stoch", "williams", "ichimoku", "fibonacci"
        ]
        
        logger.info("Technical analysis engine initialized")
    
    def fetch_price_history(self, symbol, timeframe="1h", limit=100):
        """Fetch historical price data for a symbol"""
        # Check cache first
        cache_key = f"{symbol}_{timeframe}_{limit}"
        if cache_key in self.data_cache:
            cache_time, data = self.data_cache[cache_key]
            # If data is less than 10 minutes old, use cached data
            if time.time() - cache_time < 600:
                return data
        
        try:
            # Format symbol for API
            if "/" in symbol:
                base, quote = symbol.split("/")
            else:
                # Assume last 4 characters are the quote asset
                base = symbol[:-4]
                quote = symbol[-4:]
            
            # Map timeframe to API format
            tf_map = {
                "1m": "minute", "5m": "minute", "15m": "minute", "30m": "minute",
                "1h": "hour", "4h": "hour", "1d": "day", "1w": "day"
            }
            
            # Calculate aggregation if needed
            agg = 1
            if timeframe == "5m": agg = 5
            elif timeframe == "15m": agg = 15
            elif timeframe == "30m": agg = 30
            elif timeframe == "4h": agg = 4
            elif timeframe == "1w": agg = 7
            
            # Get data from CryptoCompare
            url = "https://min-api.cryptocompare.com/data/v2/histo"
            url += f"{tf_map[timeframe]}"
            
            params = {
                "fsym": base,
                "tsym": quote,
                "limit": limit,
                "aggregate": agg
            }
            
            if self.api_keys["cryptocompare"]:
                headers = {"authorization": f"Apikey {self.api_keys['cryptocompare']}"}
            else:
                headers = {}
            
            response = requests.get(url, params=params, headers=headers)
            
            if response.status_code == 200:
                json_data = response.json()
                
                if json_data["Response"] == "Success":
                    # Parse data into DataFrame
                    ohlcv_data = json_data["Data"]["Data"]
                    
                    df = pd.DataFrame(ohlcv_data)
                    df['time'] = pd.to_datetime(df['time'], unit='s')
                    df.set_index('time', inplace=True)
                    
                    # Rename columns to standard format
                    df.rename(columns={
                        'open': 'Open',
                        'high': 'High',
                        'low': 'Low',
                        'close': 'Close',
                        'volumefrom': 'Volume'
                    }, inplace=True)
                    
                    # Cache the data with current timestamp
                    self.data_cache[cache_key] = (time.time(), df)
                    
                    return df
                else:
                    logger.warning(f"CryptoCompare API error: {json_data['Message']}")
            else:
                logger.warning(f"Failed to fetch price data: {response.text}")
        
        except Exception as e:
            logger.error(f"Error fetching price data for {symbol}: {e}")
        
        # Return empty DataFrame if fetch failed
        return pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])
    
    def calculate_all_indicators(self, symbol, timeframe="1h"):
        """Calculate all technical indicators for a symbol"""
        df = self.fetch_price_history(symbol, timeframe)
        
        if df.empty:
            logger.warning(f"No price data available for {symbol}")
            return {}
        
        results = {
            "symbol": symbol,
            "timeframe": timeframe,
            "last_price": float(df['Close'].iloc[-1]),
            "timestamp": int(time.time()),
            "indicators": {}
        }
        
        # Calculate RSI (Relative Strength Index)
        try:
            # Custom RSI implementation
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            results["indicators"]["rsi"] = {
                "current": float(rsi.iloc[-1]),
                "previous": float(rsi.iloc[-2]),
                "signal": "oversold" if rsi.iloc[-1] < 30 else "overbought" if rsi.iloc[-1] > 70 else "neutral"
            }
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
        
        # Calculate MACD (Moving Average Convergence Divergence)
        try:
            # Custom MACD implementation
            exp1 = df['Close'].ewm(span=12, adjust=False).mean()
            exp2 = df['Close'].ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            macd_signal = macd.ewm(span=9, adjust=False).mean()
            macd_hist = macd - macd_signal
            
            results["indicators"]["macd"] = {
                "macd": float(macd.iloc[-1]),
                "signal": float(macd_signal.iloc[-1]),
                "histogram": float(macd_hist.iloc[-1]),
                "signal": "bullish" if macd.iloc[-1] > macd_signal.iloc[-1] else "bearish"
            }
        except Exception as e:
            logger.error(f"Error calculating MACD: {e}")
        
        # Calculate Bollinger Bands
        try:
            # Custom Bollinger Bands implementation
            window = 20
            middle = df['Close'].rolling(window=window).mean()
            std = df['Close'].rolling(window=window).std()
            upper = middle + 2 * std
            lower = middle - 2 * std
            current_close = df['Close'].iloc[-1]
            
            # Determine position within bands
            band_width = upper.iloc[-1] - lower.iloc[-1]
            position = (current_close - lower.iloc[-1]) / band_width if band_width > 0 else 0.5
            
            results["indicators"]["bollinger"] = {
                "upper": float(upper.iloc[-1]),
                "middle": float(middle.iloc[-1]),
                "lower": float(lower.iloc[-1]),
                "width": float(band_width),
                "position": float(position),
                "signal": "oversold" if position < 0.1 else "overbought" if position > 0.9 else "neutral"
            }
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {e}")
        
        # Calculate ATR (Average True Range) - volatility indicator
        try:
            # Custom ATR implementation
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df['Close'].shift())
            low_close = np.abs(df['Low'] - df['Close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            atr = true_range.rolling(14).mean()
            atr_pct = atr.iloc[-1] / df['Close'].iloc[-1] * 100  # ATR as percentage of price
            
            results["indicators"]["atr"] = {
                "current": float(atr.iloc[-1]),
                "percentage": float(atr_pct),
                "signal": "high_volatility" if atr_pct > 5 else "low_volatility" if atr_pct < 2 else "normal_volatility"
            }
        except Exception as e:
            logger.error(f"Error calculating ATR: {e}")
        
        # Calculate OBV (On Balance Volume)
        try:
            # Custom OBV implementation
            close_diff = df['Close'].diff()
            vol = df['Volume'].copy()
            vol[close_diff > 0] = vol
            vol[close_diff < 0] = -vol
            vol[close_diff == 0] = 0
            obv = vol.cumsum()
            
            obv_change = (obv.iloc[-1] - obv.iloc[-20]) / abs(obv.iloc[-20]) if obv.iloc[-20] != 0 else 0
            
            results["indicators"]["obv"] = {
                "current": float(obv.iloc[-1]),
                "change_20p": float(obv_change),
                "signal": "bullish" if obv_change > 0.1 else "bearish" if obv_change < -0.1 else "neutral"
            }
        except Exception as e:
            logger.error(f"Error calculating OBV: {e}")
        
        # Calculate ADX (Average Directional Index) - trend strength
        try:
            # Basic approximation of ADX - simplified for demonstration
            window = 14
            
            # Calculate True Range
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df['Close'].shift())
            low_close = np.abs(df['Low'] - df['Close'].shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = tr.rolling(window=window).mean()
            
            # Simplified directional movement
            plus_dm = df['High'].diff()
            plus_dm[plus_dm < 0] = 0
            minus_dm = -df['Low'].diff()
            minus_dm[minus_dm < 0] = 0
            
            # When high change > low change, use plus_dm, otherwise use 0
            plus_dm = np.where((df['High'].diff() > -df['Low'].diff()) & (df['High'].diff() > 0), df['High'].diff(), 0)
            # When low change > high change, use minus_dm, otherwise use 0
            minus_dm = np.where((-df['Low'].diff() > df['High'].diff()) & (df['Low'].diff() < 0), -df['Low'].diff(), 0)
            
            # Calculate plus and minus directional indicators
            plus_di = 100 * pd.Series(plus_dm).rolling(window=window).mean() / atr
            minus_di = 100 * pd.Series(minus_dm).rolling(window=window).mean() / atr
            
            # Calculate directional movement index
            dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
            
            # Calculate ADX as smooth average of DX
            adx = pd.Series(dx).rolling(window=window).mean()
            
            results["indicators"]["adx"] = {
                "adx": float(adx.iloc[-1]),
                "plus_di": float(plus_di.iloc[-1]),
                "minus_di": float(minus_di.iloc[-1]),
                "trend_strength": "strong" if adx.iloc[-1] > 25 else "weak",
                "signal": "bullish" if plus_di.iloc[-1] > minus_di.iloc[-1] and adx.iloc[-1] > 20 else 
                          "bearish" if minus_di.iloc[-1] > plus_di.iloc[-1] and adx.iloc[-1] > 20 else "neutral"
            }
        except Exception as e:
            logger.error(f"Error calculating ADX: {e}")
        
        # Calculate CCI (Commodity Channel Index)
        try:
            # Custom CCI implementation
            tp = (df['High'] + df['Low'] + df['Close']) / 3
            tp_sma = tp.rolling(window=20).mean()
            tp_mean_dev = abs(tp - tp_sma).rolling(window=20).mean()
            cci = (tp - tp_sma) / (0.015 * tp_mean_dev)
            
            results["indicators"]["cci"] = {
                "current": float(cci.iloc[-1]),
                "signal": "oversold" if cci.iloc[-1] < -100 else "overbought" if cci.iloc[-1] > 100 else "neutral"
            }
        except Exception as e:
            logger.error(f"Error calculating CCI: {e}")
        
        # Calculate Stochastic Oscillator
        try:
            # Custom Stochastic implementation
            window = 14
            # Calculate %K (fast stochastic)
            lowest_low = df['Low'].rolling(window=window).min()
            highest_high = df['High'].rolling(window=window).max()
            fastk = 100 * ((df['Close'] - lowest_low) / (highest_high - lowest_low))
            
            # Calculate %D (slow stochastic)
            slowk = fastk.rolling(window=3).mean()
            slowd = slowk.rolling(window=3).mean()
            
            results["indicators"]["stoch"] = {
                "k": float(slowk.iloc[-1]),
                "d": float(slowd.iloc[-1]),
                "signal": "oversold" if slowk.iloc[-1] < 20 and slowd.iloc[-1] < 20 else 
                         "overbought" if slowk.iloc[-1] > 80 and slowd.iloc[-1] > 80 else "neutral"
            }
        except Exception as e:
            logger.error(f"Error calculating Stochastic: {e}")
        
        # Calculate Williams %R
        try:
            # Custom Williams %R implementation
            window = 14
            highest_high = df['High'].rolling(window=window).max()
            lowest_low = df['Low'].rolling(window=window).min()
            willr = -100 * (highest_high - df['Close']) / (highest_high - lowest_low)
            
            results["indicators"]["williams"] = {
                "current": float(willr.iloc[-1]),
                "signal": "oversold" if willr.iloc[-1] < -80 else "overbought" if willr.iloc[-1] > -20 else "neutral"
            }
        except Exception as e:
            logger.error(f"Error calculating Williams %R: {e}")
        
        # Calculate simple Ichimoku Cloud components
        try:
            # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
            period9_high = df['High'].rolling(window=9).max()
            period9_low = df['Low'].rolling(window=9).min()
            tenkan_sen = (period9_high + period9_low) / 2
            
            # Kijun-sen (Base Line): (26-period high + 26-period low)/2
            period26_high = df['High'].rolling(window=26).max()
            period26_low = df['Low'].rolling(window=26).min()
            kijun_sen = (period26_high + period26_low) / 2
            
            # Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2
            senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
            
            # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
            period52_high = df['High'].rolling(window=52).max()
            period52_low = df['Low'].rolling(window=52).min()
            senkou_span_b = ((period52_high + period52_low) / 2).shift(26)
            
            # Get current values (handle NaN)
            curr_price = df['Close'].iloc[-1]
            curr_tenkan = float(tenkan_sen.iloc[-1]) if not pd.isna(tenkan_sen.iloc[-1]) else None
            curr_kijun = float(kijun_sen.iloc[-1]) if not pd.isna(kijun_sen.iloc[-1]) else None
            curr_senkou_a = float(senkou_span_a.iloc[-1]) if not pd.isna(senkou_span_a.iloc[-1]) else None
            curr_senkou_b = float(senkou_span_b.iloc[-1]) if not pd.isna(senkou_span_b.iloc[-1]) else None
            
            # Determine cloud position and signal
            cloud_position = None
            signal = "neutral"
            
            if curr_senkou_a and curr_senkou_b:
                if curr_price > max(curr_senkou_a, curr_senkou_b):
                    cloud_position = "above_cloud"
                    signal = "bullish"
                elif curr_price < min(curr_senkou_a, curr_senkou_b):
                    cloud_position = "below_cloud"
                    signal = "bearish"
                else:
                    cloud_position = "in_cloud"
                    signal = "neutral"
                
                # Add cross signals
                if curr_tenkan and curr_kijun:
                    if curr_tenkan > curr_kijun and tenkan_sen.iloc[-2] <= kijun_sen.iloc[-2]:
                        signal = "bullish_cross"
                    elif curr_tenkan < curr_kijun and tenkan_sen.iloc[-2] >= kijun_sen.iloc[-2]:
                        signal = "bearish_cross"
            
            results["indicators"]["ichimoku"] = {
                "tenkan_sen": curr_tenkan,
                "kijun_sen": curr_kijun,
                "senkou_span_a": curr_senkou_a,
                "senkou_span_b": curr_senkou_b,
                "cloud_position": cloud_position,
                "signal": signal
            }
        except Exception as e:
            logger.error(f"Error calculating Ichimoku: {e}")
        
        # Calculate Fibonacci retracement levels
        try:
            # Find recent swing high and low
            window = 20  # Look back period for swing points
            
            # Get subset of data
            subset = df.iloc[-window:]
            
            # Find high and low in this window
            swing_high = subset['High'].max()
            swing_low = subset['Low'].min()
            
            # Calculate Fibonacci levels
            diff = swing_high - swing_low
            level_0 = swing_low  # 0% retracement
            level_236 = swing_low + 0.236 * diff
            level_382 = swing_low + 0.382 * diff
            level_5 = swing_low + 0.5 * diff
            level_618 = swing_low + 0.618 * diff
            level_786 = swing_low + 0.786 * diff
            level_1 = swing_high  # 100% retracement
            
            # Determine current price position relative to levels
            current_price = df['Close'].iloc[-1]
            
            # Find closest level
            levels = [
                (0, level_0), (0.236, level_236), (0.382, level_382),
                (0.5, level_5), (0.618, level_618), (0.786, level_786), (1, level_1)
            ]
            
            levels.sort(key=lambda x: abs(x[1] - current_price))
            closest_level = levels[0]
            
            # Determine if price is bouncing off or breaking through the level
            price_5d_ago = df['Close'].iloc[-6] if len(df) > 5 else df['Close'].iloc[0]
            
            signal = "neutral"
            if current_price > price_5d_ago:
                # Price is moving up
                if closest_level[0] in [0.618, 0.786] and current_price > closest_level[1]:
                    signal = "bullish_breakout"
                elif closest_level[0] in [0.618, 0.786] and current_price < closest_level[1]:
                    signal = "bullish_bounce"
            else:
                # Price is moving down
                if closest_level[0] in [0.236, 0.382] and current_price < closest_level[1]:
                    signal = "bearish_breakout"
                elif closest_level[0] in [0.236, 0.382] and current_price > closest_level[1]:
                    signal = "bearish_bounce"
            
            results["indicators"]["fibonacci"] = {
                "swing_high": float(swing_high),
                "swing_low": float(swing_low),
                "levels": {
                    "0": float(level_0),
                    "0.236": float(level_236),
                    "0.382": float(level_382),
                    "0.5": float(level_5),
                    "0.618": float(level_618),
                    "0.786": float(level_786),
                    "1": float(level_1)
                },
                "closest_level": closest_level[0],
                "closest_value": float(closest_level[1]),
                "signal": signal
            }
        except Exception as e:
            logger.error(f"Error calculating Fibonacci levels: {e}")
        
        # Cache the results
        cache_key = f"{symbol}_{timeframe}_indicators"
        self.signal_cache[cache_key] = (time.time(), results)
        
        return results
    
    def get_combined_signal(self, symbol, timeframes=None):
        """Get a combined trading signal across multiple timeframes"""
        if timeframes is None:
            timeframes = self.timeframes
        
        signals = {}
        for tf in timeframes:
            indicators = self.calculate_all_indicators(symbol, tf)
            if not indicators:
                continue
            
            signals[tf] = {
                "bullish": 0,
                "bearish": 0,
                "neutral": 0,
                "total": 0
            }
            
            # Count signals from each indicator
            for indicator, data in indicators.get("indicators", {}).items():
                signal = data.get("signal", "neutral")
                signals[tf]["total"] += 1
                
                if "bullish" in signal or "oversold" in signal:
                    signals[tf]["bullish"] += 1
                elif "bearish" in signal or "overbought" in signal:
                    signals[tf]["bearish"] += 1
                else:
                    signals[tf]["neutral"] += 1
        
        # Calculate overall signal
        overall_bullish = 0
        overall_bearish = 0
        overall_neutral = 0
        total_indicators = 0
        
        # Weight timeframes (longer timeframes have more weight)
        weights = {
            "1m": 0.2,
            "5m": 0.3,
            "15m": 0.4,
            "30m": 0.5,
            "1h": 0.7,
            "4h": 1.0,
            "1d": 1.5,
            "1w": 2.0
        }
        
        for tf, counts in signals.items():
            weight = weights.get(tf, 1.0)
            overall_bullish += counts["bullish"] * weight
            overall_bearish += counts["bearish"] * weight
            overall_neutral += counts["neutral"] * weight
            total_indicators += counts["total"] * weight
        
        if total_indicators > 0:
            bullish_pct = overall_bullish / total_indicators * 100
            bearish_pct = overall_bearish / total_indicators * 100
            neutral_pct = overall_neutral / total_indicators * 100
            
            # Determine overall signal
            if bullish_pct > 60:
                overall_signal = "strong_buy"
            elif bullish_pct > 40:
                overall_signal = "buy"
            elif bearish_pct > 60:
                overall_signal = "strong_sell"
            elif bearish_pct > 40:
                overall_signal = "sell"
            else:
                overall_signal = "neutral"
        else:
            bullish_pct = bearish_pct = neutral_pct = 0
            overall_signal = "neutral"
        
        return {
            "symbol": symbol,
            "timeframes": timeframes,
            "signals": signals,
            "overall": {
                "bullish_pct": bullish_pct,
                "bearish_pct": bearish_pct,
                "neutral_pct": neutral_pct,
                "signal": overall_signal
            },
            "timestamp": int(time.time())
        }
    
    def detect_divergence(self, symbol, timeframe="1h", indicator="rsi"):
        """Detect divergence between price and indicator"""
        df = self.fetch_price_history(symbol, timeframe)
        
        if df.empty:
            return {"divergence": "unknown", "reason": "No price data"}
        
        try:
            # Calculate indicator
            if indicator == "rsi":
                ind_values = talib.RSI(df['Close'], timeperiod=14)
            elif indicator == "macd":
                ind_values, _, _ = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
            else:
                return {"divergence": "unknown", "reason": f"Unsupported indicator: {indicator}"}
            
            # Get last 30 data points
            lookback = min(30, len(df) - 1)
            price_subset = df['Close'].iloc[-lookback:].values
            ind_subset = ind_values.iloc[-lookback:].values
            
            # Find local extrema (peaks and troughs)
            price_peaks, price_troughs = [], []
            ind_peaks, ind_troughs = [], []
            
            for i in range(1, lookback - 1):
                # Price peaks and troughs
                if price_subset[i] > price_subset[i-1] and price_subset[i] > price_subset[i+1]:
                    price_peaks.append((i, price_subset[i]))
                elif price_subset[i] < price_subset[i-1] and price_subset[i] < price_subset[i+1]:
                    price_troughs.append((i, price_subset[i]))
                
                # Indicator peaks and troughs
                if ind_subset[i] > ind_subset[i-1] and ind_subset[i] > ind_subset[i+1]:
                    ind_peaks.append((i, ind_subset[i]))
                elif ind_subset[i] < ind_subset[i-1] and ind_subset[i] < ind_subset[i+1]:
                    ind_troughs.append((i, ind_subset[i]))
            
            # Need at least 2 peaks/troughs to find divergence
            if len(price_peaks) < 2 or len(price_troughs) < 2 or len(ind_peaks) < 2 or len(ind_troughs) < 2:
                return {"divergence": "unknown", "reason": "Insufficient extrema found"}
            
            # Sort by time (index)
            price_peaks.sort(key=lambda x: x[0])
            price_troughs.sort(key=lambda x: x[0])
            ind_peaks.sort(key=lambda x: x[0])
            ind_troughs.sort(key=lambda x: x[0])
            
            # Check for bullish divergence (price makes lower lows but indicator makes higher lows)
            if (price_troughs[-1][1] < price_troughs[-2][1] and 
                ind_troughs[-1][1] > ind_troughs[-2][1]):
                return {
                    "divergence": "bullish",
                    "reason": f"Price making lower lows while {indicator} making higher lows",
                    "strength": "strong"
                }
            
            # Check for bearish divergence (price makes higher highs but indicator makes lower highs)
            if (price_peaks[-1][1] > price_peaks[-2][1] and 
                ind_peaks[-1][1] < ind_peaks[-2][1]):
                return {
                    "divergence": "bearish",
                    "reason": f"Price making higher highs while {indicator} making lower highs",
                    "strength": "strong"
                }
            
            # Check for hidden bullish divergence (price makes higher lows but indicator makes lower lows)
            if (price_troughs[-1][1] > price_troughs[-2][1] and 
                ind_troughs[-1][1] < ind_troughs[-2][1]):
                return {
                    "divergence": "hidden_bullish",
                    "reason": f"Price making higher lows while {indicator} making lower lows",
                    "strength": "moderate"
                }
            
            # Check for hidden bearish divergence (price makes lower highs but indicator makes higher highs)
            if (price_peaks[-1][1] < price_peaks[-2][1] and 
                ind_peaks[-1][1] > ind_peaks[-2][1]):
                return {
                    "divergence": "hidden_bearish",
                    "reason": f"Price making lower highs while {indicator} making higher highs",
                    "strength": "moderate"
                }
            
            return {"divergence": "none", "reason": "No divergence detected"}
            
        except Exception as e:
            logger.error(f"Error detecting divergence: {e}")
            return {"divergence": "error", "reason": str(e)}


# Example usage
if __name__ == "__main__":
    ta = TechnicalAnalysisEngine()
    
    # Calculate indicators for BTC/USDT
    print("Calculating indicators for BTC/USDT (1h)...")
    indicators = ta.calculate_all_indicators("BTC/USDT", "1h")
    print(json.dumps(indicators, indent=2))
    
    # Get combined signal
    print("\nGetting combined signal for BTC/USDT...")
    signal = ta.get_combined_signal("BTC/USDT", ["1h", "4h", "1d"])
    print(json.dumps(signal, indent=2))
    
    # Check for divergence
    print("\nChecking for RSI divergence...")
    divergence = ta.detect_divergence("BTC/USDT", "1h", "rsi")
    print(json.dumps(divergence, indent=2))
