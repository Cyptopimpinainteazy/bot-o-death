import os
import time
import json
import requests
import asyncio
import websockets
import hmac
import hashlib
import logging
from dotenv import load_dotenv
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MarketDepth")

# Load environment variables
load_dotenv()

class MarketDepthAnalyzer:
    """
    Market depth data collection and analysis module.
    Monitors orderbook data across exchanges to identify liquidity patterns.
    """
    
    def __init__(self):
        """Initialize the market depth analyzer"""
        self.market_data = {}  # Store market data by symbol
        self.depth_cache = {}  # Cache for orderbook data
        self.running = False
        self.update_interval = 5  # seconds
        
        # Configure API connections
        self.api_keys = {
            "binance": os.getenv("BINANCE_API_KEY", ""),
            "coinbase": os.getenv("COINBASE_API_KEY", ""),
            "kraken": os.getenv("KRAKEN_API_KEY", ""),
        }
        
        self.api_secrets = {
            "binance": os.getenv("BINANCE_API_SECRET", ""),
            "coinbase": os.getenv("COINBASE_API_SECRET", ""),
            "kraken": os.getenv("KRAKEN_API_SECRET", ""),
        }
        
        # Default symbols to track
        self.symbols = [
            "BTC/USDT", 
            "ETH/USDT", 
            "MATIC/USDT"
        ]
        
        logger.info("Market depth analyzer initialized")
    
    async def start(self):
        """Start the market depth analyzer"""
        self.running = True
        
        # Start background tasks for each exchange
        tasks = [
            self._binance_depth_loop(),
            self._coinbase_depth_loop(),
            # Add more exchanges as needed
        ]
        
        await asyncio.gather(*tasks)
    
    def stop(self):
        """Stop the market depth analyzer"""
        self.running = False
        logger.info("Market depth analyzer stopped")
    
    async def _binance_depth_loop(self):
        """Fetch orderbook data from Binance"""
        binance_symbols = [s.replace("/", "").lower() for s in self.symbols]
        
        while self.running:
            try:
                for symbol in binance_symbols:
                    # Fetch orderbook data (top 20 levels)
                    url = f"https://api.binance.com/api/v3/depth?symbol={symbol}&limit=20"
                    response = requests.get(url)
                    
                    if response.status_code == 200:
                        data = response.json()
                        timestamp = int(time.time() * 1000)  # use current time as Binance doesn't return timestamp in depth endpoint
                        
                        # Process and store the data
                        self._process_orderbook("binance", symbol, data, timestamp)
                    else:
                        logger.warning(f"Failed to fetch Binance depth for {symbol}: {response.text}")
                
                # Sleep before next update
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Error in Binance depth loop: {e}")
                await asyncio.sleep(self.update_interval * 2)  # Sleep longer on error
    
    async def _coinbase_depth_loop(self):
        """Fetch orderbook data from Coinbase"""
        coinbase_symbols = [s.replace("/", "-") for s in self.symbols]
        
        while self.running:
            try:
                for symbol in coinbase_symbols:
                    # Fetch orderbook data
                    url = f"https://api.exchange.coinbase.com/products/{symbol}/book?level=2"
                    response = requests.get(url)
                    
                    if response.status_code == 200:
                        data = response.json()
                        timestamp = int(time.time() * 1000)  # use current time as timestamp
                        
                        # Process and store the data
                        self._process_orderbook("coinbase", symbol, data, timestamp)
                    else:
                        logger.warning(f"Failed to fetch Coinbase depth for {symbol}: {response.text}")
                
                # Sleep before next update
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Error in Coinbase depth loop: {e}")
                await asyncio.sleep(self.update_interval * 2)  # Sleep longer on error
    
    def _process_orderbook(self, exchange, symbol, data, timestamp):
        """Process and store orderbook data"""
        try:
            # Normalize symbol format
            std_symbol = symbol.upper().replace("-", "/")
            if "/" not in std_symbol:
                # Handle Binance style symbols (BTCUSDT -> BTC/USDT)
                if "USDT" in std_symbol:
                    std_symbol = std_symbol.replace("USDT", "/USDT")
                elif "BUSD" in std_symbol:
                    std_symbol = std_symbol.replace("BUSD", "/BUSD")
            
            # Create key for the cache
            cache_key = f"{exchange}_{std_symbol}"
            
            # Extract bids and asks
            bids = []
            asks = []
            
            if exchange == "binance":
                bids = [[float(price), float(qty)] for price, qty in data.get("bids", [])]
                asks = [[float(price), float(qty)] for price, qty in data.get("asks", [])]
            elif exchange == "coinbase":
                bids = [[float(price), float(qty)] for price, qty, _ in data.get("bids", [])]
                asks = [[float(price), float(qty)] for price, qty, _ in data.get("asks", [])]
            
            # Calculate market depth metrics
            bid_depth = sum(qty for _, qty in bids)
            ask_depth = sum(qty for _, qty in asks)
            
            # Calculate weighted average prices
            bid_value = sum(price * qty for price, qty in bids)
            ask_value = sum(price * qty for price, qty in asks)
            
            bid_wavg = bid_value / bid_depth if bid_depth > 0 else 0
            ask_wavg = ask_value / ask_depth if ask_depth > 0 else 0
            
            # Calculate spread
            best_bid = max(price for price, _ in bids) if bids else 0
            best_ask = min(price for price, _ in asks) if asks else 0
            spread = best_ask - best_bid if best_bid > 0 and best_ask > 0 else 0
            spread_pct = (spread / best_bid * 100) if best_bid > 0 else 0
            
            # Store the processed data
            depth_data = {
                "timestamp": timestamp,
                "exchange": exchange,
                "symbol": std_symbol,
                "bid_depth": bid_depth,
                "ask_depth": ask_depth,
                "bid_wavg": bid_wavg,
                "ask_wavg": ask_wavg,
                "best_bid": best_bid,
                "best_ask": best_ask,
                "spread": spread,
                "spread_pct": spread_pct,
                "bids": bids[:5],  # Store only top 5 levels
                "asks": asks[:5]   # Store only top 5 levels
            }
            
            # Store in cache
            if cache_key not in self.depth_cache:
                self.depth_cache[cache_key] = []
            
            # Keep last 10 snapshots
            self.depth_cache[cache_key].append(depth_data)
            if len(self.depth_cache[cache_key]) > 10:
                self.depth_cache[cache_key].pop(0)
            
            logger.debug(f"Processed {exchange} depth for {std_symbol}")
            
        except Exception as e:
            logger.error(f"Error processing orderbook data for {exchange} {symbol}: {e}")
    
    def get_market_depth(self, symbol, exchange=None):
        """Get current market depth for a symbol"""
        result = []
        
        # Standardize symbol format
        std_symbol = symbol.upper().replace("-", "/")
        
        # Search in cache
        for key, data_list in self.depth_cache.items():
            if not data_list:
                continue
                
            cache_exchange, cache_symbol = key.split("_", 1)
            
            if std_symbol == cache_symbol and (exchange is None or exchange == cache_exchange):
                # Return the most recent data
                result.append(data_list[-1])
        
        return result
    
    def calculate_imbalance(self, symbol, exchange=None):
        """Calculate order book imbalance for a symbol"""
        depth_data = self.get_market_depth(symbol, exchange)
        
        if not depth_data:
            return 0.5  # Neutral if no data
        
        # Combine data from multiple exchanges if needed
        bid_depth_total = sum(data["bid_depth"] for data in depth_data)
        ask_depth_total = sum(data["ask_depth"] for data in depth_data)
        
        # Calculate imbalance (-1 to 1, where positive means more bids than asks)
        total_depth = bid_depth_total + ask_depth_total
        if total_depth > 0:
            imbalance = (bid_depth_total - ask_depth_total) / total_depth
        else:
            imbalance = 0
        
        return imbalance
    
    def detect_liquidity_anomalies(self, symbol, threshold=0.25):
        """Detect anomalies in liquidity for a symbol"""
        symbol_data = []
        
        # Standardize symbol format
        std_symbol = symbol.upper().replace("-", "/")
        
        # Collect data from all exchanges
        for key, data_list in self.depth_cache.items():
            if not data_list:
                continue
                
            exchange, cache_symbol = key.split("_", 1)
            if std_symbol == cache_symbol:
                symbol_data.extend(data_list)
        
        if len(symbol_data) < 3:  # Need at least 3 data points
            return {"anomaly": False, "reason": "Insufficient data"}
        
        # Sort by timestamp
        symbol_data.sort(key=lambda x: x["timestamp"])
        
        # Extract bid and ask depths
        bid_depths = [data["bid_depth"] for data in symbol_data]
        ask_depths = [data["ask_depth"] for data in symbol_data]
        spreads = [data["spread_pct"] for data in symbol_data]
        
        # Calculate percentage changes
        bid_changes = [abs((bid_depths[i] - bid_depths[i-1]) / bid_depths[i-1]) 
                      for i in range(1, len(bid_depths))]
        ask_changes = [abs((ask_depths[i] - ask_depths[i-1]) / ask_depths[i-1]) 
                      for i in range(1, len(ask_depths))]
        spread_changes = [abs((spreads[i] - spreads[i-1]) / max(0.001, spreads[i-1])) 
                         for i in range(1, len(spreads))]
        
        # Check for rapid changes
        max_bid_change = max(bid_changes) if bid_changes else 0
        max_ask_change = max(ask_changes) if ask_changes else 0
        max_spread_change = max(spread_changes) if spread_changes else 0
        
        # Detect anomalies
        anomalies = []
        if max_bid_change > threshold:
            anomalies.append(f"Bid depth changed by {max_bid_change:.2f}%")
        if max_ask_change > threshold:
            anomalies.append(f"Ask depth changed by {max_ask_change:.2f}%")
        if max_spread_change > threshold:
            anomalies.append(f"Spread changed by {max_spread_change:.2f}%")
        
        return {
            "anomaly": len(anomalies) > 0,
            "reasons": anomalies,
            "bid_depth_change": max_bid_change,
            "ask_depth_change": max_ask_change,
            "spread_change": max_spread_change,
            "timestamp": int(time.time() * 1000)
        }


# Example usage
if __name__ == "__main__":
    async def main():
        analyzer = MarketDepthAnalyzer()
        
        # Start in background
        task = asyncio.create_task(analyzer.start())
        
        # Wait a bit for data to be collected
        await asyncio.sleep(20)
        
        # Get market depth for BTC/USDT
        depth = analyzer.get_market_depth("BTC/USDT")
        print(f"Market depth for BTC/USDT: {json.dumps(depth, indent=2)}")
        
        # Calculate imbalance
        imbalance = analyzer.calculate_imbalance("BTC/USDT")
        print(f"Order book imbalance for BTC/USDT: {imbalance}")
        
        # Check for anomalies
        anomalies = analyzer.detect_liquidity_anomalies("BTC/USDT")
        print(f"Liquidity anomalies for BTC/USDT: {json.dumps(anomalies, indent=2)}")
        
        # Stop the analyzer
        analyzer.stop()
        await task
    
    asyncio.run(main())
