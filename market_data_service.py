#!/usr/bin/env python
"""
Multi-Threaded Market Data Service

This module provides high-performance price data fetching from multiple
exchanges in parallel using threading and connection pooling for lowest
latency data collection.
"""

import os
import json
import time
import logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from typing import Dict, List, Any, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger("MarketDataService")

class MarketDataService:
    """Fetches market data from multiple exchanges in parallel"""
    
    def __init__(self, config_file=None):
        """Initialize the market data service"""
        # Load configuration
        self.config_dir = Path("config")
        if config_file:
            with open(config_file, 'r') as f:
                self.config = json.load(f)
        else:
            config_path = self.config_dir / "market_data_config.json"
            if Path(config_path).exists():
                with open(config_path, 'r') as f:
                    self.config = json.load(f)
            else:
                self.config = self._create_default_config()
        
        # Thread pool for parallel requests
        self.max_workers = self.config.get("max_threads", 10)
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # Connection pools for each exchange
        self.sessions = {}
        for exchange in self.config.get("exchanges", []):
            self.sessions[exchange] = self._create_session()
        
        # Cache for market data
        self.data_cache = {}
        self.cache_lock = threading.Lock()
        self.last_fetch_time = {}
        
        # Statistics tracking
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "avg_response_time": 0,
            "last_fetch_duration": 0
        }
        
        logger.info(f"Market Data Service initialized with {self.max_workers} threads")
    
    def _create_default_config(self):
        """Create default configuration"""
        config = {
            "max_threads": 10,
            "request_timeout": 2.0,  # Timeout in seconds
            "cache_duration": 0.5,   # Cache duration in seconds
            "retry_attempts": 2,
            "exchanges": ["binance", "coinbase", "kraken", "bitfinex", "huobi"],
            "symbols": ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BTC/USD", "ETH/USD"],
            "api_endpoints": {
                "binance": "https://api.binance.com/api/v3/ticker/price",
                "coinbase": "https://api.exchange.coinbase.com/products/{}/ticker",
                "kraken": "https://api.kraken.com/0/public/Ticker?pair={}",
                "bitfinex": "https://api-pub.bitfinex.com/v2/ticker/t{}",
                "huobi": "https://api.huobi.pro/market/detail/merged?symbol={}"
            },
            "symbol_transformers": {
                "binance": lambda s: s.replace("/", ""),
                "coinbase": lambda s: s.replace("/", "-"),
                "kraken": lambda s: s.replace("/", ""),
                "bitfinex": lambda s: s.replace("/", ""),
                "huobi": lambda s: s.lower().replace("/", "")
            },
            "response_parsers": {
                "binance": "price",
                "coinbase": "price",
                "kraken": "result.{}.c.0",
                "bitfinex": "6",  # Last price is at index 6
                "huobi": "tick.close"
            }
        }
        
        # Save default config
        config_path = self.config_dir / "market_data_config.json"
        if not Path(self.config_dir).exists():
            Path(self.config_dir).mkdir(parents=True)
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Created default market data configuration at {config_path}")
        return config
    
    def _create_session(self):
        """Create an optimized session for HTTP requests"""
        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=20,
            pool_maxsize=20,
            max_retries=self.config.get("retry_attempts", 2)
        )
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        return session
    
    def _fetch_exchange_price(self, exchange: str, symbol: str) -> Dict[str, Any]:
        """
        Fetch price data for a single symbol from a specific exchange
        
        Args:
            exchange: Exchange name
            symbol: Trading pair symbol
            
        Returns:
            Dictionary with price data
        """
        start_time = time.time()
        result = {
            "exchange": exchange,
            "symbol": symbol,
            "price": None,
            "timestamp": datetime.now().isoformat(),
            "success": False,
            "error": None
        }
        
        try:
            # Track statistics
            with self.cache_lock:
                self.stats["total_requests"] += 1
            
            # Get API details
            endpoints = self.config.get("api_endpoints", {})
            transformers = self.config.get("symbol_transformers", {})
            parsers = self.config.get("response_parsers", {})
            
            if exchange not in endpoints:
                raise ValueError(f"No API endpoint configured for {exchange}")
            
            # Transform symbol to exchange format
            transform_fn = transformers.get(exchange, lambda s: s)
            transformed_symbol = transform_fn(symbol)
            
            # Build URL
            url = endpoints[exchange]
            if "{}" in url:
                url = url.format(transformed_symbol)
                params = {}
            else:
                params = {"symbol": transformed_symbol}
            
            # Make request using the exchange's session
            session = self.sessions.get(exchange, requests)
            timeout = self.config.get("request_timeout", 2.0)
            
            response = session.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            data = response.json()
            
            # Parse response based on exchange format
            parser_path = parsers.get(exchange, "")
            price = self._extract_price(data, parser_path, symbol)
            
            # Update result
            result["price"] = float(price)
            result["success"] = True
            
            # Track statistics
            with self.cache_lock:
                self.stats["successful_requests"] += 1
                response_time = time.time() - start_time
                # Update moving average of response time
                n = self.stats["successful_requests"]
                current_avg = self.stats["avg_response_time"]
                self.stats["avg_response_time"] = (current_avg * (n-1) + response_time) / n
            
        except Exception as e:
            # Track failed request
            with self.cache_lock:
                self.stats["failed_requests"] += 1
            
            result["error"] = str(e)
            logger.warning(f"Error fetching price from {exchange} for {symbol}: {str(e)}")
        
        finally:
            # Record response time
            result["response_time"] = time.time() - start_time
            return result
    
    def _extract_price(self, data, parser_path, symbol):
        """Extract price from JSON response using parser path"""
        if not parser_path:
            return data.get("price", 0.0)
        
        # Handle special case for Kraken where symbol is part of the path
        if "{}" in parser_path:
            parser_path = parser_path.format(symbol.replace("/", ""))
        
        # Navigate through nested JSON
        parts = parser_path.split(".")
        value = data
        for part in parts:
            # Handle array indexing
            if part.isdigit():
                value = value[int(part)]
            else:
                value = value.get(part, {})
        
        return value
    
    def fetch_prices(self, symbols=None, exchanges=None) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Fetch prices for multiple symbols from multiple exchanges in parallel
        
        Args:
            symbols: List of symbols to fetch, if None uses configured symbols
            exchanges: List of exchanges to fetch from, if None uses configured exchanges
            
        Returns:
            Nested dictionary of market data organized by exchange and symbol
        """
        start_time = time.time()
        
        # Use configured symbols/exchanges if not specified
        symbols = symbols or self.config.get("symbols", [])
        exchanges = exchanges or self.config.get("exchanges", [])
        
        # Check cache first (if applicable)
        cache_duration = self.config.get("cache_duration", 0.5)
        if cache_duration > 0:
            with self.cache_lock:
                if self.data_cache and (time.time() - self.last_fetch_time.get("all", 0) < cache_duration):
                    logger.debug("Returning cached market data")
                    return self.data_cache
        
        # Create a list of all (exchange, symbol) pairs to fetch
        fetch_tasks = []
        for exchange in exchanges:
            for symbol in symbols:
                fetch_tasks.append((exchange, symbol))
        
        # Submit all tasks to the thread pool
        futures = []
        for exchange, symbol in fetch_tasks:
            future = self.executor.submit(self._fetch_exchange_price, exchange, symbol)
            futures.append(future)
        
        # Process results as they complete
        market_data = {}
        for future in as_completed(futures):
            result = future.result()
            exchange = result["exchange"]
            symbol = result["symbol"]
            
            # Initialize exchange dict if needed
            if exchange not in market_data:
                market_data[exchange] = {}
            
            # Add symbol data
            market_data[exchange][symbol] = result
        
        # Update cache
        with self.cache_lock:
            self.data_cache = market_data
            self.last_fetch_time["all"] = time.time()
            self.stats["last_fetch_duration"] = time.time() - start_time
        
        logger.info(f"Fetched {len(futures)} price points in {(time.time() - start_time):.3f}s")
        return market_data
    
    def get_price_matrix(self, symbols=None, exchanges=None) -> pd.DataFrame:
        """
        Get a price matrix DataFrame with exchanges as columns and symbols as rows
        
        Args:
            symbols: List of symbols to include
            exchanges: List of exchanges to include
            
        Returns:
            DataFrame with price data
        """
        data = self.fetch_prices(symbols, exchanges)
        
        # Extract symbols and exchanges
        all_symbols = set()
        all_exchanges = set()
        for exchange, exchange_data in data.items():
            all_exchanges.add(exchange)
            for symbol in exchange_data:
                all_symbols.add(symbol)
        
        # Create DataFrame
        df = pd.DataFrame(index=sorted(all_symbols), columns=sorted(all_exchanges))
        
        # Fill with price data
        for exchange in all_exchanges:
            for symbol in all_symbols:
                if exchange in data and symbol in data[exchange]:
                    df.loc[symbol, exchange] = data[exchange][symbol].get("price")
        
        return df
    
    def get_arbitrage_opportunities(self, min_profit_pct=0.5) -> List[Dict[str, Any]]:
        """
        Identify direct arbitrage opportunities across exchanges
        
        Args:
            min_profit_pct: Minimum profit percentage to consider
            
        Returns:
            List of arbitrage opportunities
        """
        opportunities = []
        
        # Get price matrix
        df = self.get_price_matrix()
        
        # Look for opportunities
        for symbol in df.index:
            # Drop NaN prices
            prices = df.loc[symbol].dropna()
            
            if len(prices) < 2:
                continue
            
            # Find min and max prices
            min_price = prices.min()
            max_price = prices.max()
            
            # Calculate profit percentage
            profit_pct = ((max_price - min_price) / min_price) * 100
            
            if profit_pct >= min_profit_pct:
                # Find exchanges
                buy_exchange = prices.idxmin()
                sell_exchange = prices.idxmax()
                
                opportunity = {
                    "symbol": symbol,
                    "buy_exchange": buy_exchange,
                    "sell_exchange": sell_exchange,
                    "buy_price": min_price,
                    "sell_price": max_price,
                    "profit_pct": profit_pct,
                    "timestamp": datetime.now().isoformat()
                }
                
                opportunities.append(opportunity)
        
        return opportunities
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the market data service"""
        with self.cache_lock:
            stats_copy = self.stats.copy()
        
        # Add more derived stats
        if stats_copy["total_requests"] > 0:
            stats_copy["success_rate"] = (stats_copy["successful_requests"] / stats_copy["total_requests"]) * 100
        else:
            stats_copy["success_rate"] = 0
            
        return stats_copy
    
    def refresh_connections(self):
        """Refresh all connection pools"""
        for exchange in self.config.get("exchanges", []):
            self.sessions[exchange] = self._create_session()
        logger.info("Refreshed all connection pools")


# Example usage
if __name__ == "__main__":
    # Initialize the market data service
    market_service = MarketDataService()
    
    # Fetch prices
    market_data = market_service.fetch_prices()
    
    # Print price matrix
    price_df = market_service.get_price_matrix()
    print("\nPrice Matrix:")
    print(price_df)
    
    # Check for arbitrage opportunities
    opportunities = market_service.get_arbitrage_opportunities(min_profit_pct=0.1)
    print(f"\nFound {len(opportunities)} arbitrage opportunities:")
    for opp in opportunities:
        print(f"  {opp['symbol']}: Buy on {opp['buy_exchange']} at {opp['buy_price']:.2f}, "
              f"Sell on {opp['sell_exchange']} at {opp['sell_price']:.2f}, "
              f"Profit: {opp['profit_pct']:.2f}%")
    
    # Print performance stats
    stats = market_service.get_performance_stats()
    print("\nPerformance Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
