#!/usr/bin/env python
"""
Triangle Arbitrage Module for Quantum Trading

This module identifies and executes triangle arbitrage opportunities
within a single exchange by trading through three different trading pairs
to exploit price inefficiencies.
"""

import logging
import json
from datetime import datetime
from pathlib import Path
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TriangleArbitrageDetector:
    """
    Detects triangle arbitrage opportunities within exchanges
    
    This class analyzes market data to identify profitable
    triangular trading paths and calculates potential profits
    accounting for trading fees.
    """
    
    def __init__(self, config_path=None):
        """Initialize the triangle arbitrage detector with configuration"""
        # Load configuration
        self.config_dir = Path("config")
        if config_path:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            config_path = self.config_dir / "triangle_arbitrage_config.json"
            if Path(config_path).exists():
                with open(config_path, 'r') as f:
                    self.config = json.load(f)
            else:
                self.config = self._create_default_config()
        
        # Trading paths to check
        self.trading_paths = self._generate_trading_paths()
        logger.info(f"Initialized Triangle Arbitrage Detector with {len(self.trading_paths)} trading paths")
    
    def _create_default_config(self):
        """Create default configuration for triangle arbitrage"""
        config = {
            "min_profit_threshold": 0.003,  # 0.3% minimum profit
            "max_trade_amount_usd": 5000,   # Maximum trade amount in USD
            "base_currencies": ["USDT", "USD", "USDC"],
            "middle_currencies": ["BTC", "ETH", "SOL", "ADA", "XRP"],
            "trading_fee_multiplier": 1.1,  # Multiplier to account for fees in profit calculation
            "exchanges": ["binance", "coinbase", "kraken"],
            "max_triangle_depth": 3,        # Maximum number of trades in a triangle
            "check_liquidity": True,        # Whether to check for liquidity constraints
            "min_liquidity_ratio": 3.0      # Minimum ratio of available liquidity to trade size
        }
        
        # Save default config
        config_path = self.config_dir / "triangle_arbitrage_config.json"
        if not Path(self.config_dir).exists():
            Path(self.config_dir).mkdir(parents=True)
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Created default triangle arbitrage configuration at {config_path}")
        return config
    
    def _generate_trading_paths(self):
        """Generate all possible triangular trading paths to check"""
        paths = []
        
        # Get currencies from config
        base_currencies = self.config.get("base_currencies", ["USDT"])
        middle_currencies = self.config.get("middle_currencies", ["BTC", "ETH"])
        
        # Generate paths with format [start_currency, middle_currency, end_currency]
        for base in base_currencies:
            for middle in middle_currencies:
                # Skip if they're the same currency
                if base == middle:
                    continue
                    
                # Add path: base -> middle -> base
                paths.append({
                    "currencies": [base, middle, base],
                    "trading_pairs": [
                        f"{middle}/{base}",  # Buy BTC with USDT
                        f"{base}/{middle}"   # Sell BTC for USDT (or could be another pair)
                    ]
                })
                
                # Add more complex paths with multiple middle currencies
                if self.config.get("max_triangle_depth", 3) > 3:
                    for second_middle in middle_currencies:
                        if second_middle != middle and second_middle != base:
                            # Add path: base -> middle -> second_middle -> base
                            paths.append({
                                "currencies": [base, middle, second_middle, base],
                                "trading_pairs": [
                                    f"{middle}/{base}",      # Buy BTC with USDT
                                    f"{second_middle}/{middle}",  # Buy ETH with BTC
                                    f"{base}/{second_middle}"     # Sell ETH for USDT
                                ]
                            })
        
        return paths
    
    def find_triangle_opportunities(self, market_data):
        """
        Find triangle arbitrage opportunities in current market data
        
        Args:
            market_data: Dictionary of market data organized by exchange and trading pair
            
        Returns:
            List of triangle arbitrage opportunities
        """
        opportunities = []
        
        for exchange in self.config.get("exchanges", []):
            if exchange not in market_data:
                continue
                
            exchange_data = market_data[exchange]
            
            # Check each potential trading path
            for path in self.trading_paths:
                opportunity = self._check_path_profitability(exchange, path, exchange_data)
                if opportunity:
                    opportunities.append(opportunity)
        
        logger.info(f"Found {len(opportunities)} triangle arbitrage opportunities")
        return opportunities
    
    def _check_path_profitability(self, exchange, path, market_data):
        """Check if a specific trading path is profitable"""
        # Get trading pairs and expected sequence
        trading_pairs = path["trading_pairs"]
        currencies = path["currencies"]
        
        # Verify all pairs are available in the market data
        for pair in trading_pairs:
            if pair not in market_data:
                return None
        
        # Start with 1 unit of base currency
        start_amount = 1.0
        current_amount = start_amount
        
        # Track conversion rates and prices for each step
        rates = []
        prices = []
        
        # Simulate the trades
        for i, pair in enumerate(trading_pairs):
            pair_data = market_data[pair]
            
            # Get price and determine if buy or sell
            price = pair_data["price"]
            prices.append(price)
            
            # Determine buy or sell based on the currency order
            base_currency, quote_currency = pair.split("/")
            
            if currencies[i] == quote_currency:
                # We're buying the base currency with our quote currency
                rate = 1.0 / price
                # Apply fees
                fee = self.config.get("trading_fee_multiplier", 1.0) - 1.0
                rate = rate * (1.0 - fee)
            else:
                # We're selling our base currency for the quote currency
                rate = price
                # Apply fees
                fee = self.config.get("trading_fee_multiplier", 1.0) - 1.0
                rate = rate * (1.0 - fee)
            
            rates.append(rate)
            current_amount = current_amount * rate
        
        # Calculate profit percentage
        profit_pct = (current_amount - start_amount) / start_amount
        
        # Check if profit meets minimum threshold
        min_profit = self.config.get("min_profit_threshold", 0.003)
        if profit_pct > min_profit:
            # Calculate optimal trade size
            max_trade_amount = self.config.get("max_trade_amount_usd", 5000)
            
            # Check liquidity if required
            if self.config.get("check_liquidity", True):
                # Get the minimum liquidity across all pairs
                min_liquidity = float('inf')
                for pair in trading_pairs:
                    pair_data = market_data[pair]
                    min_liquidity = min(min_liquidity, pair_data.get("liquidity", float('inf')))
                
                # Calculate trade amount based on liquidity
                min_liquidity_ratio = self.config.get("min_liquidity_ratio", 3.0)
                max_trade_amount = min(max_trade_amount, min_liquidity / min_liquidity_ratio)
            
            # Create opportunity object
            opportunity = {
                "timestamp": datetime.now().isoformat(),
                "exchange": exchange,
                "type": "triangle",
                "trading_path": path,
                "currencies": currencies,
                "trading_pairs": trading_pairs,
                "rates": rates,
                "prices": prices,
                "profit_pct": profit_pct,
                "optimal_start_amount": max_trade_amount,
                "expected_final_amount": max_trade_amount * (1 + profit_pct)
            }
            
            return opportunity
        
        return None

    def assess_opportunity_risk(self, opportunity, risk_manager):
        """
        Assess the risk of a triangle arbitrage opportunity
        
        Args:
            opportunity: Triangle arbitrage opportunity to assess
            risk_manager: Risk manager instance to use for assessment
            
        Returns:
            Tuple of (viability_score, assessment_details)
        """
        # Start with default high score since all trades are on same exchange
        viability_score = 0.9
        assessment = {
            "transfer_viability": 1.0,  # No transfers needed
            "reason": "Triangle trades occur on same exchange"
        }
        
        # Check individual pair risks
        pair_viabilities = []
        exchange = opportunity["exchange"]
        
        for pair in opportunity["trading_pairs"]:
            # Create a mini opportunity for this pair to assess
            pair_opportunity = {
                "symbol": pair,
                "buy_exchange": exchange,
                "sell_exchange": exchange,
                "price_diff_pct": opportunity["profit_pct"] / len(opportunity["trading_pairs"])
            }
            
            # Check liquidity and other risks
            pair_viability, pair_assessment = risk_manager.assess_pair_viability(pair_opportunity)
            pair_viabilities.append(pair_viability)
            
            # Record assessment
            assessment[f"pair_{pair}_viability"] = pair_viability
        
        # Calculate overall viability as minimum of individual pair viabilities
        pair_viability = min(pair_viabilities) if pair_viabilities else 0.0
        
        # Combine with base viability
        viability_score = viability_score * 0.5 + pair_viability * 0.5
        
        # Check if profit is high enough to justify risk
        min_profit_threshold = self.config.get("min_profit_threshold", 0.003)
        profit_factor = opportunity["profit_pct"] / min_profit_threshold
        profit_score = min(1.0, profit_factor)
        
        # Incorporate profit into final score
        viability_score = viability_score * 0.7 + profit_score * 0.3
        
        # Add to assessment
        assessment["profit_score"] = profit_score
        assessment["overall_viability"] = viability_score
        
        return viability_score, assessment
    
    def execute_triangle_arbitrage(self, opportunity, exchange_api):
        """
        Execute a triangle arbitrage opportunity
        
        Args:
            opportunity: Triangle arbitrage opportunity to execute
            exchange_api: API interface to the exchange
            
        Returns:
            Trade execution details
        """
        exchange = opportunity["exchange"]
        trading_pairs = opportunity["trading_pairs"]
        start_amount = opportunity["optimal_start_amount"]
        
        # Prepare trade execution
        trades = []
        current_amount = start_amount
        
        try:
            # Execute each trade in sequence
            for i, pair in enumerate(trading_pairs):
                # Determine buy or sell based on the currency order
                currencies = opportunity["currencies"]
                base_currency, quote_currency = pair.split("/")
                
                is_buy = currencies[i] == quote_currency
                
                # Calculate amount for this trade
                if is_buy:
                    # We're buying the base currency with our quote currency
                    amount = current_amount / opportunity["prices"][i]
                else:
                    # We're selling our base currency for the quote currency
                    amount = current_amount
                
                # Execute the trade
                trade_result = exchange_api.create_order(
                    exchange=exchange,
                    symbol=pair,
                    order_type="market",
                    side="buy" if is_buy else "sell",
                    amount=amount
                )
                
                # Update current amount based on actual execution
                if trade_result and "filled_amount" in trade_result:
                    if is_buy:
                        current_amount = trade_result["filled_amount"]
                    else:
                        current_amount = trade_result["filled_amount"] * trade_result["price"]
                
                # Record trade
                trades.append({
                    "pair": pair,
                    "is_buy": is_buy,
                    "amount": amount,
                    "result": trade_result
                })
            
            # Calculate actual profit
            profit_pct = (current_amount - start_amount) / start_amount
            
            execution_result = {
                "timestamp": datetime.now().isoformat(),
                "exchange": exchange,
                "opportunity_id": opportunity.get("id", "unknown"),
                "start_amount": start_amount,
                "final_amount": current_amount,
                "expected_profit_pct": opportunity["profit_pct"],
                "actual_profit_pct": profit_pct,
                "trades": trades,
                "status": "completed"
            }
            
            logger.info(f"Successfully executed triangle arbitrage with {profit_pct:.4f}% profit")
            return execution_result
            
        except Exception as e:
            logger.error(f"Error executing triangle arbitrage: {str(e)}")
            return {
                "timestamp": datetime.now().isoformat(),
                "exchange": exchange,
                "opportunity_id": opportunity.get("id", "unknown"),
                "start_amount": start_amount,
                "trades": trades,
                "error": str(e),
                "status": "failed"
            }
