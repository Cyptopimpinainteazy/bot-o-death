#!/usr/bin/env python
"""
Exchange Risk Manager
--------------------
Handles practical trading concerns:
1. Transfer Delays between exchanges
2. Fee Management
3. Liquidity Assessment
"""

import os
import time
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("exchange_risk.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ExchangeRisk")

class ExchangeRiskManager:
    """Manages practical risks when trading across exchanges"""
    
    def __init__(self, config_file=None):
        """Initialize the risk manager"""
        self.config_dir = Path("config")
        self.data_dir = Path("data") / "exchange_data"
        self.results_dir = Path("results") / "risk_analysis"
        
        # Create directories
        os.makedirs(self.config_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Load configuration
        self.config = self._load_config(config_file)
        
        # Initialize trackers
        self.transfer_history = {}
        self.fee_history = {}
        self.liquidity_data = {}
        self.exchange_status = {}
        
        # Load thresholds from config
        risk_params = self.config.get("risk_parameters", {})
        self.min_profit_threshold = risk_params.get("min_profit_threshold", 0.0025)  # Minimum 0.25% profit
        self.max_acceptable_slippage = risk_params.get("max_acceptable_slippage", 0.004)  # Maximum 0.4% slippage
        self.max_liquidity_usage = 1.0 / risk_params.get("min_liquidity_ratio", 5)  # Use at most 20% of available liquidity
        self.max_transfer_wait = risk_params.get("max_transfer_wait", 120)  # Maximum 120 minutes for transfer
        self.min_confidence_score = risk_params.get("min_confidence_score", 0.6)  # Minimum 60% confidence
        
        # Initialize lock for thread safety
        self.lock = threading.Lock()
        
        logger.info("Exchange Risk Manager initialized")
    
    def _load_config(self, config_file):
        """Load configuration from file or create default"""
        # First check for config_file parameter
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                logger.info(f"Loading configuration from {config_file}")
                return json.load(f)
        
        # Then check for default config file location
        default_config_path = self.config_dir / "exchange_risk_config.json"
        if os.path.exists(default_config_path):
            with open(default_config_path, 'r') as f:
                logger.info(f"Loading existing configuration from {default_config_path}")
                return json.load(f)
        
        # Default configuration
        config = {
            "exchanges": {
                "binance": {
                    "maker_fee": 0.001,
                    "taker_fee": 0.001,
                    "withdrawal_fees": {
                        "BTC": 0.0005,
                        "ETH": 0.005,
                        "USDT": 1.0
                    },
                    "transfer_times": {
                        "internal": 1,      # minutes
                        "bitcoin": 30,      # minutes
                        "ethereum": 10,     # minutes
                        "solana": 1         # minutes
                    }
                },
                "coinbase": {
                    "maker_fee": 0.0015,
                    "taker_fee": 0.0025,
                    "withdrawal_fees": {
                        "BTC": 0.0001,
                        "ETH": 0.003,
                        "USDT": 2.0
                    },
                    "transfer_times": {
                        "internal": 1,      # minutes
                        "bitcoin": 35,      # minutes
                        "ethereum": 15,     # minutes
                        "solana": 2         # minutes
                    }
                }
            },
            "risk_parameters": {
                "min_profit_threshold": 0.005,        # Minimum 0.5% profit
                "max_acceptable_slippage": 0.002,     # Maximum 0.2% slippage
                "min_liquidity_ratio": 10,            # Trade size must be at most 1/10th of available liquidity
                "max_transfer_wait": 45,              # Maximum 45 minutes to wait for transfer
                "min_confidence_score": 0.8,          # Minimum confidence score for execution
                "max_retry_attempts": 3               # Maximum retry attempts for failed operations
            },
            "currency_networks": {
                "BTC": "bitcoin",
                "ETH": "ethereum",
                "SOL": "solana",
                "USDT_ERC20": "ethereum",
                "USDT_TRC20": "tron",
                "USDT_BEP20": "binance_smart_chain"
            }
        }
        
        # Save default config only if it doesn't exist
        config_path = self.config_dir / "exchange_risk_config.json"
        if not os.path.exists(config_path):
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            logger.info(f"Created default exchange risk configuration at {config_path}")
        else:
            logger.info(f"Using existing configuration at {config_path}")
        return config
    
    #-------------------------------------------------
    # 1. TRANSFER DELAY MANAGEMENT
    #-------------------------------------------------
    
    def estimate_transfer_time(self, from_exchange, to_exchange, currency, amount):
        """
        Estimate time required to transfer assets between exchanges
        Returns estimated time in minutes
        """
        # Get network for this currency
        currency_network = self.config.get("currency_networks", {}).get(currency, "unknown")
        
        # Get transfer times from config
        from_times = self.config.get("exchanges", {}).get(from_exchange, {}).get("transfer_times", {})
        to_times = self.config.get("exchanges", {}).get(to_exchange, {}).get("transfer_times", {})
        
        # Get network transfer time estimates
        from_time = from_times.get(currency_network, 60)  # Default 60 min if unknown
        to_time = to_times.get(currency_network, 60)      # Default 60 min if unknown
        
        # For large transfers, add processing time
        size_factor = 1.0
        if amount > 1.0 and currency == "BTC":
            size_factor = 1.2
        elif amount > 10.0 and currency == "ETH":
            size_factor = 1.2
        elif amount > 10000.0 and currency == "USDT":
            size_factor = 1.3
        
        # Calculate total transfer time (source + destination + network congestion)
        network_congestion = self._get_network_congestion(currency_network)
        total_time = (from_time + to_time) * size_factor * network_congestion
        
        # Record this estimate
        self._record_transfer_estimate(from_exchange, to_exchange, currency, amount, total_time)
        
        logger.info(f"Estimated transfer time: {total_time:.1f} minutes for {amount} {currency} from {from_exchange} to {to_exchange}")
        return total_time
    
    def _get_network_congestion(self, network):
        """Get current congestion factor for a network (1.0 = normal)"""
        # In a real system, this would query network status APIs
        # For now, we'll simulate with random values that occasionally spike
        return max(1.0, min(3.0, np.random.normal(1.1, 0.3)))
    
    def _record_transfer_estimate(self, from_exchange, to_exchange, currency, amount, time_estimate):
        """Record transfer estimate for future analysis"""
        record = {
            "timestamp": datetime.now().isoformat(),
            "from_exchange": from_exchange,
            "to_exchange": to_exchange,
            "currency": currency,
            "amount": float(amount),
            "estimated_minutes": float(time_estimate)
        }
        
        # Thread-safe update
        with self.lock:
            key = f"{from_exchange}_{to_exchange}_{currency}"
            if key not in self.transfer_history:
                self.transfer_history[key] = []
            self.transfer_history[key].append(record)
    
    def will_opportunity_persist(self, price_diff_pct, transfer_time_minutes):
        """
        Determine if an arbitrage opportunity is likely to persist
        given the current price difference and estimated transfer time
        """
        # Check if transfer time exceeds our maximum wait threshold
        if transfer_time_minutes > self.max_transfer_wait:
            logger.warning(f"Transfer time ({transfer_time_minutes:.1f} min) exceeds maximum wait time ({self.max_transfer_wait} min)")
            return 0.0  # No persistence if transfer takes too long
            
        # For RL training mode, we're more optimistic about persistence
        if self.config.get("risk_parameters", {}).get("rl_training_mode", False):
            # More lenient persistence calculation for training
            if transfer_time_minutes < 30 and price_diff_pct > 0.5:
                return 0.8  # High persistence for quick transfers
            elif transfer_time_minutes < 60:
                return 0.6  # Medium persistence for moderate transfers
            else:
                return 0.4  # Lower but still viable for longer transfers
        
        # Calculate a persistence score from 0-1
        # Higher means opportunity is more likely to last long enough
        
        # Factors:
        # 1. Initial spread size (larger = more likely to persist)
        # 2. Transfer time (longer = less likely to persist)
        # 3. Historical volatility of this spread (higher = less likely to persist)
        
        # Historical volatility factor - would come from real data in production
        # Reduce volatility factor for RL training mode to allow more opportunities
        if self.config.get("risk_parameters", {}).get("rl_training_mode", False):
            volatility_factor = 0.03  # Lower 3% change per hour for training
        else:
            volatility_factor = 0.05  # Average 5% change per hour
        
        # Convert transfer time to hours
        transfer_time_hours = transfer_time_minutes / 60.0
        
        # Calculate how much spread might close during transfer
        expected_spread_reduction = volatility_factor * transfer_time_hours
        
        # Calculate remaining spread after transfer
        expected_remaining_spread = price_diff_pct - expected_spread_reduction
        
        # Add safety factor - reduced for RL training mode
        if self.config.get("risk_parameters", {}).get("rl_training_mode", False):
            safety_margin = 0.0005  # 0.05% for training
        else:
            safety_margin = 0.002  # 0.2% for production
        expected_remaining_spread -= safety_margin
        
        # Calculate persistence score with improved formula
        raw_score = max(0, min(1, expected_remaining_spread / price_diff_pct))
        
        # Apply boost for training mode
        if self.config.get("risk_parameters", {}).get("rl_training_mode", False):
            # Apply a boost to smaller opportunities to encourage exploration
            if price_diff_pct > 0.01:  # 1% spread
                boost = min(0.3, price_diff_pct / 0.05)  # Up to 0.3 boost for large spreads
                persistence_score = min(1.0, raw_score + boost)
            else:
                persistence_score = raw_score * 1.2  # 20% boost for smaller opportunities
        else:
            persistence_score = raw_score
        
        logger.info(f"Opportunity persistence: {persistence_score:.2f} (initial spread: {price_diff_pct:.2f}%, transfer time: {transfer_time_minutes:.1f} min)")
        return persistence_score
    
    def simulate_opportunity_survival(self, price_diff_pct, transfer_time_minutes, num_simulations=100):
        """
        Run Monte Carlo simulation to estimate probability of opportunity
        surviving through the transfer time
        """
        # In a production system, this would use actual historical data
        # For now, we use a simple random walk model
        
        survival_count = 0
        volatility = 0.02  # Assumed hourly volatility
        
        # Convert to hours for simulation
        transfer_time_hours = transfer_time_minutes / 60.0
        
        # Run simulations
        for _ in range(num_simulations):
            remaining_spread = price_diff_pct
            # Simulate price movement during transfer
            for t in range(int(transfer_time_hours * 60)):  # Simulate per minute
                # Random price movement, stronger mean reversion for larger spreads
                mean_reversion = 0.0001 * (remaining_spread / 0.01)
                spread_change = np.random.normal(-mean_reversion, volatility/60)
                remaining_spread += spread_change
                
                # If spread disappears, break
                if remaining_spread <= 0:
                    break
            
            # Count successes
            if remaining_spread > 0.002:  # Minimum viable spread (0.2%)
                survival_count += 1
        
        survival_probability = survival_count / num_simulations
        logger.info(f"Opportunity survival probability: {survival_probability:.2%} based on {num_simulations} simulations")
        return survival_probability
    
    #-------------------------------------------------
    # 2. FEE MANAGEMENT
    #-------------------------------------------------
    
    def calculate_total_fees(self, from_exchange, to_exchange, buy_currency, sell_currency, amount):
        """
        Calculate all fees involved in an arbitrage transaction between exchanges
        Returns total fees as percentage of transaction size
        """
        # Get fee data from config
        from_exchange_data = self.config.get("exchanges", {}).get(from_exchange, {})
        to_exchange_data = self.config.get("exchanges", {}).get(to_exchange, {})
        
        # Trading fees (assume taker fees for arbitrage)
        from_trading_fee = from_exchange_data.get("taker_fee", 0.001)
        to_trading_fee = to_exchange_data.get("taker_fee", 0.001)
        
        # Withdrawal fee
        withdrawal_fee = from_exchange_data.get("withdrawal_fees", {}).get(buy_currency, 0)
        
        # Convert withdrawal fee to percentage
        # This would use current price data in a real system
        withdrawal_fee_pct = (withdrawal_fee / amount) if amount > 0 else 0
        
        # Network transaction fee
        # In a real system, this would be dynamically calculated based on current gas prices
        network_fee_pct = 0.0005  # Assume 0.05% network fee
        
        # Total fees
        total_fee_pct = from_trading_fee + to_trading_fee + withdrawal_fee_pct + network_fee_pct
        
        # Record for analysis
        self._record_fee_calculation(from_exchange, to_exchange, buy_currency, sell_currency, amount, total_fee_pct)
        
        logger.info(f"Total fees: {total_fee_pct:.2%} for {amount} {buy_currency}")
        return total_fee_pct
    
    def _record_fee_calculation(self, from_exchange, to_exchange, buy_currency, sell_currency, amount, total_fee_pct):
        """Record fee calculation for future analysis"""
        record = {
            "timestamp": datetime.now().isoformat(),
            "from_exchange": from_exchange,
            "to_exchange": to_exchange,
            "buy_currency": buy_currency,
            "sell_currency": sell_currency,
            "amount": float(amount),
            "total_fee_pct": float(total_fee_pct)
        }
        
        # Thread-safe update
        with self.lock:
            key = f"{from_exchange}_{to_exchange}_{buy_currency}_{sell_currency}"
            if key not in self.fee_history:
                self.fee_history[key] = []
            self.fee_history[key].append(record)
    
    def is_arbitrage_profitable(self, price_diff_pct, total_fee_pct):
        """
        Determine if an arbitrage opportunity is profitable after fees
        Returns profitability and expected profit percentage
        """
        # Calculate expected profit
        expected_profit_pct = price_diff_pct - total_fee_pct
        
        # Add safety margin for unexpected costs
        safety_margin = 0.001  # 0.1%
        adjusted_profit_pct = expected_profit_pct - safety_margin
        
        # Determine if profitable
        is_profitable = adjusted_profit_pct > self.min_profit_threshold
        
        logger.info(f"Arbitrage profitability: {is_profitable} (expected profit: {adjusted_profit_pct:.2%})")
        return is_profitable, adjusted_profit_pct
    
    #-------------------------------------------------
    # 3. LIQUIDITY MANAGEMENT
    #-------------------------------------------------
    
    def check_liquidity(self, exchange, symbol, amount, side="buy"):
        """
        Check if there's sufficient liquidity for a trade
        Returns tuple of (has_liquidity, expected_slippage, max_executable_amount)
        """
        # In a real system, this would query the order book
        # For now, we'll simulate with reasonable values
        
        # Simulate order book data (would come from exchange API)
        order_book = self._get_simulated_order_book(exchange, symbol)
        
        # Check liquidity
        if side.lower() == "buy":
            asks = order_book["asks"]  # Format: [[price, amount], ...]
            return self._calculate_execution_impact(asks, amount)
        else:  # sell
            bids = order_book["bids"]  # Format: [[price, amount], ...]
            return self._calculate_execution_impact(bids, amount)
    
    def _get_simulated_order_book(self, exchange, symbol):
        """Get simulated order book for testing"""
        # This would fetch real order book in production
        base_price = {
            "BTC/USDT": 40000,
            "ETH/USDT": 2000,
            "SOL/USDT": 100,
        }.get(symbol, 100)
        
        # Generate sample order book
        asks = []
        bids = []
        
        # Create asks (ascending prices)
        for i in range(20):
            price = base_price * (1 + 0.0001 * i)
            # Liquidity typically decreases as price increases
            amount = max(0.1, np.random.exponential(10) / (1 + i*0.2))
            asks.append([price, amount])
        
        # Create bids (descending prices)
        for i in range(20):
            price = base_price * (1 - 0.0001 * i)
            # Liquidity typically decreases as price decreases
            amount = max(0.1, np.random.exponential(10) / (1 + i*0.2))
            bids.append([price, amount])
        
        return {"asks": asks, "bids": bids}
    
    def _calculate_execution_impact(self, order_book_side, amount):
        """
        Calculate execution impact on a specific side of the order book
        Returns (has_liquidity, expected_slippage, max_executable_amount)
        
        This data can be used to train reinforcement learning models to
        optimize trade entry timing and size.
        """
        total_available = sum(order[1] for order in order_book_side)
        
        # If trying to trade more than available liquidity
        if amount > total_available:
            return False, 1.0, total_available
        
        # Calculate slippage
        executed_amount = 0
        weighted_avg_price = 0
        best_price = order_book_side[0][0]
        
        for price, order_amount in order_book_side:
            if executed_amount + order_amount >= amount:
                # This level will partially fill
                remaining = amount - executed_amount
                weighted_avg_price += price * remaining
                executed_amount = amount  # Now fully executed
                break
            else:
                # This level will completely fill
                weighted_avg_price += price * order_amount
                executed_amount += order_amount
        
        # Calculate weighted average price
        weighted_avg_price /= amount
        
        # Calculate slippage
        slippage = abs(weighted_avg_price - best_price) / best_price
        
        # Check if slippage is acceptable
        has_acceptable_liquidity = slippage <= self.max_acceptable_slippage
        
        logger.info(f"Liquidity check: {has_acceptable_liquidity} (slippage: {slippage:.2%}, available: {total_available})")
        return has_acceptable_liquidity, slippage, total_available
    
    def estimate_max_trade_size(self, exchange, symbol, max_slippage=None):
        """
        Estimate maximum trade size that can be executed with acceptable slippage
        """
        if max_slippage is None:
            max_slippage = self.max_acceptable_slippage
        
        # Get order book
        order_book = self._get_simulated_order_book(exchange, symbol)
        
        # Binary search for maximum size
        min_size = 0
        max_size = sum(order[1] for order in order_book["asks"])
        best_size = 0
        
        while max_size - min_size > 0.01:
            mid_size = (min_size + max_size) / 2
            has_liquidity, slippage, _ = self._calculate_execution_impact(order_book["asks"], mid_size)
            
            if slippage <= max_slippage:
                # This size works, try larger
                best_size = mid_size
                min_size = mid_size
            else:
                # Too large, try smaller
                max_size = mid_size
        
        logger.info(f"Maximum trade size with {max_slippage:.2%} slippage: {best_size}")
        return best_size
    
    #-------------------------------------------------
    # COMBINED RISK ASSESSMENT
    #-------------------------------------------------
    
    def assess_arbitrage_viability(self, opportunity):
        """
        Comprehensive assessment of arbitrage opportunity
        Returns viability score (0-1) and detailed breakdown
        
        This assessment provides rich features for reinforcement learning agents
        to better understand market conditions and constraints.
        """
        # Extract opportunity details
        buy_exchange = opportunity.get("buy_exchange")
        sell_exchange = opportunity.get("sell_exchange")
        symbol = opportunity.get("symbol")
        buy_price = opportunity.get("buy_price")
        sell_price = opportunity.get("sell_price")
        amount = opportunity.get("amount", 1.0)
        
        if not all([buy_exchange, sell_exchange, symbol, buy_price, sell_price]):
            logger.error("Incomplete opportunity information")
            return 0, {"error": "Incomplete information"}
        
        # Calculate price difference
        price_diff_pct = (sell_price - buy_price) / buy_price
        
        # Step 1: Check fees
        currency = symbol.split('/')[0]  # Base currency (e.g., BTC from BTC/USDT)
        quote_currency = symbol.split('/')[1]  # Quote currency (e.g., USDT from BTC/USDT)
        
        total_fee_pct = self.calculate_total_fees(
            buy_exchange, sell_exchange, currency, quote_currency, amount)
        
        is_profitable, adjusted_profit_pct = self.is_arbitrage_profitable(price_diff_pct, total_fee_pct)
        
        if not is_profitable:
            return 0, {"reason": "Not profitable after fees", "expected_profit": adjusted_profit_pct}
        
        # Step 2: Check liquidity
        has_buy_liquidity, buy_slippage, max_buy_amount = self.check_liquidity(
            buy_exchange, symbol, amount, "buy")
        
        has_sell_liquidity, sell_slippage, max_sell_amount = self.check_liquidity(
            sell_exchange, symbol, amount, "sell")
        
        max_executable_amount = min(max_buy_amount, max_sell_amount)
        
        if not (has_buy_liquidity and has_sell_liquidity):
            return 0, {
                "reason": "Insufficient liquidity", 
                "buy_slippage": buy_slippage,
                "sell_slippage": sell_slippage,
                "max_executable": max_executable_amount
            }
        
        # Step 3: Check transfer times
        transfer_time = self.estimate_transfer_time(
            buy_exchange, sell_exchange, currency, amount)
        
        persistence_score = self.will_opportunity_persist(price_diff_pct, transfer_time)
        
        if persistence_score < 0.5:
            return persistence_score, {
                "reason": "Opportunity unlikely to persist", 
                "persistence": persistence_score,
                "transfer_time": transfer_time
            }
        
        # Calculate overall viability with improved weighting
        viability_factors = [
            adjusted_profit_pct / 0.01,  # Normalize to 0-1 range assuming 1% is ideal
            1 - buy_slippage / self.max_acceptable_slippage,
            1 - sell_slippage / self.max_acceptable_slippage,
            persistence_score,
            min(1.0, amount / 10.0)  # Size factor - reward larger trades up to a point
        ]
        
        viability_score = np.mean(viability_factors)
        
        # Prepare detailed assessment
        assessment = {
            "viability_score": float(viability_score),
            "expected_profit": float(adjusted_profit_pct),
            "total_fees": float(total_fee_pct),
            "buy_slippage": float(buy_slippage),
            "sell_slippage": float(sell_slippage),
            "transfer_time_minutes": float(transfer_time),
            "persistence_probability": float(persistence_score),
            "max_executable_amount": float(max_executable_amount)
        }
        
        # Log the assessment
        logger.info(f"Arbitrage viability: {viability_score:.2f} for {symbol} between {buy_exchange}-{sell_exchange}")
        
        return viability_score, assessment
    
    def get_optimized_trade_parameters(self, opportunity):
        """
        Optimize trade parameters based on risk assessment
        Returns optimal trade amount and execution strategy
        """
        # First check if opportunity is viable
        viability, assessment = self.assess_arbitrage_viability(opportunity)
        
        if viability <= 0.3:
            return {
                "status": "rejected",
                "reason": assessment.get("reason", "Low viability"),
                "viability": viability
            }
        
        # Extract info
        symbol = opportunity["symbol"]
        buy_exchange = opportunity["buy_exchange"]
        sell_exchange = opportunity["sell_exchange"]
        max_amount = assessment["max_executable_amount"]
        
        # Calculate optimal amount based on:
        # 1. Maximum executable (liquidity)
        # 2. Risk-adjusted position size
        # 3. Probability of persistence
        
        # Start with max executable
        optimal_amount = max_amount
        
        # Reduce based on persistence probability
        persistence_factor = assessment["persistence_probability"]
        optimal_amount *= persistence_factor
        
        # Limit to reasonable percentage of liquidity
        optimal_amount = min(optimal_amount, max_amount * self.max_liquidity_usage)
        
        # Determine execution strategy
        if assessment["transfer_time_minutes"] > 30:
            # Long transfer time - use hedged strategy
            strategy = "hedged"
        else:
            # Regular transfer time - use direct strategy
            strategy = "direct"
        
        result = {
            "status": "approved",
            "optimal_amount": float(optimal_amount),
            "execution_strategy": strategy,
            "expected_profit": assessment["expected_profit"],
            "viability_score": viability,
            "risk_assessment": assessment
        }
        
        logger.info(f"Optimized parameters: amount={optimal_amount:.4f}, strategy={strategy}")
        return result
    
    def save_analysis_data(self):
        """Save all collected data for later analysis"""
        analysis_data = {
            "transfer_history": self.transfer_history,
            "fee_history": self.fee_history,
            "liquidity_data": self.liquidity_data,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save to file
        filename = f"exchange_risk_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        file_path = self.results_dir / filename
        
        with open(file_path, 'w') as f:
            json.dump(analysis_data, f, indent=2)
        
        logger.info(f"Risk analysis data saved to {file_path}")
        return file_path


# Example usage
if __name__ == "__main__":
    # Create risk manager
    risk_manager = ExchangeRiskManager()
    
    # Example arbitrage opportunity
    opportunity = {
        "buy_exchange": "binance",
        "sell_exchange": "coinbase",
        "symbol": "BTC/USDT",
        "buy_price": 40000,
        "sell_price": 40400,  # 1% difference
        "amount": 0.5  # 0.5 BTC
    }
    
    # Assess opportunity
    viability, assessment = risk_manager.assess_arbitrage_viability(opportunity)
    
    # Get optimized parameters
    optimal_params = risk_manager.get_optimized_trade_parameters(opportunity)
    
    # Print results
    print("\n=== ARBITRAGE RISK ASSESSMENT ===")
    print(f"Viability Score: {viability:.2f}")
    
    if "expected_profit" in assessment:
        print(f"Expected Profit: {assessment['expected_profit']:.2%}")
    
    if "reason" in assessment:
        print(f"Rejection Reason: {assessment['reason']}")
        
    if "transfer_time_minutes" in assessment:
        print(f"Transfer Time: {assessment['transfer_time_minutes']:.1f} minutes")
        
    if "max_executable_amount" in assessment:
        print(f"Max Executable: {assessment['max_executable_amount']:.4f} BTC")
    
    print("\n=== OPTIMIZED PARAMETERS ===")
    print(f"Status: {optimal_params['status']}")
    if optimal_params['status'] == 'approved':
        print(f"Optimal Amount: {optimal_params['optimal_amount']:.4f} BTC")
        print(f"Strategy: {optimal_params['execution_strategy']}")
    else:
        print(f"Reason: {optimal_params['reason']}")
    
    # Save analysis
    risk_manager.save_analysis_data()
