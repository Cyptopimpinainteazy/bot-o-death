#!/usr/bin/env python
"""
Fund Pre-positioning System for Quantum Trading

This module predicts where funds will be needed and proactively positions
capital across exchanges to eliminate transfer delays during arbitrage.
"""

import os
import json
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from pathlib import Path
import threading
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FundPrepositioningManager:
    """
    Predictively positions funds across exchanges to eliminate transfer delays
    during arbitrage opportunities.
    """
    
    def __init__(self, config_file=None):
        """Initialize the fund prepositioning manager"""
        self.config_dir = Path("config")
        self.config = self._load_config(config_file)
        
        # Initialize exchange balances
        self.exchange_balances = {}
        self.historical_arbitrage = []
        self.opportunity_predictions = {}
        
        # Lock for thread safety
        self.lock = threading.Lock()
        
        # Load and initialize
        self._load_exchange_balances()
        self._load_historical_arbitrage()
        
        logger.info("Fund Prepositioning Manager initialized")
    
    def _load_config(self, config_file):
        """Load configuration from file or create default"""
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                logger.info(f"Loading configuration from {config_file}")
                return json.load(f)
        
        # Check for default config
        default_config_path = self.config_dir / "fund_prepositioning_config.json"
        if os.path.exists(default_config_path):
            with open(default_config_path, 'r') as f:
                logger.info(f"Loading existing configuration from {default_config_path}")
                return json.load(f)
        
        # Create default config
        default_config = {
            "exchanges": {
                "binance": {
                    "min_balance": {
                        "BTC": 0.5,
                        "ETH": 5.0,
                        "USDT": 10000.0,
                        "SOL": 50.0
                    },
                    "target_balance": {
                        "BTC": 1.0,
                        "ETH": 10.0,
                        "USDT": 20000.0,
                        "SOL": 100.0
                    }
                },
                "coinbase": {
                    "min_balance": {
                        "BTC": 0.5,
                        "ETH": 5.0,
                        "USDT": 10000.0,
                        "SOL": 50.0
                    },
                    "target_balance": {
                        "BTC": 1.0,
                        "ETH": 10.0,
                        "USDT": 20000.0,
                        "SOL": 100.0
                    }
                },
                "kraken": {
                    "min_balance": {
                        "BTC": 0.5,
                        "ETH": 5.0,
                        "USDT": 10000.0,
                        "SOL": 50.0
                    },
                    "target_balance": {
                        "BTC": 1.0,
                        "ETH": 10.0,
                        "USDT": 20000.0,
                        "SOL": 100.0
                    }
                }
            },
            "prediction": {
                "lookback_days": 7,
                "prediction_window_hours": 24,
                "prediction_confidence_threshold": 0.7,
                "rebalance_threshold_percent": 20,
                "idle_funds_allocation": "high_volume"
            },
            "execution": {
                "max_rebalance_frequency_hours": 6,
                "preferred_transfer_times": {
                    "BTC": "weekend",
                    "ETH": "night",
                    "SOL": "anytime"
                },
                "emergency_rebalance_threshold": 0.5
            }
        }
        
        # Save default config
        if not os.path.exists(self.config_dir):
            os.makedirs(self.config_dir)
            
        with open(default_config_path, 'w') as f:
            json.dump(default_config, f, indent=2)
            
        logger.info(f"Created default fund prepositioning configuration at {default_config_path}")
        return default_config
    
    def _load_exchange_balances(self):
        """Load current exchange balances"""
        # In production, this would fetch real balances from exchange APIs
        # For now, we'll use test data
        
        # Start with config target balances as a baseline
        for exchange, config in self.config["exchanges"].items():
            self.exchange_balances[exchange] = {
                "last_updated": datetime.now().isoformat(),
                "balances": config["target_balance"].copy()
            }
            
        logger.info(f"Loaded balances for {len(self.exchange_balances)} exchanges")
    
    def _load_historical_arbitrage(self):
        """Load historical arbitrage opportunities for pattern analysis"""
        # In production, this would load from a database
        # For now, create some synthetic data
        
        arbitrage_history_path = Path("data") / "arbitrage_history.csv"
        
        if os.path.exists(arbitrage_history_path):
            try:
                self.historical_arbitrage = pd.read_csv(arbitrage_history_path)
                logger.info(f"Loaded {len(self.historical_arbitrage)} historical arbitrage records")
            except Exception as e:
                logger.error(f"Failed to load arbitrage history: {str(e)}")
                self._generate_synthetic_arbitrage_history()
        else:
            logger.info("No arbitrage history found, generating synthetic data")
            self._generate_synthetic_arbitrage_history()
    
    def _generate_synthetic_arbitrage_history(self):
        """Generate synthetic arbitrage history for testing"""
        exchanges = list(self.config["exchanges"].keys())
        symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
        
        # Generate 100 random arbitrage opportunities over the past week
        now = datetime.now()
        dates = [now - timedelta(days=7) + timedelta(minutes=i*100) for i in range(100)]
        
        data = []
        for i, date in enumerate(dates):
            # Create more opportunities between certain exchange pairs
            if i % 3 == 0:
                buy_exchange = "binance"
                sell_exchange = "coinbase"
                symbol = "BTC/USDT"
                profit = np.random.uniform(0.005, 0.02)
            elif i % 3 == 1:
                buy_exchange = "coinbase"
                sell_exchange = "kraken"
                symbol = "ETH/USDT"
                profit = np.random.uniform(0.003, 0.015)
            else:
                buy_exchange = "kraken" 
                sell_exchange = "binance"
                symbol = "SOL/USDT"
                profit = np.random.uniform(0.004, 0.018)
            
            data.append({
                "timestamp": date.isoformat(),
                "buy_exchange": buy_exchange,
                "sell_exchange": sell_exchange,
                "symbol": symbol,
                "amount": np.random.uniform(0.5, 5.0) if "BTC" in symbol else np.random.uniform(5, 50.0),
                "profit_percent": profit,
                "executed": i % 5 != 0  # 80% of opportunities executed
            })
        
        self.historical_arbitrage = pd.DataFrame(data)
        
        # Save synthetic data
        save_dir = Path("data")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        self.historical_arbitrage.to_csv(save_dir / "arbitrage_history.csv", index=False)
        logger.info(f"Generated and saved {len(self.historical_arbitrage)} synthetic arbitrage records")
    
    def predict_fund_needs(self):
        """
        Predict which exchanges and currencies will need funds in the coming hours
        based on historical arbitrage patterns and market conditions
        """
        if len(self.historical_arbitrage) == 0:
            logger.warning("No historical data available for prediction")
            return {}
        
        # Convert to pandas DataFrame if it's a list
        if isinstance(self.historical_arbitrage, list):
            df = pd.DataFrame(self.historical_arbitrage)
        else:
            df = self.historical_arbitrage.copy()
        
        # Add time features
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        
        # Calculate predictions for each exchange-currency pair
        predictions = {}
        exchanges = list(self.config["exchanges"].keys())
        symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
        
        current_hour = datetime.now().hour
        current_day = datetime.now().weekday()
        
        for symbol in symbols:
            symbol_base = symbol.split('/')[0]  # Get BTC, ETH, SOL
            
            for exchange in exchanges:
                # Filter opportunities where this exchange was the buying side
                buy_opps = df[(df['buy_exchange'] == exchange) & (df['symbol'] == symbol)]
                
                # Check if we have enough data
                if len(buy_opps) < 5:
                    continue
                
                # Basic time pattern detection
                hour_counts = buy_opps['hour'].value_counts()
                day_counts = buy_opps['day_of_week'].value_counts()
                
                # Calculate hour similarity (is current hour typically active?)
                hour_probability = 0.0
                if current_hour in hour_counts:
                    hour_probability = hour_counts[current_hour] / sum(hour_counts.values)
                
                # Calculate day similarity
                day_probability = 0.0
                if current_day in day_counts:
                    day_probability = day_counts[current_day] / sum(day_counts.values)
                
                # Calculate overall probability
                probability = (hour_probability + day_probability) / 2
                
                # Determine estimated amount needed
                if len(buy_opps) > 0:
                    avg_amount = buy_opps['amount'].mean()
                    max_amount = buy_opps['amount'].max()
                else:
                    avg_amount = self.config["exchanges"][exchange]["target_balance"].get(symbol_base, 1.0)
                    max_amount = avg_amount * 2
                
                # Store prediction
                key = f"{exchange}_{symbol_base}"
                predictions[key] = {
                    "exchange": exchange,
                    "currency": symbol_base,
                    "probability": float(probability),
                    "avg_amount": float(avg_amount),
                    "max_amount": float(max_amount),
                    "estimated_need_time": "next_24_hours" if probability > 0.3 else "next_week"
                }
        
        self.opportunity_predictions = predictions
        logger.info(f"Generated {len(predictions)} fund need predictions")
        
        return predictions
    
    def calculate_optimal_allocation(self):
        """
        Calculate optimal fund allocation across exchanges based on predictions
        and current balances
        """
        if not self.opportunity_predictions:
            self.predict_fund_needs()
        
        # Get current balances
        current_balances = self.get_exchange_balances()
        
        # Calculate allocation adjustments
        adjustments = []
        
        # Sort predictions by probability (highest first)
        sorted_predictions = sorted(
            self.opportunity_predictions.values(), 
            key=lambda x: x['probability'], 
            reverse=True
        )
        
        # Determine fund needs
        for pred in sorted_predictions:
            exchange = pred['exchange']
            currency = pred['currency']
            probability = pred['probability']
            avg_amount = pred['avg_amount']
            
            # Skip low probability predictions
            threshold = self.config["prediction"]["prediction_confidence_threshold"]
            if probability < threshold:
                continue
            
            # Get current balance
            current = current_balances.get(exchange, {}).get(currency, 0)
            
            # Get target balance from config
            target = self.config["exchanges"][exchange]["target_balance"].get(currency, 0)
            min_balance = self.config["exchanges"][exchange]["min_balance"].get(currency, 0)
            
            # Adjust target based on prediction
            adjusted_target = max(target, avg_amount * 1.5)
            
            # Calculate deficit
            deficit = adjusted_target - current
            
            # If significant deficit, find where to source funds from
            if deficit > 0 and deficit / adjusted_target > self.config["prediction"]["rebalance_threshold_percent"] / 100:
                # Find exchanges with excess of this currency
                for source_exchange in self.config["exchanges"]:
                    if source_exchange == exchange:
                        continue
                    
                    # Get source balance
                    source_balance = current_balances.get(source_exchange, {}).get(currency, 0)
                    source_min = self.config["exchanges"][source_exchange]["min_balance"].get(currency, 0)
                    source_target = self.config["exchanges"][source_exchange]["target_balance"].get(currency, 0)
                    
                    # Calculate excess at source
                    excess = source_balance - max(source_target, source_min)
                    
                    # If source has excess, plan a transfer
                    if excess > 0:
                        transfer_amount = min(excess, deficit)
                        
                        if transfer_amount > 0:
                            # Add transfer to adjustments
                            adjustments.append({
                                "from_exchange": source_exchange,
                                "to_exchange": exchange,
                                "currency": currency,
                                "amount": transfer_amount,
                                "priority": probability,
                                "reason": f"Predicted opportunity with {probability:.1%} confidence"
                            })
                            
                            # Update deficits for next iteration
                            deficit -= transfer_amount
                            
                            # If deficit resolved, break
                            if deficit <= 0:
                                break
        
        # Sort adjustments by priority
        adjustments = sorted(adjustments, key=lambda x: x['priority'], reverse=True)
        
        logger.info(f"Calculated {len(adjustments)} fund adjustment recommendations")
        return adjustments
    
    def execute_fund_transfers(self, dry_run=True):
        """
        Execute the recommended fund transfers between exchanges
        
        Args:
            dry_run: If True, just simulate transfers without execution
        
        Returns:
            List of executed transfers
        """
        # Get transfer recommendations
        transfer_plan = self.calculate_optimal_allocation()
        
        executed_transfers = []
        
        for transfer in transfer_plan:
            from_exchange = transfer['from_exchange']
            to_exchange = transfer['to_exchange']
            currency = transfer['currency']
            amount = transfer['amount']
            
            logger.info(f"{'SIMULATION: ' if dry_run else ''}Transferring {amount} {currency} from {from_exchange} to {to_exchange}")
            
            if not dry_run:
                # In production, this would call APIs to execute the transfer
                # For now, we'll update our simulated balances
                try:
                    with self.lock:
                        # Deduct from source
                        if from_exchange in self.exchange_balances:
                            if currency in self.exchange_balances[from_exchange]['balances']:
                                self.exchange_balances[from_exchange]['balances'][currency] -= amount
                        
                        # Add to destination
                        if to_exchange in self.exchange_balances:
                            if currency in self.exchange_balances[to_exchange]['balances']:
                                self.exchange_balances[to_exchange]['balances'][currency] += amount
                            else:
                                self.exchange_balances[to_exchange]['balances'][currency] = amount
                        
                        # Update timestamp
                        self.exchange_balances[from_exchange]['last_updated'] = datetime.now().isoformat()
                        self.exchange_balances[to_exchange]['last_updated'] = datetime.now().isoformat()
                    
                    # Record execution
                    transfer['timestamp'] = datetime.now().isoformat()
                    transfer['status'] = 'completed'
                    executed_transfers.append(transfer)
                    
                except Exception as e:
                    logger.error(f"Failed to execute transfer: {str(e)}")
                    transfer['status'] = 'failed'
                    transfer['error'] = str(e)
                    executed_transfers.append(transfer)
            else:
                # For dry run, mark as simulated
                transfer['timestamp'] = datetime.now().isoformat()
                transfer['status'] = 'simulated'
                executed_transfers.append(transfer)
        
        # Save transfer history
        self._save_transfer_history(executed_transfers)
        
        return executed_transfers
    
    def _save_transfer_history(self, transfers):
        """Save transfer history to file"""
        history_dir = Path("results") / "fund_transfers"
        if not os.path.exists(history_dir):
            os.makedirs(history_dir)
            
        # Generate filename with timestamp
        filename = f"transfers_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(history_dir / filename, 'w') as f:
            json.dump(transfers, f, indent=2)
    
    def get_exchange_balances(self):
        """Get current exchange balances"""
        balances = {}
        for exchange, data in self.exchange_balances.items():
            balances[exchange] = data['balances']
        return balances
    
    def get_balance_status(self):
        """Get balance status report with predictions"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "balances": self.get_exchange_balances(),
            "predictions": self.opportunity_predictions,
            "status": {}
        }
        
        # Calculate status for each exchange and currency
        for exchange, data in self.config["exchanges"].items():
            report["status"][exchange] = {}
            
            for currency, min_balance in data["min_balance"].items():
                current = self.exchange_balances.get(exchange, {}).get("balances", {}).get(currency, 0)
                target = data["target_balance"].get(currency, 0)
                
                status = "optimal"
                if current < min_balance:
                    status = "critical"
                elif current < target * 0.8:
                    status = "low"
                elif current > target * 1.5:
                    status = "excess"
                
                report["status"][exchange][currency] = {
                    "status": status,
                    "current": current,
                    "minimum": min_balance,
                    "target": target,
                    "percent_of_target": (current / target) * 100 if target > 0 else 0
                }
        
        return report
    
    def start_automatic_rebalancing(self, interval_minutes=60, dry_run=True):
        """
        Start automatic rebalancing of funds in the background
        
        Args:
            interval_minutes: How often to check and rebalance
            dry_run: If True, simulate transfers without execution
        """
        def rebalance_job():
            while True:
                try:
                    logger.info("Running automatic fund rebalancing")
                    self.predict_fund_needs()
                    transfers = self.execute_fund_transfers(dry_run=dry_run)
                    
                    logger.info(f"Completed rebalancing cycle with {len(transfers)} transfers")
                    
                    # Sleep until next cycle
                    time.sleep(interval_minutes * 60)
                except Exception as e:
                    logger.error(f"Error in rebalancing job: {str(e)}")
                    time.sleep(300)  # Sleep for 5 minutes on error
        
        # Start background thread
        thread = threading.Thread(target=rebalance_job, daemon=True)
        thread.start()
        
        logger.info(f"Started automatic rebalancing every {interval_minutes} minutes")
        return thread


# Sample usage
if __name__ == "__main__":
    # Initialize fund prepositioning manager
    fund_manager = FundPrepositioningManager()
    
    # Predict fund needs
    predictions = fund_manager.predict_fund_needs()
    
    # Calculate optimal allocation
    allocation = fund_manager.calculate_optimal_allocation()
    
    # Execute transfers (simulation mode)
    executed = fund_manager.execute_fund_transfers(dry_run=True)
    
    # Get balance status report
    status = fund_manager.get_balance_status()
    
    # Print summary
    print("\n=== FUND PREPOSITIONING SUMMARY ===")
    print(f"Predicted {len(predictions)} potential opportunities")
    print(f"Recommended {len(allocation)} fund transfers")
    print(f"Executed {len([t for t in executed if t['status'] == 'simulated'])} simulated transfers")
    
    # Print current balances
    print("\n=== CURRENT BALANCES ===")
    for exchange, currencies in status["balances"].items():
        print(f"{exchange}:")
        for currency, amount in currencies.items():
            status_code = status["status"][exchange][currency]["status"]
            status_marker = "✅" if status_code == "optimal" else "⚠️" if status_code == "low" else "❌" if status_code == "critical" else "⬆️"
            print(f"  {currency}: {amount} {status_marker}")
    
    print("\nPre-positioned funds will eliminate transfer delays during arbitrage execution")
