import os
import sys
import time
import json
import asyncio
import logging
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import our custom modules
from market_depth import MarketDepthAnalyzer
from technical_analysis import TechnicalAnalysisEngine
from risk_management import RiskManager
from tx_fee_optimizer import TxFeeOptimizer
from slippage_control import SlippageController
from mempool_monitor import MempoolMonitor
from quantum import quantum_trade_strategy, create_quantum_circuit
from trade_execution import execute_trade

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("quantum_trading.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("EnhancedQuantumTrading")

# Load environment variables
load_dotenv()

class EnhancedQuantumTrading:
    """
    Enhanced Quantum Trading System that integrates:
    - Mempool monitoring for trading opportunities
    - Market depth data analysis
    - Technical analysis indicators
    - Risk management constraints
    - Transaction fee optimization
    - Slippage control mechanisms
    """
    
    def __init__(self):
        """Initialize the enhanced quantum trading system"""
        logger.info("Initializing Enhanced Quantum Trading System")
        
        # Initialize component modules
        self.market_depth = MarketDepthAnalyzer()
        self.technical_analysis = TechnicalAnalysisEngine()
        self.risk_manager = RiskManager()
        self.fee_optimizer = TxFeeOptimizer()
        self.slippage_controller = SlippageController()
        
        # Trading parameters
        self.trading_active = False
        self.monitoring_active = False
        self.target_tokens = []
        self.target_pairs = []
        
        # Configure trading
        self._load_configuration()
        
        # Mempool monitor (will be initialized in start_monitoring)
        self.mempool_monitor = None
        
        logger.info("Enhanced Quantum Trading System initialized")
    
    def _load_configuration(self):
        """Load trading configuration"""
        try:
            # Default configuration
            self.config = {
                "chains": ["ethereum", "polygon"],
                "default_chain": "polygon",
                "target_dexs": ["uniswap_v2", "sushiswap", "quickswap"],
                "min_order_value_usd": 1000,
                "max_slippage": 0.02,  # 2%
                "gas_priority": "medium",
                "risk_level": "moderate",
                "position_size_pct": 0.05,  # 5% of portfolio per position
                "take_profit_pct": 0.05,  # 5% take profit
                "stop_loss_pct": 0.02,   # 2% stop loss
                "quantum_circuit_config": {
                    "shots": 1024,
                    "depth": 4
                }
            }
            
            # Try to load custom config
            config_path = os.path.join(os.path.dirname(__file__), "config", "trading_config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    custom_config = json.load(f)
                    # Update config with custom values
                    self.config.update(custom_config)
            
            # Load target tokens and pairs
            self._load_target_tokens()
            
            # Configure risk management based on config
            self.risk_manager.set_risk_parameters({
                "max_position_size_pct": self.config["position_size_pct"],
                "max_daily_drawdown_pct": 0.03,  # 3% max daily drawdown
                "max_total_exposure_pct": 0.30,  # 30% max total exposure
                "stop_loss_pct": self.config["stop_loss_pct"],
                "take_profit_pct": self.config["take_profit_pct"]
            })
            
            logger.info(f"Configuration loaded: {json.dumps(self.config, indent=2)}")
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            # Use default config if loading fails
    
    def _load_target_tokens(self):
        """Load target tokens and trading pairs"""
        # Default tokens
        self.target_tokens = [
            # Ethereum tokens
            {"address": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2", "symbol": "WETH", "chain": "ethereum", "decimals": 18},
            {"address": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48", "symbol": "USDC", "chain": "ethereum", "decimals": 6},
            {"address": "0xdAC17F958D2ee523a2206206994597C13D831ec7", "symbol": "USDT", "chain": "ethereum", "decimals": 6},
            {"address": "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599", "symbol": "WBTC", "chain": "ethereum", "decimals": 8},
            
            # Polygon tokens
            {"address": "0x0d500B1d8E8eF31E21C99d1Db9A6444d3ADf1270", "symbol": "WMATIC", "chain": "polygon", "decimals": 18},
            {"address": "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174", "symbol": "USDC", "chain": "polygon", "decimals": 6},
            {"address": "0xc2132D05D31c914a87C6611C10748AEb04B58e8F", "symbol": "USDT", "chain": "polygon", "decimals": 6},
            {"address": "0x1BFD67037B42Cf73acF2047067bd4F2C47D9BfD6", "symbol": "WBTC", "chain": "polygon", "decimals": 8}
        ]
        
        # Create trading pairs
        self.target_pairs = []
        
        # Add common pairs for each chain
        for chain in self.config["chains"]:
            chain_tokens = [token for token in self.target_tokens if token["chain"] == chain]
            
            # Get base tokens (typically stablecoins or native tokens)
            base_tokens = [token for token in chain_tokens if token["symbol"] in ["USDC", "USDT", "WETH", "WMATIC", "WBTC"]]
            
            # Create pairs
            for base in base_tokens:
                for token in chain_tokens:
                    if base["address"] != token["address"]:
                        self.target_pairs.append({
                            "chain": chain,
                            "token_in": base["address"],
                            "token_out": token["address"],
                            "token_in_symbol": base["symbol"],
                            "token_out_symbol": token["symbol"],
                            "pair_name": f"{token['symbol']}/{base['symbol']}",
                            "decimals_in": base["decimals"],
                            "decimals_out": token["decimals"]
                        })
        
        logger.info(f"Loaded {len(self.target_tokens)} tokens and {len(self.target_pairs)} trading pairs")
    
    async def start_monitoring(self):
        """Start monitoring mempool for opportunities"""
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return False
        
        try:
            logger.info("Starting enhanced trading monitors")
            
            # Start market depth analyzer
            market_depth_task = asyncio.create_task(self.market_depth.start())
            
            # We use create_task instead of await to allow it to run concurrently
            
            # Start mempool monitoring
            self.mempool_monitor = MempoolMonitor(chain=self.config["default_chain"])
            self.mempool_monitor.register_opportunity_callback(self.handle_mempool_opportunity)
            self.mempool_monitor.start_monitoring()
            
            self.monitoring_active = True
            
            logger.info("All monitors started successfully")
            
            # Keep monitors running
            while self.monitoring_active:
                # Periodically check market data and update risk parameters
                await self._update_market_analysis()
                await asyncio.sleep(60)  # Update analysis every minute
            
            # Cleanup when stopped
            self.mempool_monitor.stop()
            self.market_depth.stop()
            await market_depth_task
            
            return True
            
        except Exception as e:
            logger.error(f"Error starting monitoring: {e}")
            self.monitoring_active = False
            return False
    
    def stop_monitoring(self):
        """Stop all monitoring components"""
        logger.info("Stopping all monitors")
        self.monitoring_active = False
        
        if self.mempool_monitor:
            self.mempool_monitor.stop_monitoring()
        
        self.market_depth.stop()
    
    async def _update_market_analysis(self):
        """Update market analysis for all target pairs"""
        try:
            # Choose a subset of pairs to analyze (to avoid too many API calls)
            sample_pairs = self.target_pairs[:min(5, len(self.target_pairs))]
            
            for pair in sample_pairs:
                # Get market depth
                depth_data = self.market_depth.get_market_depth(
                    f"{pair['token_out_symbol']}/{pair['token_in_symbol']}"
                )
                
                # Get technical indicators
                indicators = self.technical_analysis.calculate_all_indicators(
                    f"{pair['token_out_symbol']}/{pair['token_in_symbol']}", 
                    "1h"
                )
                
                # Check for anomalies
                anomalies = self.market_depth.detect_liquidity_anomalies(
                    f"{pair['token_out_symbol']}/{pair['token_in_symbol']}"
                )
                
                # If we found strong signals, log them
                if indicators and "indicators" in indicators:
                    # Check for notable signals
                    rsi = indicators["indicators"].get("rsi", {}).get("current", 50)
                    if rsi < 30 or rsi > 70:
                        logger.info(f"Notable RSI for {pair['pair_name']}: {rsi}")
                
                if anomalies and anomalies.get("anomaly", False):
                    logger.info(f"Liquidity anomaly detected for {pair['pair_name']}: {anomalies['reasons']}")
                
                # Sleep briefly between pairs to avoid API rate limits
                await asyncio.sleep(1)
            
            # Update risk parameters based on market conditions
            # Use volatility from ATR indicator and trend from ADX
            volatility = 0.5  # Default moderate volatility
            trend = 0  # Default neutral trend
            
            # If we have indicator data, calculate average volatility and trend
            if indicators and "indicators" in indicators:
                # Extract volatility from ATR
                atr = indicators["indicators"].get("atr", {})
                if atr and "percentage" in atr:
                    # Normalize to 0-1 range (0-10% ATR)
                    volatility = min(1.0, atr["percentage"] / 10.0)
                
                # Extract trend from ADX
                adx = indicators["indicators"].get("adx", {})
                if adx and "adx" in adx and "plus_di" in adx and "minus_di" in adx:
                    # Calculate trend direction (-1 to 1)
                    if adx["adx"] > 20:  # Only consider strong trends
                        if adx["plus_di"] > adx["minus_di"]:
                            trend = min(1.0, (adx["adx"] - 20) / 30)  # Normalize to 0-1
                        else:
                            trend = max(-1.0, -(adx["adx"] - 20) / 30)  # Normalize to -1-0
            
            # Adjust risk parameters
            self.risk_manager.adjust_for_market_conditions(volatility, trend)
            
        except Exception as e:
            logger.error(f"Error updating market analysis: {e}")
    
    def handle_mempool_opportunity(self, tx_data):
        """Handle trading opportunity from mempool"""
        if not self.trading_active:
            logger.info(f"Trading opportunity detected, but trading is disabled: {tx_data['hash']}")
            return
        
        try:
            logger.info(f"Processing trading opportunity from mempool: {tx_data['hash']}")
            
            # Extract relevant data from the transaction
            chain = tx_data.get("chain", self.config["default_chain"])
            token_in = tx_data.get("token_in")
            token_out = tx_data.get("token_out")
            amount_in = tx_data.get("amount_in", 0)
            
            # Skip if we don't have enough information
            if not token_in or not token_out or amount_in <= 0:
                logger.warning(f"Insufficient transaction data: {json.dumps(tx_data)}")
                return
            
            # Find token symbols
            token_in_symbol = next((token["symbol"] for token in self.target_tokens 
                                  if token["address"].lower() == token_in.lower() and token["chain"] == chain), "Unknown")
            token_out_symbol = next((token["symbol"] for token in self.target_tokens 
                                   if token["address"].lower() == token_out.lower() and token["chain"] == chain), "Unknown")
            
            logger.info(f"Detected swap: {amount_in} {token_in_symbol} to {token_out_symbol} on {chain}")
            
            # Run enhanced opportunity analysis
            asyncio.create_task(self.analyze_and_execute_opportunity(
                chain, token_in, token_out, amount_in, tx_data
            ))
            
        except Exception as e:
            logger.error(f"Error handling mempool opportunity: {e}")
    
    async def analyze_and_execute_opportunity(self, chain, token_in, token_out, amount_in, tx_data=None):
        """Analyze trading opportunity with all enhanced features and execute if profitable"""
        try:
            # 1. Get market depth data
            pair_symbol = "Unknown/Unknown"
            for pair in self.target_pairs:
                if (pair["chain"] == chain and 
                    pair["token_in"].lower() == token_in.lower() and 
                    pair["token_out"].lower() == token_out.lower()):
                    pair_symbol = pair["pair_name"]
                    break
            
            depth_data = self.market_depth.get_market_depth(pair_symbol)
            
            # Check order book imbalance
            imbalance = self.market_depth.calculate_imbalance(pair_symbol)
            
            # 2. Get technical indicators
            indicators = self.technical_analysis.calculate_all_indicators(pair_symbol, "1h")
            
            # Get combined signal across multiple timeframes
            signal = self.technical_analysis.get_combined_signal(pair_symbol, ["1h", "4h"])
            
            # Check for divergence (possible reversal)
            divergence = self.technical_analysis.detect_divergence(pair_symbol, "1h", "rsi")
            
            # 3. Run quantum analysis
            # Create a quantum circuit with market data influence
            quantum_params = {
                "depth": self.config["quantum_circuit_config"]["depth"],
                "shots": self.config["quantum_circuit_config"]["shots"]
            }
            
            # Add market factors to quantum parameters
            if indicators and "indicators" in indicators:
                # Use RSI as a quantum parameter
                rsi = indicators["indicators"].get("rsi", {}).get("current", 50)
                quantum_params["rsi"] = rsi / 100  # Normalize to 0-1
                
                # Use MACD histogram
                macd = indicators["indicators"].get("macd", {})
                if macd and "histogram" in macd:
                    # Normalize MACD histogram (roughly -10 to +10 range)
                    quantum_params["macd"] = max(-1, min(1, macd["histogram"] / 10))
            
            # Add order book imbalance
            quantum_params["imbalance"] = imbalance
            
            # Run quantum circuit
            quantum_circuit = create_quantum_circuit(**quantum_params)
            quantum_result = quantum_trade_strategy(quantum_circuit)
            
            # Extract quantum factors
            quantum_buy = quantum_result.get("buy_probability", 0)
            quantum_sell = quantum_result.get("sell_probability", 0)
            quantum_factor = quantum_buy - quantum_sell  # Range from -1 to 1
            
            # 4. Risk analysis
            # Adjust risk based on quantum signal
            self.risk_manager.adjust_quantum_risk(quantum_factor)
            
            # Calculate appropriate position size
            position = self.risk_manager.calculate_position_size(
                pair_symbol.split('/')[0],  # Token symbol
                price=1.0,  # Will be updated with real price later
                strategy_type="standard"
            )
            
            # 5. Determine overall trading action
            # Combine quantum and technical signals
            overall_signal = "hold"
            signal_strength = abs(quantum_factor) * 0.7 + (signal["overall"]["bullish_pct"] / 100) * 0.3
            
            if quantum_factor > 0.2 and signal["overall"]["bullish_pct"] > 40:
                overall_signal = "buy"
            elif quantum_factor < -0.2 and signal["overall"]["bearish_pct"] > 40:
                overall_signal = "sell"
            
            # Log analysis results
            logger.info(f"Analysis for {pair_symbol}:")
            logger.info(f"  - Order book imbalance: {imbalance:.2f}")
            logger.info(f"  - Technical signal: {signal['overall']['signal']}")
            logger.info(f"  - Divergence: {divergence.get('divergence', 'none')}")
            logger.info(f"  - Quantum factor: {quantum_factor:.2f}")
            logger.info(f"  - Overall signal: {overall_signal} (strength: {signal_strength:.2f})")
            
            # If we have a clear signal, prepare to execute
            if overall_signal != "hold" and signal_strength > 0.5:
                # 6. Optimize transaction parameters
                # Optimize gas price
                gas_data = self.fee_optimizer.optimize_gas_price(
                    chain, 
                    priority=self.config["gas_priority"]
                )
                
                # Optimize slippage
                if overall_signal == "buy":
                    tokens = [token_in, token_out]  # Buy token_out with token_in
                else:
                    tokens = [token_out, token_in]  # Sell token_out for token_in
                
                # Find best trading route
                route = self.slippage_controller.optimize_trading_route(
                    chain, tokens, amount_in
                )
                
                # Calculate optimal trade size to minimize slippage
                trade_sizing = self.slippage_controller.calculate_optimal_trade_size(
                    chain, route["dex"] if route else "uniswap_v2", 
                    tokens[0], tokens[1], amount_in
                )
                
                # Recommend slippage tolerance
                slippage = self.slippage_controller.recommend_slippage_tolerance(
                    chain, route["dex"] if route else "uniswap_v2", 
                    tokens[0], tokens[1], amount_in
                )
                
                logger.info(f"Execution parameters:")
                logger.info(f"  - Gas price: {gas_data['readableMaxFee'] if gas_data else 'Unknown'}")
                logger.info(f"  - Optimal route: {route['dex'] if route else 'direct'}")
                logger.info(f"  - Recommended slippage: {slippage * 100:.2f}%")
                logger.info(f"  - Trade sizing: {len(trade_sizing['trade_sizes'])} trades")
                
                # 7. Execute the trade
                if self.trading_active:
                    # Validate trade against risk management
                    trade_params = {
                        "symbol": pair_symbol.split('/')[0],
                        "price": 1.0,  # Placeholder, would use actual price in real implementation
                        "units": position["units"],
                        "side": overall_signal,
                        "strategy": "standard"
                    }
                    
                    validation = self.risk_manager.validate_trade(trade_params)
                    
                    if validation["valid"]:
                        logger.info(f"Executing {overall_signal} trade for {pair_symbol}")
                        
                        # Call trade execution with all enhanced parameters
                        # This is a simplified example - real execution would include more parameters
                        execute_trade(
                            buy_chain=chain if overall_signal == "buy" else None,
                            sell_chain=chain if overall_signal == "sell" else None,
                            amount_in=position["value"],
                            slippage_tolerance=slippage,
                            gas_price=gas_data["maxFeePerGas"] if gas_data and "maxFeePerGas" in gas_data else None
                        )
                        
                        # Record trade in risk manager
                        self.risk_manager.record_trade(trade_params)
                        
                        logger.info(f"Trade executed successfully")
                    else:
                        logger.warning(f"Trade validation failed: {validation['reason']}")
                else:
                    logger.info("Trade would be executed, but trading is disabled")
            else:
                logger.info(f"No clear trading signal or signal too weak ({signal_strength:.2f})")
        
        except Exception as e:
            logger.error(f"Error analyzing opportunity: {e}")
    
    def start_trading(self):
        """Enable automated trading based on signals"""
        logger.info("Enabling automated trading")
        self.trading_active = True
    
    def stop_trading(self):
        """Disable automated trading"""
        logger.info("Disabling automated trading")
        self.trading_active = False
    
    def get_portfolio_summary(self):
        """Get current portfolio summary"""
        return self.risk_manager.get_portfolio_summary()


# Example usage
if __name__ == "__main__":
    async def main():
        # Initialize enhanced trading system
        trading = EnhancedQuantumTrading()
        
        # Start in monitoring-only mode (no automated trading)
        monitor_task = asyncio.create_task(trading.start_monitoring())
        
        try:
            # Run for a while to gather data
            logger.info("Running in monitoring mode for 2 minutes")
            await asyncio.sleep(120)
            
            # Enable automated trading
            trading.start_trading()
            logger.info("Automated trading enabled")
            
            # Run with trading enabled
            logger.info("Running with trading enabled for 5 minutes")
            await asyncio.sleep(300)
            
            # Disable trading but keep monitoring
            trading.stop_trading()
            logger.info("Automated trading disabled")
            
            # Run a bit longer in monitoring mode
            logger.info("Running in monitoring mode for 1 more minute")
            await asyncio.sleep(60)
            
        finally:
            # Stop everything
            trading.stop_monitoring()
            await monitor_task
    
    # Run the example
    asyncio.run(main())
