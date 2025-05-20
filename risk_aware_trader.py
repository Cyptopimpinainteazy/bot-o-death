#!/usr/bin/env python
"""
Risk-Aware Quantum Trader
------------------------
Integrates the Exchange Risk Manager with the Quantum Trading System
to make trading decisions that account for exchange risks:
- Transfer delays between exchanges
- Fee management
- Liquidity assessment
"""

import os
import logging
import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

# Import our modules
from exchange_risk_manager import ExchangeRiskManager
from quantum_ensemble import QuantumEnsembleTrader
from technical_analysis import TechnicalAnalysisEngine
from fund_prepositioning import FundPrepositioningManager
from triangle_arbitrage import TriangleArbitrageDetector
from market_data_service import MarketDataService
from mev_bundle_manager import MEVBundleManager
from flashloan_manager import FlashloanManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("risk_aware_trading.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("RiskAwareTrader")

class RiskAwareQuantumTrader:
    """Integrates risk management with quantum trading strategies"""
    
    def __init__(self, config_file=None):
        """Initialize the risk-aware trader"""
        self.config_dir = Path("config")
        self.results_dir = Path("results") / "risk_trading"
        
        # Create directories
        os.makedirs(self.config_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Load configuration
        self.config = self._load_config(config_file)
        
        # Initialize risk manager
        self.risk_manager = ExchangeRiskManager()
        
        # Initialize quantum trading components
        self.ta_engine = TechnicalAnalysisEngine()
        self.ensemble_trader = QuantumEnsembleTrader()
        
        # Initialize fund prepositioning system
        # TODO: This will be made into a configurable option in the GUI mode
        self.fund_manager = FundPrepositioningManager()
        self.use_fund_prepositioning = self.config.get("trading", {}).get("use_fund_prepositioning", True)
        
        # Initialize triangle arbitrage detector
        # TODO: This will be made into a configurable option in the GUI mode
        self.triangle_detector = TriangleArbitrageDetector()
        self.use_triangle_arbitrage = True  # Default enabled
        
        # Initialize multi-threaded market data service
        self.market_data_service = MarketDataService()
        self.use_parallel_data_fetching = True  # Default enabled
        
        # Initialize MEV bundle manager
        self.mev_bundle_manager = MEVBundleManager()
        self.use_mev_bundles = True  # Default enabled
        
        # Initialize flashloan manager
        self.flashloan_manager = FlashloanManager()
        self.use_flashloans = self.config.get("trading", {}).get("use_flashloans", True)
        
        # Trading stats
        self.trades = []
        self.rejected_opportunities = []
        
        logger.info("Risk-Aware Quantum Trader initialized")
    
    def _check_fund_positioning(self):
        """Check and manage fund positioning across exchanges"""
        # TODO: This will be made into a configurable option in the GUI mode
        
        # Update predictions based on recent trading patterns
        predictions = self.fund_manager.predict_fund_needs()
        
        # Check if we need to rebalance
        if predictions:
            # Calculate recommended adjustments
            adjustments = self.fund_manager.calculate_optimal_allocation()
            if adjustments:
                logger.info(f"Fund prepositioning identified {len(adjustments)} recommended transfers")
                
                # TODO: In GUI mode, we will offer manual approval option for transfers
                transfers = self.fund_manager.execute_fund_transfers(dry_run=False)
                logger.info(f"Executed {len(transfers)} fund transfers to prepare for trading")
            else:
                logger.info("Fund positioning optimal - no transfers needed")
    
    def _load_config(self, config_file):
        """Load configuration from file or create default"""
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                return json.load(f)
        
        # Default configuration
        config = {
            "trading": {
                "max_position_size_usd": 10000,
                "min_confidence_threshold": 0.7,
                "min_viable_profit": 0.005,  # 0.5%
                "min_transfer_viability": 0.6,
                "use_fund_prepositioning": True,  # Enable fund prepositioning by default
                "exchanges": ["binance", "coinbase", "kraken"]
            },
            "risk_management": {
                "max_open_positions": 5,
                "max_drawdown_percent": 5,
                "position_sizing_method": "volatility",  # fixed, volatility, kelly
                "max_slippage_tolerance": 0.003  # 0.3%
            },
            "market_data": {
                "symbols": ["BTC/USDT", "ETH/USDT", "SOL/USDT"],
                "update_interval_seconds": 60,
                "lookback_periods": 24  # Hours
            }
        }
        
        # Save default config
        config_path = self.config_dir / "risk_aware_trading_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Created default risk aware trading configuration at {config_path}")
        return config
    
    def identify_arbitrage_opportunities(self, market_data):
        """
        Identify potential arbitrage opportunities across exchanges
        """
        # Get direct exchange-to-exchange arbitrage opportunities
        direct_opportunities = self._identify_direct_arbitrage(market_data)
        
        # Get triangle arbitrage opportunities if enabled
        triangle_opportunities = []
        if self.use_triangle_arbitrage:
            triangle_opportunities = self._identify_triangle_arbitrage(market_data)
        
        # Combine all opportunities
        opportunities = direct_opportunities + triangle_opportunities
        logger.info(f"Identified {len(opportunities)} potential arbitrage opportunities: " 
                   f"{len(direct_opportunities)} direct, {len(triangle_opportunities)} triangle")
        
        return opportunities
    
    def _identify_direct_arbitrage(self, market_data):
        """
        Identify direct arbitrage opportunities between different exchanges
        """
        opportunities = []
        symbols = self.config["market_data"]["symbols"]
        exchanges = self.config["trading"]["exchanges"]
        
        # For each symbol and exchange pair
        for symbol in symbols:
            # Extract latest prices across all exchanges
            prices = {}
            for exchange in exchanges:
                if exchange in market_data and symbol in market_data[exchange]:
                    prices[exchange] = market_data[exchange][symbol]["price"]
            
            # Need at least 2 exchanges to compare
            if len(prices) < 2:
                continue
            
            # Find arbitrage opportunities
            for buy_exchange in prices:
                for sell_exchange in prices:
                    if buy_exchange == sell_exchange:
                        continue
                    
                    buy_price = prices[buy_exchange]
                    sell_price = prices[sell_exchange]
                    
                    # If we can buy lower on one exchange and sell higher on another
                    if sell_price > buy_price:
                        price_diff_pct = (sell_price - buy_price) / buy_price
                        
                        # Only consider significant price differences
                        if price_diff_pct >= self.config["trading"]["min_viable_profit"]:
                            opportunity = {
                                "timestamp": datetime.now().isoformat(),
                                "type": "direct", 
                                "symbol": symbol,
                                "buy_exchange": buy_exchange,
                                "sell_exchange": sell_exchange,
                                "buy_price": buy_price,
                                "sell_price": sell_price,
                                "price_diff_pct": price_diff_pct,
                                "amount": 1.0  # Default amount, will be optimized later
                            }
                            opportunities.append(opportunity)
        
        logger.info(f"Identified {len(opportunities)} potential direct arbitrage opportunities")
        return opportunities
        
    def _identify_triangle_arbitrage(self, market_data):
        """
        Identify triangle arbitrage opportunities within single exchanges
        """
        # Use the triangle arbitrage detector to find opportunities
        triangle_opportunities = self.triangle_detector.find_triangle_opportunities(market_data)
        
        # Process and filter opportunities
        processed_opportunities = []
        min_profit = self.config["trading"].get("min_viable_profit", 0.005)
        
        for opp in triangle_opportunities:
            # Only consider opportunities that meet our minimum profit threshold
            if opp["profit_pct"] >= min_profit:
                # Add standard fields for compatibility with our system
                opp["price_diff_pct"] = opp["profit_pct"]
                opp["buy_exchange"] = opp["exchange"]  # Same exchange for all legs
                opp["sell_exchange"] = opp["exchange"]  # Same exchange for all legs
                
                processed_opportunities.append(opp)
        
        logger.info(f"Identified {len(processed_opportunities)} viable triangle arbitrage opportunities")
        return processed_opportunities
    
    def filter_opportunities_by_risk(self, opportunities):
        """
        Filter arbitrage opportunities based on risk assessment
        """
        viable_opportunities = []
        
        for opportunity in opportunities:
            # Get opportunity type
            opportunity_type = opportunity.get("type", "direct")
            
            # Assess viability based on opportunity type
            if opportunity_type == "triangle":
                # For triangle arbitrage, use specialized assessment
                viability, assessment = self.triangle_detector.assess_opportunity_risk(
                    opportunity, self.risk_manager
                )
            else:
                # For direct arbitrage use standard assessment
                viability, assessment = self.risk_manager.assess_arbitrage_viability(opportunity)
            
            # Check if viable based on our threshold
            min_viability = self.config["trading"]["min_transfer_viability"]
            if viability >= min_viability:
                # Add risk assessment to the opportunity
                opportunity["risk_assessment"] = assessment
                opportunity["viability_score"] = viability
                viable_opportunities.append(opportunity)
            else:
                # Record rejected opportunity
                opportunity["risk_assessment"] = assessment
                opportunity["viability_score"] = viability
                opportunity["rejection_reason"] = assessment.get("reason", "Low viability score")
                self.rejected_opportunities.append(opportunity)
        
        # Log results by type
        direct_count = sum(1 for o in viable_opportunities if o.get("type") == "direct")
        triangle_count = sum(1 for o in viable_opportunities if o.get("type") == "triangle")
        logger.info(
            f"Filtered to {len(viable_opportunities)} viable opportunities after risk assessment: "
            f"{direct_count} direct, {triangle_count} triangle"
        )
        return viable_opportunities
    
    def optimize_trade_parameters(self, opportunities):
        """
        Optimize trade parameters for viable opportunities
        """
        optimized_opportunities = []
        
        for opportunity in opportunities:
            # Get optimized parameters
            params = self.risk_manager.get_optimized_trade_parameters(opportunity)
            
            if params["status"] == "approved":
                # Update opportunity with optimized parameters
                original_amount = params["optimal_amount"]
                execution_strategy = params["execution_strategy"]
                
                # If fund prepositioning is enabled, check if funds are available
                if self.use_fund_prepositioning:
                    # Check if we have sufficient funds at the buy exchange
                    buy_exchange = opportunity['buy_exchange']
                    symbol_base = opportunity['symbol'].split('/')[0]  # Get BTC, ETH, SOL
                    
                    balances = self.fund_manager.get_exchange_balances()
                    available_balance = balances.get(buy_exchange, {}).get(symbol_base, 0)
                    
                    if available_balance < original_amount:
                        logger.warning(f"Insufficient {symbol_base} on {buy_exchange} for trade: " 
                                     f"needed {original_amount}, available {available_balance}")
                        
                        if available_balance <= 0:
                            # No funds available, reject opportunity
                            opportunity["rejection_reason"] = f"No {symbol_base} available on {buy_exchange}"
                            self.rejected_opportunities.append(opportunity)
                            continue
                        else:
                            # Adjust amount to available balance
                            original_amount = available_balance
                            logger.info(f"Adjusted trade amount to available balance: {available_balance} {symbol_base}")
                    else:
                        logger.info(f"Sufficient funds available on {buy_exchange}: {available_balance} {symbol_base}")
                
                opportunity["optimized_amount"] = original_amount
                opportunity["execution_strategy"] = execution_strategy
                opportunity["expected_profit"] = params["expected_profit"]
                optimized_opportunities.append(opportunity)
            else:
                # Record rejected opportunity
                opportunity["rejection_reason"] = params.get("reason", "Failed optimization")
                self.rejected_opportunities.append(opportunity)
        
        logger.info(f"Optimized {len(optimized_opportunities)} trading opportunities")
        return optimized_opportunities
    
    def execute_trades(self, optimized_opportunities):
        """
        Execute trades for optimized opportunities, using MEV bundles where appropriate
        
        Args:
            optimized_opportunities: List of optimized trading opportunities
            
        Returns:
            List of executed trades
        """
        if not optimized_opportunities:
            logger.info("No opportunities to execute")
            return []
        
        executed_trades = []
        
        # Group opportunities by network/chain to enable bundling
        network_opportunities = {}
        non_bundleable = []
        
        # If MEV bundling is disabled, execute trades individually
        if not self.use_mev_bundles:
            for opportunity in optimized_opportunities:
                trade = self._execute_single_trade(opportunity)
                if trade:
                    self.trades.append(trade)
                    executed_trades.append(trade)
            return executed_trades
        
        # Pre-process opportunities to check for flashloan enhancement
        enhanced_opportunities = []
        for opportunity in optimized_opportunities:
            # Check if opportunity can be enhanced with flashloans
            if self.use_flashloans:
                # Determine if flashloan is suitable for this opportunity
                is_suitable, loan_amount, provider = self.flashloan_manager.is_flashloan_suitable(opportunity)
                
                if is_suitable:
                    # Create a copy to avoid modifying the original
                    enhanced_opp = opportunity.copy()
                    
                    # Add flashloan details
                    enhanced_opp["use_flashloan"] = True
                    enhanced_opp["flashloan_amount"] = loan_amount
                    enhanced_opp["flashloan_provider"] = provider
                    enhanced_opp["original_amount"] = opportunity.get("optimized_amount", 0)
                    
                    # Calculate flashloan fee
                    if provider == "aave_v3":
                        fee_rate = 0.0009  # 0.09% for Aave v3
                    elif provider == "balancer":
                        fee_rate = 0.0006  # 0.06% for Balancer
                    else:
                        fee_rate = 0.001   # Default 0.1%
                    
                    enhanced_opp["flashloan_fee"] = loan_amount * fee_rate
                    
                    # Scale profit based on amplified amount minus fee
                    base_profit = opportunity.get("expected_profit", 0)
                    amplification_factor = loan_amount / opportunity.get("optimized_amount", 1)
                    enhanced_opp["expected_profit"] = (base_profit * amplification_factor) - enhanced_opp["flashloan_fee"]
                    
                    logger.info(f"Enhanced opportunity with {loan_amount} {opportunity.get('symbol', '').split('/')[0] if 'symbol' in opportunity else 'ETH'} " +
                               f"flashloan using {provider}. Profit amplified to {enhanced_opp['expected_profit']:.6f}")
                    
                    enhanced_opportunities.append(enhanced_opp)
                else:
                    # No flashloan, use original opportunity
                    enhanced_opp = opportunity.copy()
                    enhanced_opp["use_flashloan"] = False
                    enhanced_opportunities.append(enhanced_opp)
            else:
                # Flashloans disabled, use original opportunity
                enhanced_opp = opportunity.copy()
                enhanced_opp["use_flashloan"] = False
                enhanced_opportunities.append(enhanced_opp)
        
        # Replace original opportunities with enhanced ones
        optimized_opportunities = enhanced_opportunities
        
        # Group opportunities by network for bundling
        for opportunity in optimized_opportunities:
            network = opportunity.get("network", "ethereum")
            
            # Check if opportunity is compatible with MEV bundling
            if self._is_bundleable(opportunity):
                if network not in network_opportunities:
                    network_opportunities[network] = []
                network_opportunities[network].append(opportunity)
            else:
                non_bundleable.append(opportunity)
        
        # Execute bundles for each network
        for network, network_opps in network_opportunities.items():
            if len(network_opps) > 1:
                logger.info(f"Creating MEV bundle with {len(network_opps)} opportunities on {network}")
                
                # Convert opportunities to format expected by MEV bundle manager
                bundle_opportunities = [self._convert_to_bundle_format(opp) for opp in network_opps]
                
                # Create and submit bundle
                try:
                    bundle = self.mev_bundle_manager.create_arbitrage_bundle(
                        opportunities=bundle_opportunities,
                        network=network
                    )
                    
                    if bundle:
                        result = self.mev_bundle_manager.submit_bundle(bundle)
                        
                        if result["any_success"]:
                            logger.info(f"Successfully submitted MEV bundle for {len(network_opps)} opportunities")
                            
                            # Record each opportunity in the bundle as a trade
                            for opp in network_opps:
                                trade = self._create_trade_record(opp, "executed", bundle_id=bundle["bundle_hash"])
                                self.trades.append(trade)
                                executed_trades.append(trade)
                        else:
                            logger.warning(f"Failed to submit MEV bundle: {result.get('results')}")
                            # Fall back to individual execution if bundle submission fails
                            for opp in network_opps:
                                non_bundleable.append(opp)
                    else:
                        logger.warning("Failed to create MEV bundle, falling back to individual execution")
                        for opp in network_opps:
                            non_bundleable.append(opp)
                            
                except Exception as e:
                    logger.error(f"Error creating/submitting MEV bundle: {e}")
                    # Fall back to individual execution
                    for opp in network_opps:
                        non_bundleable.append(opp)
            else:
                # Only one opportunity for this network, execute individually
                for opp in network_opps:
                    non_bundleable.append(opp)
        
        # Execute remaining opportunities individually
        for opportunity in non_bundleable:
            trade = self._execute_single_trade(opportunity)
            if trade:
                self.trades.append(trade)
                executed_trades.append(trade)
        
        # Record trade data for AI training
        self._record_trade_data_for_training(executed_trades)
        
        return executed_trades
        
    def _execute_single_trade(self, opportunity):
        """Execute a single trade opportunity"""
        opportunity_type = opportunity.get("type", "direct")
        
        # Execute based on opportunity type
        if opportunity_type == "triangle":
            trade = self._execute_triangle_arbitrage(opportunity)
        else:
            trade = self._execute_direct_arbitrage(opportunity)
            
        if trade:
            logger.info(f"Executed {opportunity_type} trade: {trade['trade_id']} - {trade.get('trading_path', trade.get('symbol'))}")
            
        return trade
        
    def _execute_direct_arbitrage(self, opportunity):
        """
        Execute a direct arbitrage trade between exchanges
        
        Args:
            opportunity: The optimized trading opportunity
            
        Returns:
            Dictionary with trade details
        """
        # In a real system, this would call exchange APIs
        # For now, we'll simulate execution
        
        trade = {
            "timestamp": datetime.now().isoformat(),
            "trade_id": f"trade_{len(self.trades) + 1}",
            "type": "direct",
            "symbol": opportunity["symbol"],
            "buy_exchange": opportunity["buy_exchange"],
            "sell_exchange": opportunity["sell_exchange"],
            "amount": opportunity["optimized_amount"],
            "buy_price": opportunity["buy_price"],
            "sell_price": opportunity["sell_price"],
            "expected_profit": opportunity["expected_profit"],
            "status": "executed",
            "execution_strategy": opportunity["execution_strategy"]
        }
        
        return trade
        
    def _execute_triangle_arbitrage(self, opportunity):
        """
        Execute a triangle arbitrage trade within a single exchange
        
        Args:
            opportunity: The optimized triangle trading opportunity
            
        Returns:
            Dictionary with trade details
        """
        # In a real system, this would be handled by the triangle detector's execute method
        # which would interact with the exchange API
        
        # Get exchange and trading path
        exchange = opportunity["exchange"]
        trading_path = opportunity.get("trading_path", {})
        
        # Mock execution by calling the triangle arbitrage executor
        # In a real system, we would do: result = self.triangle_detector.execute_triangle_arbitrage(opportunity, exchange_api)
        
        # For now, simulate a successful trade
        trade = {
            "timestamp": datetime.now().isoformat(),
            "trade_id": f"trade_{len(self.trades) + 1}",
            "type": "triangle",
            "exchange": exchange,
            "trading_path": trading_path,
            "currencies": opportunity.get("currencies", []),
            "trading_pairs": opportunity.get("trading_pairs", []),
            "amount": opportunity.get("optimal_start_amount", 1.0),
            "expected_profit": opportunity.get("profit_pct", 0),
            "status": "executed",
            "execution_strategy": "triangle"
        }
        
        return trade
    
    def monitor_trades(self):
        """
        Monitor ongoing trades and update their status
        This would poll exchange APIs in a real system
        """
        active_trades = [trade for trade in self.trades if trade["status"] in ["executed", "partial"]]
        
        for trade in active_trades:
            # In a real system, this would query exchange APIs
            # For now, we'll simulate with random completions
            
            # Simulate trade progress (25% chance of completion per check)
            if np.random.random() < 0.25:
                trade["status"] = "completed"
                trade["completion_time"] = datetime.now().isoformat()
                
                # Simulate actual profit (may vary from expected)
                profit_variance = np.random.normal(0, 0.002)
                trade["actual_profit"] = trade["expected_profit"] + profit_variance
                
                logger.info(f"Trade completed: {trade['trade_id']} with profit {trade['actual_profit']:.2%}")
        
        completed = [t for t in self.trades if t["status"] == "completed"]
        active = [t for t in self.trades if t["status"] in ["executed", "partial"]]
        
        logger.info(f"Trade status: {len(completed)} completed, {len(active)} active")
        return completed, active
    
    def generate_trading_report(self):
        """
        Generate a comprehensive trading report
        """
        completed_trades = [trade for trade in self.trades if trade["status"] == "completed"]
        
        if not completed_trades:
            return {
                "status": "No completed trades",
                "timestamp": datetime.now().isoformat()
            }
        
        # Calculate trading statistics
        total_profit = sum(trade.get("actual_profit", 0) for trade in completed_trades)
        avg_profit = total_profit / len(completed_trades) if completed_trades else 0
        
        profitable_trades = [t for t in completed_trades if t.get("actual_profit", 0) > 0]
        win_rate = len(profitable_trades) / len(completed_trades) if completed_trades else 0
        
        # Get exchange distribution
        exchange_pairs = {}
        for trade in completed_trades:
            pair = f"{trade['buy_exchange']}-{trade['sell_exchange']}"
            if pair not in exchange_pairs:
                exchange_pairs[pair] = 0
            exchange_pairs[pair] += 1
        
        # Reasons for rejected opportunities
        rejection_reasons = {}
        for opp in self.rejected_opportunities:
            reason = opp.get("rejection_reason", "Unknown")
            if reason not in rejection_reasons:
                rejection_reasons[reason] = 0
            rejection_reasons[reason] += 1
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_completed_trades": len(completed_trades),
            "total_profit": float(total_profit),
            "average_profit": float(avg_profit),
            "win_rate": float(win_rate),
            "active_trades": len([t for t in self.trades if t["status"] in ["executed", "partial"]]),
            "rejected_opportunities": len(self.rejected_opportunities),
            "exchange_distribution": exchange_pairs,
            "rejection_reasons": rejection_reasons,
            "trade_history": self.trades[-10:],  # Last 10 trades
        }
        
        # Save report
        report_path = self.results_dir / f"trading_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Trading report generated and saved to {report_path}")
        return report
    
    def run_trading_cycle(self, market_data=None):
        """
        Run a complete trading cycle:
        1. Fetch market data (using multi-threading if enabled)
        2. Identify opportunities
        3. Assess risks
        4. Optimize parameters
        5. Execute trades
        6. Monitor existing trades
        7. Generate reports
        
        Args:
            market_data: Optional market data. If None and use_parallel_data_fetching is True,
                         data will be fetched in parallel using MarketDataService
        """
        logger.info("Starting trading cycle")
        
        # Check and rebalance funds across exchanges if enabled
        if self.use_fund_prepositioning:
            self._check_fund_positioning()
        
        # Step 1: Fetch market data if not provided
        start_time = time.time()
        if market_data is None and self.use_parallel_data_fetching:
            logger.info("Fetching market data using multi-threaded service")
            market_data = self.market_data_service.fetch_prices()
            logger.info(f"Fetched market data in {time.time() - start_time:.3f}s")
        
        # Step 2: Identify opportunities
        opportunities = self.identify_arbitrage_opportunities(market_data)
        
        # Step 3: Filter by risk assessment
        viable_opportunities = self.filter_opportunities_by_risk(opportunities)
        
        # Step 4: Optimize parameters
        optimized_opportunities = self.optimize_trade_parameters(viable_opportunities)
        
        # Step 5: Execute trades
        executed_trades = self.execute_trades(optimized_opportunities)
        
        # Step 6: Monitor existing trades
        completed_trades, active_trades = self.monitor_trades()
        
        # Step 7: Generate report
        report = self.generate_trading_report()
        
        cycle_summary = {
            "timestamp": datetime.now().isoformat(),
            "opportunities_found": len(opportunities),
            "opportunities_viable": len(viable_opportunities),
            "trades_executed": len(executed_trades),
            "trades_completed": len(completed_trades),
            "trades_active": len(active_trades),
            "report": report
        }
        
        logger.info(f"Trading cycle completed: {len(executed_trades)} new trades executed")
        return cycle_summary
        
    def _is_bundleable(self, opportunity):
        """Check if an opportunity is compatible with MEV bundling"""
        # Conditions for bundling:
        # 1. Must be on-chain (not CEX-to-CEX)
        # 2. Must have network/chain information
        # 3. Must be compatible with smart contract execution
        
        opportunity_type = opportunity.get("type")
        execution_venue = opportunity.get("execution_venue", "onchain")  # Default to onchain
        has_network = "network" in opportunity
        
        # Triangle arbitrage within a single DEX is always bundleable
        if opportunity_type == "triangle" and execution_venue == "onchain":
            return True
            
        # Direct arbitrage between DEXes on the same chain is bundleable
        if opportunity_type == "direct" and execution_venue == "onchain":
            buy_venue_type = opportunity.get("buy_venue_type", "unknown")
            sell_venue_type = opportunity.get("sell_venue_type", "unknown")
            return buy_venue_type == "dex" and sell_venue_type == "dex" and has_network
            
        return False
        
    def _convert_to_bundle_format(self, opportunity):
        """Convert internal opportunity format to MEV bundle format"""
        opportunity_type = opportunity.get("type")
        
        if opportunity_type == "direct":
            return {
                "type": "direct_arbitrage",
                "buy_exchange": opportunity.get("buy_exchange"),
                "sell_exchange": opportunity.get("sell_exchange"),
                "symbol": opportunity.get("symbol"),
                "buy_price": opportunity.get("buy_price"),
                "sell_price": opportunity.get("sell_price"),
                "amount": opportunity.get("optimized_amount"),
                "slippage": opportunity.get("slippage", 0.005),
                "buy_path": opportunity.get("buy_path", []),
                "sell_path": opportunity.get("sell_path", []),
                "network": opportunity.get("network", "ethereum"),
                "buy_router_address": opportunity.get("buy_router_address"),
                "sell_router_address": opportunity.get("sell_router_address")
            }
        elif opportunity_type == "triangle":
            return {
                "type": "triangle_arbitrage",
                "exchange": opportunity.get("exchange"),
                "trading_path": opportunity.get("trading_path", []),
                "prices": opportunity.get("prices", []),
                "amount": opportunity.get("optimized_amount"),
                "expected_profit": opportunity.get("expected_profit"),
                "slippage": opportunity.get("slippage", 0.005),
                "network": opportunity.get("network", "ethereum"),
                "router_address": opportunity.get("router_address")
            }
        else:
            return opportunity
    
    def _create_trade_record(self, opportunity, status, bundle_id=None):
        """Create a trade record based on opportunity and execution status"""
        opportunity_type = opportunity.get("type", "unknown")
        
        trade = {
            "timestamp": datetime.now().isoformat(),
            "trade_id": f"trade_{len(self.trades) + 1}",
            "type": opportunity_type,
            "status": status,
            "execution_strategy": opportunity.get("execution_strategy", "standard"),
            "expected_profit": opportunity.get("expected_profit"),
            "network": opportunity.get("network", "ethereum")
        }
        
        # Add flashloan information if used
        # This is important for AI training to learn optimal capital leverage strategies
        if opportunity.get("use_flashloan", False):
            trade["used_flashloan"] = True
            trade["flashloan_amount"] = opportunity.get("flashloan_amount", 0)
            trade["flashloan_provider"] = opportunity.get("flashloan_provider", "")
            trade["flashloan_fee"] = opportunity.get("flashloan_fee", 0)
            trade["original_amount"] = opportunity.get("original_amount", 0)
            # Capital efficiency metric for AI training
            if opportunity.get("original_amount", 0) > 0:
                trade["capital_efficiency"] = opportunity.get("flashloan_amount", 0) / opportunity.get("original_amount", 1)
            else:
                trade["capital_efficiency"] = 0
        else:
            trade["used_flashloan"] = False
        
        # Add bundle info if applicable
        if bundle_id:
            trade["bundle_id"] = bundle_id
            trade["execution_method"] = "mev_bundle"
        else:
            trade["execution_method"] = "individual"
            
        # Add type-specific fields
        if opportunity_type == "direct":
            trade.update({
                "symbol": opportunity.get("symbol"),
                "buy_exchange": opportunity.get("buy_exchange"),
                "sell_exchange": opportunity.get("sell_exchange"),
                "amount": opportunity.get("optimized_amount"),
                "buy_price": opportunity.get("buy_price"),
                "sell_price": opportunity.get("sell_price")
            })
        elif opportunity_type == "triangle":
            trade.update({
                "exchange": opportunity.get("exchange"),
                "trading_path": opportunity.get("trading_path"),
                "amount": opportunity.get("optimized_amount"),
                "prices": opportunity.get("prices")
            })
            
        return trade
        
    def _record_trade_data_for_training(self, executed_trades):
        """Record trade data for AI training purposes"""
        if not executed_trades:
            return
            
        # Save trade data to a file for AI training
        training_dir = Path("data/training")
        if not training_dir.exists():
            training_dir.mkdir(parents=True)
            
        # Format: date_trades.json
        date_str = datetime.now().strftime("%Y%m%d")
        trades_file = training_dir / f"{date_str}_trades.json"
        
        # Append to existing file or create new one
        existing_trades = []
        if trades_file.exists():
            with open(trades_file, 'r') as f:
                try:
                    existing_trades = json.load(f)
                except json.JSONDecodeError:
                    logger.warning(f"Error reading existing trades file: {trades_file}")
        
        # Add market conditions to each trade for better ML context
        enriched_trades = []
        for trade in executed_trades:
            trade_copy = trade.copy()
            # Add market conditions at time of execution for ML training
            try:
                trade_copy["market_conditions"] = {
                    "timestamp": datetime.now().isoformat(),
                    "volatility": self.ta_engine.get_volatility(trade.get("symbol", "BTC/USDT")),
                    "trend": self.ta_engine.get_trend_strength(trade.get("symbol", "BTC/USDT")),
                    "market_volume": self.ta_engine.get_market_volume(trade.get("symbol", "BTC/USDT"))
                }
            except AttributeError:
                # If technical_analysis is not available, add basic info
                trade_copy["market_conditions"] = {
                    "timestamp": datetime.now().isoformat()
                }
            enriched_trades.append(trade_copy)
        
        # Combine and save
        all_trades = existing_trades + enriched_trades
        with open(trades_file, 'w') as f:
            json.dump(all_trades, f, indent=2)
            
        logger.info(f"Recorded {len(enriched_trades)} trades for AI training")


def simulate_market_data():
    """
    Generate simulated market data for testing
    In a real system, this would come from exchange APIs
    """

    exchanges = ["binance", "coinbase", "kraken"]
    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    
    market_data = {}
    
    # Base prices
    base_prices = {
        "BTC/USDT": 40000,
        "ETH/USDT": 2000,
        "SOL/USDT": 100
    }
    
    # Generate data for each exchange
    for exchange in exchanges:
        market_data[exchange] = {}
        
        for symbol in symbols:
            # Add random variation to price
            base_price = base_prices[symbol]
            price_variation = np.random.normal(0, base_price * 0.005)  # 0.5% standard deviation
            price = base_price + price_variation
            
            # Add volume information
            volume = np.random.lognormal(mean=10, sigma=1)  # Random volume
            
            # Add to market data
            market_data[exchange][symbol] = {
                "price": price,
                "volume": volume,
                "timestamp": datetime.now().isoformat()
            }
    
    # Introduce a specific arbitrage opportunity
    # Make BTC cheaper on binance and more expensive on coinbase
    market_data["binance"]["BTC/USDT"]["price"] = base_prices["BTC/USDT"] * 0.99  # 1% cheaper
    market_data["coinbase"]["BTC/USDT"]["price"] = base_prices["BTC/USDT"] * 1.01  # 1% more expensive
    
    return market_data


# Example usage
if __name__ == "__main__":
    # Create risk-aware trader
    trader = RiskAwareQuantumTrader()
    
    # Get simulated market data
    market_data = simulate_market_data()
    
    # Check fund positioning status
    fund_status = trader.fund_manager.get_balance_status()
    print("\n=== FUND POSITIONING STATUS ===")
    print(f"Auto-positioning: {'Enabled' if trader.use_fund_prepositioning else 'Disabled'}")
    
    # Run a trading cycle
    results = trader.run_trading_cycle(market_data)
    
    # Print summary
    print("\n=== TRADING CYCLE SUMMARY ===")
    print(f"Opportunities Found: {results['opportunities_found']}")
    print(f"Viable After Risk Assessment: {results['opportunities_viable']}")
    print(f"Trades Executed: {results['trades_executed']}")
    print(f"Active Trades: {results['trades_active']}")
    
    # Print profit if any trades completed
    report = results["report"]
    if isinstance(report, dict) and "status" not in report:
        print(f"\nWin Rate: {report['win_rate']:.2%}")
        print(f"Total Profit: {report['total_profit']:.2%}")
        
        if "rejection_reasons" in report:
            print("\nRejection Reasons:")
            for reason, count in report["rejection_reasons"].items():
                print(f"- {reason}: {count}")
    else:
        print("\nNo completed trades yet")
