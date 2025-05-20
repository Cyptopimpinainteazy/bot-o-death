#!/usr/bin/env python
"""
Quantum Trading System Optimizer
--------------------------------
Optimizes the performance of the quantum trading system by integrating
the ensemble model, feature optimization, and hyperparameter tuning.
"""

import os
import numpy as np
import pandas as pd
import json
import logging
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import pickle
import time

# Local imports
from technical_analysis import TechnicalAnalysisEngine
from quantum_ensemble import QuantumEnsembleTrader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("quantum_optimizer.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("QuantumOptimizer")

class QuantumTradingOptimizer:
    """
    Master optimizer for quantum trading system - integrates all optimization components
    to create a unified, high-performance trading system.
    """
    
    def __init__(self):
        """Initialize the quantum trading optimizer"""
        # Paths
        self.results_dir = Path("results")
        self.models_dir = Path("models")
        self.config_dir = Path("config")
        self.optimizer_dir = self.results_dir / "optimizer"
        
        # Create directories
        os.makedirs(self.optimizer_dir, exist_ok=True)
        os.makedirs(self.config_dir, exist_ok=True)
        
        # Load components
        self.ta_engine = TechnicalAnalysisEngine()
        self.ensemble = QuantumEnsembleTrader()
        
        # Load optimization settings
        self.settings = self._load_settings()
        
        # Initialize metrics tracking
        self.metrics = {
            "win_rate": [],
            "profit": [],
            "accuracy": [],
            "drawdown": [],
            "sharpe_ratio": [],
            "timestamp": []
        }
        
        logger.info("Initialized Quantum Trading Optimizer")
    
    def _load_settings(self):
        """Load optimizer settings"""
        settings_path = self.config_dir / "optimizer_settings.json"
        
        if settings_path.exists():
            try:
                with open(settings_path, 'r') as f:
                    settings = json.load(f)
                logger.info(f"Loaded optimizer settings from {settings_path}")
                return settings
            except Exception as e:
                logger.error(f"Error loading settings: {str(e)}")
        
        # Default settings
        settings = {
            "backtesting": {
                "initial_capital": 10000,
                "position_size": 0.1,  # 10% of capital per trade
                "max_open_positions": 3,
                "stop_loss": 0.03,     # 3% stop loss
                "take_profit": 0.05,   # 5% take profit
            },
            "optimization": {
                "feature_weight_decay": 0.95,  # How much to discount feature importance over time
                "learning_rate": 0.01,
                "batch_size": 32,
                "epochs": 10,
                "validation_split": 0.2
            },
            "ensemble": {
                "model_weights": {
                    "rf": 1.0,
                    "xgb": 1.0,
                    "quantum": 1.5  # Higher weight for quantum model
                },
                "confidence_threshold": 0.65  # Minimum confidence to execute trade
            },
            "risk_management": {
                "max_daily_drawdown": 0.05,  # 5% max daily drawdown
                "max_drawdown": 0.15,        # 15% max overall drawdown
                "max_leverage": 2.0,         # Maximum leverage
                "volatility_scaling": True,  # Scale position size by volatility
            }
        }
        
        # Save default settings
        with open(settings_path, 'w') as f:
            json.dump(settings, f, indent=2)
        
        logger.info(f"Created default optimizer settings at {settings_path}")
        return settings
    
    def load_feature_weights(self):
        """Load optimized feature weights"""
        weights_path = self.results_dir / "feature_analysis" / "quantum_recommendations.json"
        
        if weights_path.exists():
            try:
                with open(weights_path, 'r') as f:
                    data = json.load(f)
                weights = data.get("feature_weights", {})
                logger.info(f"Loaded feature weights from {weights_path}")
                return weights
            except Exception as e:
                logger.error(f"Error loading feature weights: {str(e)}")
        
        logger.warning("No optimized feature weights found")
        return None
    
    def load_quantum_params(self):
        """Load optimized quantum parameters"""
        params_path = self.results_dir / "quantum_tuning" / "best_quantum_params.json"
        
        if params_path.exists():
            try:
                with open(params_path, 'r') as f:
                    data = json.load(f)
                params = data.get("best_params", {})
                logger.info(f"Loaded quantum parameters from {params_path}")
                return params
            except Exception as e:
                logger.error(f"Error loading quantum parameters: {str(e)}")
        
        logger.warning("No optimized quantum parameters found")
        return None
    
    def integrate_optimizations(self):
        """Integrate all optimization components"""
        logger.info("Integrating optimization components")
        
        # Load optimized components
        feature_weights = self.load_feature_weights()
        quantum_params = self.load_quantum_params()
        
        # Create integrated configuration
        config = {
            "version": "1.0",
            "timestamp": datetime.now().isoformat(),
            "feature_weights": feature_weights,
            "quantum_params": quantum_params,
            "ensemble_settings": self.settings["ensemble"],
            "risk_management": self.settings["risk_management"],
            "backtesting": self.settings["backtesting"]
        }
        
        # Save integrated configuration
        config_path = self.config_dir / "integrated_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Created integrated configuration at {config_path}")
        return config
    
    def optimize_trading_parameters(self, market_data=None):
        """Optimize trading parameters based on historical data"""
        logger.info("Optimizing trading parameters")
        
        # Load market data if not provided
        if market_data is None:
            data_path = self.results_dir / "training_data.csv"
            try:
                market_data = pd.read_csv(data_path)
                logger.info(f"Loaded market data from {data_path}")
            except Exception as e:
                logger.error(f"Error loading market data: {str(e)}")
                return None
        
        # Optimize position sizing
        self._optimize_position_sizing(market_data)
        
        # Optimize stop loss and take profit levels
        self._optimize_stop_loss_take_profit(market_data)
        
        # Optimize confidence thresholds
        self._optimize_confidence_threshold(market_data)
        
        logger.info("Trading parameters optimization complete")
        return self.settings
    
    def _optimize_position_sizing(self, market_data):
        """Optimize position sizing based on volatility and expected return"""
        logger.info("Optimizing position sizing")
        
        # Calculate historical volatility
        if 'price' in market_data.columns:
            returns = market_data['price'].pct_change().dropna()
            volatility = returns.std()
            
            # Adjust position size based on volatility
            base_position_size = self.settings["backtesting"]["position_size"]
            
            # Higher volatility = smaller position size
            if volatility > 0.03:  # High volatility
                new_position_size = base_position_size * 0.7
            elif volatility < 0.01:  # Low volatility
                new_position_size = base_position_size * 1.3
            else:
                new_position_size = base_position_size
            
            # Cap at reasonable limits
            new_position_size = min(0.25, max(0.05, new_position_size))
            
            self.settings["backtesting"]["position_size"] = float(new_position_size)
            logger.info(f"Adjusted position size to {new_position_size:.2f} based on volatility {volatility:.4f}")
    
    def _optimize_stop_loss_take_profit(self, market_data):
        """Optimize stop loss and take profit levels based on market volatility"""
        logger.info("Optimizing stop loss and take profit levels")
        
        if 'price' in market_data.columns:
            returns = market_data['price'].pct_change().dropna()
            daily_range = abs(returns).mean() * 3  # Average daily range, multiplied for safety
            
            # Set stop loss based on daily range
            new_stop_loss = max(0.02, min(0.05, float(daily_range)))
            
            # Set take profit at 1.5-2x stop loss for positive expectancy
            new_take_profit = float(new_stop_loss * 1.8)
            
            self.settings["backtesting"]["stop_loss"] = new_stop_loss
            self.settings["backtesting"]["take_profit"] = new_take_profit
            
            logger.info(f"Optimized stop loss: {new_stop_loss:.3f}, take profit: {new_take_profit:.3f}")
    
    def _optimize_confidence_threshold(self, market_data):
        """Optimize confidence threshold for trade execution"""
        # Run backtests with different confidence thresholds
        thresholds = [0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
        results = {}
        
        # Simple validation to select best threshold
        # For a production system, we would do cross-validation
        for threshold in thresholds:
            self.settings["ensemble"]["confidence_threshold"] = threshold
            profit = self._quick_backtest(market_data)
            results[threshold] = profit
        
        # Select best threshold
        best_threshold = max(results, key=results.get)
        self.settings["ensemble"]["confidence_threshold"] = best_threshold
        
        logger.info(f"Optimized confidence threshold: {best_threshold} with profit: {results[best_threshold]:.4f}")
    
    def _quick_backtest(self, market_data):
        """Run a quick backtest to evaluate a configuration"""
        # This is a simplified backtest
        # In production, we'd use a more sophisticated backtesting framework
        initial_capital = self.settings["backtesting"]["initial_capital"]
        position_size = self.settings["backtesting"]["position_size"]
        threshold = self.settings["ensemble"]["confidence_threshold"]
        
        capital = initial_capital
        position = None
        entry_price = 0
        
        for i in range(1, len(market_data)):
            prev_data = market_data.iloc[i-1].to_dict()
            current_data = market_data.iloc[i].to_dict()
            
            # Get recommendation from ensemble
            if self.ensemble.ensemble_model is None:
                # Make sure ensemble model is loaded/built
                self.ensemble.build_ensemble_model()
            
            action, confidence = self.ensemble.predict_trade_action(prev_data)
            
            # Current price
            if 'price' in current_data:
                current_price = current_data['price']
            else:
                continue  # Skip if no price data
            
            # Execute trades based on recommendations and threshold
            if position is None:  # No position
                if action == 'buy' and confidence >= threshold:
                    # Open long position
                    position = 'long'
                    entry_price = current_price
                    # Calculate position size based on capital and settings
                    size = capital * position_size / current_price
                elif action == 'sell' and confidence >= threshold:
                    # Open short position (simplified)
                    position = 'short'
                    entry_price = current_price
                    size = capital * position_size / current_price
            else:  # Have position
                if position == 'long':
                    # Close if sell signal or stop loss or take profit hit
                    pnl = (current_price - entry_price) / entry_price
                    if action == 'sell' or pnl <= -self.settings["backtesting"]["stop_loss"] or pnl >= self.settings["backtesting"]["take_profit"]:
                        # Close long position
                        profit = size * (current_price - entry_price)
                        capital += profit
                        position = None
                elif position == 'short':
                    # Close if buy signal or stop loss or take profit hit
                    pnl = (entry_price - current_price) / entry_price
                    if action == 'buy' or pnl <= -self.settings["backtesting"]["stop_loss"] or pnl >= self.settings["backtesting"]["take_profit"]:
                        # Close short position
                        profit = size * (entry_price - current_price)
                        capital += profit
                        position = None
        
        # Close any open position at the end
        if position is not None:
            if position == 'long':
                profit = size * (current_price - entry_price)
                capital += profit
            elif position == 'short':
                profit = size * (entry_price - current_price)
                capital += profit
        
        # Calculate profit percentage
        profit_pct = (capital - initial_capital) / initial_capital
        return profit_pct
    
    def evaluate_model(self, market_data=None):
        """Run a comprehensive evaluation of the trading model"""
        logger.info("Running comprehensive model evaluation")
        
        # Load market data if not provided
        if market_data is None:
            data_path = self.results_dir / "training_data.csv"
            try:
                market_data = pd.read_csv(data_path)
                logger.info(f"Loaded market data from {data_path}")
            except Exception as e:
                logger.error(f"Error loading market data: {str(e)}")
                return None
        
        # Ensure we have the ensemble model
        if self.ensemble.ensemble_model is None:
            self.ensemble.build_ensemble_model()
        
        # Split data for out-of-sample testing
        split_idx = int(len(market_data) * 0.8)
        train_data = market_data.iloc[:split_idx]
        test_data = market_data.iloc[split_idx:]
        
        # Initialize metrics
        trades = []
        equity_curve = [self.settings["backtesting"]["initial_capital"]]
        positions = []
        current_position = None
        entry_price = 0
        entry_time = None
        capital = self.settings["backtesting"]["initial_capital"]
        
        # Run simulation
        for i in range(1, len(test_data)):
            prev_data = test_data.iloc[i-1].to_dict()
            current_data = test_data.iloc[i].to_dict()
            
            # Get recommendation
            action, confidence = self.ensemble.predict_trade_action(prev_data)
            
            # Current price and time
            if 'price' in current_data:
                current_price = current_data['price']
            else:
                continue
                
            if 'timestamp' in current_data:
                current_time = current_data['timestamp']
            else:
                current_time = i
            
            # Position management logic
            if current_position is None:  # No position
                if action == 'buy' and confidence >= self.settings["ensemble"]["confidence_threshold"]:
                    # Open long position
                    current_position = 'long'
                    entry_price = current_price
                    entry_time = current_time
                    # Position sizing
                    size = capital * self.settings["backtesting"]["position_size"] / current_price
                    positions.append({
                        'type': 'entry',
                        'position': 'long',
                        'price': current_price,
                        'time': current_time,
                        'confidence': confidence,
                        'size': size
                    })
                elif action == 'sell' and confidence >= self.settings["ensemble"]["confidence_threshold"]:
                    # Open short position
                    current_position = 'short'
                    entry_price = current_price
                    entry_time = current_time
                    # Position sizing
                    size = capital * self.settings["backtesting"]["position_size"] / current_price
                    positions.append({
                        'type': 'entry',
                        'position': 'short',
                        'price': current_price,
                        'time': current_time,
                        'confidence': confidence,
                        'size': size
                    })
            else:  # Have position
                if current_position == 'long':
                    # Calculate current P&L
                    pnl = (current_price - entry_price) / entry_price
                    
                    # Exit conditions
                    exit_reason = None
                    if pnl <= -self.settings["backtesting"]["stop_loss"]:
                        exit_reason = "stop_loss"
                    elif pnl >= self.settings["backtesting"]["take_profit"]:
                        exit_reason = "take_profit"
                    elif action == 'sell' and confidence >= self.settings["ensemble"]["confidence_threshold"]:
                        exit_reason = "signal"
                    
                    if exit_reason:
                        # Close long position
                        profit = size * (current_price - entry_price)
                        capital += profit
                        
                        # Record trade
                        trade = {
                            'entry_price': entry_price,
                            'exit_price': current_price,
                            'entry_time': entry_time,
                            'exit_time': current_time,
                            'position': current_position,
                            'profit': profit,
                            'profit_pct': pnl,
                            'exit_reason': exit_reason
                        }
                        trades.append(trade)
                        
                        # Record exit in positions
                        positions.append({
                            'type': 'exit',
                            'position': 'long',
                            'price': current_price,
                            'time': current_time,
                            'pnl': pnl,
                            'reason': exit_reason
                        })
                        
                        current_position = None
                
                elif current_position == 'short':
                    # Calculate current P&L
                    pnl = (entry_price - current_price) / entry_price
                    
                    # Exit conditions
                    exit_reason = None
                    if pnl <= -self.settings["backtesting"]["stop_loss"]:
                        exit_reason = "stop_loss"
                    elif pnl >= self.settings["backtesting"]["take_profit"]:
                        exit_reason = "take_profit"
                    elif action == 'buy' and confidence >= self.settings["ensemble"]["confidence_threshold"]:
                        exit_reason = "signal"
                    
                    if exit_reason:
                        # Close short position
                        profit = size * (entry_price - current_price)
                        capital += profit
                        
                        # Record trade
                        trade = {
                            'entry_price': entry_price,
                            'exit_price': current_price,
                            'entry_time': entry_time,
                            'exit_time': current_time,
                            'position': current_position,
                            'profit': profit,
                            'profit_pct': pnl,
                            'exit_reason': exit_reason
                        }
                        trades.append(trade)
                        
                        # Record exit in positions
                        positions.append({
                            'type': 'exit',
                            'position': 'short',
                            'price': current_price,
                            'time': current_time,
                            'pnl': pnl,
                            'reason': exit_reason
                        })
                        
                        current_position = None
            
            # Update equity curve
            equity_curve.append(capital)
        
        # Close any remaining position at the end
        if current_position is not None:
            if current_position == 'long':
                pnl = (current_price - entry_price) / entry_price
                profit = size * (current_price - entry_price)
                capital += profit
            elif current_position == 'short':
                pnl = (entry_price - current_price) / entry_price
                profit = size * (entry_price - current_price)
                capital += profit
            
            trade = {
                'entry_price': entry_price,
                'exit_price': current_price,
                'entry_time': entry_time,
                'exit_time': current_time,
                'position': current_position,
                'profit': profit,
                'profit_pct': pnl,
                'exit_reason': 'end_of_data'
            }
            trades.append(trade)
            
            positions.append({
                'type': 'exit',
                'position': current_position,
                'price': current_price,
                'time': current_time,
                'pnl': pnl,
                'reason': 'end_of_data'
            })
        
        # Calculate performance metrics
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t['profit'] > 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        profit_pct = (capital - self.settings["backtesting"]["initial_capital"]) / self.settings["backtesting"]["initial_capital"]
        
        # Calculate drawdown
        max_equity = 0
        max_drawdown = 0
        for equity in equity_curve:
            max_equity = max(max_equity, equity)
            drawdown = (max_equity - equity) / max_equity
            max_drawdown = max(max_drawdown, drawdown)
        
        # Calculate Sharpe ratio (simplified)
        returns = np.diff(equity_curve) / np.array(equity_curve[:-1])
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        
        # Record results
        results = {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "win_rate": float(win_rate),
            "profit_pct": float(profit_pct),
            "max_drawdown": float(max_drawdown),
            "sharpe_ratio": float(sharpe_ratio),
            "final_capital": float(capital),
            "trades": trades,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save results
        with open(self.optimizer_dir / "evaluation_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        # Generate visualization
        self._plot_results(equity_curve, positions, results)
        
        # Update metrics history
        self.metrics["win_rate"].append(win_rate)
        self.metrics["profit"].append(profit_pct)
        self.metrics["accuracy"].append(win_rate)  # Using win rate as proxy for accuracy
        self.metrics["drawdown"].append(max_drawdown)
        self.metrics["sharpe_ratio"].append(sharpe_ratio)
        self.metrics["timestamp"].append(datetime.now().isoformat())
        
        # Save metrics history
        with open(self.optimizer_dir / "metrics_history.json", 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        logger.info(f"Evaluation complete - Win Rate: {win_rate:.4f}, Profit: {profit_pct:.4f}, Sharpe: {sharpe_ratio:.4f}")
        return results
    
    def _plot_results(self, equity_curve, positions, results):
        """Create visualization of trading results"""
        try:
            # Create figure with multiple subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
            
            # Plot equity curve
            ax1.plot(equity_curve, label='Equity Curve', color='blue')
            ax1.set_title('Quantum Trading Optimization Results')
            ax1.set_ylabel('Capital')
            ax1.grid(True)
            
            # Add annotations for trade entries and exits
            for pos in positions:
                if pos['type'] == 'entry':
                    if pos['position'] == 'long':
                        ax1.scatter(pos['time'], equity_curve[int(pos['time']) if isinstance(pos['time'], (int, float)) else 0], 
                                 marker='^', color='green', s=100)
                    else:
                        ax1.scatter(pos['time'], equity_curve[int(pos['time']) if isinstance(pos['time'], (int, float)) else 0], 
                                 marker='v', color='red', s=100)
                elif pos['type'] == 'exit':
                    if pos['position'] == 'long':
                        ax1.scatter(pos['time'], equity_curve[int(pos['time']) if isinstance(pos['time'], (int, float)) else 0], 
                                 marker='o', color='orange', s=80)
                    else:
                        ax1.scatter(pos['time'], equity_curve[int(pos['time']) if isinstance(pos['time'], (int, float)) else 0], 
                                 marker='o', color='purple', s=80)
            
            # Plot trade P&L
            trade_pnls = [t['profit_pct'] for t in results['trades']]
            trade_times = [t['exit_time'] for t in results['trades']]
            
            colors = ['green' if pnl >= 0 else 'red' for pnl in trade_pnls]
            ax2.bar(range(len(trade_pnls)), trade_pnls, color=colors)
            ax2.set_title('Trade P&L (%)')
            ax2.set_xlabel('Trade Number')
            ax2.set_ylabel('P&L %')
            ax2.grid(True)
            
            # Add summary text
            summary = (
                f"Win Rate: {results['win_rate']:.2%}\n"
                f"Total Trades: {results['total_trades']}\n"
                f"Profit: {results['profit_pct']:.2%}\n"
                f"Max Drawdown: {results['max_drawdown']:.2%}\n"
                f"Sharpe Ratio: {results['sharpe_ratio']:.2f}"
            )
            
            ax1.text(0.02, 0.97, summary, transform=ax1.transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            
            # Save plot
            plot_path = self.optimizer_dir / f"trade_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(plot_path)
            
            logger.info(f"Results plot saved to {plot_path}")
            
            plt.close(fig)
        except Exception as e:
            logger.error(f"Error creating results plot: {str(e)}")
    
    def run_full_optimization(self):
        """Run the complete optimization process"""
        logger.info("Starting full quantum trading optimization process")
        
        # Integrate existing optimization components
        integrated_config = self.integrate_optimizations()
        
        # Optimize trading parameters
        optimized_settings = self.optimize_trading_parameters()
        
        # Run comprehensive evaluation
        evaluation_results = self.evaluate_model()
        
        # Generate summary report
        self._generate_optimization_report(integrated_config, optimized_settings, evaluation_results)
        
        logger.info("Full optimization process complete")
        return evaluation_results
    
    def _generate_optimization_report(self, config, settings, results):
        """Generate a comprehensive optimization report"""
        report = {
            "version": "1.0",
            "generated_at": datetime.now().isoformat(),
            "configuration": config,
            "optimized_settings": settings,
            "evaluation_results": results,
            "recommendations": self._generate_recommendations(results)
        }
        
        # Save report
        report_path = self.optimizer_dir / f"optimization_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Optimization report saved to {report_path}")
        return report
    
    def _generate_recommendations(self, results):
        """Generate trading recommendations based on evaluation results"""
        recommendations = []
        
        # Win rate recommendations
        if results["win_rate"] < 0.5:
            recommendations.append({
                "area": "Signal Quality",
                "issue": "Low win rate",
                "recommendation": "Consider increasing confidence threshold or revising feature weights"
            })
        elif results["win_rate"] > 0.8:
            recommendations.append({
                "area": "Position Sizing",
                "issue": "High win rate may allow more aggressive position sizing",
                "recommendation": "Consider increasing position size by 20-30%"
            })
        
        # Drawdown recommendations
        if results["max_drawdown"] > 0.2:
            recommendations.append({
                "area": "Risk Management",
                "issue": "High drawdown",
                "recommendation": "Tighten stop loss levels and reduce position sizing"
            })
        
        # Sharpe ratio recommendations
        if results["sharpe_ratio"] < 1.0:
            recommendations.append({
                "area": "Risk/Reward",
                "issue": "Low Sharpe ratio",
                "recommendation": "Review take profit levels and trade frequency"
            })
        
        # Profit recommendations
        if results["profit_pct"] < 0.05:
            recommendations.append({
                "area": "Profitability",
                "issue": "Low overall profit",
                "recommendation": "Consider more aggressive trade selection and reducing trading costs"
            })
        
        return recommendations

if __name__ == "__main__":
    print("=== Starting Quantum Trading System Optimizer ===")
    optimizer = QuantumTradingOptimizer()
    
    print("Running full optimization process...")
    results = optimizer.run_full_optimization()
    
    if results:
        print(f"\nOptimization Results Summary:")
        print(f"Win Rate: {results['win_rate']:.2%}")
        print(f"Total Profit: {results['profit_pct']:.2%}")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {results['max_drawdown']:.2%}")
        print(f"Total Trades: {results['total_trades']}")
    
    print("=== Optimization Complete ===")
