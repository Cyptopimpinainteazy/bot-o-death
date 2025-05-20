#!/usr/bin/env python3
"""
Maximum Profit Trading Bot Runner

Launches the trading bot in maximum profit mode with minimal restrictions.
"""

import os
import sys
import time
import json
import logging
import signal
import argparse
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)
base_dir = os.path.dirname(parent_dir)
sys.path.append(base_dir)

# Set up directory paths
logs_dir = os.path.join(parent_dir, 'logs')
data_dir = os.path.join(parent_dir, 'data')
models_dir = os.path.join(parent_dir, 'models')
reports_dir = os.path.join(parent_dir, 'reports')

# Ensure all required directories exist
os.makedirs(logs_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)
os.makedirs(reports_dir, exist_ok=True)

# Import necessary modules
from src.aggressive_trader import initialize_max_profit_trader
from ai_optimization.reinforcement_trainer import ReinforcementTrainer
from examples.ai_optimized_trading import generate_sample_opportunities

# Setup logging
log_file = os.path.join(logs_dir, f'trading_bot_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("MaxProfitBot")

def signal_handler(sig, frame):
    """Handle interrupt signals gracefully"""
    logger.info("🛑 Shutdown signal received. Closing bot gracefully...")
    # Save final performance metrics
    if 'trader' in globals():
        save_performance_metrics(trader)
    logger.info("💰 Bot shutdown complete. Final performance saved.")
    sys.exit(0)

def save_performance_metrics(trader):
    """Save performance metrics to disk"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_file = os.path.join(parent_dir, 'data', f'performance_metrics_{timestamp}.json')
    
    try:
        # Create a copy of performance history for saving
        metrics = trader.performance_history.copy()
        
        # Convert any non-serializable objects
        if 'trade_history' in metrics:
            for trade in metrics['trade_history']:
                for key, value in list(trade.items()):
                    if not isinstance(value, (str, int, float, bool, list, dict)) and value is not None:
                        trade[key] = str(value)
        
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Performance metrics saved to {metrics_file}")
    except Exception as e:
        logger.error(f"Failed to save performance metrics: {str(e)}")

def display_status_banner(trader, runtime):
    """Display a status banner with key performance metrics"""
    print("\n" + "="*80)
    print(f"🚀 MAX PROFIT TRADING BOT - RUNNING FOR {runtime}")
    print("="*80)
    print(f"💰 Total Profit: ${trader.performance_history['total_profit']:.2f}")
    print(f"📊 Trades: {trader.performance_history['total_trades']} total, " + 
          f"{trader.performance_history['successful_trades']} successful, " +
          f"{trader.performance_history['failed_trades']} failed")
    
    success_rate = 0
    if trader.performance_history['total_trades'] > 0:
        success_rate = trader.performance_history['successful_trades'] / trader.performance_history['total_trades'] * 100
    
    print(f"✅ Success Rate: {success_rate:.1f}%")
    
    if hasattr(trader, 'today_trade_count'):
        print(f"📈 Today's Trades: {trader.today_trade_count}/{trader.risk_settings['max_daily_trades']}")
    
    if hasattr(trader, 'circuit_breaker_active') and trader.circuit_breaker_active:
        print(f"⚠️ CIRCUIT BREAKER ACTIVE - Will auto-reset")
    
    print(f"🌎 Market Conditions: {trader.market_conditions.upper()}")
    print("="*80 + "\n")

def run_trading_bot(args):
    """Run the trading bot with specified settings"""
    global trader
    
    logger.info("🚀 Initializing MaxProfitTrader...")
    model_path = args.model_path
    trader = initialize_max_profit_trader(model_path)
    
    # Set initial capital
    trader.initial_capital = args.initial_capital
    trader.performance_history['peak_capital'] = args.initial_capital
    
    # Initialize trading day tracking
    trader.today_trade_count = 0
    trader.last_trade_day = datetime.now().date()
    
    # Trading loop setup
    start_time = datetime.now()
    opportunity_count = 0
    trade_count = 0
    
    # Status display interval
    status_interval = timedelta(minutes=args.status_interval)
    last_status = datetime.now()
    
    opportunity_batch_size = args.batch_size
    sleep_between_batches = args.sleep
    
    logger.info(f"Starting trading with {trader.initial_capital:.2f} initial capital")
    logger.info(f"Risk level: {trader.risk_level}, Max daily trades: {trader.risk_settings['max_daily_trades']}")
    
    try:
        while True:
            # Check if we should generate a status update
            now = datetime.now()
            if now - last_status >= status_interval:
                runtime = now - start_time
                hours, remainder = divmod(runtime.seconds, 3600)
                minutes, seconds = divmod(remainder, 60)
                runtime_str = f"{runtime.days}d {hours}h {minutes}m {seconds}s"
                display_status_banner(trader, runtime_str)
                last_status = now
            
            # Generate a batch of trading opportunities
            opportunities = generate_sample_opportunities(count=opportunity_batch_size)
            opportunity_count += len(opportunities)
            
            # Process the opportunities
            for opportunity in opportunities:
                # Add metadata to opportunity for better tracking
                opportunity['timestamp'] = datetime.now().isoformat()
                opportunity['opportunity_id'] = f"opp_{opportunity_count}"
                
                # Enhance the opportunity with AI optimization
                enhanced = trader.enhance_opportunity(opportunity)
                
                if enhanced:
                    # Execute the trade with the chosen strategy
                    if enhanced.get('execution_strategy') == 'flashloan':
                        # Simulate flashloan execution
                        profit = enhanced.get('flashloan_profit', 0)
                        success = profit > 0
                        
                        if success:
                            trader.performance_history['flashloan_trades'] += 1
                    else:
                        # Simulate standard execution
                        profit = enhanced.get('standard_profit', 0)
                        success = profit > 0
                        
                        if success:
                            trader.performance_history['standard_trades'] += 1
                    
                    # Update trade counts and performance
                    trade_count += 1
                    trader.performance_history['total_trades'] += 1
                    trader.today_trade_count += 1
                    
                    if success:
                        trader.performance_history['successful_trades'] += 1
                        trader.performance_history['total_profit'] += profit
                        logger.info(f"Trade {trade_count} executed successfully: ${profit:.2f} profit")
                    else:
                        trader.performance_history['failed_trades'] += 1
                        logger.warning(f"Trade {trade_count} failed to yield profit")
                    
                    # Update peak capital if needed
                    current_capital = trader.initial_capital + trader.performance_history['total_profit']
                    if current_capital > trader.performance_history['peak_capital']:
                        trader.performance_history['peak_capital'] = current_capital
                    
                    # Track trade in history
                    trader.performance_history['trade_history'].append({
                        'trade_id': trade_count,
                        'timestamp': datetime.now().isoformat(),
                        'asset': enhanced.get('asset', 'unknown'),
                        'execution_strategy': enhanced.get('execution_strategy', 'standard'),
                        'profit': profit,
                        'success': success
                    })
                    
                    # Save metrics periodically
                    if trade_count % 50 == 0:
                        save_performance_metrics(trader)
            
            # Sleep between batches to avoid overwhelming the system
            time.sleep(sleep_between_batches)
    
    except KeyboardInterrupt:
        logger.info("Manual interruption received. Shutting down...")
    except Exception as e:
        logger.error(f"Error in trading loop: {str(e)}", exc_info=True)
    finally:
        # Save final performance data
        save_performance_metrics(trader)
        logger.info("Trading bot has been shut down")

if __name__ == "__main__":
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run the max profit trading bot")
    parser.add_argument("--model-path", type=str, default=None, help="Path to a pre-trained model")
    parser.add_argument("--initial-capital", type=float, default=10000.0, help="Initial capital amount")
    parser.add_argument("--batch-size", type=int, default=10, help="Number of opportunities to generate in each batch")
    parser.add_argument("--sleep", type=float, default=5.0, help="Sleep time between batches in seconds")
    parser.add_argument("--status-interval", type=int, default=5, help="Status display interval in minutes")
    
    args = parser.parse_args()
    
    # Run the bot
    run_trading_bot(args)
