#!/usr/bin/env python3
"""
24-Hour Trading Bot Simulation Test
This script runs a full 24-hour simulated trading period with the enhanced risk-aware
trading bot to evaluate performance in various market conditions.
"""

import os
import sys
import json
import time
import logging
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_simulation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import trading modules
from ai_optimization.trader_optimization import AIOptimizedTrader
from ai_optimization.reinforcement_trainer import ReinforcementTrainer

def generate_realistic_market_conditions(hours=24, interval_mins=5):
    """
    Generate realistic market conditions for a 24-hour period
    with varying volatility, gas prices, and risk levels
    """
    # Calculate number of intervals
    intervals = int((hours * 60) // interval_mins)
    
    # Time series for the simulation
    start_time = datetime.now() - timedelta(hours=hours)
    timestamps = [start_time + timedelta(minutes=i*interval_mins) for i in range(intervals)]
    
    # Base market conditions
    base_volatility = 0.02  # 2% baseline volatility
    base_gas_price = 50     # 50 gwei baseline
    
    # Generate market condition patterns
    conditions = []
    
    for i in range(intervals):
        # Time of day effects (higher volatility during market open/close)
        hour = timestamps[i].hour
        time_factor = 1.0
        
        # More volatility during market hours (8am-4pm) and especially at open/close
        if 8 <= hour < 16:
            time_factor = 1.2
            if hour in [8, 9, 15]:  # Market open/close hours
                time_factor = 1.5
        
        # Weekend effect (less activity)
        if timestamps[i].weekday() >= 5:  # Saturday or Sunday
            time_factor *= 0.7
        
        # Random volatility with occasional spikes
        volatility_multiplier = np.random.lognormal(0, 0.3)  # Random factor, sometimes spikes
        if np.random.random() < 0.05:  # 5% chance of volatility spike
            volatility_multiplier *= 3
            
        volatility = base_volatility * time_factor * volatility_multiplier
        
        # Gas price with gradual changes and occasional spikes
        if i > 0:
            # Gas tends to change gradually
            gas_change = np.random.normal(0, 5)
            if np.random.random() < 0.03:  # 3% chance of gas price spike
                gas_change = np.random.normal(50, 20)
                
            gas_price = max(10, min(300, conditions[-1]['gas_price_gwei'] + gas_change))
        else:
            gas_price = base_gas_price
            
        # Market conditions for this interval
        condition = {
            'timestamp': timestamps[i].strftime('%Y-%m-%d %H:%M:%S'),
            'volatility': volatility,
            'gas_price_gwei': gas_price,
            'trading_volume': np.random.lognormal(0, 0.4) * 100000,  # Random trading volume
            'market_trend': np.random.normal(0, 0.01),  # Slight market drift
            'risk_level': min(1.0, max(0.1, np.random.normal(0.5, 0.15)))  # Risk level between 0.1-1.0
        }
        
        conditions.append(condition)
    
    return conditions

def generate_trading_opportunities(market_conditions, opportunity_count=400):
    """
    Generate trading opportunities based on market conditions
    """
    opportunities = []
    
    # Asset options for diversification
    assets = ['ETH/USDT', 'BTC/USDT', 'MATIC/USDT', 'SOL/USDT', 'AVAX/USDT', 
              'LINK/USDT', 'UNI/USDT', 'AAVE/USDT', 'MKR/USDT', 'CRV/USDT']
    
    # For each market condition period, generate some opportunities
    for i, condition in enumerate(market_conditions):
        # Number of opportunities in this period (more during volatile periods)
        volatility = condition['volatility']
        period_opportunity_count = np.random.poisson(opportunity_count / len(market_conditions) * (1 + volatility*5))
        
        # Gas price from market conditions
        gas_price = condition['gas_price_gwei']
        
        for j in range(period_opportunity_count):
            # Select random assets for this opportunity
            asset = np.random.choice(assets)
            
            # Base values
            base_amount = np.random.lognormal(8, 1)  # Random position size
            base_profit = np.random.lognormal(-3, 1) * (1 + volatility*10)  # Higher profit in volatile markets
            
            # Risk is influenced by volatility and specific opportunity factors
            risk_score = min(0.95, max(0.1, condition['risk_level'] * np.random.normal(1, 0.2)))
            
            # Gas costs fluctuate with gas price
            standard_gas_cost = 0.001 + 0.00001 * gas_price
            flashloan_gas_cost = 0.003 + 0.00003 * gas_price
            
            # Calculate raw profits
            standard_raw_profit = base_profit
            
            # Flashloan multiplier (between 3-10x)
            flashloan_multiplier = np.random.uniform(3, 10)
            
            # Market impact factor reduces profit as trade size increases
            market_impact = 0.1 * (base_amount / 1000)
            market_impact = min(0.8, market_impact)  # Cap at 80% reduction
            
            # Apply market impact to flashloan profit
            flashloan_raw_profit = standard_raw_profit * flashloan_multiplier * (1 - market_impact)
            
            # Flashloan fee (typically 0.09% of the borrowed amount)
            flashloan_fee_rate = 0.0009
            flashloan_fee_amount = flashloan_raw_profit * flashloan_fee_rate
            
            # Final profits
            standard_profit = standard_raw_profit - standard_gas_cost
            flashloan_profit = flashloan_raw_profit - flashloan_fee_amount - flashloan_gas_cost
            
            # Create the opportunity
            opportunity = {
                'id': f"opp_{i}_{j}",
                'timestamp': condition['timestamp'],
                'asset': asset,
                'amount': base_amount,
                'position_size': base_amount * np.random.uniform(100, 1000),
                'risk_score': risk_score,
                'volatility': condition['volatility'],
                'gas_price_gwei': gas_price,
                'market_trend': condition['market_trend'],
                'standard_gas_cost': standard_gas_cost,
                'flashloan_gas_cost': flashloan_gas_cost,
                'standard_raw_profit': standard_raw_profit,
                'flashloan_raw_profit': flashloan_raw_profit,
                'flashloan_fee_amount': flashloan_fee_amount,
                'standard_profit': standard_profit,
                'flashloan_profit': flashloan_profit,
                'expected_profit': max(standard_profit, flashloan_profit),
                'assets': [asset.split('/')[0], asset.split('/')[1]]
            }
            
            opportunities.append(opportunity)
    
    return opportunities

def run_simulation(model_path=None, risk_level='moderate', hours=24, seed=None):
    """
    Run a full trading simulation with realistic market conditions
    """
    if seed is not None:
        np.random.seed(seed)
        
    logging.info(f"Starting {hours}-hour trading simulation with risk level: {risk_level}")
    
    # Generate market conditions
    logging.info("Generating realistic market conditions...")
    market_conditions = generate_realistic_market_conditions(hours=hours)
    
    # Generate trading opportunities
    logging.info("Generating trading opportunities...")
    opportunities = generate_trading_opportunities(market_conditions)
    logging.info(f"Generated {len(opportunities)} potential trading opportunities")
    
    # Initialize AI trader
    logging.info(f"Initializing AI trader with risk level: {risk_level}")
    trader = AIOptimizedTrader(input_features=10, model_path=model_path, risk_level=risk_level)
    trader.initial_capital = 10000.0  # Starting capital
    trader.performance_history['peak_capital'] = trader.initial_capital
    
    # Track simulation results
    simulation_results = {
        'market_conditions': market_conditions,
        'opportunities': len(opportunities),
        'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'trades': [],
        'capital_history': [trader.initial_capital],
        'timestamps': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        'hourly_profits': [0] * hours
    }
    
    # Process opportunities in time order
    sorted_opportunities = sorted(opportunities, key=lambda x: x['timestamp'])
    
    # Group by hour for reporting
    hour_profits = {}
    
    # Track current capital
    current_capital = trader.initial_capital
    trader.today_trade_count = 0
    
    # Process each opportunity
    for i, opportunity in enumerate(sorted_opportunities):
        # Reset daily trade count at the start of a new day
        opp_time = datetime.strptime(opportunity['timestamp'], '%Y-%m-%d %H:%M:%S')
        hour_key = opp_time.strftime('%Y-%m-%d %H')
        
        if hour_key not in hour_profits:
            hour_profits[hour_key] = 0
            
        # Simulate daily trade count limit
        if opp_time.hour == 0 and opp_time.minute == 0:
            trader.today_trade_count = 0
        
        # Track trade count
        trader.today_trade_count += 1
        
        # Optimize the opportunity with risk management
        enhanced = trader.enhance_opportunity(opportunity)
        
        # Skip rejected opportunities
        if enhanced is None:
            continue
            
        # Simulate trade execution
        strategy = enhanced['ai_recommended_strategy']
        confidence = enhanced['ai_confidence']
        
        if strategy == 'standard':
            actual_profit = opportunity['standard_profit']
        else:  # flashloan
            actual_profit = opportunity['flashloan_profit']
        
        # Update capital and track in history
        current_capital += actual_profit
        simulation_results['capital_history'].append(current_capital)
        simulation_results['timestamps'].append(opportunity['timestamp'])
        
        # Update hourly profits
        hour_idx = opp_time.hour
        simulation_results['hourly_profits'][hour_idx] += actual_profit
        hour_profits[hour_key] += actual_profit
        
        # Update trader's performance history
        trader.performance_history['total_profit'] += actual_profit
        if current_capital > trader.performance_history['peak_capital']:
            trader.performance_history['peak_capital'] = current_capital
            
        # Record trade details
        trade_record = {
            'id': opportunity['id'],
            'timestamp': opportunity['timestamp'],
            'asset': opportunity['asset'],
            'strategy': strategy,
            'confidence': confidence,
            'profit': actual_profit,
            'gas_price': opportunity['gas_price_gwei'],
            'risk_score': opportunity['risk_score']
        }
        
        simulation_results['trades'].append(trade_record)
        
        # Add to trader's trade history
        trader.performance_history['trade_history'].append({
            'timestamp': opportunity['timestamp'],
            'strategy': strategy,
            'profit': actual_profit
        })
        
        # Log progress occasionally
        if i % 50 == 0:
            logging.info(f"Processed {i}/{len(sorted_opportunities)} opportunities, " +
                        f"current capital: ${current_capital:.2f}")
    
    # Calculate final results
    final_capital = current_capital
    total_profit = final_capital - trader.initial_capital
    profit_percentage = (total_profit / trader.initial_capital) * 100
    executed_trades = len(simulation_results['trades'])
    
    simulation_results['final_capital'] = final_capital
    simulation_results['total_profit'] = total_profit
    simulation_results['profit_percentage'] = profit_percentage
    simulation_results['executed_trades'] = executed_trades
    simulation_results['opportunity_capture_rate'] = executed_trades / len(opportunities) if opportunities else 0
    
    # Log hourly profits
    logging.info("Hourly profit breakdown:")
    for hour, profit in hour_profits.items():
        logging.info(f"{hour}: ${profit:.2f}")
    
    # Log final results
    logging.info(f"Simulation completed - Results:")
    logging.info(f"Starting capital: ${trader.initial_capital:.2f}")
    logging.info(f"Final capital: ${final_capital:.2f}")
    logging.info(f"Total profit: ${total_profit:.2f} ({profit_percentage:.2f}%)")
    logging.info(f"Executed trades: {executed_trades} out of {len(opportunities)} opportunities")
    logging.info(f"Opportunity capture rate: {simulation_results['opportunity_capture_rate']:.2f}")
    
    # Generate performance visualization
    generate_performance_charts(simulation_results, trader)
    
    return simulation_results, trader

def generate_performance_charts(results, trader):
    """
    Generate performance visualization charts for the simulation
    """
    # Create output directory
    os.makedirs('reports', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = f'reports/simulation_results_{timestamp}'
    
    # Create figure with multiple subplots
    plt.figure(figsize=(20, 16))
    
    # 1. Capital Growth Chart
    plt.subplot(3, 2, 1)
    plt.plot(results['capital_history'], linewidth=2)
    plt.title('Capital Growth Over Time', fontsize=14)
    plt.xlabel('Trade Number', fontsize=12)
    plt.ylabel('Capital ($)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 2. Hourly Profit Breakdown
    plt.subplot(3, 2, 2)
    x = range(len(results['hourly_profits']))
    plt.bar(x, results['hourly_profits'], color='green')
    plt.title('Hourly Profit Distribution', fontsize=14)
    plt.xlabel('Hour of Day', fontsize=12)
    plt.ylabel('Profit ($)', fontsize=12)
    plt.xticks(x)
    plt.grid(axis='y', alpha=0.3)
    
    # 3. Strategy Distribution
    plt.subplot(3, 2, 3)
    strategies = [trade['strategy'] for trade in results['trades']]
    strategy_counts = {'standard': strategies.count('standard'), 
                      'flashloan': strategies.count('flashloan')}
    plt.pie([strategy_counts['standard'], strategy_counts['flashloan']],
            labels=['Standard', 'Flashloan'], autopct='%1.1f%%',
            colors=['#66b3ff', '#99ff99'])
    plt.title('Trading Strategy Distribution', fontsize=14)
    
    # 4. Profit by Strategy
    plt.subplot(3, 2, 4)
    standard_profits = [t['profit'] for t in results['trades'] if t['strategy'] == 'standard']
    flashloan_profits = [t['profit'] for t in results['trades'] if t['strategy'] == 'flashloan']
    
    profit_data = [sum(standard_profits), sum(flashloan_profits)]
    plt.bar(['Standard', 'Flashloan'], profit_data, color=['#66b3ff', '#99ff99'])
    plt.title('Total Profit by Strategy', fontsize=14)
    plt.ylabel('Profit ($)', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    
    # 5. Gas Price vs. Trade Count
    plt.subplot(3, 2, 5)
    gas_prices = [t['gas_price'] for t in results['trades']]
    gas_bins = np.linspace(min(gas_prices), max(gas_prices), 20)
    plt.hist(gas_prices, bins=gas_bins, alpha=0.7)
    plt.title('Gas Price Distribution for Executed Trades', fontsize=14)
    plt.xlabel('Gas Price (Gwei)', fontsize=12)
    plt.ylabel('Number of Trades', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 6. Risk Score vs. Profit Scatter
    plt.subplot(3, 2, 6)
    risk_scores = [t['risk_score'] for t in results['trades']]
    profits = [t['profit'] for t in results['trades']]
    
    plt.scatter(risk_scores, profits, alpha=0.6)
    plt.title('Risk Score vs. Profit', fontsize=14)
    plt.xlabel('Risk Score', fontsize=12)
    plt.ylabel('Profit ($)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{report_path}.png', dpi=300)
    plt.close()
    
    # Save results as JSON for future analysis
    with open(f'{report_path}.json', 'w') as f:
        # Convert objects that aren't JSON serializable
        serializable_results = results.copy()
        serializable_results['trades'] = results['trades']
        json.dump(serializable_results, f, indent=2)
    
    logging.info(f"Performance charts saved to {report_path}.png")
    logging.info(f"Results data saved to {report_path}.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run 24-hour trading bot simulation')
    parser.add_argument('--model', type=str, help='Path to pre-trained model', default=None)
    parser.add_argument('--risk', type=str, choices=['conservative', 'moderate', 'aggressive'], 
                        default='moderate', help='Risk level for trading')
    parser.add_argument('--hours', type=int, default=24, help='Number of hours to simulate')
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility', default=None)
    
    args = parser.parse_args()
    
    print(f"Starting {args.hours}-hour trading simulation with {args.risk} risk profile")
    results, trader = run_simulation(
        model_path=args.model,
        risk_level=args.risk,
        hours=args.hours,
        seed=args.seed
    )
    
    print(f"Simulation completed successfully. Results available in the reports directory.")
    print(f"Total profit: ${results['total_profit']:.2f} ({results['profit_percentage']:.2f}%)")
