#!/usr/bin/env python
"""
Test Quantum Enhanced Trading
----------------------------
This script tests the enhanced quantum trading features and RL integration.
It runs a simple simulated trading scenario to verify functionality.
"""

import numpy as np
import matplotlib.pyplot as plt
from rl_agent import TradingEnv, train_rl_agent
from quantum_enhancements import EnhancedQuantumTrading, extract_market_conditions
import logging
import random
import os
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("test_quantum_enhanced.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("TestQuantumEnhanced")

def generate_mock_market_data(num_steps=50, chains=None):
    """Generate mock market data for testing"""
    if chains is None:
        chains = [{'name': 'Polygon', 'dex': 'QuickSwap'}, {'name': 'Solana', 'dex': 'Raydium'}]
    
    # Initialize with base prices
    base_prices = {
        'Polygon': 1.0,
        'Solana': 25.0,
        'Ethereum': 1800.0,
        'Avalanche': 12.0
    }
    
    market_data_series = []
    
    # Create price trends with some randomness
    for step in range(num_steps):
        chain_data = {}
        indicators = {}
        
        # Market cycle phase (0 to 2π)
        cycle_phase = (step / num_steps) * 2 * np.pi
        
        # Global market trend - sine wave with noise
        global_trend = np.sin(cycle_phase) * 0.05 + random.uniform(-0.02, 0.02)
        
        # Generate data for each chain
        for chain in chains:
            chain_name = chain['name']
            base_price = base_prices.get(chain_name, 1.0)
            
            # Chain-specific trend with correlation to global trend
            chain_specific = random.uniform(-0.03, 0.03)
            price_change = global_trend + chain_specific
            
            # Update base price with compound effect
            if step > 0:
                base_prices[chain_name] *= (1 + price_change)
            
            price = base_prices[chain_name]
            
            # Generate other market data
            volume = max(100, price * random.uniform(1000, 5000) * (1 + abs(price_change) * 10))
            depth = max(10, volume * random.uniform(0.2, 0.5))
            volatility = max(0.01, abs(price_change) * 5)
            
            # Store chain data
            chain_data[chain_name] = {
                'price': price,
                'volume': volume,
                'depth': depth,
                'volatility': volatility,
                'price_change': price_change
            }
        
        # Calculate technical indicators based on price action
        rsi_base = 50 + global_trend * 500  # Convert to 0-100 scale
        rsi = max(0, min(100, rsi_base))
        
        macd_signal = global_trend * 2
        macd = macd_signal + chain_specific
        
        order_imbalance = global_trend * 50 + random.uniform(-20, 20)
        
        indicators = {
            'rsi': rsi,
            'macd': macd,
            'macd_signal': macd_signal,
            'order_imbalance': order_imbalance
        }
        
        # Extract market conditions
        market_conditions = extract_market_conditions(chain_data, indicators)
        
        # Combine all data
        market_data = {
            'step': step,
            'chain_data': chain_data,
            'indicators': indicators
        }
        market_data.update(market_conditions)
        
        market_data_series.append(market_data)
    
    return market_data_series

def test_enhanced_quantum_trading():
    """Test the enhanced quantum trading system in isolation"""
    logger.info("Testing Enhanced Quantum Trading System")
    
    # Initialize the quantum trading system
    quantum_system = EnhancedQuantumTrading()
    
    # Generate mock market data
    market_data_series = generate_mock_market_data(num_steps=50)
    
    # Execute quantum strategy on each market data point
    results = []
    for market_data in market_data_series:
        result = quantum_system.execute_enhanced_quantum_strategy(market_data)
        results.append(result)
        
        logger.info(f"Step {market_data['step']}: Action={result.get('action', 'unknown')}, " + 
                   f"Q-Factor={result.get('enhanced_quantum_factor', 0):.4f}, " +
                   f"Circuit Depth={result.get('circuit_params', {}).get('depth', 0)}")
    
    # Analyze results
    actions = [r.get('action', 'unknown') for r in results]
    q_factors = [r.get('enhanced_quantum_factor', 0) for r in results]
    buy_probs = [r.get('buy_probability', 0) for r in results]
    sell_probs = [r.get('sell_probability', 0) for r in results]
    hold_probs = [r.get('hold_probability', 0) for r in results]
    
    # Create output directory
    os.makedirs("results/test", exist_ok=True)
    
    # Plot results
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(q_factors, label="Quantum Factor")
    plt.title("Enhanced Quantum Factors Over Time")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(buy_probs, label="Buy Probability")
    plt.plot(sell_probs, label="Sell Probability")
    plt.plot(hold_probs, label="Hold Probability")
    plt.title("Action Probabilities Over Time")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("results/test/quantum_enhanced_test.png")
    
    # Count actions
    action_counts = {}
    for action in actions:
        if action not in action_counts:
            action_counts[action] = 0
        action_counts[action] += 1
    
    logger.info(f"Action distribution: {action_counts}")
    logger.info(f"Average quantum factor: {np.mean(q_factors):.4f}")
    logger.info(f"Enhanced Quantum Trading test completed successfully")
    
    return results

def test_rl_agent_integration():
    """Test the RL agent integration with enhanced quantum trading"""
    logger.info("Testing RL Agent Integration with Enhanced Quantum Trading")
    
    # Create RL environment
    env = TradingEnv()
    
    # Run for a few episodes manually to test functionality
    num_episodes = 3
    max_steps_per_episode = 20
    
    for episode in range(num_episodes):
        obs = env.reset()
        total_reward = 0
        
        for step in range(max_steps_per_episode):
            # Take random actions for testing
            action = random.randint(0, 3)  # 0: Hold, 1: Buy, 2: Sell, 3: Sandwich
            obs, reward, done, info = env.step(action)
            
            total_reward += reward
            
            logger.info(f"Episode {episode+1}, Step {step+1}, Action: {action}, " +
                       f"Reward: {reward:.4f}, Portfolio: ${info['portfolio_value']:.2f}, " +
                       f"Q-Factor: {info['quantum_factor']:.4f}")
            
            if done:
                break
        
        logger.info(f"Episode {episode+1} finished with total reward: {total_reward:.4f}")
    
    logger.info("Quick RL Agent integration test completed successfully")
    
    # Now test a very short training run
    logger.info("Testing short RL Agent training...")
    try:
        model = train_rl_agent(total_timesteps=1000)  # Very short training just to test
        logger.info("RL training test completed successfully")
    except Exception as e:
        logger.error(f"Error during RL training test: {str(e)}")
    
    return True

if __name__ == "__main__":
    logger.info("Starting Enhanced Quantum Trading System Tests")
    
    # First test the enhanced quantum trading in isolation
    quantum_results = test_enhanced_quantum_trading()
    
    # Then test the RL agent integration
    rl_integration = test_rl_agent_integration()
    
    logger.info("All tests completed!")
