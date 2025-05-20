#!/usr/bin/env python
"""
Evaluate Quantum Trading Profitability
-------------------------------------
This script loads a trained model and evaluates its profitability
on historical or simulated market data.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import logging
from stable_baselines3 import DQN, PPO, A2C
from rl_agent import TradingEnv, CHAINS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("evaluation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ProfitabilityEvaluator")

def find_latest_model(algorithm="DQN"):
    """Find the most recently trained model"""
    models_dir = "models"
    if not os.path.exists(models_dir):
        logger.error(f"Models directory {models_dir} does not exist")
        return None
        
    # Get all model files for the specified algorithm
    model_files = [f for f in os.listdir(models_dir) if f.startswith(algorithm)]
    if not model_files:
        logger.error(f"No models found for algorithm {algorithm}")
        return None
        
    # Sort by timestamp (most recent last)
    model_files.sort()
    latest_model = os.path.join(models_dir, model_files[-1])
    logger.info(f"Found latest model: {latest_model}")
    return latest_model

def evaluate_profitability(model_path=None, algorithm="DQN", episodes=5, 
                         steps_per_episode=100, chains=None):
    """Evaluate the profitability of a trained model"""
    if chains is None:
        chains = CHAINS
        
    # Find latest model if not specified
    if model_path is None:
        model_path = find_latest_model(algorithm)
        if model_path is None:
            return None
            
    # Create environment
    env = TradingEnv(chains)
    
    # Load model
    try:
        if algorithm == "DQN":
            model = DQN.load(model_path)
        elif algorithm == "PPO":
            model = PPO.load(model_path)
        elif algorithm == "A2C":
            model = A2C.load(model_path)
        else:
            logger.error(f"Unsupported algorithm {algorithm}")
            return None
    except Exception as e:
        logger.error(f"Error loading model {model_path}: {str(e)}")
        return None
        
    logger.info(f"Evaluating profitability of model {model_path} for {episodes} episodes")
    
    results = []
    portfolio_values = []
    returns = []
    win_count = 0
    
    for episode in range(episodes):
        obs = env.reset()
        done = False
        step = 0
        episode_portfolio_values = []
        trades = []
        
        # Store initial portfolio value
        initial_portfolio = env.balance
        episode_portfolio_values.append(initial_portfolio)
        
        while not done and step < steps_per_episode:
            # Model makes prediction
            action, _states = model.predict(obs, deterministic=True)
            
            # Execute action
            obs, reward, done, info = env.step(action)
            
            # Track portfolio value
            portfolio_value = info.get('portfolio_value', 0)
            episode_portfolio_values.append(portfolio_value)
            
            # Track trade
            if info.get('trade_executed', False):
                trades.append({
                    'step': step,
                    'action': action,
                    'price': info.get('price', 0),
                    'quantity': info.get('quantity', 0),
                    'portfolio_value': portfolio_value,
                    'reward': reward
                })
            
            step += 1
            
        # Calculate episode return
        final_portfolio = episode_portfolio_values[-1]
        episode_return = (final_portfolio - initial_portfolio) / initial_portfolio * 100
        returns.append(episode_return)
        
        if episode_return > 0:
            win_count += 1
        
        # Store episode results
        results.append({
            'episode': episode,
            'steps': step,
            'initial_portfolio': initial_portfolio,
            'final_portfolio': final_portfolio,
            'return_pct': episode_return,
            'max_portfolio': max(episode_portfolio_values),
            'min_portfolio': min(episode_portfolio_values),
            'trade_count': len(trades)
        })
        
        # Append to overall portfolio values
        portfolio_values.extend([(episode, s, v) for s, v in enumerate(episode_portfolio_values)])
        
        logger.info(f"Episode {episode+1}: Return: {episode_return:.2f}%, " +
                   f"Portfolio: ${final_portfolio:.2f}, Trades: {len(trades)}")
    
    # Calculate overall statistics
    avg_return = np.mean(returns)
    win_rate = win_count / episodes * 100
    
    logger.info(f"Evaluation complete. Average return: {avg_return:.2f}%, Win rate: {win_rate:.2f}%")
    
    # Save results
    results_dir = "results/evaluation"
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f"{results_dir}/profit_evaluation_{timestamp}.json"
    
    evaluation_results = {
        'model_path': model_path,
        'algorithm': algorithm,
        'episodes': episodes,
        'steps_per_episode': steps_per_episode,
        'avg_return_pct': avg_return,
        'win_rate': win_rate,
        'episode_results': results,
        'timestamp': datetime.now().isoformat()
    }
    
    # Save as JSON
    with open(results_file, 'w') as f:
        import json
        json.dump(evaluation_results, f, indent=4)
    
    # Plot portfolio values
    plt.figure(figsize=(12, 6))
    episode_colors = plt.cm.jet(np.linspace(0, 1, episodes))
    
    for episode in range(episodes):
        episode_data = [(s, v) for e, s, v in portfolio_values if e == episode]
        steps, values = zip(*episode_data)
        plt.plot(steps, values, color=episode_colors[episode], label=f"Episode {episode+1}")
    
    plt.title("Portfolio Value Over Time")
    plt.xlabel("Step")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{results_dir}/portfolio_values_{timestamp}.png")
    
    return evaluation_results
    
if __name__ == "__main__":
    # Use the most recent DQN model
    chains = [
        {'name': 'Polygon', 'dex': 'QuickSwap'},
        {'name': 'Solana', 'dex': 'Raydium'}
    ]
    
    results = evaluate_profitability(
        algorithm="DQN",
        episodes=3,
        steps_per_episode=50,
        chains=chains
    )
    
    if results:
        print("\n===== PROFITABILITY SUMMARY =====")
        print(f"Average Return: {results['avg_return_pct']:.2f}%")
        print(f"Win Rate: {results['win_rate']:.2f}%")
        print(f"Results saved to: results/evaluation/")
