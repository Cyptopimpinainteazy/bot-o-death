#!/usr/bin/env python
"""
Train Quantum RL Agent
---------------------
Training script for the reinforcement learning agent with enhanced quantum trading features.
This script trains the RL model on historical market data and saves performance metrics.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import logging
from rl_agent import train_rl_agent, TradingEnv, CHAINS
from quantum_enhancements import EnhancedQuantumTrading
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("QuantumRLTrainer")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Quantum-Enhanced RL Trading Agent')
    parser.add_argument('--timesteps', type=int, default=100000,
                        help='Total timesteps for training (default: 100000)')
    parser.add_argument('--algorithm', type=str, default='DQN',
                        help='RL algorithm to use (default: DQN)')
    parser.add_argument('--chains', type=str, nargs='+', default=['Polygon', 'Solana'],
                        help='List of chains to train on (default: Polygon Solana)')
    parser.add_argument('--save_interval', type=int, default=10000,
                        help='Save model checkpoint every n steps (default: 10000)')
    parser.add_argument('--output_dir', type=str, default='results/training',
                        help='Directory to save results (default: results/training)')
    parser.add_argument('--evaluate', action='store_true',
                        help='Run evaluation after training')
    
    return parser.parse_args()

def setup_environment(args):
    """Setup training environment and directories"""
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(f"{args.output_dir}/models", exist_ok=True)
    os.makedirs(f"{args.output_dir}/plots", exist_ok=True)
    os.makedirs(f"{args.output_dir}/metrics", exist_ok=True)
    
    # Format chains for the environment
    formatted_chains = []
    for chain in args.chains:
        if chain == 'Polygon':
            formatted_chains.append({'name': 'Polygon', 'dex': 'QuickSwap'})
        elif chain == 'Solana':
            formatted_chains.append({'name': 'Solana', 'dex': 'Raydium'})
        elif chain == 'Ethereum':
            formatted_chains.append({'name': 'Ethereum', 'dex': 'Uniswap'})
        elif chain == 'Avalanche':
            formatted_chains.append({'name': 'Avalanche', 'dex': 'Trader Joe'})
        else:
            formatted_chains.append({'name': chain, 'dex': 'Unknown'})
    
    # Save configuration
    config = {
        'timestamp': datetime.now().isoformat(),
        'algorithm': args.algorithm,
        'timesteps': args.timesteps,
        'chains': formatted_chains,
        'save_interval': args.save_interval
    }
    
    with open(f"{args.output_dir}/training_config.json", 'w') as f:
        json.dump(config, f, indent=4)
    
    return formatted_chains, config

def evaluate_model(model, env, num_episodes=10):
    """Evaluate the trained model"""
    logger.info(f"Evaluating model for {num_episodes} episodes")
    
    episode_rewards = []
    portfolio_values = []
    quantum_factors = []
    
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        quantum_impact = []
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            
            # Track metrics
            portfolio_values.append(info['portfolio_value'])
            quantum_factors.append(info['quantum_factor'])
            
        logger.info(f"Episode {episode+1} finished with total reward: {total_reward:.2f}")
        episode_rewards.append(total_reward)
    
    # Return evaluation metrics
    return {
        'episode_rewards': episode_rewards,
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'portfolio_values': portfolio_values,
        'quantum_factors': quantum_factors
    }

def plot_evaluation_results(eval_metrics, output_dir):
    """Generate plots from evaluation results"""
    # Plot episode rewards
    plt.figure(figsize=(10, 6))
    plt.plot(eval_metrics['episode_rewards'])
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/plots/episode_rewards.png")
    plt.close()
    
    # Plot portfolio value progression
    plt.figure(figsize=(12, 6))
    plt.plot(eval_metrics['portfolio_values'])
    plt.title('Portfolio Value Over Time')
    plt.xlabel('Step')
    plt.ylabel('Portfolio Value ($)')
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/plots/portfolio_values.png")
    plt.close()
    
    # Plot quantum factor distribution
    plt.figure(figsize=(10, 6))
    plt.hist(eval_metrics['quantum_factors'], bins=20, alpha=0.7)
    plt.title('Quantum Factor Distribution')
    plt.xlabel('Quantum Factor')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/plots/quantum_factors.png")
    plt.close()
    
    logger.info(f"Evaluation plots saved to {output_dir}/plots/")

def main():
    """Main training function"""
    # Parse arguments
    args = parse_arguments()
    
    # Setup environment
    chains, config = setup_environment(args)
    logger.info(f"Training on chains: {', '.join([chain['name'] for chain in chains])}")
    
    # Train the agent
    logger.info(f"Starting training for {args.timesteps} timesteps using {args.algorithm}")
    try:
        model = train_rl_agent(
            chains=chains,
            total_timesteps=args.timesteps,
            algorithm=args.algorithm
        )
        
        # Save the final model
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = f"{args.output_dir}/models/{args.algorithm}_{timestamp}_final"
        model.save(model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Run evaluation if requested
        if args.evaluate:
            # Create evaluation environment
            eval_env = TradingEnv(chains)
            
            # Evaluate the model
            eval_metrics = evaluate_model(model, eval_env)
            
            # Save evaluation metrics
            with open(f"{args.output_dir}/metrics/evaluation_{timestamp}.json", 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                for key, value in eval_metrics.items():
                    if isinstance(value, np.ndarray):
                        eval_metrics[key] = value.tolist()
                json.dump(eval_metrics, f, indent=4)
            
            # Plot results
            plot_evaluation_results(eval_metrics, args.output_dir)
            
            logger.info(f"Evaluation results: Mean reward: {eval_metrics['mean_reward']:.2f} ± {eval_metrics['std_reward']:.2f}")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise
    
    logger.info("Training completed successfully")

if __name__ == "__main__":
    main()
