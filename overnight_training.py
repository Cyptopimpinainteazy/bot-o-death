#!/usr/bin/env python
"""
Overnight Quantum Trading RL Training
------------------------------------
Extended training session for the quantum-enhanced trading model
"""

import os
import logging
import argparse
from datetime import datetime
import numpy as np
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from rl_agent import train_rl_agent, TradingEnv, CHAINS

# Configure logging
import sys
log_file = "overnight_training.log"

# Clear the log file first
with open(log_file, 'w') as f:
    f.write(f"Starting overnight training at {datetime.now().isoformat()}\n")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Define handlers
file_handler = logging.FileHandler(log_file)
stream_handler = logging.StreamHandler(sys.stdout)

# Set format for handlers
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

# Get logger and add handlers
logger = logging.getLogger("OvernightTraining")
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

def create_results_dirs():
    """Create all necessary directories for storing results"""
    dirs = [
        "results/metrics",
        "results/evaluation",
        "results/plots",
        "models",
        "checkpoints",
        "results/tensorboard"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        logger.info(f"Created directory: {d}")

def main():
    # Create necessary directories
    create_results_dirs()
    
    # Define chains to trade on
    all_chains = [
        {'name': 'Polygon', 'dex': 'QuickSwap'},
        {'name': 'Solana', 'dex': 'Raydium'},
        {'name': 'Ethereum', 'dex': 'Uniswap'}
    ]
    
    # Set up training parameters
    timesteps = 100000  # Substantial training
    algorithm = "PPO"   # PPO tends to work better for trading
    
    # Create timestamp for identifying this run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Set up callbacks for evaluation and checkpoints
    checkpoint_dir = "checkpoints"
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=checkpoint_dir,
        name_prefix=f"{algorithm}_{timestamp}"
    )
    
    # Create separate evaluation environment
    eval_env = TradingEnv(all_chains)
    
    # Manual monitoring without using Monitor wrapper
    # (compatible with our custom gym environment)
    eval_dir = "results/evaluation"
    os.makedirs(eval_dir, exist_ok=True)
    
    # Custom evaluation function
    def evaluate_callback(model, freq):
        if model.num_timesteps % freq == 0:
            logger.info(f"Evaluating model at {model.num_timesteps} timesteps")
            obs = eval_env.reset()
            done = False
            rewards = []
            total_reward = 0
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = eval_env.step(action)
                total_reward += reward
                if done:
                    rewards.append(total_reward)
                    total_reward = 0
                    
            avg_reward = np.mean(rewards) if rewards else 0
            logger.info(f"Evaluation: Average reward: {avg_reward:.2f}")
            
            # Save if it's the best model so far
            model_path = f"models/best_{algorithm}_{timestamp}.zip"
            model.save(model_path)
            logger.info(f"Saved model checkpoint to {model_path}")
            
    # Custom callback wrapper
    class CustomCallback:
        def __init__(self, eval_func, freq):
            self.eval_func = eval_func
            self.freq = freq
            
        def __call__(self, locals_dict, globals_dict):
            model = locals_dict['self']
            self.eval_func(model, self.freq)
            return True
    
    # Combine callbacks
    custom_eval = CustomCallback(evaluate_callback, 10000)
    callbacks = [checkpoint_callback]
    
    logger.info(f"Starting extended overnight training with {algorithm} for {timesteps} steps")
    logger.info(f"Training on chains: {', '.join([c['name'] for c in all_chains])}")
    
    try:
        # Train the model
        model = train_rl_agent(
            chains=all_chains,
            total_timesteps=timesteps,
            algorithm=algorithm,
            callback=callbacks,
            learning_rate=0.0003,
            gamma=0.99,
            exploration_fraction=0.2,
            checkpoint_freq=10000,
            tensorboard_log=f"./results/tensorboard/{algorithm}_{timestamp}"
        )
        
        logger.info("Extended training complete")
        
        # Save final model
        final_model_path = f"models/{algorithm}_{timestamp}_final.zip"
        model.save(final_model_path)
        logger.info(f"Saved final model to {final_model_path}")
        
        # Evaluate final model performance
        logger.info("Evaluating final model performance...")
        rewards = []
        
        for _ in range(10):  # 10 evaluation episodes
            obs = eval_env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = eval_env.step(action)
                episode_reward += reward
            
            rewards.append(episode_reward)
        
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        logger.info(f"Final evaluation: Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
        
        # Record final metrics
        final_metrics_path = f"results/metrics/{algorithm}_{timestamp}_final_metrics.json"
        import json
        with open(final_metrics_path, 'w') as f:
            json.dump({
                'algorithm': algorithm,
                'total_timesteps': timesteps,
                'chains': [c['name'] for c in all_chains],
                'mean_reward': float(mean_reward),
                'std_reward': float(std_reward),
                'final_model_path': final_model_path,
                'training_completed': True,
                'completion_time': datetime.now().isoformat()
            }, f, indent=4)
        
        logger.info(f"Training results saved to {final_metrics_path}")
        
    except Exception as e:
        logger.error(f"Error during extended training: {str(e)}")
        raise

if __name__ == "__main__":
    main()
