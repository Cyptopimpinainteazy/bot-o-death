#!/usr/bin/env python3
"""
Trading Bot Model Trainer

This script integrates the data collector with reinforcement learning to train
trading models based on collected market data and trading history.
"""

import os
import sys
import json
import time
import logging
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
import tensorflow as tf
from pathlib import Path

# Add parent directory to path to import modules
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

# Import our modules
from scripts.data_collector import TradingDataCollector
from scripts.ai_optimization.reinforcement_trainer import ReinforcementTrainer
from scripts.ai_optimization.reinforcement_trainer_extension import *  # Import extensions

# Configure logging
logs_dir = os.path.join(project_root, "logs")
os.makedirs(logs_dir, exist_ok=True)

log_file = os.path.join(logs_dir, "training.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("TrainingModule")

class TradingEnvironment:
    """
    Environment for reinforcement learning that simulates trading scenarios
    based on collected data.
    """
    
    def __init__(self, data, window_size=10, initial_balance=1000):
        """
        Initialize the trading environment.
        
        Args:
            data: DataFrame with trading data
            window_size: Size of observation window
            initial_balance: Initial account balance
        """
        self.data = data
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.reset()
        
        # Feature columns (adjust based on your data)
        self.feature_columns = [col for col in data.columns 
                               if col not in ['timestamp', 'trade_id', 'profit']]
        self.n_features = len(self.feature_columns)
        
        logger.info(f"Trading environment initialized with {len(data)} data points")
        logger.info(f"Using features: {self.feature_columns}")
    
    def reset(self):
        """
        Reset the environment to initial state.
        
        Returns:
            Initial observation
        """
        self.balance = self.initial_balance
        self.position = 0  # No position
        self.current_step = self.window_size
        self.done = False
        return self._get_observation()
    
    def _get_observation(self):
        """
        Get the current observation (state).
        
        Returns:
            Current state features
        """
        start = max(0, self.current_step - self.window_size)
        end = self.current_step
        
        # Get window of data
        window_data = self.data.iloc[start:end]
        
        # Extract features
        features = window_data[self.feature_columns].values.flatten()
        
        # Add account state features
        features = np.append(features, [self.balance, self.position])
        
        return features
    
    def step(self, action):
        """
        Take an action in the environment.
        
        Args:
            action: 0 (standard execution) or 1 (flashloan execution)
            
        Returns:
            (observation, reward, done, info)
        """
        if self.done:
            return self._get_observation(), 0, True, {}
        
        # Get the current row data
        current_data = self.data.iloc[self.current_step]
        
        # Calculate profit based on action
        profit = current_data.get('profit', 0)
        
        # Adjust profit based on action (flashloan may have higher profit but higher risk)
        if action == 1:  # Flashloan execution
            profit = profit * 1.2  # 20% higher profit potential
            risk_factor = 0.05     # 5% risk of failure
            
            # Simulate flashloan risk
            if np.random.random() < risk_factor:
                profit = -profit * 2  # Failed flashloan results in losses
        
        # Update balance
        self.balance += profit
        
        # Update position (simplified)
        self.position = 1 if profit > 0 else 0
        
        # Define reward as profit
        reward = profit
        
        # Move to next step
        self.current_step += 1
        
        # Check if done
        if self.current_step >= len(self.data) - 1:
            self.done = True
        
        return self._get_observation(), reward, self.done, {'profit': profit}

def load_trading_data(data_path):
    """
    Load and preprocess trading data.
    
    Args:
        data_path: Path to trading data CSV
        
    Returns:
        Processed DataFrame
    """
    try:
        df = pd.read_csv(data_path)
        logger.info(f"Loaded {len(df)} rows from {data_path}")
        
        # Basic preprocessing
        if 'timestamp' in df.columns:
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Fill missing values
        df = df.fillna(0)
        
        return df
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        return None

def collect_training_data():
    """
    Collect and prepare training data using the TradingDataCollector.
    
    Returns:
        Path to collected training data
    """
    logger.info("Starting data collection process...")
    
    try:
        # Initialize data collector
        collector = TradingDataCollector()
        
        # Collect data from log files
        log_dir = os.path.join(project_root, "logs")
        collector.collect_from_log_directory(log_dir)
        
        # Process and save the collected data
        output_dir = os.path.join(project_root, "data")
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"trading_data_{timestamp}.csv")
        
        collector.save_to_csv(output_path)
        logger.info(f"Training data saved to {output_path}")
        
        return output_path
    except Exception as e:
        logger.error(f"Error collecting training data: {str(e)}")
        return None

def train_rl_model(data_path, model_output_path=None, epochs=100):
    """
    Train a reinforcement learning model on trading data.
    
    Args:
        data_path: Path to training data CSV
        model_output_path: Where to save the trained model
        epochs: Number of training epochs
        
    Returns:
        Trained model
    """
    logger.info(f"Starting RL model training with {epochs} epochs")
    
    # Load data
    data = load_trading_data(data_path)
    if data is None:
        logger.error("Failed to load training data")
        return None
    
    # Create environment
    window_size = 10  # Look at last 10 data points for decision
    env = TradingEnvironment(data, window_size=window_size)
    
    # Calculate input dimension
    input_dim = env.n_features * window_size + 2  # +2 for balance and position
    
    # Initialize trainer
    trainer = ReinforcementTrainer(input_dim=input_dim)
    
    # Check for existing model
    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, "dqn_model_final.h5")
    if os.path.exists(model_path):
        logger.info(f"Loading existing model from {model_path}")
        trainer.load(model_path)
    else:
        # Create baseline if no model exists
        logger.info("No existing model found, creating baseline")
        trainer.create_baseline_model()
    
    # Train model
    logger.info("Training model...")
    metrics = trainer.train(
        env, 
        episodes=epochs, 
        max_steps=min(1000, len(data)), 
        batch_size=32,
        update_target_every=10,
        save_every=max(1, epochs // 10)  # Save 10 times during training
    )
    
    # Save final model
    if model_output_path is None:
        model_output_path = os.path.join(models_dir, "dqn_model_final.h5")
    
    trainer.save(model_output_path)
    logger.info(f"Model saved to {model_output_path}")
    
    # Save training metrics
    results_dir = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    metrics_path = os.path.join(results_dir, "training_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f)
    
    logger.info(f"Training metrics saved to {metrics_path}")
    return trainer

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Train trading bot models")
    parser.add_argument("--data-path", type=str, help="Path to training data CSV (optional)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--collect-data", action="store_true", help="Collect fresh training data")
    parser.add_argument("--output-model", type=str, help="Path to save trained model")
    
    args = parser.parse_args()
    
    # Print banner
    print("=" * 80)
    print("  QUANTUM TRADING BOT TRAINER")
    print("=" * 80)
    print(f"Starting at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Collect data if requested or if no data path provided
    data_path = args.data_path
    if args.collect_data or data_path is None:
        data_path = collect_training_data()
        if data_path is None:
            logger.error("Data collection failed, exiting")
            return
    
    # Train model
    trainer = train_rl_model(
        data_path=data_path,
        model_output_path=args.output_model,
        epochs=args.epochs
    )
    
    if trainer is None:
        logger.error("Training failed")
        return
    
    print("\n" + "=" * 80)
    print("  TRAINING COMPLETE")
    print("=" * 80)
    print(f"Finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

if __name__ == "__main__":
    main()
