#!/usr/bin/env python
"""
Reinforcement Learning Environment for Quantum Trading with Risk Management

This module creates a reinforcement learning environment that integrates
with the risk-aware quantum trading system, allowing AI bots to be
trained on historical market data with risk parameters.
"""

import os
import numpy as np
import pandas as pd
import gym
from gym import spaces
import json
import logging
from pathlib import Path
from datetime import datetime

from exchange_risk_manager import ExchangeRiskManager
from technical_analysis import TechnicalAnalysisEngine
from quantum_ensemble import QuantumEnsembleTrader
from fund_prepositioning import FundPrepositioningManager
from triangle_arbitrage import TriangleArbitrageDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class QuantumTradingEnv(gym.Env):
    """
    Reinforcement Learning Environment for Quantum Trading
    
    This environment allows RL algorithms to learn optimal trading strategies
    while taking into account risk parameters such as transfer delays,
    fees, and liquidity issues.
    """
    
    def __init__(self, config_path=None, historical_data_path=None):
        """Initialize the RL trading environment with configuration"""
        super(QuantumTradingEnv, self).__init__()
        
        # Load configuration
        self.config_dir = Path("config")
        if config_path:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            config_path = self.config_dir / "rl_trading_config.json"
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    self.config = json.load(f)
            else:
                self.config = self._create_default_config()
        
        # Load risk-aware trading configuration
        risk_config_path = self.config_dir / "risk_aware_trading_config.json"
        if os.path.exists(risk_config_path):
            with open(risk_config_path, 'r') as f:
                self.risk_trading_config = json.load(f)
        else:
            self.risk_trading_config = {"trading": {"use_fund_prepositioning": True}}
        
        # Initialize risk manager and trading components
        self.risk_manager = ExchangeRiskManager()
        self.technical_engine = TechnicalAnalysisEngine()
        self.quantum_trader = QuantumEnsembleTrader()
        
        # Initialize fund prepositioning manager if enabled
        self.use_fund_prepositioning = self.risk_trading_config.get("trading", {}).get("use_fund_prepositioning", True)
        self.fund_manager = FundPrepositioningManager() if self.use_fund_prepositioning else None
        
        logger.info(f"Fund prepositioning is {'enabled' if self.use_fund_prepositioning else 'disabled'} for RL training")
        
        # Load historical data
        self.historical_data_path = historical_data_path or "data/historical_market_data.csv"
        self._load_historical_data()
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(3)  # 0: hold, 1: buy, 2: sell
        
        # Observation space includes:
        # - Market features (price, volume, technical indicators)
        # - Risk metrics (transfer times, fees, liquidity)
        # - Quantum predictions
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(20,),  # 20 features in our state
            dtype=np.float32
        )
        
        # Initialize state
        self.current_step = 0
        self.current_balance = self.config.get("initial_balance", 10000.0)
        self.current_position = 0.0
        self.trade_history = []
        self.reward_history = []
        
    def _create_default_config(self):
        """Create default configuration for RL training"""
        config = {
            "initial_balance": 10000.0,
            "trading_fee": 0.001,
            "slippage": 0.001,
            "episode_length": 1000,
            "reward_scaling": 0.01,
            "risk_weight": 0.5,
            "use_fund_prepositioning": True,
            "rl_parameters": {
                "learning_rate": 0.001,
                "discount_factor": 0.99,
                "exploration_rate": 0.1,
                "batch_size": 64,
                "training_iterations": 100,
                "reward_function": "sharpe",   # Options: sharpe, profit, custom
                "max_drawdown_penalty": 0.5
            },
            "fund_prepositioning": {
                "enabled": True,
                "min_balance_factor": 1.2,    # Multiplier for minimum balance requirements
                "prediction_confidence": 0.5,  # Lower threshold for training
                "simulated_transfer_speedup": 5.0  # Speed up transfers for faster training
            },
            "market_features": [
                "price", "volume", "rsi", "macd", "bollinger"
            ],
            "risk_features": [
                "transfer_time", "fee_impact", "liquidity_score", "slippage_estimate", "fund_availability"
            ],
            "quantum_features": [
                "ensemble_prediction", "confidence_score", "quantum_uncertainty"
            ]
        }
        
        # Save default config
        config_path = self.config_dir / "rl_trading_config.json"
        if not os.path.exists(self.config_dir):
            os.makedirs(self.config_dir)
            
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Created default RL trading configuration at {config_path}")
        return config
    
    def _load_historical_data(self):
        """Load historical market data for training"""
        try:
            self.data = pd.read_csv(self.historical_data_path)
            logger.info(f"Loaded historical data with {len(self.data)} entries")
        except FileNotFoundError:
            logger.warning(f"Historical data file not found: {self.historical_data_path}")
            # Create dummy data for testing
            self._create_dummy_data()
    
    def _create_dummy_data(self):
        """Create dummy data for testing when no historical data is available"""
        logger.info("Creating dummy data for testing")
        dates = pd.date_range(start="2022-01-01", periods=1000, freq="1H")
        
        # Generate random walk price data
        price = 100 * np.ones(1000)
        for i in range(1, 1000):
            price[i] = price[i-1] * (1 + np.random.normal(0, 0.01))
        
        # Generate random volume
        volume = np.random.normal(1000, 300, 1000)
        volume = np.abs(volume)
        
        # Generate other features
        self.data = pd.DataFrame({
            'timestamp': dates,
            'price': price,
            'volume': volume,
            'exchange': ['binance'] * 1000,
            'symbol': ['BTC/USDT'] * 1000
        })
        
        # Save dummy data
        os.makedirs(os.path.dirname(self.historical_data_path), exist_ok=True)
        self.data.to_csv(self.historical_data_path, index=False)
        logger.info(f"Saved dummy data to {self.historical_data_path}")
    
    def _get_observation(self):
        """Get current market state as observation vector"""
        if self.current_step >= len(self.data):
            return np.zeros(self.observation_space.shape[0])
        
        # Get current market data
        current_data = self.data.iloc[self.current_step]
        
        # Calculate technical indicators
        technical_indicators = self.technical_engine.calculate_indicators(
            self.data.iloc[:self.current_step+1], 
            features=['rsi', 'macd', 'bollinger_bands']
        )
        
        # Get risk metrics from risk manager
        risk_metrics = self._get_risk_metrics(current_data)
        
        # Get quantum predictions
        quantum_predictions = self._get_quantum_predictions(current_data)
        
        # Combine all features
        observation = np.concatenate([
            # Market features
            [
                current_data['price'],
                current_data['volume'],
                technical_indicators.get('rsi', 50),
                technical_indicators.get('macd', 0),
                technical_indicators.get('bollinger_upper', current_data['price']) - current_data['price'],
                technical_indicators.get('bollinger_lower', current_data['price']) - current_data['price'],
            ],
            
            # Portfolio features
            [
                self.current_balance,
                self.current_position,
                self.current_position * current_data['price'],
            ],
            
            # Risk metrics
            risk_metrics,
            
            # Quantum predictions
            quantum_predictions
        ])
        
        return observation[:self.observation_space.shape[0]]
    
    def _get_risk_metrics(self, current_data):
        """Get risk metrics from the risk manager"""
        # Calculate transfer times between exchanges
        transfer_time = self.risk_manager.estimate_transfer_time(
            "binance", "coinbase", "BTC", 1.0
        )
        
        # Calculate fee impact
        fee_impact = self.risk_manager.calculate_fee_impact(
            "binance", "coinbase", "BTC", 1.0, current_data['price']
        )
        
        # Assess liquidity
        liquidity_score = 0.8  # Placeholder - would come from risk manager
        slippage_estimate = 0.002  # Placeholder
        
        return np.array([
            transfer_time / 60.0,  # Normalize to hours
            fee_impact,
            liquidity_score,
            slippage_estimate
        ])
    
    def _get_quantum_predictions(self, current_data):
        """Get predictions from quantum ensemble trader"""
        # In real implementation, this would call the quantum trader
        # For now, we'll use placeholder values
        ensemble_prediction = 0.0  # -1 to 1 (sell to buy)
        confidence_score = 0.0
        quantum_uncertainty = 0.0
        
        try:
            # Try to get real predictions from quantum trader
            prediction_result = self.quantum_trader.predict(
                symbol=current_data['symbol'],
                exchange=current_data['exchange'],
                current_price=current_data['price']
            )
            
            ensemble_prediction = prediction_result.get('signal', 0.0)
            confidence_score = prediction_result.get('confidence', 0.7)
            quantum_uncertainty = prediction_result.get('uncertainty', 0.3)
        except Exception as e:
            logger.warning(f"Could not get quantum predictions: {str(e)}")
        
        return np.array([
            ensemble_prediction,
            confidence_score,
            quantum_uncertainty
        ])
        
    def _estimate_transfer_time(self, exchange, asset):
        """Estimate transfer time between exchanges for a specific asset"""
        # Get network type for asset
        network_mapping = self.risk_manager.config.get('currency_networks', {})
        asset_network = network_mapping.get(asset, 'bitcoin')
        
        # Get transfer times from exchange risk config
        try:
            transfer_times = self.risk_manager.config['exchanges'][exchange]['transfer_times']
            transfer_time_minutes = transfer_times.get(asset_network, 30)  # Default 30 minutes
            
            # Apply speedup for faster RL training if enabled
            if self.use_fund_prepositioning and 'fund_prepositioning' in self.config:
                speedup = self.config['fund_prepositioning'].get('simulated_transfer_speedup', 1.0)
                transfer_time_minutes = transfer_time_minutes / speedup
                
            return max(1, transfer_time_minutes)  # Minimum 1 minute
            
        except (KeyError, TypeError):
            # Default transfer times if not found
            default_times = {
                'bitcoin': 20,
                'ethereum': 10,
                'solana': 2,
                'internal': 1
            }
            return default_times.get(asset_network, 15)
    
    def reset(self):
        """Reset the environment to starting state"""
        self.current_step = 0
        self.current_balance = self.config.get("initial_balance", 10000.0)
        self.current_position = 0.0
        self.trade_history = []
        self.reward_history = []
        
        return self._get_observation()
    
    def step(self, action):
        """
        Take a step in the environment based on the action
        
        Args:
            action (int): 0 = hold, 1 = buy, 2 = sell
            
        Returns:
            tuple: (observation, reward, done, info)
        """
        if self.current_step >= len(self.data) - 1:
            # Episode is done
            return self._get_observation(), 0, True, {}
        
        # Get current price data
        current_price = self.data.iloc[self.current_step]['price']
        next_price = self.data.iloc[self.current_step + 1]['price']
        current_exchange = self.data.iloc[self.current_step]['exchange']
        current_symbol = self.data.iloc[self.current_step]['symbol']
        symbol_base = current_symbol.split('/')[0]  # e.g., BTC from BTC/USDT
        
        # If fund prepositioning is enabled, check fund availability
        fund_availability = 1.0  # Default full availability
        if self.use_fund_prepositioning and self.fund_manager:
            # Check for fund availability in current exchange
            balances = self.fund_manager.get_exchange_balances()
            exchange_balances = balances.get(current_exchange, {})
            symbol_balance = exchange_balances.get(symbol_base, 0)
            
            # Calculate availability as a ratio (1.0 = full availability, 0.0 = no funds)
            min_balance_required = self.fund_manager.config['exchanges'].get(current_exchange, {}).get('min_balance', {}).get(symbol_base, 0)
            if min_balance_required > 0:
                fund_availability = min(symbol_balance / min_balance_required, 1.0)
        
        # Execute action
        reward = 0.0
        info = {
            'action': action,
            'price': current_price,
            'balance': self.current_balance,
            'position': self.current_position,
            'exchange': current_exchange,
            'symbol': current_symbol,
            'fund_availability': fund_availability
        }
        
        if action == 1:  # Buy
            if self.current_balance > 0:
                # Apply fund availability if prepositioning is enabled
                effective_balance = self.current_balance
                if self.use_fund_prepositioning:
                    # Adjust available balance based on fund availability
                    effective_balance = self.current_balance * fund_availability
                    if fund_availability < 0.1:  # Almost no funds available
                        # Apply severe penalty for trying to trade with insufficient funds
                        reward = -2.0 * self.config.get("reward_scaling", 0.01)
                        info['warning'] = f"Insufficient funds on {current_exchange} for {symbol_base}"
                        
                # Calculate how much to buy with effective balance
                amount_to_buy = effective_balance / current_price
                
                if amount_to_buy > 0:
                    # Apply trading fee
                    fee = amount_to_buy * current_price * self.config.get("trading_fee", 0.001)
                    self.current_balance -= fee
                    
                    # Apply slippage
                    slippage_cost = amount_to_buy * current_price * self.config.get("slippage", 0.001)
                    self.current_balance -= slippage_cost
                    
                    # Apply transfer delay cost if fund availability is low
                    transfer_penalty = 0
                    if fund_availability < 0.5:
                        # Simulate transfer delay cost
                        transfer_time = self._estimate_transfer_time(current_exchange, symbol_base)
                        transfer_penalty = amount_to_buy * current_price * (transfer_time / 1440) * 0.01  # 1% per day
                        self.current_balance -= transfer_penalty
                    
                    # Update position and balance
                    self.current_position += amount_to_buy
                    self.current_balance -= amount_to_buy * current_price
                    
                    # Record trade
                    self.trade_history.append({
                        'step': self.current_step,
                        'action': 'buy',
                        'exchange': current_exchange,
                        'symbol': current_symbol,
                        'price': current_price,
                        'amount': amount_to_buy,
                        'fee': fee,
                        'slippage': slippage_cost,
                        'transfer_penalty': transfer_penalty,
                        'fund_availability': fund_availability
                    })
                    
                    # Calculate immediate reward (costs + fund positioning reward/penalty)
                    trading_costs = fee + slippage_cost + transfer_penalty
                    fund_positioning_reward = fund_availability * 0.005 * self.config.get("reward_scaling", 0.01)  # Reward for good positioning
                    
                    reward = -trading_costs * self.config.get("reward_scaling", 0.01) + fund_positioning_reward
                
        elif action == 2:  # Sell
            if self.current_position > 0:
                # Calculate sell value
                sell_value = self.current_position * current_price
                
                # Apply trading fee
                fee = sell_value * self.config.get("trading_fee", 0.001)
                sell_value -= fee
                
                # Apply slippage
                slippage_cost = sell_value * self.config.get("slippage", 0.001)
                sell_value -= slippage_cost
                
                # Update position and balance
                self.current_balance += sell_value
                self.current_position = 0
                
                # Record trade
                self.trade_history.append({
                    'step': self.current_step,
                    'action': 'sell',
                    'price': current_price,
                    'amount': self.current_position,
                    'fee': fee,
                    'slippage': slippage_cost
                })
                
                # Calculate immediate reward based on profit/loss
                reward = (sell_value / (self.trade_history[-2]['amount'] * self.trade_history[-2]['price']) - 1.0) 
                reward *= self.config.get("reward_scaling", 0.01)
        
        # Calculate portfolio value change for hold action
        prev_portfolio_value = self.current_balance + (self.current_position * current_price)
        next_portfolio_value = self.current_balance + (self.current_position * next_price)
        
        # Add unrealized profit/loss change to reward for all actions
        reward += (next_portfolio_value - prev_portfolio_value) * self.config.get("reward_scaling", 0.01)
        
        # Record reward
        self.reward_history.append(reward)
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= min(len(self.data) - 1, self.config.get("episode_length", 1000))
        
        # Add additional info for analysis
        info['portfolio_value'] = next_portfolio_value
        info['reward'] = reward
        
        return self._get_observation(), reward, done, info
    
    def render(self, mode='human'):
        """Render the current state of the environment"""
        if mode == 'human':
            portfolio_value = self.current_balance + (self.current_position * self.data.iloc[self.current_step]['price'])
            print(f"Step: {self.current_step}")
            print(f"Price: {self.data.iloc[self.current_step]['price']:.2f}")
            print(f"Balance: {self.current_balance:.2f}")
            print(f"Position: {self.current_position:.6f}")
            print(f"Portfolio Value: {portfolio_value:.2f}")
            print(f"Cumulative Reward: {sum(self.reward_history):.2f}")
            print("-" * 50)
    
    def close(self):
        """Clean up resources"""
        pass


class QuantumTradingRLTrainer:
    """
    Trainer for RL algorithms on the Quantum Trading Environment
    
    Supports training with multiple RL algorithms including DQN, PPO, A2C.
    """
    
    def __init__(self, algorithm='ppo', config_path=None):
        """Initialize the RL trainer with algorithm and config"""
        self.algorithm = algorithm.lower()
        
        # Load configuration
        if config_path:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            config_path = Path("config") / "rl_training_config.json"
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    self.config = json.load(f)
            else:
                self.config = self._create_default_training_config()
        
        # Initialize environment
        self.env = QuantumTradingEnv()
        
        # Initialize metrics
        self.training_results = {
            'algorithm': self.algorithm,
            'episodes': [],
            'rewards': [],
            'portfolio_values': [],
            'trade_counts': []
        }
    
    def _create_default_training_config(self):
        """Create default RL training configuration"""
        config = {
            "algorithms": {
                "ppo": {
                    "learning_rate": 0.0003,
                    "n_steps": 2048,
                    "batch_size": 64,
                    "n_epochs": 10,
                    "gamma": 0.99,
                    "gae_lambda": 0.95,
                    "clip_range": 0.2,
                    "ent_coef": 0.01,
                    "vf_coef": 0.5,
                    "max_grad_norm": 0.5
                },
                "dqn": {
                    "learning_rate": 0.0005,
                    "buffer_size": 10000,
                    "learning_starts": 1000,
                    "batch_size": 32,
                    "gamma": 0.99,
                    "target_update_interval": 500,
                    "train_freq": 1,
                    "gradient_steps": 1,
                    "exploration_fraction": 0.1,
                    "exploration_final_eps": 0.05
                },
                "a2c": {
                    "learning_rate": 0.0007,
                    "n_steps": 5,
                    "gamma": 0.99,
                    "gae_lambda": 1.0,
                    "ent_coef": 0.01,
                    "vf_coef": 0.5,
                    "max_grad_norm": 0.5,
                    "rms_prop_eps": 1e-5
                }
            },
            "training": {
                "total_timesteps": 100000,
                "eval_episodes": 10,
                "save_freq": 10000,
                "log_freq": 1000,
                "random_seed": 42
            }
        }
        
        # Save default config
        config_path = Path("config") / "rl_training_config.json"
        if not os.path.exists(config_path.parent):
            os.makedirs(config_path.parent)
            
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Created default RL training configuration at {config_path}")
        return config
    
    def train(self, total_timesteps=None):
        """Train the RL agent"""
        try:
            from stable_baselines3 import PPO, DQN, A2C
            
            # Set random seed
            np.random.seed(self.config["training"].get("random_seed", 42))
            
            # Create the RL model based on selected algorithm
            if self.algorithm == 'ppo':
                algo_config = self.config["algorithms"].get("ppo", {})
                model = PPO(
                    "MlpPolicy", 
                    self.env,
                    learning_rate=algo_config.get("learning_rate", 0.0003),
                    n_steps=algo_config.get("n_steps", 2048),
                    batch_size=algo_config.get("batch_size", 64),
                    n_epochs=algo_config.get("n_epochs", 10),
                    gamma=algo_config.get("gamma", 0.99),
                    gae_lambda=algo_config.get("gae_lambda", 0.95),
                    clip_range=algo_config.get("clip_range", 0.2),
                    ent_coef=algo_config.get("ent_coef", 0.01),
                    vf_coef=algo_config.get("vf_coef", 0.5),
                    max_grad_norm=algo_config.get("max_grad_norm", 0.5),
                    verbose=1
                )
            elif self.algorithm == 'dqn':
                algo_config = self.config["algorithms"].get("dqn", {})
                model = DQN(
                    "MlpPolicy", 
                    self.env,
                    learning_rate=algo_config.get("learning_rate", 0.0005),
                    buffer_size=algo_config.get("buffer_size", 10000),
                    learning_starts=algo_config.get("learning_starts", 1000),
                    batch_size=algo_config.get("batch_size", 32),
                    gamma=algo_config.get("gamma", 0.99),
                    target_update_interval=algo_config.get("target_update_interval", 500),
                    train_freq=algo_config.get("train_freq", 1),
                    gradient_steps=algo_config.get("gradient_steps", 1),
                    exploration_fraction=algo_config.get("exploration_fraction", 0.1),
                    exploration_final_eps=algo_config.get("exploration_final_eps", 0.05),
                    verbose=1
                )
            elif self.algorithm == 'a2c':
                algo_config = self.config["algorithms"].get("a2c", {})
                model = A2C(
                    "MlpPolicy", 
                    self.env,
                    learning_rate=algo_config.get("learning_rate", 0.0007),
                    n_steps=algo_config.get("n_steps", 5),
                    gamma=algo_config.get("gamma", 0.99),
                    gae_lambda=algo_config.get("gae_lambda", 1.0),
                    ent_coef=algo_config.get("ent_coef", 0.01),
                    vf_coef=algo_config.get("vf_coef", 0.5),
                    max_grad_norm=algo_config.get("max_grad_norm", 0.5),
                    rms_prop_eps=algo_config.get("rms_prop_eps", 1e-5),
                    verbose=1
                )
            else:
                raise ValueError(f"Unsupported algorithm: {self.algorithm}. Choose from: ppo, dqn, a2c")
            
            # Start training
            time_steps = total_timesteps or self.config["training"].get("total_timesteps", 100000)
            logger.info(f"Starting training with {self.algorithm.upper()} for {time_steps} timesteps")
            
            model.learn(total_timesteps=time_steps)
            
            # Save the trained model
            model_path = Path("models") / f"quantum_trading_{self.algorithm}_{time_steps}.zip"
            if not os.path.exists(model_path.parent):
                os.makedirs(model_path.parent)
                
            model.save(model_path)
            logger.info(f"Saved trained model to {model_path}")
            
            # Evaluate the trained model
            self._evaluate_model(model)
            
            return model, self.training_results
            
        except ImportError:
            logger.error("Could not import stable-baselines3. Please install it with: pip install stable-baselines3")
            logger.info("For now, running a simple random agent simulation...")
            return self._train_random_agent(total_timesteps)
    
    def _train_random_agent(self, total_timesteps=None):
        """Train a random agent as fallback without stable-baselines"""
        time_steps = total_timesteps or self.config["training"].get("total_timesteps", 100000)
        logger.info(f"Training random agent for {time_steps} timesteps")
        
        episodes = 0
        total_steps = 0
        
        while total_steps < time_steps:
            obs = self.env.reset()
            done = False
            episode_reward = 0
            trade_count = 0
            
            while not done and total_steps < time_steps:
                action = np.random.randint(0, 3)  # Random action
                obs, reward, done, info = self.env.step(action)
                
                episode_reward += reward
                if action in [1, 2]:  # Buy or sell actions
                    trade_count += 1
                
                total_steps += 1
                
                if total_steps % 1000 == 0:
                    logger.info(f"Completed {total_steps}/{time_steps} steps")
            
            # Log episode results
            episodes += 1
            portfolio_value = self.env.current_balance + (self.env.current_position * self.env.data.iloc[min(self.env.current_step, len(self.env.data)-1)]['price'])
            
            self.training_results['episodes'].append(episodes)
            self.training_results['rewards'].append(episode_reward)
            self.training_results['portfolio_values'].append(portfolio_value)
            self.training_results['trade_counts'].append(trade_count)
            
            logger.info(f"Episode {episodes}: Reward={episode_reward:.2f}, Portfolio={portfolio_value:.2f}, Trades={trade_count}")
        
        return None, self.training_results
    
    def _evaluate_model(self, model, episodes=10):
        """Evaluate the trained model"""
        logger.info(f"Evaluating model for {episodes} episodes")
        
        for i in range(episodes):
            obs = self.env.reset()
            done = False
            episode_reward = 0
            trade_count = 0
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = self.env.step(action)
                
                episode_reward += reward
                if action in [1, 2]:  # Buy or sell actions
                    trade_count += 1
            
            # Log evaluation results
            portfolio_value = self.env.current_balance + (self.env.current_position * self.env.data.iloc[min(self.env.current_step, len(self.env.data)-1)]['price'])
            
            logger.info(f"Eval Episode {i+1}: Reward={episode_reward:.2f}, Portfolio={portfolio_value:.2f}, Trades={trade_count}")
    
    def save_results(self):
        """Save training results to disk"""
        results_path = Path("results") / "rl_training" / f"{self.algorithm}_results.json"
        if not os.path.exists(results_path.parent):
            os.makedirs(results_path.parent)
            
        with open(results_path, 'w') as f:
            json.dump(self.training_results, f, indent=2)
        
        logger.info(f"Saved training results to {results_path}")


if __name__ == "__main__":
    # Example usage
    trainer = QuantumTradingRLTrainer(algorithm='ppo')
    model, results = trainer.train(total_timesteps=10000)
    trainer.save_results()
