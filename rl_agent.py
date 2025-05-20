import numpy as np
import torch
import logging
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from typing import Dict, Tuple
import gymnasium as gym
from gymnasium import spaces
import pandas as pd

logger = logging.getLogger(__name__)

class TradingEnv(gym.Env):
    """Custom Trading Environment that follows gym interface"""
    def __init__(self, df, window_size=20):
        super(TradingEnv, self).__init__()
        
        self.df = df
        self.window_size = window_size
        self.current_step = window_size
        
        # Define action space (0: hold, 1: buy, 2: sell)
        self.action_space = spaces.Discrete(3)
        
        # Define observation space
        n_features = len(df.columns) - 1  # Exclude timestamp
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(window_size, n_features),
            dtype=np.float32
        )
        
        self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.total_reward = 0
        self.position = 0  # 0: no position, 1: long, -1: short
        # Return observation and an empty info dict
        observation = self._next_observation()
        info = {}
        return observation, info
        
    def _next_observation(self):
        obs = self.df.iloc[self.current_step-self.window_size:self.current_step, 1:].values
        return obs.astype(np.float32)
        
    def step(self, action):
        try:
            # Validate action
            if not isinstance(action, (int, np.integer)) or action not in [0, 1, 2]:
                raise ValueError(f"Invalid action: {action}")
                
            # Take action
            if action == 1 and self.position <= 0:  # Buy
                self.position = 1
            elif action == 2 and self.position >= 0:  # Sell
                self.position = -1
            elif action == 0:  # Hold
                pass
                
            # Move to next step
            self.current_step += 1
            
            # Get new observation
            obs = self._next_observation()
            
            # Ensure observation shape matches expected shape
            if obs.shape != self.observation_space.shape:
                obs = obs.reshape(self.observation_space.shape)
            
            # Calculate reward
            reward = self._calculate_reward()
            self.total_reward += reward
            
            # Check if episode is done
            done = self.current_step >= len(self.df)
            
            return obs, reward, done, False, {}
            
        except Exception as e:
            logger.error(f"Error in step: {str(e)}")
            return np.zeros(self.observation_space.shape), 0, True, False, {}
            
    def _calculate_reward(self):
        if self.current_step >= len(self.df):
            return 0
            
        current_price = self.df.iloc[self.current_step]['close']
        prev_price = self.df.iloc[self.current_step-1]['close']
        price_change = (current_price - prev_price) / prev_price
        
        if self.position == 1:  # Long position
            return price_change
        elif self.position == -1:  # Short position
            return -price_change
        else:  # No position
            return 0

class RLAgent:
    def __init__(self, state_size: int, action_size: int, window_size: int = 20,
                 memory_size: int = 1000,
                 batch_size: int = 64, gamma: float = 0.99, epsilon: float = 1.0,
                 epsilon_min: float = 0.01, epsilon_decay: float = 0.995,
                 learning_rate: float = 0.001):
        """Initialize RL Agent with PPO algorithm"""
        self.state_size = state_size
        self.action_size = action_size
        self.window_size = window_size
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.model = None
        
    def create_env(self, data: pd.DataFrame) -> gym.Env:
        """Create a vectorized environment for training"""
        env = TradingEnv(data, window_size=self.window_size)
        return DummyVecEnv([lambda: env])
        
    def train(self, train_data: pd.DataFrame, val_data: pd.DataFrame) -> Dict:
        """Train the agent using PPO algorithm"""
        try:
            # Create environments for training and validation using the DataFrames
            train_env = self.create_env(train_data)
            val_env = self.create_env(val_data)
            
            # Initialize the PPO model if not already initialized
            if self.model is None:
                self.model = PPO(
                    "MlpPolicy",
                    train_env,
                    learning_rate=self.learning_rate,
                    gamma=self.gamma,
                    batch_size=self.batch_size,
                    verbose=1
                )
            
            # Create evaluation callback
            eval_callback = EvalCallback(
                val_env,
                best_model_save_path="models/best_rl_model",
                log_path="logs/rl_results",
                eval_freq=1000,
                deterministic=True,
                render=False
            )
            
            # Train the model
            self.model.learn(
                total_timesteps=100000,
                callback=eval_callback
            )
            
            metrics = {
                "train_reward": train_env.get_attr("total_reward")[0],
                "val_reward": val_env.get_attr("total_reward")[0]
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error during RL training: {str(e)}")
            raise
            
    def predict(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make a prediction for the given state"""
        if self.model is None:
            raise ValueError("Model not trained yet")
            
        action, _ = self.model.predict(state, deterministic=True)
        return action
        
    def save(self, path: str):
        """Save the model"""
        if self.model is not None:
            self.model.save(path)
            
    def load(self, path: str):
        """Load the model"""
        self.model = PPO.load(path) 