"""
Reinforcement Learning Trainer for Quantum Trading Optimization

This module implements a reinforcement learning-based training system
that optimizes trading strategies based on historical data.
"""

import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input, LSTM, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from collections import deque
import random
import matplotlib.pyplot as plt
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class ReinforcementTrainer:
    """
    Reinforcement Learning Trainer for optimizing quantum trading strategies.
    Uses a Deep Q-Network approach to learn optimal trade execution strategies.
    """
    
    def __init__(self, input_dim=10, memory_size=1000, batch_size=32, gamma=0.95):
        """
        Initialize the Reinforcement Learning Trainer
        
        Args:
            input_dim: Dimension of input features
            memory_size: Size of replay memory buffer
            batch_size: Size of minibatch for training
            gamma: Discount factor for future rewards
        """
        self.input_dim = input_dim
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.gamma = gamma
        
        # Possible actions (0: standard execution, 1: flashloan execution)
        self.action_space = [0, 1]
        self.action_size = len(self.action_space)
        
        # Replay memory
        self.memory = deque(maxlen=memory_size)
        
        # Exploration parameters
        self.epsilon = 1.0  # Initial exploration rate
        self.epsilon_min = 0.01  # Minimum exploration rate
        self.epsilon_decay = 0.995  # Decay rate for exploration
        
        # Build models
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        
        # Training metrics
        self.metrics = {
            'loss': [],
            'reward': [],
            'epsilon': []
        }
        
        # Create necessary directories
        os.makedirs('models', exist_ok=True)
        os.makedirs('results', exist_ok=True)
        
        logger = logging.getLogger(__name__)
        logger.info("Reinforcement Trainer initialized successfully")
    
    def _build_model(self):
        """
        Build a deep Q-learning model
        
        Returns:
            Keras model for Q-value prediction
        """
        model = Sequential()
        model.add(Dense(64, input_dim=self.input_dim, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(64, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
        return model
    
    def update_target_model(self):
        """Update target model with weights from the main model"""
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        """
        Store experience in replay memory
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, training=True):
        """
        Choose action based on epsilon-greedy policy
        
        Args:
            state: Current state
            training: Whether in training mode (exploration enabled)
            
        Returns:
            Selected action
        """
        if training and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])
    
    def replay(self, batch_size=None):
        """
        Train model using experience replay
        
        Args:
            batch_size: Size of minibatch for training
            
        Returns:
            Training loss
        """
        if batch_size is None:
            batch_size = self.batch_size
            
        if len(self.memory) < batch_size:
            return 0
        
        minibatch = random.sample(self.memory, batch_size)
        
        states = []
        targets = []
        
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state, verbose=0)[0]
            
            if done:
                target[action] = reward
            else:
                t = self.target_model.predict(next_state, verbose=0)[0]
                target[action] = reward + self.gamma * np.amax(t)
                
            states.append(state[0])
            targets.append(target)
            
        history = self.model.fit(np.array(states), np.array(targets), 
                                epochs=1, verbose=0)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return history.history['loss'][0]
    
    def load(self, model_path):
        """
        Load model weights
        
        Args:
            model_path: Path to saved model
        """
        if os.path.exists(model_path):
            self.model.load_weights(model_path)
            self.update_target_model()
            logging.info(f"Model loaded from {model_path}")
        else:
            logging.warning(f"Model file {model_path} not found")
    
    def save(self, model_path):
        """
        Save model weights
        
        Args:
            model_path: Path to save model
        """
        self.model.save_weights(model_path)
        logging.info(f"Model saved to {model_path}")
    
    def train(self, env, episodes=1000, max_steps=200, batch_size=None, 
              update_target_every=10, save_every=100, render=False):
        """
        Train the model using the provided environment
        
        Args:
            env: Training environment that follows gym-like API
            episodes: Number of episodes to train
            max_steps: Maximum steps per episode
            batch_size: Size of minibatch for training
            update_target_every: Update target model every n episodes
            save_every: Save model every n episodes
            render: Whether to render the environment
            
        Returns:
            Training metrics
        """
        if batch_size is None:
            batch_size = self.batch_size
            
        rewards = []
        
        for e in range(episodes):
            state = env.reset()
            state = np.reshape(state, [1, self.input_dim])
            
            total_reward = 0
            avg_loss = []
            
            for step in range(max_steps):
                if render:
                    env.render()
                    
                # Select action
                action = self.act(state)
                
                # Take action
                next_state, reward, done, _ = env.step(action)
                next_state = np.reshape(next_state, [1, self.input_dim])
                
                # Store in replay memory
                self.remember(state, action, reward, next_state, done)
                
                # Train model
                loss = self.replay(batch_size)
                if loss > 0:
                    avg_loss.append(loss)
                
                total_reward += reward
                state = next_state
                
                if done:
                    break
                    
            # Update target model
            if e % update_target_every == 0:
                self.update_target_model()
                
            # Save model
            if e % save_every == 0:
                self.save(f"models/dqn_model_ep{e}.h5")
                
            # Logging
            avg_loss_val = np.mean(avg_loss) if avg_loss else 0
            rewards.append(total_reward)
            
            self.metrics['loss'].append(avg_loss_val)
            self.metrics['reward'].append(total_reward)
            self.metrics['epsilon'].append(self.epsilon)
            
            msg = f"Episode: {e+1}/{episodes}, Reward: {total_reward}, "
            msg += f"Loss: {avg_loss_val:.4f}, Epsilon: {self.epsilon:.4f}"
            logging.info(msg)
            
            # Save training metrics
            if e % save_every == 0:
                self._save_metrics(f"results/training_metrics_ep{e}.json")
                self._plot_metrics(f"results/training_plot_ep{e}.png")
                
        # Final save
        self.save("models/dqn_model_final.h5")
        self._save_metrics("results/training_metrics_final.json")
        self._plot_metrics("results/training_plot_final.png")
        
        return self.metrics
    
    def _save_metrics(self, filepath):
        """
        Save training metrics to JSON file
        
        Args:
            filepath: Path to save metrics
        """
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f)
        
    def _plot_metrics(self, filepath):
        """
        Plot training metrics
        
        Args:
            filepath: Path to save plot
        """
        plt.figure(figsize=(15, 10))
        
        plt.subplot(3, 1, 1)
        plt.plot(self.metrics['reward'])
        plt.title('Reward per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        
        plt.subplot(3, 1, 2)
        plt.plot(self.metrics['loss'])
        plt.title('Loss per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        
        plt.subplot(3, 1, 3)
        plt.plot(self.metrics['epsilon'])
        plt.title('Epsilon per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Epsilon')
        
        plt.tight_layout()
        plt.savefig(filepath)
        plt.close()
