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
        
        # Experience replay memory
        self.memory = deque(maxlen=memory_size)
        
        # Exploration parameters
        self.epsilon = 1.0  # initial exploration rate
        self.epsilon_min = 0.01  # minimum exploration rate
        self.epsilon_decay = 0.995  # decay rate for exploration
        
        # Build the neural network model
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        
        # Training history
        self.loss_history = []
        self.reward_history = []
        self.epsilon_history = []
        
        logging.info("Reinforcement Trainer initialized")
    
    def _build_model(self):
        """
        Build a neural network model for deep Q-learning
        
        Returns:
            Keras Model for predicting Q-values
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
        """Update the target model to match the primary model weights"""
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """
        Choose an action based on current state
        
        Args:
            state: Current state vector
            
        Returns:
            Selected action index
        """
        # Epsilon-greedy action selection
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state = np.reshape(state, [1, self.input_dim])
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])
    
    def replay(self, batch_size=None):
        """
        Train the model using random samples from memory
        
        Args:
            batch_size: Size of minibatch to train on
        
        Returns:
            Training loss value
        """
        if batch_size is None:
            batch_size = self.batch_size
            
        if len(self.memory) < batch_size:
            return 0
        
        # Sample random minibatch from memory
        minibatch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            state = np.reshape(state, [1, self.input_dim])
            next_state = np.reshape(next_state, [1, self.input_dim])
            
            # Get target Q value
            if done:
                target = reward
            else:
                target = reward + self.gamma * np.amax(
                    self.target_model.predict(next_state, verbose=0)[0]
                )
            
            # Get current Q values and update the target for chosen action
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            
            # Train the network
            history = self.model.fit(state, target_f, epochs=1, verbose=0)
            loss = history.history['loss'][0]
        
        # Decay epsilon after each replay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        self.loss_history.append(loss)
        self.epsilon_history.append(self.epsilon)
        
        return loss
    
    def train_on_dataset(self, dataset_path, num_epochs=100, update_target_every=10):
        """
        Train the model on historical trade data
        
        Args:
            dataset_path: Path to the dataset file (CSV)
            num_epochs: Number of training epochs
            update_target_every: Update target network every N epochs
            
        Returns:
            Training history metrics
        """
        logging.info(f"Starting training on dataset: {dataset_path}")
        
        # Load and preprocess dataset
        try:
            if dataset_path.endswith('.csv'):
                df = pd.read_csv(dataset_path)
            elif dataset_path.endswith('.json'):
                with open(dataset_path, 'r') as f:
                    data = json.load(f)
                df = pd.DataFrame(data)
            else:
                raise ValueError("Unsupported file format. Use CSV or JSON.")
        except Exception as e:
            logging.error(f"Error loading dataset: {str(e)}")
            return None
        
        # Feature engineering and normalization
        X, y, rewards = self._preprocess_data(df)
        
        if X is None or len(X) == 0:
            logging.error("Failed to preprocess data")
            return None
        
        logging.info(f"Training on {len(X)} samples for {num_epochs} epochs")
        
        # Initialize training metrics
        epoch_rewards = []
        epoch_losses = []
        
        # Main training loop
        for epoch in range(num_epochs):
            epoch_reward = 0
            epoch_loss = 0
            
            # Iterate through each sample in the dataset
            for i in range(len(X)):
                state = X[i]
                
                # Choose action using epsilon-greedy policy
                action = self.act(state)
                
                # Calculate reward based on chosen action
                # If action=1 (flashloan) use flashloan_profit, else use standard_profit
                reward = rewards[i][action]
                epoch_reward += reward
                
                # For simplicity, next_state is the same as current state in this offline training
                # In a real environment, next_state would be different
                next_state = state
                done = (i == len(X) - 1)  # done if last sample
                
                # Store experience in replay memory
                self.remember(state, action, reward, next_state, done)
                
                # Train on a batch of samples
                loss = self.replay()
                if loss > 0:  # Only count non-zero loss values
                    epoch_loss += loss
            
            # Update target network periodically
            if epoch % update_target_every == 0:
                self.update_target_model()
                logging.info(f"Epoch {epoch+1}/{num_epochs}, Updated target network")
            
            avg_reward = epoch_reward / len(X)
            avg_loss = epoch_loss / len(X) if epoch_loss > 0 else 0
            
            self.reward_history.append(avg_reward)
            
            logging.info(f"Epoch {epoch+1}/{num_epochs}, Avg Reward: {avg_reward:.4f}, "
                         f"Avg Loss: {avg_loss:.4f}, Epsilon: {self.epsilon:.4f}")
        
        # Save training history
        history = {
            'loss': self.loss_history,
            'reward': self.reward_history,
            'epsilon': self.epsilon_history
        }
        
        # Generate training plots
        self._plot_training_results(history)
        
        return history
    
    def _preprocess_data(self, df):
        """
        Preprocess and normalize training data
        
        Args:
            df: Pandas DataFrame with trade data
            
        Returns:
            X: Feature vectors
            y: Target vectors
            rewards: Reward values for each action
        """
        logging.info("Preprocessing dataset...")
        
        try:
            # Check required columns
            required_cols = ['expected_profit', 'risk_score', 'actual_profit', 
                           'standard_profit', 'flashloan_profit']
            
            for col in required_cols:
                if col not in df.columns:
                    logging.warning(f"Missing column: {col}")
            
            # Extract features (customize based on available columns)
            features = [
                'expected_profit', 
                'risk_score',
                'optimized_amount'
            ]
            
            # Add network features if available
            if 'network' in df.columns:
                networks = pd.get_dummies(df['network'], prefix='network')
                df = pd.concat([df, networks], axis=1)
                features.extend(networks.columns.tolist())
            
            # Add trade type features if available
            if 'type' in df.columns:
                trade_types = pd.get_dummies(df['type'], prefix='type')
                df = pd.concat([df, trade_types], axis=1)
                features.extend(trade_types.columns.tolist())
            
            # Ensure all required columns exist
            X_cols = [col for col in features if col in df.columns]
            
            if not X_cols:
                logging.error("No valid feature columns found")
                return None, None, None
            
            # Create feature vectors
            X = df[X_cols].values
            
            # Normalize feature values
            X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-10)
            
            # For targets, use actual profits as the reward signal
            standard_profits = df['standard_profit'].values if 'standard_profit' in df.columns else df['actual_profit'].values
            flashloan_profits = df['flashloan_profit'].values if 'flashloan_profit' in df.columns else standard_profits * 1.2
            
            # Target is binary: which strategy performed better
            y = (flashloan_profits > standard_profits).astype(int)
            
            # Create reward matrix [standard_reward, flashloan_reward] for each sample
            rewards = np.column_stack((standard_profits, flashloan_profits))
            
            logging.info(f"Processed {len(X)} samples with {len(X_cols)} features")
            
            return X, y, rewards
            
        except Exception as e:
            logging.error(f"Error preprocessing data: {str(e)}")
            return None, None, None
    
    def _plot_training_results(self, history):
        """
        Plot and save training metrics
        
        Args:
            history: Dictionary containing training history
        """
        try:
            # Create output directory if it doesn't exist
            output_dir = 'results/ai_training'
            os.makedirs(output_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Plot reward history
            plt.figure(figsize=(10, 6))
            plt.plot(history['reward'])
            plt.title('Average Reward per Epoch')
            plt.xlabel('Epoch')
            plt.ylabel('Reward')
            plt.grid(True)
            plt.savefig(f"{output_dir}/reward_history_{timestamp}.png")
            
            # Plot loss history
            if history['loss']:
                plt.figure(figsize=(10, 6))
                plt.plot(history['loss'])
                plt.title('Training Loss per Batch')
                plt.xlabel('Batch')
                plt.ylabel('Loss')
                plt.grid(True)
                plt.savefig(f"{output_dir}/loss_history_{timestamp}.png")
            
            # Plot epsilon decay
            plt.figure(figsize=(10, 6))
            plt.plot(history['epsilon'])
            plt.title('Exploration Rate (Epsilon) Decay')
            plt.xlabel('Batch')
            plt.ylabel('Epsilon')
            plt.grid(True)
            plt.savefig(f"{output_dir}/epsilon_decay_{timestamp}.png")
            
            logging.info(f"Training plots saved to {output_dir}")
            
        except Exception as e:
            logging.error(f"Error generating plots: {str(e)}")
    
    def save_model(self, filepath):
        """Save the model to disk"""
        try:
            self.model.save(filepath)
            logging.info(f"Model saved to {filepath}")
            return True
        except Exception as e:
            logging.error(f"Error saving model: {str(e)}")
            return False
    
    def load_model(self, filepath):
        """Load the model from disk"""
        try:
            self.model = load_model(filepath)
            self.update_target_model()
            logging.info(f"Model loaded from {filepath}")
            return True
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            return False
    
    def predict_best_action(self, trade_opportunity):
        """
        Predict the best action for a given trade opportunity
        
        Args:
            trade_opportunity: Dictionary with trade features
            
        Returns:
            best_action: 0 for standard execution, 1 for flashloan execution
            confidence: Confidence score for the prediction
        """
        try:
            # Preprocess the trade opportunity
            features = self._extract_features_from_opportunity(trade_opportunity)
            
            if features is None:
                return 0, 0.0
            
            # Make prediction
            state = np.reshape(features, [1, self.input_dim])
            q_values = self.model.predict(state, verbose=0)[0]
            best_action = np.argmax(q_values)
            
            # Calculate confidence (normalized difference between action values)
            confidence = np.abs(q_values[1] - q_values[0]) / (np.sum(np.abs(q_values)) + 1e-10)
            
            return best_action, float(confidence)
            
        except Exception as e:
            logging.error(f"Error in prediction: {str(e)}")
            return 0, 0.0  # Default to standard execution on error
    
    def _extract_features_from_opportunity(self, opportunity):
        """
        Extract and normalize features from a trade opportunity
        
        Args:
            opportunity: Dictionary with trade opportunity data
            
        Returns:
            Normalized feature vector
        """
        try:
            # Define feature extraction based on your model's input requirements
            features = []
            
            # Add expected profit
            if 'expected_profit' in opportunity:
                features.append(float(opportunity['expected_profit']))
            else:
                features.append(0.0)
            
            # Add risk score
            if 'risk_score' in opportunity:
                features.append(float(opportunity['risk_score']))
            else:
                features.append(0.5)  # Default risk score
            
            # Add trade amount
            if 'optimized_amount' in opportunity:
                features.append(float(opportunity['optimized_amount']))
            else:
                features.append(0.0)
            
            # Add one-hot encoded features for networks
            networks = ['ethereum', 'polygon', 'arbitrum', 'optimism', 'base']
            network_features = [1.0 if opportunity.get('network', '').lower() == net else 0.0 
                               for net in networks]
            features.extend(network_features)
            
            # Add one-hot encoded features for trade types
            trade_types = ['direct', 'triangle']
            type_features = [1.0 if opportunity.get('type', '').lower() == t else 0.0 
                            for t in trade_types]
            features.extend(type_features)
            
            # Ensure we have the right number of features
            if len(features) != self.input_dim:
                logging.warning(f"Feature dimension mismatch: got {len(features)}, expected {self.input_dim}")
                # Pad with zeros if necessary
                features.extend([0.0] * (self.input_dim - len(features)))
            
            return np.array(features)
            
        except Exception as e:
            logging.error(f"Error extracting features: {str(e)}")
            return None
