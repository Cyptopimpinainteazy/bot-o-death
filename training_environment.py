import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional
import asyncio
from dataclasses import dataclass
import json
import os
from pathlib import Path
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import yaml
import pickle
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    train_start_date: datetime
    train_end_date: datetime
    val_start_date: datetime
    val_end_date: datetime
    test_start_date: datetime
    test_end_date: datetime
    batch_size: int
    epochs: int
    learning_rate: float
    validation_frequency: int
    early_stopping_patience: int
    model_save_path: str
    lookback_period: int
    n_qubits: int
    max_risk: float
    var_confidence: float

class TrainingEnvironment:
    def __init__(self, config_path: str = 'config/models_config.yaml'):
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        self.models = {}
        self.metrics = {}
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            raise ValueError(f"Error loading config: {str(e)}")
            
    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger('TrainingEnvironment')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
        
    async def train(self, train_data: dict, val_data: dict) -> dict:
        """Main training loop for all models."""
        try:
            training_results = {}
            
            # Train Quantum Neural Network
            self.logger.info("Training Quantum Neural Network")
            qnn_metrics, qnn_model = await self._train_quantum_nn(train_data, val_data)
            training_results['quantum_nn'] = qnn_metrics
            self.models['quantum_nn'] = qnn_model
            
            # Train RL Agent
            self.logger.info("Training RL Agent")
            rl_metrics, rl_model = await self._train_rl_agent(train_data, val_data)
            training_results['rl_agent'] = rl_metrics
            self.models['rl_agent'] = rl_model
            
            # Train Risk Management System
            self.logger.info("Training Risk Management System")
            risk_metrics, risk_model = await self._train_risk_management(train_data, val_data)
            training_results['risk_management'] = risk_metrics
            self.models['risk_management'] = risk_model
            
            # Train Multi-Agent System
            self.logger.info("Training Multi-Agent System")
            multi_agent_metrics, multi_agent_model = await self._train_multi_agent(train_data, val_data)
            training_results['multi_agent'] = multi_agent_metrics
            self.models['multi_agent'] = multi_agent_model
            
            # Save final models
            self._save_models()
            
            return training_results
            
        except Exception as e:
            self.logger.error(f"Error in training process: {str(e)}")
            raise
            
    def _save_models(self):
        """Save all trained models."""
        try:
            for model_name, model in self.models.items():
                save_path = os.path.join(
                    self.config['model_save_path'],
                    f'{model_name}_final.pth'
                )
                if isinstance(model, torch.nn.Module):
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'config': self.config[model_name]
                    }, save_path)
                else:
                    # Handle non-PyTorch models
                    with open(save_path, 'wb') as f:
                        pickle.dump(model, f)
                self.logger.info(f"Saved {model_name} model to {save_path}")
        except Exception as e:
            self.logger.error(f"Error saving models: {str(e)}")
            raise
    
    async def prepare_data(self, data: Dict[str, pd.DataFrame]) -> Tuple[Dict, Dict, Dict]:
        """
        Prepare training, validation, and test datasets
        """
        try:
            train_data = {}
            val_data = {}
            test_data = {}
            
            # Convert date strings to datetime objects
            train_start = pd.to_datetime(self.config['training']['train_start_date'])
            train_end = pd.to_datetime(self.config['training']['train_end_date'])
            val_start = pd.to_datetime(self.config['training']['val_start_date'])
            val_end = pd.to_datetime(self.config['training']['val_end_date'])
            test_start = pd.to_datetime(self.config['training']['test_start_date'])
            test_end = pd.to_datetime(self.config['training']['test_end_date'])
            
            for symbol, df in data.items():
                # Ensure timestamp column is datetime
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # Split data into train, validation, and test sets
                train_mask = (df['timestamp'] >= train_start) & (df['timestamp'] <= train_end)
                val_mask = (df['timestamp'] >= val_start) & (df['timestamp'] <= val_end)
                test_mask = (df['timestamp'] >= test_start) & (df['timestamp'] <= test_end)
                
                train_data[symbol] = df[train_mask].copy()
                val_data[symbol] = df[val_mask].copy()
                test_data[symbol] = df[test_mask].copy()
                
                # Log the shapes of the splits
                self.logger.info(
                    f"Data split for {symbol}: "
                    f"Train: {train_data[symbol].shape}, "
                    f"Val: {val_data[symbol].shape}, "
                    f"Test: {test_data[symbol].shape}"
                )
            
            self.logger.info("Data preparation completed successfully")
            return train_data, val_data, test_data
        
        except Exception as e:
            self.logger.error(f"Error preparing data: {str(e)}")
            raise
    
    def _preprocess_features(self, df: pd.DataFrame, feature_cols: List[str], scaler: Optional[StandardScaler] = None) -> Tuple[np.ndarray, StandardScaler]:
        """
        Preprocess features with proper handling of NaN and infinite values
        """
        # Extract features
        features = df[feature_cols].values
        
        # Replace infinite values with NaN
        features = np.where(np.isinf(features), np.nan, features)
        
        # Calculate statistics for each column
        column_means = np.nanmean(features, axis=0)
        column_stds = np.nanstd(features, axis=0)
        
        # Handle zero standard deviation and NaN values
        column_stds = np.where(column_stds == 0, 1, column_stds)
        column_means = np.where(np.isnan(column_means), 0, column_means)
        
        # Replace NaN with column means
        for i in range(features.shape[1]):
            mask = np.isnan(features[:, i])
            features[mask, i] = column_means[i]
        
        # Clip extreme values (3 standard deviations)
        for i in range(features.shape[1]):
            lower_bound = column_means[i] - 3 * column_stds[i]
            upper_bound = column_means[i] + 3 * column_stds[i]
            features[:, i] = np.clip(features[:, i], lower_bound, upper_bound)
        
        # Normalize features
        if scaler is None:
            scaler = StandardScaler()
            features = scaler.fit_transform(features)
        else:
            features = scaler.transform(features)
        
        return features, scaler
    
    async def _train_quantum_nn(self,
                               train_data: Dict[str, pd.DataFrame],
                               val_data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Train Quantum Neural Network
        """
        try:
            # Initialize metrics tracking
            best_val_loss = float('inf')
            patience_counter = 0
            train_losses = []
            val_losses = []
            
            # Get training parameters from config
            batch_size = self.config['quantum_nn']['batch_size']
            n_epochs = self.config['quantum_nn']['epochs']
            patience = self.config['training']['early_stopping_patience']
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Convert DataFrames to feature arrays
            train_features = []
            train_targets = []
            val_features = []
            val_targets = []
            
            for symbol in train_data.keys():
                # Extract features and targets
                train_df = train_data[symbol]
                val_df = val_data[symbol]
                
                # Get feature columns
                feature_cols = [col for col in train_df.columns if col not in ['timestamp', 'target']]
                
                # Preprocess features
                train_feat, scaler = self._preprocess_features(train_df, feature_cols)
                val_feat, _ = self._preprocess_features(val_df, feature_cols, scaler)
                
                # Prepare targets
                train_targets.append(train_df['target'].values.reshape(-1, 1))
                val_targets.append(val_df['target'].values.reshape(-1, 1))
                
                # Append features
                train_features.append(train_feat)
                val_features.append(val_feat)
            
            # Concatenate features and targets
            train_features = np.concatenate(train_features)
            train_targets = np.concatenate(train_targets)
            val_features = np.concatenate(val_features)
            val_targets = np.concatenate(val_targets)
            
            # Prepare data loaders
            train_dataset = TensorDataset(
                torch.tensor(train_features, dtype=torch.float32),
                torch.tensor(train_targets, dtype=torch.float32)
            )
            val_dataset = TensorDataset(
                torch.tensor(val_features, dtype=torch.float32),
                torch.tensor(val_targets, dtype=torch.float32)
            )
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            
            # Initialize model
            from .quantum_nn import QuantumNeuralNetwork
            model = QuantumNeuralNetwork(
                input_dim=train_features.shape[1],
                hidden_dims=self.config['quantum_nn']['hidden_dims'],
                n_qubits=self.config['quantum_nn']['n_qubits'],
                n_layers=self.config['quantum_nn']['quantum_layers'],
                output_dim=1  # Binary classification
            ).to(device)
            
            # Initialize optimizer
            optimizer = optim.Adam(
                model.parameters(),
                lr=self.config['quantum_nn']['learning_rate'],
                weight_decay=0.01
            )
            
            criterion = nn.MSELoss()
            
            # Training loop
            for epoch in range(n_epochs):
                model.train()
                epoch_train_loss = 0.0
                
                # Training phase
                for batch_features, batch_labels in train_loader:
                    batch_features = batch_features.to(device)
                    batch_labels = batch_labels.to(device)
                    
                    optimizer.zero_grad()
                    outputs = model(batch_features)
                    loss = criterion(outputs, batch_labels)
                    
                    # Check for NaN loss
                    if torch.isnan(loss):
                        self.logger.warning(f"NaN loss detected in epoch {epoch+1}")
                        continue
                    
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    epoch_train_loss += loss.item()
                
                # Validation phase
                model.eval()
                epoch_val_loss = 0.0
                
                with torch.no_grad():
                    for batch_features, batch_labels in val_loader:
                        batch_features = batch_features.to(device)
                        batch_labels = batch_labels.to(device)
                        
                        outputs = model(batch_features)
                        val_loss = criterion(outputs, batch_labels)
                        epoch_val_loss += val_loss.item()
                
                # Calculate average losses
                avg_train_loss = epoch_train_loss / len(train_loader)
                avg_val_loss = epoch_val_loss / len(val_loader)
                
                train_losses.append(avg_train_loss)
                val_losses.append(avg_val_loss)
                
                # Log progress
                self.logger.info(
                    f"Epoch {epoch+1}/{n_epochs} - "
                    f"Train Loss: {avg_train_loss:.6f} - "
                    f"Val Loss: {avg_val_loss:.6f}"
                )
                
                # Early stopping check
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    
                    # Save best model
                    model_save_path = os.path.join(
                        self.config['training']['model_checkpoint_path'],
                        'quantum_nn_best.pth'
                    )
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_loss': avg_train_loss,
                        'val_loss': avg_val_loss,
                        'config': model.get_config(),
                        'scaler': scaler
                    }, model_save_path)
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        self.logger.info(
                            f"Early stopping triggered after {epoch+1} epochs"
                        )
                        break
            
            # Save training history
            history = {
                'train_loss': train_losses,
                'val_loss': val_losses,
                'best_val_loss': best_val_loss,
                'epochs_trained': epoch + 1
            }
            
            return history, model
        
        except Exception as e:
            self.logger.error(f"Error in quantum neural network training: {str(e)}")
            raise
    
    async def _train_rl_agent(self,
                             train_data: Dict[str, pd.DataFrame],
                             val_data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Train RL Agent
        """
        try:
            # Initialize scaler for consistency
            scaler = None
            train_features_list = []
            val_features_list = []
            train_returns_list = []
            val_returns_list = []
            
            for symbol in train_data.keys():
                train_df = train_data[symbol]
                val_df = val_data[symbol]
                
                # Get feature columns (exclude timestamp and target)
                feature_cols = [col for col in train_df.columns if col not in ['timestamp', 'target']]
                
                # Preprocess features using the same method as Quantum NN
                train_feat, scaler = self._preprocess_features(train_df, feature_cols, scaler)
                val_feat, _ = self._preprocess_features(val_df, feature_cols, scaler)
                
                # Ensure no NaN/infinite values
                train_feat = np.nan_to_num(train_feat, nan=0.0, posinf=1.0, neginf=-1.0)
                val_feat = np.nan_to_num(val_feat, nan=0.0, posinf=1.0, neginf=-1.0)
                
                train_features_list.append(train_feat)
                val_features_list.append(val_feat)
                
                # Calculate returns safely
                train_prices = train_df['close'].values if 'close' in train_df else train_df[feature_cols[-1]].values
                val_prices = val_df['close'].values if 'close' in val_df else val_df[feature_cols[-1]].values
                
                # Handle zero prices
                train_prices = np.where(train_prices == 0, 1e-10, train_prices)
                val_prices = np.where(val_prices == 0, 1e-10, val_prices)
                
                # Calculate price differences
                train_diff = np.diff(train_prices, prepend=train_prices[0])
                val_diff = np.diff(val_prices, prepend=val_prices[0])
                
                # Calculate returns
                train_returns = train_diff / train_prices
                val_returns = val_diff / val_prices
                
                # Handle any remaining NaN/infinite values
                train_returns = np.nan_to_num(train_returns, nan=0.0, posinf=1.0, neginf=-1.0)
                val_returns = np.nan_to_num(val_returns, nan=0.0, posinf=1.0, neginf=-1.0)
                
                train_returns_list.append(train_returns)
                val_returns_list.append(val_returns)
                
                # Log feature statistics for debugging
                logger.debug(f"Processed features for {symbol}:")
                logger.debug(f"Train features shape: {train_feat.shape}")
                logger.debug(f"Train features min: {np.min(train_feat)}, max: {np.max(train_feat)}")
                logger.debug(f"Train returns min: {np.min(train_returns)}, max: {np.max(train_returns)}")
            
            # Concatenate features and returns
            train_features = np.concatenate(train_features_list)
            val_features = np.concatenate(val_features_list)
            train_returns = np.concatenate(train_returns_list)
            val_returns = np.concatenate(val_returns_list)
            
            # Final validation of concatenated data
            if not np.all(np.isfinite(train_features)) or not np.all(np.isfinite(val_features)):
                raise ValueError("Concatenated features contain NaN or infinite values")
            if not np.all(np.isfinite(train_returns)) or not np.all(np.isfinite(val_returns)):
                raise ValueError("Concatenated returns contain NaN or infinite values")
            
            # Prepare data dictionary
            train_dict = {
                'features': train_features,
                'returns': train_returns
            }
            val_dict = {
                'features': val_features,
                'returns': val_returns
            }
            
            # Initialize metrics
            metrics = {
                'episode_rewards': [],
                'episode_lengths': [],
                'val_rewards': []
            }
            
            # Initialize RL agent
            from .rl_agent import RLAgent
            agent = RLAgent(
                state_size=train_features.shape[1],
                action_size=3,  # Buy, Sell, Hold
                memory_size=self.config['rl_agent']['memory_size'],
                batch_size=self.config['rl_agent']['batch_size'],
                gamma=self.config['rl_agent']['gamma'],
                epsilon=self.config['rl_agent']['epsilon'],
                epsilon_min=self.config['rl_agent']['epsilon_min'],
                epsilon_decay=self.config['rl_agent']['epsilon_decay'],
                learning_rate=self.config['rl_agent']['learning_rate']
            )
            
            # Training loop
            batch_metrics = await agent.train(train_dict, val_dict)
            
            # Update metrics
            metrics.update(batch_metrics)
            
            # Save model
            model_path = os.path.join(self.config['training']['model_checkpoint_path'], 'rl_agent.zip')
            agent.save(model_path)
            
            return metrics, agent
        
        except Exception as e:
            logger.error(f"Error training RL agent: {str(e)}")
            raise
    
    async def _train_risk_management(self,
                                    train_data: Dict[str, pd.DataFrame],
                                    val_data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Train Risk Management System
        """
        try:
            # Convert data to numpy arrays
            train_features = np.concatenate([df.drop(['timestamp', 'target'], axis=1).values 
                                          for df in train_data.values()])
            val_features = np.concatenate([df.drop(['timestamp', 'target'], axis=1).values 
                                        for df in val_data.values()])
            
            # Calculate returns and volatility for risk metrics
            train_returns = np.diff(train_features[:, 3]) / train_features[:-1, 3]  # Using close price
            val_returns = np.diff(val_features[:, 3]) / val_features[:-1, 3]
            
            train_volatility = np.std(train_returns, axis=0)
            val_volatility = np.std(val_returns, axis=0)
            
            # Prepare data dictionary
            train_dict = {
                'features': train_features,
                'returns': train_returns,
                'volatility': train_volatility
            }
            val_dict = {
                'features': val_features,
                'returns': val_returns,
                'volatility': val_volatility
            }
            
            # Initialize metrics
            metrics = {
                'train_loss': [],
                'val_loss': [],
                'var_estimates': [],
                'risk_scores': []
            }
            
            # Initialize risk management system
            from .risk_management import RiskManagementSystem
            risk_system = RiskManagementSystem(
                lookback_period=self.config.lookback_period,
                n_qubits=self.config.n_qubits,
                max_risk=self.config.max_risk,
                var_confidence=self.config.var_confidence,
                learning_rate=self.config.learning_rate
            )
            
            # Training loop
            batch_metrics = await risk_system.train(train_dict, val_dict)
            
            # Update metrics
            metrics.update(batch_metrics)
            
            # Save model
            model_path = os.path.join(self.config.model_save_path, 'risk_management.pth')
            risk_system.save(model_path)
            
            return metrics
        
        except Exception as e:
            logger.error(f"Error training risk management system: {str(e)}")
            raise
    
    async def _train_multi_agent(self,
                                train_data: Dict[str, pd.DataFrame],
                                val_data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Train Multi-Agent System
        """
        try:
            # Implement multi-agent system training
            metrics = {
                'consensus_accuracy': [],
                'individual_performance': {},
                'system_performance': []
            }
            
            # Training loop implementation will go here
            
            return metrics
        
        except Exception as e:
            logger.error(f"Error training multi-agent system: {str(e)}")
            raise
    
    def save_training_results(self, results: Dict, filename: str = 'training_results.json'):
        """
        Save training results
        """
        try:
            filepath = os.path.join(self.config.model_save_path, filename)
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=4, default=str)
            
            logger.info(f"Training results saved to {filepath}")
        
        except Exception as e:
            logger.error(f"Error saving training results: {str(e)}")
            raise
    
    def load_training_results(self, filename: str = 'training_results.json') -> Dict:
        """
        Load training results
        """
        try:
            filepath = os.path.join(self.config.model_save_path, filename)
            with open(filepath, 'r') as f:
                results = json.load(f)
            
            logger.info(f"Training results loaded from {filepath}")
            return results
        
        except Exception as e:
            logger.error(f"Error loading training results: {str(e)}")
            raise 