import yaml
import logging
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
import pennylane as qml
from .rl_agent import RLAgent
from .quantum_nn import QuantumNeuralNetwork
from .risk_management import RiskManagementSystem
from .multi_agent import MultiAgentSystem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, config_path: str = 'config/models_config.yaml'):
        self.config_path = config_path
        self.load_config()
        self.models = {}
        self._initialize_models()
        
    def load_config(self):
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            logger.info("Configuration loaded successfully")
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            raise
    
    def _initialize_models(self):
        """Initialize all models with enhanced configuration"""
        try:
            # RL Agent with enhanced parameters
            self.models['rl_agent'] = RLAgent(
                state_size=self.config['rl_agent']['state_size'],
                action_size=self.config['rl_agent']['action_size'],
                memory_size=self.config['rl_agent']['memory_size'],
                batch_size=self.config['rl_agent']['batch_size'],
                gamma=self.config['rl_agent']['gamma'],
                epsilon=self.config['rl_agent']['epsilon'],
                epsilon_min=self.config['rl_agent']['epsilon_min'],
                epsilon_decay=self.config['rl_agent']['epsilon_decay'],
                learning_rate=self.config['rl_agent']['learning_rate']
            )
            
            # Quantum Neural Network with enhanced parameters
            self.models['quantum_nn'] = QuantumNeuralNetwork(
                n_qubits=self.config['quantum_nn']['n_qubits'],
                input_dim=self.config['quantum_nn']['input_dim'],
                hidden_dims=self.config['quantum_nn']['hidden_dims'],
                output_dim=self.config['quantum_nn']['output_dim'],
                learning_rate=self.config['quantum_nn']['learning_rate'],
                dropout_rate=self.config['quantum_nn']['dropout_rate']
            )
            
            # Risk Management System
            self.models['risk_management'] = RiskManagementSystem(
                lookback_period=self.config['risk_management']['lookback_period'],
                n_qubits=self.config['risk_management']['n_qubits'],
                max_risk=self.config['risk_management']['max_risk'],
                var_confidence=self.config['risk_management']['var_confidence'],
                learning_rate=self.config['risk_management']['learning_rate']
            )
            
            # Multi-Agent System
            self.models['multi_agent'] = MultiAgentSystem(
                n_agents=self.config['multi_agent']['n_agents'],
                agent_types=self.config['multi_agent']['agent_types'],
                state_size=self.config['multi_agent']['state_size'],
                action_size=self.config['multi_agent']['action_size'],
                learning_rate=self.config['multi_agent']['learning_rate']
            )
            
            logger.info("Models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
            raise
    
    def train_rl_agent(self, train_data: Dict[str, np.ndarray], val_data: Dict[str, np.ndarray]) -> Dict:
        """Train RL Agent"""
        try:
            metrics = self.models['rl_agent'].train(train_data, val_data)
            logger.info("RL Agent training completed")
            return metrics
        except Exception as e:
            logger.error(f"Error training RL Agent: {str(e)}")
            raise
    
    def train_quantum_nn(self, train_data: Dict[str, np.ndarray], val_data: Dict[str, np.ndarray]) -> Dict:
        """Train Quantum Neural Network"""
        try:
            metrics = self.models['quantum_nn'].train(train_data, val_data)
            logger.info("Quantum Neural Network training completed")
            return metrics
        except Exception as e:
            logger.error(f"Error training Quantum Neural Network: {str(e)}")
            raise
    
    def train_risk_management(self, train_data: Dict[str, np.ndarray], val_data: Dict[str, np.ndarray]) -> Dict:
        """Train Risk Management System"""
        try:
            metrics = self.models['risk_management'].train(train_data, val_data)
            logger.info("Risk Management System training completed")
            return metrics
        except Exception as e:
            logger.error(f"Error training Risk Management System: {str(e)}")
            raise
    
    def train_multi_agent(self, train_data: Dict[str, np.ndarray], val_data: Dict[str, np.ndarray]) -> Dict:
        """Train Multi-Agent System"""
        try:
            metrics = self.models['multi_agent'].train(train_data, val_data)
            logger.info("Multi-Agent System training completed")
            return metrics
        except Exception as e:
            logger.error(f"Error training Multi-Agent System: {str(e)}")
            raise
    
    def save_models(self, save_path: str = 'models/'):
        """Save all models"""
        try:
            Path(save_path).mkdir(parents=True, exist_ok=True)
            
            for model_name, model in self.models.items():
                model_path = Path(save_path) / f"{model_name}.pth"
                model.save(str(model_path))
                
            logger.info(f"Models saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Error saving models: {str(e)}")
            raise
    
    def load_models(self, load_path: str = 'models/'):
        """Load all models"""
        try:
            for model_name, model in self.models.items():
                model_path = Path(load_path) / f"{model_name}.pth"
                if model_path.exists():
                    model.load(str(model_path))
                    
            logger.info(f"Models loaded from {load_path}")
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise 