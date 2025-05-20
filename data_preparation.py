import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import yaml
import logging
from datetime import datetime, timedelta
import asyncio
import os
from pathlib import Path
import talib
from sklearn.preprocessing import MinMaxScaler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketDataProcessor:
    def __init__(self, config_path: str = 'config/models_config.yaml'):
        # Initialize logger first
        self.logger = self._setup_logging()
        # Then load config
        self.config = self._load_config(config_path)
        self.scaler = MinMaxScaler()
        
    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger('MarketDataProcessor')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            self.logger.info("Configuration loaded successfully")
            return config
        except Exception as e:
            self.logger.error(f"Error loading config: {str(e)}")
            raise
            
    async def prepare_market_data(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Prepare market data for training."""
        try:
            processed_data = {}
            
            for symbol, df in data.items():
                self.logger.info(f"Processing data for {symbol}")
                
                # Ensure required columns exist
                required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                if not all(col in df.columns for col in required_cols):
                    raise ValueError(f"Missing required columns for {symbol}")
                
                # Calculate technical indicators
                df = self._calculate_technical_indicators(df)
                
                # Add additional features
                df = self._add_features(df)
                
                # Normalize features
                df = self._normalize_features(df)
                
                # Store processed data
                processed_data[symbol] = df
                
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Error preparing market data: {str(e)}")
            raise
            
    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators."""
        try:
            # RSI
            df['rsi'] = talib.RSI(df['close'], timeperiod=14)
            
            # MACD
            macd, signal, hist = talib.MACD(
                df['close'],
                fastperiod=12,
                slowperiod=26,
                signalperiod=9
            )
            df['macd'] = macd
            df['macd_signal'] = signal
            df['macd_hist'] = hist
            
            # Bollinger Bands
            upper, middle, lower = talib.BBANDS(
                df['close'],
                timeperiod=20,
                nbdevup=2,
                nbdevdn=2
            )
            df['bb_upper'] = upper
            df['bb_middle'] = middle
            df['bb_lower'] = lower
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating technical indicators: {str(e)}")
            raise
            
    def _add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add additional features."""
        try:
            # Price changes
            df['price_change'] = df['close'].pct_change()
            
            # Volume changes
            df['volume_change'] = df['volume'].pct_change()
            
            # Volatility
            df['volatility'] = df['close'].rolling(window=20).std()
            
            # Moving averages
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()
            df['sma_200'] = df['close'].rolling(window=200).mean()
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding features: {str(e)}")
            raise
            
    def _normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize features."""
        try:
            # Get feature columns (exclude timestamp and target)
            feature_cols = [col for col in df.columns if col not in ['timestamp', 'target']]
            
            # Replace infinite values with NaN
            df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
            
            # Calculate mean and std for each feature
            means = df[feature_cols].mean()
            stds = df[feature_cols].std()
            
            # Handle zero std
            stds = stds.replace(0, 1)
            
            # Normalize features
            df[feature_cols] = (df[feature_cols] - means) / stds
            
            # Replace NaN values with 0
            df[feature_cols] = df[feature_cols].fillna(0)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error normalizing features: {str(e)}")
            raise
            
    async def _load_market_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load market data from data sources
        """
        try:
            # This is a placeholder. Implement actual data loading logic
            # based on your data sources (e.g., CSV files, APIs, etc.)
            data = {}
            
            # Example: Load from CSV files
            data_dir = Path("data/market_data")
            for file in data_dir.glob("*.csv"):
                symbol = file.stem
                df = pd.read_csv(file)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # Convert numeric columns to float64
                numeric_columns = ['open', 'high', 'low', 'close', 'volume']
                for col in numeric_columns:
                    df[col] = df[col].astype(np.float64)
                
                data[symbol] = df
                self.logger.info(f"Loaded data for {symbol} with shape {df.shape}")
                
            return data
            
        except Exception as e:
            self.logger.error(f"Error loading market data: {str(e)}")
            raise
            
    async def _process_single_symbol(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process single symbol data
        """
        try:
            # Calculate technical indicators
            df = self._calculate_technical_indicators(df)
            
            # Normalize features
            df = self._normalize_features(df)
            
            # Add any additional features
            df = self._add_features(df)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error processing symbol data: {str(e)}")
            raise
            
    def prepare_training_data(self, df: pd.DataFrame, lookback: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for model training
        """
        try:
            # Select features for training
            feature_columns = [col for col in df.columns if col not in ['timestamp', 'target']]
            
            # Create sequences
            sequences = []
            targets = []
            
            for i in range(len(df) - lookback):
                sequence = df[feature_columns].iloc[i:(i + lookback)].values
                target = df['target'].iloc[i + lookback]
                sequences.append(sequence)
                targets.append(target)
                
            return np.array(sequences), np.array(targets)
            
        except Exception as e:
            self.logger.error(f"Error preparing training data: {str(e)}")
            raise 