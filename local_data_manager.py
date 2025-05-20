import os
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import logging
from typing import Dict, List, Optional, Union
import json
import pickle
from polygon import RESTClient

class LocalDataManager:
    def __init__(self, config_path: str = 'config/local_config.yaml'):
        self.config_path = config_path
        self.load_config()
        self.logger = logging.getLogger(__name__)
        
        # Initialize Polygon REST client if API key is available
        polygon_api_key = os.getenv("POLYGON_API_KEY")
        if polygon_api_key:
            self.polygon_client = RESTClient(api_key=polygon_api_key)
        else:
            self.polygon_client = None
        
    def load_config(self):
        """Load local configuration"""
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            self.logger.info("Local configuration loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading local configuration: {str(e)}")
            raise
            
    def save_market_data(self, 
                        data: pd.DataFrame, 
                        symbol: str, 
                        timeframe: str = '1h') -> None:
        """Save market data to local storage"""
        try:
            directory = Path(self.config['data_storage']['market_data']['directory'])
            filename = f"{symbol}_{timeframe}.{self.config['data_storage']['market_data']['format']}"
            filepath = directory / filename
            
            data.to_csv(filepath, index=False)
            self.logger.info(f"Market data saved for {symbol} at {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving market data: {str(e)}")
            raise
            
    def load_market_data(self, 
                        symbol: str, 
                        timeframe: str = '1h') -> pd.DataFrame:
        """Load market data from local storage"""
        try:
            directory = Path(self.config['data_storage']['market_data']['directory'])
            filename = f"{symbol}_{timeframe}.{self.config['data_storage']['market_data']['format']}"
            filepath = directory / filename
            
            if not filepath.exists():
                raise FileNotFoundError(f"No data found for {symbol} at {filepath}")
                
            data = pd.read_csv(filepath)
            self.logger.info(f"Market data loaded for {symbol} from {filepath}")
            return data
            
        except Exception as e:
            self.logger.error(f"Error loading market data: {str(e)}")
            raise
            
    def save_model(self, 
                  model: object, 
                  model_name: str, 
                  version: str = 'latest') -> None:
        """Save model to local storage"""
        try:
            directory = Path(self.config['data_storage']['models']['directory'])
            filename = f"{model_name}_{version}.{self.config['data_storage']['models']['format']}"
            filepath = directory / filename
            
            if self.config['data_storage']['models']['format'] == 'pth':
                import torch
                torch.save(model.state_dict(), filepath)
            else:
                with open(filepath, 'wb') as f:
                    pickle.dump(model, f)
                    
            self.logger.info(f"Model {model_name} saved at {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise
            
    def load_model(self, 
                  model_name: str, 
                  version: str = 'latest') -> object:
        """Load model from local storage"""
        try:
            directory = Path(self.config['data_storage']['models']['directory'])
            filename = f"{model_name}_{version}.{self.config['data_storage']['models']['format']}"
            filepath = directory / filename
            
            if not filepath.exists():
                raise FileNotFoundError(f"No model found at {filepath}")
                
            if self.config['data_storage']['models']['format'] == 'pth':
                import torch
                model = torch.load(filepath)
            else:
                with open(filepath, 'rb') as f:
                    model = pickle.load(f)
                    
            self.logger.info(f"Model {model_name} loaded from {filepath}")
            return model
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise
            
    def save_training_results(self, 
                            results: Dict, 
                            model_name: str) -> None:
        """Save training results to local storage"""
        try:
            directory = Path(self.config['data_storage']['logs']['directory'])
            filename = f"{model_name}_training_results.json"
            filepath = directory / filename
            
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=4)
                
            self.logger.info(f"Training results saved for {model_name} at {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving training results: {str(e)}")
            raise
            
    def load_training_results(self, 
                            model_name: str) -> Dict:
        """Load training results from local storage"""
        try:
            directory = Path(self.config['data_storage']['logs']['directory'])
            filename = f"{model_name}_training_results.json"
            filepath = directory / filename
            
            if not filepath.exists():
                raise FileNotFoundError(f"No training results found at {filepath}")
                
            with open(filepath, 'r') as f:
                results = json.load(f)
                
            self.logger.info(f"Training results loaded for {model_name} from {filepath}")
            return results
            
        except Exception as e:
            self.logger.error(f"Error loading training results: {str(e)}")
            raise
            
    def clear_cache(self) -> None:
        """Clear temporary cache"""
        try:
            cache_dir = Path(self.config['local_settings']['cache_directory'])
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
                cache_dir.mkdir()
                self.logger.info("Cache cleared successfully")
                
        except Exception as e:
            self.logger.error(f"Error clearing cache: {str(e)}")
            raise

    def fetch_market_data_polygon(self, symbol: str, start: str, end: str, multiplier: int = 1, timespan: str = "day", limit: int = 50000) -> pd.DataFrame:
        """Fetch aggregated market data for a symbol from Polygon.io and return as a DataFrame."""
        if not self.polygon_client:
            raise RuntimeError("Polygon REST client is not initialized: missing POLYGON_API_KEY")
        try:
            # Fetch aggregates from Polygon
            aggs = self.polygon_client.list_aggs(
                ticker=symbol,
                multiplier=multiplier,
                timespan=timespan,
                from_=start,
                to=end,
                limit=limit
            )
            # Convert to list of dicts and DataFrame
            data = [agg.dict() for agg in aggs]
            df = pd.DataFrame(data)
            # Convert timestamp and rename columns
            df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
            df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'}, inplace=True)
            return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        except Exception as e:
            self.logger.error(f"Error fetching market data from Polygon: {str(e)}")
            raise 