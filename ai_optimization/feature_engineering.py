"""
Feature Engineering for AI Trading Optimization

This module processes raw trading data into engineered features
suitable for machine learning models.
"""

import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class TradingFeatureEngineer:
    """
    Processes raw trading data into engineered features for ML models.
    """
    
    def __init__(self, feature_config=None):
        """
        Initialize the feature engineer with optional configuration
        
        Args:
            feature_config: Dictionary of feature engineering configuration
        """
        self.scalers = {}
        self.feature_config = feature_config or self._default_config()
        self.fitted = False
        
        logging.info("Trading Feature Engineer initialized")
    
    def _default_config(self):
        """Default feature engineering configuration"""
        return {
            'numerical_features': [
                'expected_profit', 
                'risk_score',
                'optimized_amount',
                'buy_price',
                'sell_price'
            ],
            'categorical_features': [
                'network',
                'type',
                'execution_venue',
                'buy_venue_type',
                'sell_venue_type'
            ],
            'target_features': [
                'standard_profit',
                'flashloan_profit'
            ],
            'scaling_method': 'standard',  # 'standard' or 'minmax'
            'drop_features': [
                'id',
                'timestamp'
            ]
        }
    
    def load_data(self, filepath):
        """
        Load data from CSV or JSON file
        
        Args:
            filepath: Path to data file
            
        Returns:
            Pandas DataFrame
        """
        try:
            if filepath.endswith('.csv'):
                df = pd.read_csv(filepath)
            elif filepath.endswith('.json'):
                with open(filepath, 'r') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    df = pd.DataFrame(data)
                else:
                    df = pd.DataFrame([data])
            else:
                raise ValueError("Unsupported file format. Use CSV or JSON.")
            
            logging.info(f"Loaded data with {len(df)} records and {len(df.columns)} columns")
            return df
        
        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            return pd.DataFrame()
    
    def process_features(self, df, fit=True):
        """
        Process raw data into model-ready features
        
        Args:
            df: Pandas DataFrame with raw data
            fit: Whether to fit scalers or use existing transformations
            
        Returns:
            X: Feature matrix (numpy array)
            y: Target variable (numpy array)
            feature_names: List of feature names
        """
        try:
            if df.empty:
                return None, None, []
            
            # Make a copy to avoid modifying the original
            data = df.copy()
            
            # Clean data
            data = self._clean_data(data)
            
            # Process numerical features
            num_features = [f for f in self.feature_config['numerical_features'] 
                            if f in data.columns]
            
            if num_features:
                if fit or not self.fitted:
                    if self.feature_config['scaling_method'] == 'standard':
                        self.scalers['numerical'] = StandardScaler()
                    else:
                        self.scalers['numerical'] = MinMaxScaler()
                    
                    data[num_features] = self.scalers['numerical'].fit_transform(
                        data[num_features].values
                    )
                else:
                    data[num_features] = self.scalers['numerical'].transform(
                        data[num_features].values
                    )
            
            # Process categorical features (one-hot encoding)
            cat_features = [f for f in self.feature_config['categorical_features'] 
                           if f in data.columns]
            
            encoded_cats = []
            for feature in cat_features:
                encoded = pd.get_dummies(data[feature], prefix=feature)
                encoded_cats.append(encoded)
                
                # Store unique categories during fit
                if fit or not self.fitted:
                    self.scalers[f'categories_{feature}'] = data[feature].unique().tolist()
            
            # Combine all features
            feature_dfs = [data[num_features]] + encoded_cats
            features = pd.concat(feature_dfs, axis=1)
            
            # Extract targets if they exist
            target_features = [f for f in self.feature_config['target_features'] 
                              if f in data.columns]
            
            if target_features:
                y = data[target_features].values
            else:
                y = None
            
            # Record feature names
            feature_names = features.columns.tolist()
            
            # Convert to numpy arrays
            X = features.values
            
            if fit:
                self.fitted = True
                
            logging.info(f"Processed {X.shape[1]} features from {len(data)} records")
            
            return X, y, feature_names
            
        except Exception as e:
            logging.error(f"Error processing features: {str(e)}")
            return None, None, []
    
    def _clean_data(self, df):
        """
        Clean and prepare data for feature engineering
        
        Args:
            df: Pandas DataFrame to clean
            
        Returns:
            Cleaned DataFrame
        """
        # Remove specified columns
        drop_cols = [f for f in self.feature_config['drop_features'] if f in df.columns]
        if drop_cols:
            df = df.drop(columns=drop_cols)
        
        # Handle missing values in numerical columns
        num_features = [f for f in self.feature_config['numerical_features'] 
                        if f in df.columns]
        
        for col in num_features:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
        
        # Handle missing values in categorical columns
        cat_features = [f for f in self.feature_config['categorical_features'] 
                        if f in df.columns]
        
        for col in cat_features:
            if df[col].isnull().any():
                df[col] = df[col].fillna('unknown')
        
        # Convert categorical variables to strings to ensure proper encoding
        for col in cat_features:
            df[col] = df[col].astype(str)
        
        return df
    
    def process_single_opportunity(self, opportunity):
        """
        Process a single trading opportunity for prediction
        
        Args:
            opportunity: Dictionary with opportunity data
            
        Returns:
            Feature vector (numpy array)
        """
        # Convert to DataFrame for consistent processing
        df = pd.DataFrame([opportunity])
        
        # Process using existing transformations
        X, _, _ = self.process_features(df, fit=False)
        
        return X
    
    def save_feature_config(self, filepath):
        """
        Save feature engineering configuration and scalers
        
        Args:
            filepath: Path to save configuration
            
        Returns:
            Boolean indicating success or failure
        """
        try:
            # We can't directly serialize sklearn objects
            # So we'll save the parameters instead
            config_to_save = {
                'feature_config': self.feature_config,
                'fitted': self.fitted
            }
            
            # For each scaler, save its parameters
            scaler_params = {}
            for key, scaler in self.scalers.items():
                if key == 'numerical':
                    if hasattr(scaler, 'mean_') and hasattr(scaler, 'scale_'):
                        scaler_params[key] = {
                            'type': self.feature_config['scaling_method'],
                            'mean': scaler.mean_.tolist() if hasattr(scaler, 'mean_') else None,
                            'scale': scaler.scale_.tolist() if hasattr(scaler, 'scale_') else None,
                            'min': scaler.data_min_.tolist() if hasattr(scaler, 'data_min_') else None,
                            'max': scaler.data_max_.tolist() if hasattr(scaler, 'data_max_') else None
                        }
                else:
                    scaler_params[key] = scaler
            
            config_to_save['scaler_params'] = scaler_params
            
            with open(filepath, 'w') as f:
                json.dump(config_to_save, f, indent=2)
                
            logging.info(f"Feature configuration saved to {filepath}")
            return True
            
        except Exception as e:
            logging.error(f"Error saving feature configuration: {str(e)}")
            return False
    
    def load_feature_config(self, filepath):
        """
        Load feature engineering configuration and scalers
        
        Args:
            filepath: Path to load configuration from
            
        Returns:
            Boolean indicating success or failure
        """
        try:
            with open(filepath, 'r') as f:
                config = json.load(f)
            
            self.feature_config = config['feature_config']
            self.fitted = config['fitted']
            
            # Reconstruct scalers
            scaler_params = config['scaler_params']
            self.scalers = {}
            
            for key, params in scaler_params.items():
                if key == 'numerical':
                    if params['type'] == 'standard':
                        scaler = StandardScaler()
                        if params['mean'] is not None and params['scale'] is not None:
                            scaler.mean_ = np.array(params['mean'])
                            scaler.scale_ = np.array(params['scale'])
                            scaler.n_features_in_ = len(params['mean'])
                    else:  # minmax
                        scaler = MinMaxScaler()
                        if params['min'] is not None and params['max'] is not None:
                            scaler.data_min_ = np.array(params['min'])
                            scaler.data_max_ = np.array(params['max'])
                            scaler.n_features_in_ = len(params['min'])
                    
                    self.scalers[key] = scaler
                else:
                    self.scalers[key] = params
            
            logging.info(f"Feature configuration loaded from {filepath}")
            return True
            
        except Exception as e:
            logging.error(f"Error loading feature configuration: {str(e)}")
            return False
    
    def analyze_feature_importance(self, df, model=None):
        """
        Analyze the importance of different features
        
        Args:
            df: Pandas DataFrame with trading data
            model: Trained model with feature_importances_ attribute
            
        Returns:
            DataFrame with feature importance
        """
        try:
            # Process features
            X, y, feature_names = self.process_features(df)
            
            if X is None or len(feature_names) == 0:
                return pd.DataFrame()
            
            # If model is provided and has feature_importances_
            if model is not None and hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                
                # Create DataFrame of feature importances
                feature_imp = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importances
                })
                
                # Sort by importance
                feature_imp = feature_imp.sort_values('importance', ascending=False)
                
                return feature_imp
            else:
                # Perform basic correlation analysis with targets
                if y is not None and y.shape[1] > 0:
                    df_corr = pd.DataFrame(X, columns=feature_names)
                    
                    # Add target columns
                    for i, target in enumerate(self.feature_config['target_features']):
                        if i < y.shape[1]:
                            df_corr[target] = y[:, i]
                    
                    # Calculate correlations with targets
                    correlations = {}
                    for target in self.feature_config['target_features']:
                        if target in df_corr.columns:
                            target_corrs = df_corr.corr()[target].sort_values(ascending=False)
                            correlations[target] = target_corrs
                    
                    # Feature importance based on average absolute correlation with targets
                    avg_corr = pd.DataFrame(index=feature_names)
                    
                    for target, corrs in correlations.items():
                        avg_corr[f'corr_{target}'] = corrs[feature_names].values
                    
                    avg_corr['avg_abs_corr'] = avg_corr.abs().mean(axis=1)
                    avg_corr = avg_corr.sort_values('avg_abs_corr', ascending=False)
                    
                    return avg_corr
                
                return pd.DataFrame({'feature': feature_names})
        
        except Exception as e:
            logging.error(f"Error analyzing feature importance: {str(e)}")
            return pd.DataFrame()
    
    def generate_dataset_stats(self, df):
        """
        Generate statistics about the dataset
        
        Args:
            df: Pandas DataFrame with trading data
            
        Returns:
            Dictionary with dataset statistics
        """
        try:
            stats = {}
            
            # Basic dataset info
            stats['record_count'] = len(df)
            stats['feature_count'] = len(df.columns)
            
            # Statistics for numerical features
            num_features = [f for f in self.feature_config['numerical_features'] 
                           if f in df.columns]
            
            if num_features:
                stats['numerical'] = {}
                
                for feature in num_features:
                    stats['numerical'][feature] = {
                        'mean': float(df[feature].mean()),
                        'median': float(df[feature].median()),
                        'min': float(df[feature].min()),
                        'max': float(df[feature].max()),
                        'std': float(df[feature].std())
                    }
            
            # Statistics for categorical features
            cat_features = [f for f in self.feature_config['categorical_features'] 
                           if f in df.columns]
            
            if cat_features:
                stats['categorical'] = {}
                
                for feature in cat_features:
                    value_counts = df[feature].value_counts()
                    stats['categorical'][feature] = {
                        'unique_count': len(value_counts),
                        'most_common': value_counts.index[0],
                        'most_common_count': int(value_counts.iloc[0]),
                        'distribution': {str(k): int(v) for k, v in value_counts.items()}
                    }
            
            # Statistics for target features
            target_features = [f for f in self.feature_config['target_features'] 
                              if f in df.columns]
            
            if target_features:
                stats['targets'] = {}
                
                for feature in target_features:
                    stats['targets'][feature] = {
                        'mean': float(df[feature].mean()),
                        'median': float(df[feature].median()),
                        'min': float(df[feature].min()),
                        'max': float(df[feature].max()),
                        'std': float(df[feature].std())
                    }
                
                # Calculate profit improvement statistics
                if 'standard_profit' in df.columns and 'flashloan_profit' in df.columns:
                    improvement = (df['flashloan_profit'] - df['standard_profit']) / df['standard_profit'].abs()
                    stats['profit_improvement'] = {
                        'mean': float(improvement.mean()),
                        'median': float(improvement.median()),
                        'min': float(improvement.min()),
                        'max': float(improvement.max()),
                        'std': float(improvement.std())
                    }
            
            return stats
            
        except Exception as e:
            logging.error(f"Error generating dataset stats: {str(e)}")
            return {}
