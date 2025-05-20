#!/usr/bin/env python
"""
Quantum Feature Optimizer
-------------------------
Identifies and optimizes the most influential quantum features for trading.
"""

import os
import numpy as np
import pandas as pd
import json
import logging
from datetime import datetime
from sklearn.feature_selection import mutual_info_regression, SelectKBest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("feature_optimizer.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("FeatureOptimizer")

class QuantumFeatureOptimizer:
    """Optimize quantum feature selection and tuning for trading bots"""
    
    def __init__(self, data_path=None):
        """Initialize the optimizer"""
        self.data_path = data_path or "results/training_data.csv"
        self.results_dir = "results/feature_analysis"
        os.makedirs(self.results_dir, exist_ok=True)
        logger.info(f"Initialized Quantum Feature Optimizer")
    
    def load_data(self):
        """Load and prepare the training data"""
        if not os.path.exists(self.data_path):
            logger.error(f"Training data not found at {self.data_path}")
            return None
        
        try:
            df = pd.read_csv(self.data_path)
            logger.info(f"Loaded {len(df)} records from {self.data_path}")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return None
    
    def analyze_feature_importance(self, df, target='target'):
        """Analyze feature importance using mutual information"""
        if target not in df.columns:
            # If target is not available, use price change as target
            logger.info("Target column not found, calculating price change")
            for chain in df['chain'].unique():
                chain_df = df[df['chain'] == chain].sort_values('timestamp')
                df.loc[df['chain'] == chain, 'price_change'] = chain_df['price'].pct_change(1)
            
            target = 'price_change'
            df = df.dropna(subset=[target])
        
        # Select numeric features
        numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        numeric_features = [f for f in numeric_features if f != target and 'timestamp' not in f.lower()]
        
        if not numeric_features:
            logger.error("No numeric features found for analysis")
            return None
        
        # Prepare data
        X = df[numeric_features]
        y = df[target]
        
        # Handle missing values
        X = X.fillna(0)
        
        # Calculate mutual information scores
        mi_scores = mutual_info_regression(X, y)
        mi_results = pd.DataFrame({'Feature': numeric_features, 'MI_Score': mi_scores})
        mi_results = mi_results.sort_values('MI_Score', ascending=False)
        
        # Create visualization
        plt.figure(figsize=(12, len(numeric_features) * 0.4))
        sns.barplot(x='MI_Score', y='Feature', data=mi_results)
        plt.title('Feature Importance (Mutual Information)')
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/feature_importance.png")
        logger.info(f"Feature importance analysis saved to {self.results_dir}/feature_importance.png")
        
        # Save results to JSON
        with open(f"{self.results_dir}/feature_importance.json", 'w') as f:
            json.dump(mi_results.to_dict('records'), f, indent=2)
        
        return mi_results
    
    def find_optimal_parameters(self, df, top_features=5):
        """Find optimal parameters for quantum circuit based on top features"""
        # Analyze feature importance first
        importance = self.analyze_feature_importance(df)
        if importance is None:
            return None
        
        # Select top features
        top_features = min(top_features, len(importance))
        top_feature_names = importance['Feature'].tolist()[:top_features]
        logger.info(f"Top {top_features} features: {', '.join(top_feature_names)}")
        
        # Map features to quantum parameters
        quantum_params = {
            "depth": 5,  # Default
            "shots": 2048,  # Default
            "feature_weights": {}
        }
        
        # Calculate normalized weights for each feature
        total_importance = importance['MI_Score'].sum()
        if total_importance > 0:
            for feature, score in zip(importance['Feature'], importance['MI_Score']):
                quantum_params["feature_weights"][feature] = float(score / total_importance)
        
        # Generate recommended circuit depth based on data complexity
        feature_count = len(df.columns)
        if feature_count > 20:
            quantum_params["depth"] = 7
        elif feature_count > 10:
            quantum_params["depth"] = 5
        else:
            quantum_params["depth"] = 3
        
        # Generate recommended shots based on data size
        data_size = len(df)
        if data_size > 10000:
            quantum_params["shots"] = 4096
        elif data_size > 5000:
            quantum_params["shots"] = 2048
        else:
            quantum_params["shots"] = 1024
        
        # Save recommendations
        logger.info(f"Recommended quantum parameters: depth={quantum_params['depth']}, shots={quantum_params['shots']}")
        
        with open(f"{self.results_dir}/quantum_recommendations.json", 'w') as f:
            json.dump(quantum_params, f, indent=2)
        
        return quantum_params
    
    def run_optimization(self):
        """Run the complete optimization process"""
        logger.info("Starting quantum feature optimization")
        
        # Load data
        df = self.load_data()
        if df is None:
            return False
        
        # Find optimal parameters
        params = self.find_optimal_parameters(df)
        if params is None:
            return False
        
        logger.info("Optimization complete - see results in feature_optimizer.log")
        return True


if __name__ == "__main__":
    print("=== Starting Quantum Feature Optimization ===")
    optimizer = QuantumFeatureOptimizer()
    optimizer.run_optimization()
    print("=== Optimization Complete ===")
