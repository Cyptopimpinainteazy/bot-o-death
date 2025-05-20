#!/usr/bin/env python
"""
Quantum Circuit Hyperparameter Optimizer
----------------------------------------
Optimizes quantum circuit parameters for enhanced trading performance.
"""

import os
import numpy as np
import pandas as pd
import json
import logging
import optuna
from datetime import datetime
from quantum import create_quantum_circuit, quantum_trade_strategy
from technical_analysis import TechnicalAnalysisEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("quantum_hyperopt.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("QuantumHyperopt")

class QuantumCircuitOptimizer:
    """Optimize quantum circuit hyperparameters for maximum trading advantage"""
    
    def __init__(self, market_data_path=None, n_trials=50):
        """Initialize the optimizer"""
        self.market_data_path = market_data_path or "results/training_data.csv"
        self.results_dir = "results/quantum_tuning"
        self.n_trials = n_trials
        self.best_params = None
        os.makedirs(self.results_dir, exist_ok=True)
        self.ta_engine = TechnicalAnalysisEngine()
        logger.info(f"Initialized Quantum Circuit Optimizer with {n_trials} trials")
    
    def load_market_data(self):
        """Load market data for optimization"""
        if not os.path.exists(self.market_data_path):
            logger.error(f"Market data not found at {self.market_data_path}")
            return None
        
        try:
            df = pd.read_csv(self.market_data_path)
            logger.info(f"Loaded {len(df)} market data points from {self.market_data_path}")
            return df
        except Exception as e:
            logger.error(f"Error loading market data: {str(e)}")
            return None
    
    def prepare_test_scenarios(self, df, n_scenarios=20):
        """Prepare representative test scenarios from market data"""
        if 'chain' not in df.columns:
            logger.warning("No 'chain' column in data, treating all data as single chain")
            df['chain'] = 'default'
        
        scenarios = []
        
        # Get representative scenarios across different market conditions
        for chain in df['chain'].unique():
            chain_df = df[df['chain'] == chain].copy()
            
            # Add technical indicators if not present
            if 'rsi' not in chain_df.columns:
                chain_df = self.add_technical_indicators(chain_df)
            
            # Find scenarios with different market conditions
            # Bullish scenarios
            bullish = chain_df[chain_df['price'].pct_change(5) > 0.02].sample(min(5, len(chain_df)))
            # Bearish scenarios
            bearish = chain_df[chain_df['price'].pct_change(5) < -0.02].sample(min(5, len(chain_df)))
            # Sideways scenarios
            sideways = chain_df[
                (chain_df['price'].pct_change(5) > -0.01) & 
                (chain_df['price'].pct_change(5) < 0.01)
            ].sample(min(5, len(chain_df)))
            
            # Volatile scenarios
            if 'volatility' in chain_df.columns:
                volatile = chain_df.nlargest(5, 'volatility')
            else:
                volatile = chain_df.nlargest(5, 'price').head(2).append(chain_df.nsmallest(3, 'price'))
            
            # Combine all scenarios
            chain_scenarios = pd.concat([bullish, bearish, sideways, volatile])
            scenarios.append(chain_scenarios)
        
        # Combine all chain scenarios
        all_scenarios = pd.concat(scenarios)
        
        # If we have more scenarios than needed, sample them
        if len(all_scenarios) > n_scenarios:
            all_scenarios = all_scenarios.sample(n_scenarios)
        
        logger.info(f"Prepared {len(all_scenarios)} test scenarios for optimization")
        return all_scenarios
    
    def add_technical_indicators(self, df):
        """Add technical indicators to the dataframe if not present"""
        # Sort by timestamp if available
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp')
        
        # Add RSI if not present
        if 'rsi' not in df.columns and 'price' in df.columns:
            # Calculate RSI directly since we don't have a calculate_rsi method
            price = df['price'].values
            delta = np.diff(price)
            delta = np.append(delta, 0)
            
            # Calculate gains and losses
            gains = np.where(delta > 0, delta, 0)
            losses = np.where(delta < 0, -delta, 0)
            
            # Calculate average gains and losses over 14 periods
            avg_gain = np.zeros_like(gains)
            avg_loss = np.zeros_like(losses)
            
            for i in range(14, len(gains)):
                avg_gain[i] = np.mean(gains[i-14:i])
                avg_loss[i] = np.mean(losses[i-14:i])
            
            # Calculate RS and RSI
            rs = np.zeros_like(price)
            mask = avg_loss != 0
            rs[mask] = avg_gain[mask] / avg_loss[mask]
            rsi = 100 - (100 / (1 + rs))
            
            # Fill initial values
            rsi[:14] = 50
            df['rsi'] = rsi
        
        # Add MACD if not present
        if 'macd' not in df.columns and 'price' in df.columns:
            # Calculate MACD directly
            price = df['price'].values
            ema12 = np.zeros_like(price)
            ema26 = np.zeros_like(price)
            
            # Initialize
            ema12[0] = price[0]
            ema26[0] = price[0]
            
            # EMA calculation
            alpha12 = 2 / (12 + 1)
            alpha26 = 2 / (26 + 1)
            
            for i in range(1, len(price)):
                ema12[i] = price[i] * alpha12 + ema12[i-1] * (1 - alpha12)
                ema26[i] = price[i] * alpha26 + ema26[i-1] * (1 - alpha26)
            
            # MACD line
            macd = ema12 - ema26
            df['macd'] = macd
            
            # Normalize MACD to -1 to 1 range
            max_macd = max(abs(df['macd'].max()), abs(df['macd'].min()))
            if max_macd > 0:
                df['macd'] = df['macd'] / max_macd
        
        # Add order book imbalance if not present
        if 'imbalance' not in df.columns:
            if 'bids' in df.columns and 'asks' in df.columns:
                df['imbalance'] = (df['bids'] - df['asks']) / (df['bids'] + df['asks'])
            else:
                # Generate synthetic imbalance based on price changes
                df['imbalance'] = df['price'].pct_change(3)
                df['imbalance'] = df['imbalance'].fillna(0)
                # Normalize to -1 to 1
                max_imb = max(abs(df['imbalance'].max()), abs(df['imbalance'].min()))
                if max_imb > 0:
                    df['imbalance'] = df['imbalance'] / max_imb
        
        return df.fillna(0)
    
    def objective(self, trial):
        """Optuna objective function for circuit optimization"""
        # Get test scenarios
        if not hasattr(self, 'test_scenarios'):
            df = self.load_market_data()
            self.test_scenarios = self.prepare_test_scenarios(df)
        
        if self.test_scenarios is None or len(self.test_scenarios) == 0:
            logger.error("No test scenarios available for optimization")
            return float('-inf')
        
        # Hyperparameters to optimize
        depth = trial.suggest_int('depth', 2, 10)
        shots = trial.suggest_categorical('shots', [1024, 2048, 4096, 8192])
        
        # Circuit characteristics
        entanglement_strategy = trial.suggest_categorical('entanglement', ['linear', 'full', 'circular'])
        rotation_blocks = trial.suggest_int('rotation_blocks', 1, 4)
        
        # Feature mapping
        rsi_scaling = trial.suggest_float('rsi_scaling', 0.5, 2.0)
        macd_scaling = trial.suggest_float('macd_scaling', 0.5, 2.0)
        imbalance_scaling = trial.suggest_float('imbalance_scaling', 0.5, 2.0)
        
        # Run test scenarios
        profits = []
        
        for _, scenario in self.test_scenarios.iterrows():
            # Extract or compute features
            rsi = scenario.get('rsi', 0.5)
            macd = scenario.get('macd', 0)
            imbalance = scenario.get('imbalance', 0)
            
            # Apply feature scaling
            rsi = min(1, max(0, rsi * rsi_scaling))
            macd = min(1, max(-1, macd * macd_scaling))
            imbalance = min(1, max(-1, imbalance * imbalance_scaling))
            
            # Create quantum circuit
            circuit_config = create_quantum_circuit(
                depth=depth,
                shots=shots,
                rsi=rsi,
                macd=macd,
                imbalance=imbalance
            )
            
            # Add custom params to circuit config
            circuit_config['entanglement'] = entanglement_strategy
            circuit_config['rotation_blocks'] = rotation_blocks
            
            # Get trading strategy recommendation
            result = quantum_trade_strategy(circuit_config)
            
            # Calculate profit based on next price movement
            next_price_change = scenario.get('price_change', 0)
            if next_price_change == 0 and 'next_price' in scenario:
                next_price_change = (scenario['next_price'] / scenario['price']) - 1
            
            # Determine action based on probabilities
            if result['buy_probability'] > result['sell_probability']:
                action = 'buy'
                profit = next_price_change  # Profit is positive if price goes up after buying
            elif result['sell_probability'] > result['buy_probability']:
                action = 'sell'
                profit = -next_price_change  # Profit is positive if price goes down after selling
            else:
                action = 'hold'
                profit = 0
            
            profits.append(profit)
        
        # Calculate mean profit across all scenarios
        mean_profit = np.mean(profits)
        
        # Logging
        logger.info(f"Trial {trial.number}: mean_profit={mean_profit:.4f}, depth={depth}, shots={shots}")
        
        return mean_profit
    
    def run_optimization(self):
        """Run the hyperparameter optimization process"""
        logger.info(f"Starting quantum circuit hyperparameter optimization with {self.n_trials} trials")
        
        # Create Optuna study
        study = optuna.create_study(direction='maximize')
        
        try:
            # Run optimization
            study.optimize(self.objective, n_trials=self.n_trials)
            
            # Get best parameters
            self.best_params = study.best_params
            best_value = study.best_value
            
            logger.info(f"Best mean profit: {best_value:.4f}")
            logger.info(f"Best parameters: {self.best_params}")
            
            # Save results
            results = {
                "best_params": self.best_params,
                "best_value": float(best_value),
                "timestamp": datetime.now().isoformat(),
                "n_trials": self.n_trials
            }
            
            with open(f"{self.results_dir}/best_quantum_params.json", 'w') as f:
                json.dump(results, f, indent=2)
            
            # Create visualization
            try:
                from optuna.visualization import plot_param_importances, plot_optimization_history
                import matplotlib.pyplot as plt
                
                # Parameter importance
                fig = plot_param_importances(study)
                fig.write_image(f"{self.results_dir}/param_importance.png")
                
                # Optimization history
                fig = plot_optimization_history(study)
                fig.write_image(f"{self.results_dir}/optimization_history.png")
                
                logger.info(f"Saved optimization visualizations to {self.results_dir}")
            except Exception as e:
                logger.warning(f"Could not create visualizations: {str(e)}")
            
            return self.best_params
            
        except Exception as e:
            logger.error(f"Error in optimization: {str(e)}")
            return None
    
    def apply_best_params(self, config_file="config/quantum_config.json"):
        """Apply the best parameters to the quantum config file"""
        if self.best_params is None:
            logger.error("No best parameters available to apply")
            return False
        
        # Create config directory if needed
        os.makedirs(os.path.dirname(config_file), exist_ok=True)
        
        # Save config
        config = {
            "circuit": {
                "depth": self.best_params.get("depth", 5),
                "shots": self.best_params.get("shots", 2048),
                "entanglement": self.best_params.get("entanglement", "linear"),
                "rotation_blocks": self.best_params.get("rotation_blocks", 2)
            },
            "features": {
                "rsi_scaling": self.best_params.get("rsi_scaling", 1.0),
                "macd_scaling": self.best_params.get("macd_scaling", 1.0),
                "imbalance_scaling": self.best_params.get("imbalance_scaling", 1.0)
            },
            "last_updated": datetime.now().isoformat(),
            "optimization_value": study.best_value if hasattr(self, 'study') else None
        }
        
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Applied best parameters to {config_file}")
        return True


if __name__ == "__main__":
    print("=== Starting Quantum Circuit Hyperparameter Optimization ===")
    optimizer = QuantumCircuitOptimizer(n_trials=25)  # Use fewer trials for faster results
    best_params = optimizer.run_optimization()
    
    if best_params:
        optimizer.apply_best_params()
        print(f"=== Optimization Complete - Best Profit: {optimizer.best_value:.4f} ===")
    else:
        print("=== Optimization Failed ===")
