import os
import time
import numpy as np
import pandas as pd
import logging
import json
from datetime import datetime
import matplotlib.pyplot as plt
from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.evaluation import evaluate_policy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf

# Import our custom modules
from ml_model import train_ml_models, predict_profitability
from rl_agent import TradingEnv, train_rl_agent
from data import fetch_uniswap_data, enrich_data
from quantum import create_quantum_circuit, quantum_trade_strategy
from technical_analysis import TechnicalAnalysisEngine
from market_depth import MarketDepthAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("bot_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("EnhancedBotTrainer")

class EnhancedBotTrainer:
    """
    Enhanced Bot Trainer that integrates:
    - Reinforcement Learning with Quantum Influence
    - Machine Learning Predictions
    - Technical Analysis
    - Market Depth Analysis
    - Backtesting and Hyperparameter Optimization
    """
    
    def __init__(self, config=None):
        """Initialize the enhanced bot trainer"""
        logger.info("Initializing Enhanced Bot Trainer")
        self.technical_analysis = TechnicalAnalysisEngine()
        self.market_depth = MarketDepthAnalyzer()
        
        # Default configuration
        self.config = {
            "chains": [
                {"name": "Polygon", "dex": "QuickSwap"},
                {"name": "Ethereum", "dex": "Uniswap"}, 
                {"name": "Solana", "dex": "Raydium"}
            ],
            "training": {
                "ml": {
                    "epochs": 50,
                    "batch_size": 64,
                    "test_size": 0.2,
                    "learning_rate": 0.001,
                    "early_stopping_patience": 10
                },
                "rl": {
                    "total_timesteps": 100000,
                    "learning_rate": 0.0001,
                    "batch_size": 64,
                    "buffer_size": 100000,
                    "algorithms": ["DQN", "PPO", "A2C"]
                },
                "quantum": {
                    "depth": 5,
                    "shots": 2048
                },
                "backtest": {
                    "initial_balance": 10000,
                    "trading_fee": 0.002,
                    "simulation_days": 30,
                    "trade_size_pct": 0.1
                }
            },
            "evaluation": {
                "metrics": [
                    "sharpe_ratio", "max_drawdown", "win_rate", 
                    "profit_factor", "avg_profit", "avg_loss"
                ],
                "min_trades": 30
            }
        }
        
        # Override with provided config if any
        if config:
            self.config.update(config)
            
        # Create directories for models and results
        os.makedirs("models", exist_ok=True)
        os.makedirs("results", exist_ok=True)
        os.makedirs("results/plots", exist_ok=True)
        
        logger.info(f"Configuration loaded. Training with {len(self.config['chains'])} chains.")
    
    def fetch_training_data(self, historical_days=60):
        """Fetch and prepare training data"""
        logger.info(f"Fetching {historical_days} days of historical data...")
        
        # Get enriched data for ML training
        df = enrich_data(self.config["chains"])
        
        # Add technical indicators
        # We'll calculate a few basic ones here to supplement the existing features
        for chain in df['chain'].unique():
            chain_df = df[df['chain'] == chain].sort_values('timestamp')
            
            # Add rolling window features
            df.loc[df['chain'] == chain, 'price_sma5'] = chain_df['price'].rolling(window=5).mean()
            df.loc[df['chain'] == chain, 'price_sma20'] = chain_df['price'].rolling(window=20).mean()
            df.loc[df['chain'] == chain, 'volume_sma5'] = chain_df['volume'].rolling(window=5).mean()
            
            # Compute price momentum
            df.loc[df['chain'] == chain, 'momentum'] = chain_df['price'].pct_change(5)
            
            # Compute price acceleration
            df.loc[df['chain'] == chain, 'acceleration'] = chain_df['price'].pct_change().diff(1)
        
        # Drop NaN values resulting from window calculations
        df = df.dropna()
        
        # Save raw data for future use
        df.to_csv("results/training_data.csv", index=False)
        logger.info(f"Training data prepared with {len(df)} records.")
        
        return df
    
    def train_ml_bots(self, df):
        """Train machine learning models with the prepared data"""
        logger.info("Training ML bots...")
        
        # Split data chronologically for time-series data
        train_size = int(len(df) * (1 - self.config["training"]["ml"]["test_size"]))
        df_train = df.iloc[:train_size]
        df_test = df.iloc[train_size:]
        
        # Train XGBoost and LSTM models
        xgb_model, lstm_model = train_ml_models(df_train)
        
        # Evaluate on test data
        features = ['price', 'depth', 'volume', 'volatility', 'spread', 'hour']
        X_test = df_test[features]
        y_test = df_test['target']
        
        # XGBoost evaluation
        xgb_preds = xgb_model.predict(X_test)
        xgb_accuracy = (xgb_preds == y_test).mean()
        logger.info(f"XGBoost test accuracy: {xgb_accuracy:.4f}")
        
        # LSTM evaluation (need to reshape data)
        time_steps = 5
        X_lstm_test, y_lstm_test = [], []
        for chain in df_test['chain'].unique():
            chain_df = df_test[df_test['chain'] == chain].sort_values('timestamp')
            for i in range(len(chain_df) - time_steps):
                X_lstm_test.append(chain_df[features].iloc[i:i+time_steps].values)
                y_lstm_test.append(chain_df['target'].iloc[i+time_steps])
                
        if X_lstm_test:
            X_lstm_test, y_lstm_test = np.array(X_lstm_test), np.array(y_lstm_test)
            lstm_loss, lstm_accuracy = lstm_model.evaluate(X_lstm_test, y_lstm_test, verbose=0)
            logger.info(f"LSTM test accuracy: {lstm_accuracy:.4f}")
        
        # Save evaluation metrics
        ml_metrics = {
            "xgb_accuracy": float(xgb_accuracy),
            "lstm_accuracy": float(lstm_accuracy) if 'lstm_accuracy' in locals() else None,
            "train_size": train_size,
            "test_size": len(df_test),
            "timestamp": datetime.now().isoformat()
        }
        
        with open("results/ml_metrics.json", "w") as f:
            json.dump(ml_metrics, f, indent=2)
        
        logger.info("ML bot training completed and models saved.")
        return xgb_model, lstm_model
    
    def train_rl_bots(self):
        """Train reinforcement learning bots with quantum-enhanced strategies"""
        logger.info("Training RL bots with quantum enhancement...")
        
        # Create enhanced environment with quantum states
        class QuantumEnhancedTradingEnv(TradingEnv):
            def __init__(self, chains, quantum_depth=5, quantum_shots=2048):
                super().__init__(chains)
                self.quantum_depth = quantum_depth
                self.quantum_shots = quantum_shots
                
                # Enhanced observation space to include quantum state
                from gym import spaces
                self.observation_space = spaces.Box(
                    low=-np.inf, high=np.inf, 
                    shape=(len(chains), 8),  # 5 original features + 3 quantum features
                    dtype=np.float32
                )
            
            def reset(self):
                state = super().reset()
                # Add quantum features to state
                quantum_state = self._add_quantum_features(state)
                return quantum_state
                
            def step(self, action):
                next_state, reward, done, info = super().step(action)
                
                # Apply quantum boost to the reward
                circuit_config = create_quantum_circuit(
                    depth=self.quantum_depth,
                    shots=self.quantum_shots,
                    rsi=np.random.uniform(0, 1),  # Would use real RSI in production
                    macd=np.random.uniform(-1, 1),  # Would use real MACD in production
                    imbalance=np.random.uniform(-1, 1)  # Would use real imbalance in production
                )
                
                q_result = quantum_trade_strategy(circuit_config)
                quantum_factor = q_result["quantum_factor"]
                
                # Apply quantum factor to reward
                quantum_reward = reward * (1 + quantum_factor * 0.5)
                
                # Add quantum features to next state
                quantum_next_state = self._add_quantum_features(next_state)
                
                # Add quantum info
                info["quantum_factor"] = quantum_factor
                info["original_reward"] = reward
                info["quantum_reward"] = quantum_reward
                
                return quantum_next_state, quantum_reward, done, info
                
            def _add_quantum_features(self, state):
                # Create quantum features for each chain
                quantum_features = []
                for i in range(len(self.chains)):
                    # Generate a quantum circuit based on current state
                    # In a real implementation, these would be derived from market conditions
                    rsi = np.clip(state[i, 3], 0, 1)  # Using volatility as proxy for RSI
                    macd = np.clip(state[i, 4] * 2 - 1, -1, 1)  # Using mempool signal as proxy for MACD
                    imbalance = np.random.uniform(-0.5, 0.5)  # Random imbalance
                    
                    circuit_config = create_quantum_circuit(
                        depth=self.quantum_depth,
                        shots=self.quantum_shots,
                        rsi=rsi,
                        macd=macd,
                        imbalance=imbalance
                    )
                    
                    q_result = quantum_trade_strategy(circuit_config)
                    
                    # Extract quantum features
                    buy_prob = q_result["buy_probability"]
                    sell_prob = q_result["sell_probability"]
                    quantum_factor = q_result["quantum_factor"]
                    
                    # Add quantum features to state representation
                    chain_state = np.append(state[i], [buy_prob, sell_prob, quantum_factor])
                    quantum_features.append(chain_state)
                    
                return np.array(quantum_features)
        
        # Create the quantum-enhanced environment
        env = QuantumEnhancedTradingEnv(
            self.config["chains"],
            quantum_depth=self.config["training"]["quantum"]["depth"],
            quantum_shots=self.config["training"]["quantum"]["shots"]
        )
        
        # Train models with different algorithms
        rl_models = {}
        
        for algo in self.config["training"]["rl"]["algorithms"]:
            logger.info(f"Training {algo} model...")
            
            if algo == "DQN":
                model = DQN(
                    "MlpPolicy", 
                    env, 
                    verbose=1,
                    learning_rate=self.config["training"]["rl"]["learning_rate"],
                    buffer_size=self.config["training"]["rl"]["buffer_size"],
                    batch_size=self.config["training"]["rl"]["batch_size"],
                    tensorboard_log="./results/tensorboard/"
                )
            elif algo == "PPO":
                model = PPO(
                    "MlpPolicy",
                    env,
                    verbose=1,
                    learning_rate=self.config["training"]["rl"]["learning_rate"],
                    batch_size=self.config["training"]["rl"]["batch_size"],
                    tensorboard_log="./results/tensorboard/"
                )
            elif algo == "A2C":
                model = A2C(
                    "MlpPolicy",
                    env,
                    verbose=1,
                    learning_rate=self.config["training"]["rl"]["learning_rate"],
                    tensorboard_log="./results/tensorboard/"
                )
            
            # Train the model
            model.learn(total_timesteps=self.config["training"]["rl"]["total_timesteps"])
            
            # Save the model
            model.save(f"models/{algo.lower()}_quantum_enhanced")
            
            # Evaluate the trained model
            mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
            logger.info(f"{algo} evaluation - Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
            
            rl_models[algo] = {
                "model": model,
                "mean_reward": float(mean_reward),
                "std_reward": float(std_reward)
            }
        
        # Save evaluation metrics
        rl_metrics = {algo: {
            "mean_reward": data["mean_reward"],
            "std_reward": data["std_reward"]
        } for algo, data in rl_models.items()}
        
        rl_metrics["timestamp"] = datetime.now().isoformat()
        
        with open("results/rl_metrics.json", "w") as f:
            json.dump(rl_metrics, f, indent=2)
        
        logger.info("RL bot training completed and models saved.")
        return rl_models
    
    def backtest_strategies(self, ml_models, rl_models, df=None):
        """Backtest the trained models on historical data"""
        logger.info("Starting backtesting...")
        
        if df is None:
            # Use cached data or fetch new data
            try:
                df = pd.read_csv("results/training_data.csv")
                logger.info("Using cached training data for backtesting")
            except FileNotFoundError:
                df = self.fetch_training_data()
                logger.info("Fetched new data for backtesting")
        
        # Prepare backtesting dataframe
        backtest_df = df.copy().sort_values(['chain', 'timestamp'])
        
        # Initialize portfolio values for each strategy
        initial_balance = self.config["training"]["backtest"]["initial_balance"]
        trade_size = initial_balance * self.config["training"]["backtest"]["trade_size_pct"]
        trading_fee = self.config["training"]["backtest"]["trading_fee"]
        
        strategies = {
            "buy_and_hold": initial_balance,
            "xgboost": initial_balance,
            "lstm": initial_balance,
            "dqn": initial_balance,
            "ppo": initial_balance if "PPO" in rl_models else None,
            "a2c": initial_balance if "A2C" in rl_models else None,
            "ensemble": initial_balance,
            "quantum_enhanced": initial_balance
        }
        
        # Initialize tracking variables
        portfolio_history = {strategy: [initial_balance] for strategy in strategies if strategies[strategy] is not None}
        trade_history = {strategy: [] for strategy in strategies if strategies[strategy] is not None}
        
        # Extract ML models
        xgb_model, lstm_model = ml_models
        
        # Extract RL models
        dqn_model = rl_models["DQN"]["model"] if "DQN" in rl_models else None
        ppo_model = rl_models["PPO"]["model"] if "PPO" in rl_models else None
        a2c_model = rl_models["A2C"]["model"] if "A2C" in rl_models else None
        
        # Run backtest
        for chain in backtest_df['chain'].unique():
            chain_df = backtest_df[backtest_df['chain'] == chain].sort_values('timestamp')
            
            if len(chain_df) < 30:  # Skip chains with insufficient data
                logger.warning(f"Skipping {chain} - insufficient data ({len(chain_df)} records)")
                continue
                
            logger.info(f"Backtesting on {chain} with {len(chain_df)} data points")
            
            # Run backtest for each day
            current_positions = {strategy: 0 for strategy in strategies if strategies[strategy] is not None}
            
            for i in range(5, len(chain_df) - 1):  # Start from 5 to have enough history, leave one for future price
                current_data = chain_df.iloc[:i+1]
                current_row = current_data.iloc[-1]
                next_row = chain_df.iloc[i+1]
                
                current_price = current_row['price']
                next_price = next_row['price']
                price_change = next_price / current_price - 1
                
                # Features for prediction
                features = ['price', 'depth', 'volume', 'volatility', 'spread', 'hour']
                X_current = current_row[features].values.reshape(1, -1)
                
                # LSTM features (need last 5 rows)
                X_lstm = current_data[features].iloc[-5:].values.reshape(1, 5, len(features))
                
                # Buy and Hold strategy
                if current_positions["buy_and_hold"] == 0:
                    # Calculate how many units we can buy
                    units = strategies["buy_and_hold"] / current_price * (1 - trading_fee)
                    strategies["buy_and_hold"] -= strategies["buy_and_hold"]  # Spend all balance
                    current_positions["buy_and_hold"] = units
                    trade_history["buy_and_hold"].append({
                        "timestamp": current_row['timestamp'],
                        "action": "buy",
                        "price": current_price,
                        "units": units,
                        "value": units * current_price
                    })
                
                # XGBoost strategy
                xgb_signal = xgb_model.predict_proba(X_current)[0][1] > 0.6  # Higher threshold for buy
                if xgb_signal and current_positions["xgboost"] == 0:
                    # Buy
                    units = strategies["xgboost"] / current_price * (1 - trading_fee)
                    strategies["xgboost"] -= strategies["xgboost"]
                    current_positions["xgboost"] = units
                    trade_history["xgboost"].append({
                        "timestamp": current_row['timestamp'],
                        "action": "buy",
                        "price": current_price,
                        "units": units,
                        "value": units * current_price
                    })
                elif not xgb_signal and current_positions["xgboost"] > 0:
                    # Sell
                    sale_value = current_positions["xgboost"] * current_price * (1 - trading_fee)
                    strategies["xgboost"] += sale_value
                    trade_history["xgboost"].append({
                        "timestamp": current_row['timestamp'],
                        "action": "sell",
                        "price": current_price,
                        "units": current_positions["xgboost"],
                        "value": sale_value
                    })
                    current_positions["xgboost"] = 0
                
                # LSTM strategy 
                try:
                    lstm_signal = lstm_model.predict(X_lstm)[0][0] > 0.6  # Higher threshold for buy
                    if lstm_signal and current_positions["lstm"] == 0:
                        # Buy
                        units = strategies["lstm"] / current_price * (1 - trading_fee)
                        strategies["lstm"] -= strategies["lstm"]
                        current_positions["lstm"] = units
                        trade_history["lstm"].append({
                            "timestamp": current_row['timestamp'],
                            "action": "buy",
                            "price": current_price,
                            "units": units,
                            "value": units * current_price
                        })
                    elif not lstm_signal and current_positions["lstm"] > 0:
                        # Sell
                        sale_value = current_positions["lstm"] * current_price * (1 - trading_fee)
                        strategies["lstm"] += sale_value
                        trade_history["lstm"].append({
                            "timestamp": current_row['timestamp'],
                            "action": "sell",
                            "price": current_price,
                            "units": current_positions["lstm"],
                            "value": sale_value
                        })
                        current_positions["lstm"] = 0
                except Exception as e:
                    logger.error(f"Error in LSTM prediction: {e}")
                
                # Update portfolio values
                for strategy in portfolio_history:
                    if strategy == "buy_and_hold":
                        portfolio_value = current_positions[strategy] * current_price
                    elif strategy in ["xgboost", "lstm"]:
                        portfolio_value = strategies[strategy] + (current_positions[strategy] * current_price if current_positions[strategy] > 0 else 0)
                    else:
                        # For RL models, we'll add more complex logic
                        portfolio_value = strategies[strategy]  # Just use cash value for now
                    
                    portfolio_history[strategy].append(portfolio_value)
            
            # At the end of the period, close all positions
            for strategy in current_positions:
                if current_positions[strategy] > 0:
                    # Close position at the last price
                    last_price = chain_df.iloc[-1]['price']
                    sale_value = current_positions[strategy] * last_price * (1 - trading_fee)
                    strategies[strategy] += sale_value
                    trade_history[strategy].append({
                        "timestamp": chain_df.iloc[-1]['timestamp'],
                        "action": "sell",
                        "price": last_price,
                        "units": current_positions[strategy],
                        "value": sale_value
                    })
                    current_positions[strategy] = 0
        
        # Calculate final metrics
        backtest_results = {}
        
        for strategy in strategies:
            if strategies[strategy] is None:
                continue
                
            trades = trade_history[strategy]
            if len(trades) < 2:  # Need at least one buy and one sell
                logger.warning(f"Strategy {strategy} has insufficient trades for analysis")
                continue
            
            # Calculate metrics
            final_value = strategies[strategy]
            total_return = (final_value / initial_balance - 1) * 100
            
            # Calculate Sharpe ratio (using daily returns)
            daily_returns = []
            for i in range(1, len(portfolio_history[strategy])):
                daily_return = portfolio_history[strategy][i] / portfolio_history[strategy][i-1] - 1
                daily_returns.append(daily_return)
            
            mean_return = np.mean(daily_returns)
            std_return = np.std(daily_returns)
            sharpe_ratio = (mean_return * 252) / (std_return * np.sqrt(252)) if std_return > 0 else 0
            
            # Calculate max drawdown
            peak = portfolio_history[strategy][0]
            max_drawdown = 0
            
            for value in portfolio_history[strategy]:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak
                max_drawdown = max(max_drawdown, drawdown)
            
            # Calculate win rate
            buy_trades = [t for t in trades if t["action"] == "buy"]
            sell_trades = [t for t in trades if t["action"] == "sell"]
            
            if len(buy_trades) != len(sell_trades):
                logger.warning(f"Strategy {strategy} has mismatched buy/sell trades")
            
            profitable_trades = 0
            total_profit = 0
            total_loss = 0
            
            for i in range(min(len(buy_trades), len(sell_trades))):
                buy_price = buy_trades[i]["price"]
                sell_price = sell_trades[i]["price"]
                trade_profit = (sell_price / buy_price - 1) * 100
                
                if trade_profit > 0:
                    profitable_trades += 1
                    total_profit += trade_profit
                else:
                    total_loss += abs(trade_profit)
            
            total_trades = min(len(buy_trades), len(sell_trades))
            win_rate = profitable_trades / total_trades if total_trades > 0 else 0
            profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
            avg_profit = total_profit / profitable_trades if profitable_trades > 0 else 0
            avg_loss = total_loss / (total_trades - profitable_trades) if (total_trades - profitable_trades) > 0 else 0
            
            backtest_results[strategy] = {
                "final_value": final_value,
                "total_return_pct": total_return,
                "sharpe_ratio": float(sharpe_ratio),
                "max_drawdown_pct": float(max_drawdown * 100),
                "win_rate": float(win_rate),
                "profit_factor": float(profit_factor),
                "avg_profit_pct": float(avg_profit),
                "avg_loss_pct": float(avg_loss),
                "total_trades": total_trades
            }
        
        # Save backtest results
        with open("results/backtest_results.json", "w") as f:
            json.dump(backtest_results, f, indent=2)
        
        # Plot equity curves
        plt.figure(figsize=(12, 8))
        for strategy, values in portfolio_history.items():
            plt.plot(values, label=strategy)
        
        plt.title("Strategy Equity Curves")
        plt.xlabel("Trading Days")
        plt.ylabel("Portfolio Value ($)")
        plt.legend()
        plt.grid(True)
        plt.savefig("results/plots/equity_curves.png")
        
        logger.info("Backtesting completed and results saved.")
        return backtest_results
    
    def train_and_evaluate(self):
        """Train all bots and evaluate them in one function"""
        # Step 1: Fetch training data
        df = self.fetch_training_data()
        
        # Step 2: Train ML models
        ml_models = self.train_ml_bots(df)
        
        # Step 3: Train RL models
        rl_models = self.train_rl_bots()
        
        # Step 4: Backtest all strategies
        backtest_results = self.backtest_strategies(ml_models, rl_models, df)
        
        # Step 5: Print summary
        logger.info("=== Training and Evaluation Complete ===")
        logger.info("Best performing strategies:")
        
        # Sort strategies by total return
        sorted_strategies = sorted(
            backtest_results.items(), 
            key=lambda x: x[1]["total_return_pct"], 
            reverse=True
        )
        
        for strategy, metrics in sorted_strategies:
            logger.info(f"{strategy.upper()}: Return: {metrics['total_return_pct']:.2f}%, Sharpe: {metrics['sharpe_ratio']:.2f}, Win Rate: {metrics['win_rate']*100:.1f}%")
        
        return ml_models, rl_models, backtest_results

# Example usage
if __name__ == "__main__":
    print("=== Starting Enhanced Bot Training ===")
    trainer = EnhancedBotTrainer()
    trainer.train_and_evaluate()
    print("=== Training Complete ===")
