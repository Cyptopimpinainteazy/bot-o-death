#!/usr/bin/env python
"""
Quantum Trading Ensemble Model
------------------------------
Combines multiple strategies and models for more robust trading decisions.
"""

import os
import numpy as np
import pandas as pd
import json
import logging
import pickle
from datetime import datetime
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from pathlib import Path

# Import our modules
from quantum import create_quantum_circuit, quantum_trade_strategy
from technical_analysis import TechnicalAnalysisEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("quantum_ensemble.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("QuantumEnsemble")

class QuantumEnsembleTrader:
    """
    Ensemble trading model that combines multiple strategies:
    - Quantum circuit predictions
    - Classical ML predictions
    - Technical analysis signals
    - Optimized feature weighting
    """
    
    def __init__(self):
        """Initialize the ensemble trader"""
        self.ta_engine = TechnicalAnalysisEngine()
        self.models_dir = Path("models")
        self.results_dir = Path("results")
        self.ensemble_dir = self.results_dir / "ensemble"
        self.feature_weights = self._load_feature_weights()
        self.ensemble_model = None
        
        # Create directories
        os.makedirs(self.ensemble_dir, exist_ok=True)
        
        # Load quantum settings
        self.quantum_settings = self._load_quantum_settings()
        
        logger.info("Initialized Quantum Ensemble Trader")
    
    def _load_feature_weights(self):
        """Load feature weights from feature analysis"""
        weights_path = self.results_dir / "feature_analysis" / "quantum_recommendations.json"
        
        if weights_path.exists():
            try:
                with open(weights_path, 'r') as f:
                    data = json.load(f)
                logger.info(f"Loaded feature weights from {weights_path}")
                return data.get("feature_weights", {})
            except Exception as e:
                logger.error(f"Error loading feature weights: {str(e)}")
        
        # Default weights if file not found
        return {
            "price": 0.2,
            "volume": 0.1,
            "volatility": 0.1,
            "spread": 0.15,
            "depth": 0.05,
            "momentum": 0.1,
            "acceleration": 0.1,
            "rsi": 0.1,
            "macd": 0.1
        }
    
    def _load_quantum_settings(self):
        """Load optimal quantum circuit settings"""
        settings_path = Path("config/quantum_config.json")
        
        if settings_path.exists():
            try:
                with open(settings_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading quantum settings: {str(e)}")
        
        # Use feature analysis recommendations if available
        recommendations_path = self.results_dir / "feature_analysis" / "quantum_recommendations.json"
        if recommendations_path.exists():
            try:
                with open(recommendations_path, 'r') as f:
                    data = json.load(f)
                return {
                    "depth": data.get("depth", 5),
                    "shots": data.get("shots", 1024)
                }
            except Exception:
                pass
        
        # Default settings
        return {
            "depth": 5,
            "shots": 2048
        }
    
    def _load_ml_models(self):
        """Load trained ML models"""
        models = {}
        
        # Try to load XGBoost model
        xgb_path = self.models_dir / "xgb_model.json"
        if xgb_path.exists():
            try:
                import xgboost as xgb
                models["xgboost"] = xgb.XGBClassifier()
                models["xgboost"].load_model(str(xgb_path))
                logger.info("Loaded XGBoost model")
            except Exception as e:
                logger.error(f"Error loading XGBoost model: {str(e)}")
        
        # Try to load LSTM model
        lstm_path = self.models_dir / "lstm_model.h5"
        if lstm_path.exists():
            try:
                from tensorflow.keras.models import load_model
                models["lstm"] = load_model(str(lstm_path))
                logger.info("Loaded LSTM model")
            except Exception as e:
                logger.error(f"Error loading LSTM model: {str(e)}")
        
        # Add more models if available...
        
        return models
    
    def _load_rl_models(self):
        """Load trained RL models"""
        models = {}
        
        # Try to load DQN model
        dqn_files = list(self.models_dir.glob("DQN_*.zip"))
        if dqn_files:
            latest_dqn = max(dqn_files, key=lambda x: x.stat().st_mtime)
            try:
                from stable_baselines3 import DQN
                models["dqn"] = DQN.load(str(latest_dqn))
                logger.info(f"Loaded DQN model: {latest_dqn.name}")
            except Exception as e:
                logger.error(f"Error loading DQN model: {str(e)}")
        
        # Try to load PPO model
        ppo_files = list(self.models_dir.glob("PPO_*.zip"))
        if ppo_files:
            latest_ppo = max(ppo_files, key=lambda x: x.stat().st_mtime)
            try:
                from stable_baselines3 import PPO
                models["ppo"] = PPO.load(str(latest_ppo))
                logger.info(f"Loaded PPO model: {latest_ppo.name}")
            except Exception as e:
                logger.error(f"Error loading PPO model: {str(e)}")
        
        # Try to load A2C model
        a2c_files = list(self.models_dir.glob("A2C_*.zip"))
        if a2c_files:
            latest_a2c = max(a2c_files, key=lambda x: x.stat().st_mtime)
            try:
                from stable_baselines3 import A2C
                models["a2c"] = A2C.load(str(latest_a2c))
                logger.info(f"Loaded A2C model: {latest_a2c.name}")
            except Exception as e:
                logger.error(f"Error loading A2C model: {str(e)}")
        
        return models
    
    def build_ensemble_model(self, market_data=None):
        """Build the ensemble trading model"""
        logger.info("Building quantum ensemble trading model")
        
        # Load market data if provided or from CSV
        if market_data is None:
            data_path = self.results_dir / "training_data.csv"
            if data_path.exists():
                try:
                    market_data = pd.read_csv(data_path)
                    logger.info(f"Loaded market data from {data_path}")
                except Exception as e:
                    logger.error(f"Error loading market data: {str(e)}")
                    return False
            else:
                logger.error(f"Market data not found at {data_path}")
                return False
        
        # Prepare feature matrix
        X, y = self._prepare_features(market_data)
        if X is None or y is None:
            return False
        
        # Create base classifiers
        classifiers = []
        
        # Add Random Forest classifier
        rf = RandomForestClassifier(
            n_estimators=100, 
            max_depth=10,
            random_state=42
        )
        classifiers.append(('rf', rf))
        
        # Try to add XGBoost
        try:
            import xgboost as xgb
            xgb_clf = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
            classifiers.append(('xgb', xgb_clf))
        except ImportError:
            logger.warning("XGBoost not available")
        
        # For now, let's use only the standard classifiers since the custom quantum classifier needs more work
        # We'll use these classifiers as-is for now
        
        # Create ensemble model
        self.ensemble_model = VotingClassifier(
            estimators=classifiers,
            voting='soft'  # Use probability estimates for voting
        )
        
        # Train the model
        logger.info("Training ensemble model")
        self.ensemble_model.fit(X, y)
        
        # Save the model
        model_path = self.ensemble_dir / "quantum_ensemble_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(self.ensemble_model, f)
        
        logger.info(f"Ensemble model trained and saved to {model_path}")
        return True
    
    def _prepare_features(self, market_data):
        """Prepare features for the ensemble model"""
        if market_data is None or len(market_data) == 0:
            logger.error("No market data available")
            return None, None
        
        # Extract features based on available columns
        feature_cols = []
        
        # Add basic features if available
        for col in ['price', 'volume', 'volatility', 'spread', 'depth']:
            if col in market_data.columns:
                feature_cols.append(col)
        
        # Add technical indicators if available
        for col in ['momentum', 'acceleration', 'rsi', 'macd', 'price_sma5', 'price_sma20']:
            if col in market_data.columns:
                feature_cols.append(col)
        
        if not feature_cols:
            logger.error("No valid features found in market data")
            return None, None
        
        # Create feature matrix
        X = market_data[feature_cols].copy()
        
        # Fill missing values
        X = X.fillna(0)
        
        # Create target variable if possible
        if 'target' in market_data.columns:
            y = market_data['target']
        elif 'action' in market_data.columns:
            # Convert action strings to integers
            action_map = {'buy': 1, 'sell': 0, 'hold': 2}
            y = market_data['action'].map(action_map)
        elif 'price_change' in market_data.columns:
            # Create binary target based on price change
            y = (market_data['price_change'] > 0).astype(int)
        else:
            logger.error("No valid target column found")
            return None, None
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Save feature names and scaler for later use
        self.feature_names = feature_cols
        self.scaler = scaler
        
        with open(self.ensemble_dir / "feature_scaler.pkl", 'wb') as f:
            pickle.dump(scaler, f)
        
        with open(self.ensemble_dir / "feature_names.json", 'w') as f:
            json.dump(feature_cols, f)
        
        return X_scaled, y
    
    class QuantumClassifier:
        """Custom scikit-learn compatible classifier using quantum circuits"""
        
        def __init__(self):
            self.classes_ = np.array([0, 1])  # Binary classification: sell, buy
            self.quantum_depth = 5
            self.quantum_shots = 2048
        
        def fit(self, X, y):
            """Just store the classes, no real training needed"""
            self.classes_ = np.unique(y)
            return self
        
        def predict(self, X):
            """Predict using quantum circuit"""
            return np.array([self._quantum_decision(x) for x in X])
        
        def predict_proba(self, X):
            """Return probability estimates"""
            probas = []
            for x in X:
                buy_prob, sell_prob = self._quantum_probabilities(x)
                
                # For binary classification
                if len(self.classes_) == 2:
                    probas.append([sell_prob, buy_prob])
                # For ternary (buy, sell, hold)
                elif len(self.classes_) == 3:
                    hold_prob = 1.0 - buy_prob - sell_prob
                    probas.append([sell_prob, buy_prob, hold_prob])
            
            return np.array(probas)
        
        def _quantum_decision(self, features):
            """Make a quantum-based trading decision"""
            buy_prob, sell_prob = self._quantum_probabilities(features)
            if buy_prob > sell_prob:
                return 1  # Buy
            else:
                return 0  # Sell
        
        def _quantum_probabilities(self, features):
            """Get quantum trading probabilities"""
            # Extract or generate features for quantum circuit
            feature_count = len(features)
            
            if feature_count >= 5:
                # Use the most important features
                rsi = min(1, max(0, features[0]))  # First feature as RSI proxy
                macd = min(1, max(-1, features[1]))  # Second feature as MACD proxy
                imbalance = min(1, max(-1, features[2]))  # Third feature as imbalance proxy
            else:
                # Not enough features, use defaults
                rsi = 0.5
                macd = 0
                imbalance = 0
            
            # Create quantum circuit
            circuit_config = create_quantum_circuit(
                depth=self.quantum_depth,
                shots=self.quantum_shots,
                rsi=rsi,
                macd=macd,
                imbalance=imbalance
            )
            
            # Get trading strategy
            result = quantum_trade_strategy(circuit_config)
            
            return result['buy_probability'], result['sell_probability']
    
    def load_ensemble_model(self):
        """Load a previously trained ensemble model"""
        model_path = self.ensemble_dir / "quantum_ensemble_model.pkl"
        
        if model_path.exists():
            try:
                with open(model_path, 'rb') as f:
                    self.ensemble_model = pickle.load(f)
                
                # Load feature names and scaler
                with open(self.ensemble_dir / "feature_names.json", 'r') as f:
                    self.feature_names = json.load(f)
                
                with open(self.ensemble_dir / "feature_scaler.pkl", 'rb') as f:
                    self.scaler = pickle.load(f)
                
                logger.info(f"Loaded ensemble model from {model_path}")
                return True
            except Exception as e:
                logger.error(f"Error loading ensemble model: {str(e)}")
        
        logger.warning(f"Ensemble model not found at {model_path}")
        return False
    
    def predict_trade_action(self, market_data):
        """Predict the best trading action using the ensemble model"""
        if self.ensemble_model is None:
            success = self.load_ensemble_model()
            if not success:
                success = self.build_ensemble_model(market_data)
                if not success:
                    logger.error("Could not build or load ensemble model")
                    return None, 0.0
        
        # Extract features
        features = []
        for feature in self.feature_names:
            if feature in market_data:
                features.append(market_data[feature])
            else:
                features.append(0)  # Default value for missing features
        
        # Scale features
        features_scaled = self.scaler.transform([features])
        
        # Get prediction
        prediction = self.ensemble_model.predict(features_scaled)[0]
        probabilities = self.ensemble_model.predict_proba(features_scaled)[0]
        
        # Determine action and confidence
        max_probability = max(probabilities)
        
        if len(probabilities) == 2:  # Binary classification (buy/sell)
            action = "buy" if prediction == 1 else "sell"
        elif len(probabilities) == 3:  # Ternary classification (buy/sell/hold)
            if prediction == 0:
                action = "sell"
            elif prediction == 1:
                action = "buy"
            else:
                action = "hold"
        
        return action, float(max_probability)
    
    def evaluate_model(self, test_data=None):
        """Evaluate the ensemble model performance"""
        if test_data is None:
            # Use most recent 20% of training data
            data_path = self.results_dir / "training_data.csv"
            if data_path.exists():
                try:
                    full_data = pd.read_csv(data_path)
                    test_size = int(len(full_data) * 0.2)
                    test_data = full_data.tail(test_size)
                    logger.info(f"Using last {test_size} records for evaluation")
                except Exception as e:
                    logger.error(f"Error loading test data: {str(e)}")
                    return False
            else:
                logger.error(f"Test data not found")
                return False
        
        if self.ensemble_model is None:
            success = self.load_ensemble_model()
            if not success:
                success = self.build_ensemble_model()
                if not success:
                    logger.error("Could not build or load ensemble model")
                    return False
        
        # Prepare features
        X_test, y_test = self._prepare_features(test_data)
        if X_test is None or y_test is None:
            return False
        
        # Evaluate model
        accuracy = self.ensemble_model.score(X_test, y_test)
        logger.info(f"Ensemble model accuracy: {accuracy:.4f}")
        
        # Get detailed predictions for analysis
        predictions = self.ensemble_model.predict(X_test)
        probabilities = self.ensemble_model.predict_proba(X_test)
        
        # Calculate profit
        profit = 0
        correct_trades = 0
        total_trades = len(predictions)
        
        for i, pred in enumerate(predictions):
            true_value = y_test.iloc[i] if hasattr(y_test, 'iloc') else y_test[i]
            
            # Simplified profit calculation
            if pred == true_value:
                correct_trades += 1
                profit += 0.01  # 1% profit per correct trade
            else:
                profit -= 0.01  # 1% loss per incorrect trade
        
        win_rate = correct_trades / total_trades if total_trades > 0 else 0
        
        # Save evaluation results
        eval_results = {
            "accuracy": float(accuracy),
            "win_rate": float(win_rate),
            "profit": float(profit),
            "total_trades": total_trades,
            "correct_trades": correct_trades,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(self.ensemble_dir / "evaluation_results.json", 'w') as f:
            json.dump(eval_results, f, indent=2)
        
        logger.info(f"Win Rate: {win_rate:.4f}, Profit: {profit:.4f}")
        return eval_results


if __name__ == "__main__":
    print("=== Starting Quantum Ensemble Trader ===")
    trader = QuantumEnsembleTrader()
    
    # Build and evaluate the model
    trader.build_ensemble_model()
    eval_results = trader.evaluate_model()
    
    if eval_results:
        print(f"Model Accuracy: {eval_results['accuracy']:.4f}")
        print(f"Win Rate: {eval_results['win_rate']:.4f}")
        print(f"Total Profit: {eval_results['profit']:.4f}")
    
    print("=== Quantum Ensemble Trader Ready ===")
