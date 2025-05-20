import gym
import numpy as np
from gym import spaces
from stable_baselines3 import DQN
import pandas as pd
from datetime import datetime
from data import fetch_uniswap_data, enrich_data
from quantum import quantum_trade_strategy, create_quantum_circuit
from quantum_enhancements import EnhancedQuantumTrading, extract_market_conditions
import random
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("rl_agent.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("RL_Agent")

CHAINS = [{'name': 'Polygon', 'dex': 'QuickSwap'}, {'name': 'Solana', 'dex': 'Raydium'}]

class TradingEnv(gym.Env):
    def __init__(self, chains=CHAINS):
        super().__init__()
        self.chains = chains
        self.action_space = spaces.Discrete(4)  # 0: Hold, 1: Buy, 2: Sell, 3: Buy+Sell (sandwich)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(len(chains), 5), dtype=np.float32)
        self.state = None
        self.step_count = 0
        self.max_steps = 100
        
        # Initialize portfolio tracking
        self.balance = 10000  # Starting balance in USD
        self.positions = {chain['name']: 0 for chain in chains}  # Holdings per chain
        
        # Initialize the enhanced quantum trading system
        self.quantum_system = EnhancedQuantumTrading()
        
        # Initialize technical indicators for market analysis
        self.indicators = {
            "rsi": 50,  # Neutral RSI
            "macd": 0,
            "macd_signal": 0,
            "order_imbalance": 0
        }
        
        logger.info("Trading environment initialized with enhanced quantum system")

    def reset(self):
        """Reset the environment to start a new episode"""
        # Reset portfolio
        self.balance = 10000
        self.positions = {chain['name']: 0 for chain in self.chains}
        
        # Reset indicators
        self.indicators = {
            "rsi": 50,
            "macd": 0,
            "macd_signal": 0,
            "order_imbalance": 0
        }
        
        # Reset step counter
        self.step_count = 0
        
        # Initialize transaction history
        self.transaction_history = []
        
        # Get fresh market data
        try:
            chain_data = enrich_data(self.chains)
        except Exception as e:
            # Fallback to basic data if enriched data fails
            logger.warning(f"Enriched data unavailable, falling back to basic data: {str(e)}")
            chain_data = None
            
        # Use fetch_uniswap_data directly if enrich_data isn't ready
        raw_data = fetch_uniswap_data(self.chains)
        
        # Calculate volatility based on recent price changes (mocked here)
        volatility = np.array([random.uniform(0.01, 0.2) for _ in self.chains])
        
        # Calculate mempool signals (mocked here)
        mempool_signals = np.array([random.uniform(0, 1) for _ in self.chains])
        
        # Construct state
        self.state = np.array([[raw_data[chain['name']]['price'],
                               raw_data[chain['name']]['depth'],
                               raw_data[chain['name']]['volume'],
                               volatility[i],
                               mempool_signals[i]]
                              for i, chain in enumerate(self.chains)])
        
        # Reset quantum system for new episode
        if hasattr(self, 'quantum_system'):
            self.quantum_system.reset_state()
        
        return self.state

    def step(self, action):
        """Execute one step in the environment"""
        self.step_count += 1
        reward = -0.01  # Small penalty for each step (encourages efficient trading)
        done = self.step_count >= self.max_steps

        # Fetch chain data
        chain_data = fetch_uniswap_data(self.chains)
        
        # Update technical indicators based on action and history
        # These would normally come from technical analysis, but we simulate them here
        if action == 1:  # Buy
            self.indicators["rsi"] = min(90, self.indicators["rsi"] + random.randint(5, 15))  # Increasing RSI
            self.indicators["macd"] = min(2.0, self.indicators["macd"] + random.uniform(0.1, 0.3))
            self.indicators["order_imbalance"] = min(90, self.indicators["order_imbalance"] + random.randint(5, 20))
        elif action == 2:  # Sell
            self.indicators["rsi"] = max(10, self.indicators["rsi"] - random.randint(5, 15))  # Decreasing RSI
            self.indicators["macd"] = max(-2.0, self.indicators["macd"] - random.uniform(0.1, 0.3))
            self.indicators["order_imbalance"] = max(-90, self.indicators["order_imbalance"] - random.randint(5, 20))
        elif action == 3:  # Sandwich
            # Sandwich trades often happen during high volatility
            self.indicators["rsi"] = 50 + random.randint(-15, 15)  # Neutral with noise
            self.indicators["macd"] = self.indicators["macd"] + random.uniform(-0.2, 0.2)
            self.indicators["order_imbalance"] = min(100, max(-100, self.indicators["order_imbalance"] + random.randint(-30, 30)))
        
        # Gradually move MACD signal line towards MACD line (equilibrium)
        self.indicators["macd_signal"] = self.indicators["macd_signal"] + (self.indicators["macd"] - self.indicators["macd_signal"]) * 0.2
        
        # Prepare market data for enhanced quantum strategy
        market_data = {
            "chain_data": {chain["name"]: chain_data[chain["name"]] for chain in self.chains},
            "indicators": self.indicators
        }
        
        # Extract market conditions like volatility, trend strength, etc.
        market_conditions = extract_market_conditions(market_data["chain_data"], market_data["indicators"])
        market_data.update(market_conditions)
        
        # Execute enhanced quantum trading strategy
        q_result = self.quantum_system.execute_enhanced_quantum_strategy(market_data)
        
        # Use the enhanced quantum factor
        quantum_boost = q_result.get("enhanced_quantum_factor", 0.0)
        
        # Log quantum strategy details
        logger.debug(f"Action: {action}, Quantum factor: {quantum_boost:.4f}, Circuit depth: {q_result['circuit_params']['depth']}")
        
        # Execute trade based on action with quantum-aware decisions
        if action == 1:  # Buy
            reward = self.execute_buy(chain_data, q_result)
        elif action == 2:  # Sell
            reward = self.execute_sell(chain_data, q_result)
        elif action == 3:  # Sandwich
            reward = self.execute_sandwich(chain_data, q_result)
        else:  # Hold
            reward = self.execute_hold(chain_data, q_result)

        # Apply quantum boost to reward with adaptive scaling based on action confidence
        action_probability = max(q_result.get("buy_probability", 0.0), q_result.get("sell_probability", 0.0), q_result.get("hold_probability", 0.0))
        confidence_multiplier = (action_probability - 0.33) * 3  # Scale from 0 to 2
        confidence_multiplier = max(0.5, min(2.0, confidence_multiplier))  # Clip between 0.5 and 2.0
        
        # Apply quantum boost with confidence scaling
        reward *= (1 + quantum_boost * confidence_multiplier)
        
        # Log the quantum-adjusted reward
        logger.debug(f"Original reward: {reward/(1 + quantum_boost * confidence_multiplier):.4f}, Quantum-adjusted: {reward:.4f}")
        # Update state with current market data and add quantum features
        # Calculate volatility based on price changes
        volatility = np.array([random.uniform(0.01, 0.2) * (1 + abs(quantum_boost)) for _ in self.chains])
        
        # Calculate mempool signals - now influenced by quantum factors
        mempool_signals = np.array([random.uniform(0, 1) * (1 + q_result.get("entanglement_factor", 0)) for _ in self.chains])
        
        # Construct new state with quantum-enhanced features
        self.state = np.array([[chain_data[c['name']]['price'],
                                chain_data[c['name']]['depth'],
                                chain_data[c['name']]['volume'],
                                volatility[i],
                                mempool_signals[i]]
                              for i, c in enumerate(self.chains)])
        
        # Create info dict with useful debugging info
        info = {
            "portfolio_value": self.calculate_portfolio_value(chain_data),
            "quantum_factor": quantum_boost,
            "confidence": action_probability,
            "circuit_depth": q_result['circuit_params']['depth']
        }
        
        # Record trade in transaction history
        self.record_transaction(action, reward, chain_data, q_result)
        
        return self.state, reward, done, info

    def execute_buy(self, chain_data, q_result):
        """Execute a buy action with quantum-enhanced decision making"""
        # Choose which chain to buy based on quantum probabilities
        chain_probs = []
        for chain in self.chains:
            chain_name = chain['name']
            # Higher price momentum and higher quantum buy probability leads to higher purchase probability
            momentum = random.uniform(-0.1, 0.2)  # Simulated price momentum
            buy_prob = q_result.get("buy_probability", 0.5)
            chain_probs.append(momentum * buy_prob)
        
        # Normalized probabilities
        chain_probs = np.array(chain_probs)
        chain_probs = np.exp(chain_probs * 5)  # Apply softmax-like scaling
        chain_probs = chain_probs / chain_probs.sum()
        
        # Choose chain based on probabilities
        selected_idx = np.random.choice(len(self.chains), p=chain_probs)
        selected_chain = self.chains[selected_idx]['name']
        
        # Check if we have enough balance
        price = chain_data[selected_chain]['price']
        if self.balance < price:
            logger.debug(f"Insufficient balance to buy {selected_chain}")
            return -0.05  # Penalty for trying to buy without funds
        
        # Calculate how much to buy (10% of available balance)
        amount_to_spend = self.balance * 0.1
        quantity = amount_to_spend / price
        
        # Update portfolio
        self.balance -= amount_to_spend
        self.positions[selected_chain] += quantity
        
        # Calculate reward based on buy confidence and market conditions
        reward = 0.01  # Base reward for action
        
        # Add bonus if quantum system is highly confident in buy
        if q_result.get("buy_probability", 0) > 0.6:
            reward += 0.05
        
        logger.info(f"Bought {quantity:.4f} {selected_chain} at ${price:.2f}")
        return reward
    
    def execute_sell(self, chain_data, q_result):
        """Execute a sell action with quantum-enhanced decision making"""
        # Choose which chain to sell based on quantum probabilities and current positions
        chain_probs = []
        for chain in self.chains:
            chain_name = chain['name']
            
            # Higher position value and higher quantum sell probability leads to higher sell probability
            position_value = self.positions[chain_name] * chain_data[chain_name]['price']
            sell_prob = q_result.get("sell_probability", 0.5)
            
            # Cannot sell what we don't have
            if self.positions[chain_name] <= 0:
                chain_probs.append(0.0)
            else:
                chain_probs.append(position_value * sell_prob)
        
        # Check if we have anything to sell
        if sum(chain_probs) <= 0:
            logger.debug("No positions to sell")
            return -0.05  # Penalty for trying to sell with no positions
        
        # Normalized probabilities
        chain_probs = np.array(chain_probs)
        if chain_probs.sum() > 0:  # Ensure we have something to sell
            chain_probs = chain_probs / chain_probs.sum()
            
            # Choose chain based on probabilities
            selected_idx = np.random.choice(len(self.chains), p=chain_probs)
            selected_chain = self.chains[selected_idx]['name']
            
            # Calculate how much to sell (50% of position)
            quantity_to_sell = self.positions[selected_chain] * 0.5
            price = chain_data[selected_chain]['price']
            sale_value = quantity_to_sell * price
            
            # Update portfolio
            self.balance += sale_value
            self.positions[selected_chain] -= quantity_to_sell
            
            # Calculate reward based on sell confidence and market conditions
            reward = 0.01  # Base reward for action
            
            # Add bonus if quantum system is highly confident in sell
            if q_result.get("sell_probability", 0) > 0.6:
                reward += 0.05
            
            logger.info(f"Sold {quantity_to_sell:.4f} {selected_chain} at ${price:.2f}")
            return reward
        
        return -0.01  # Small penalty for unsuccessful sell attempt
    
    def execute_sandwich(self, chain_data, q_result):
        """Execute a sandwich trade (buy+sell) with quantum-enhanced decision making"""
        # Sandwich trades are complex and require special market conditions
        # They're profitable during high volatility and when order imbalance is high
        
        # Choose which chain for sandwich based on quantum factors
        entanglement = q_result.get("entanglement_factor", 0.0)
        volatility = q_result.get("market_volatility", 0.5)
        
        # Sandwich trades are most profitable in high volatility, high liquidity markets
        chain_scores = []
        for chain in self.chains:
            chain_name = chain['name']
            volume = chain_data[chain_name]['volume']
            depth = chain_data[chain_name]['depth']
            
            # Score based on trading conditions
            score = (volume / 1000) * (volatility * 2) * (entanglement + 0.5)
            chain_scores.append(score)
        
        # Normalize scores
        chain_scores = np.array(chain_scores)
        if chain_scores.sum() > 0:
            chain_scores = chain_scores / chain_scores.sum()
            
            # Select chain for sandwich trade
            selected_idx = np.random.choice(len(self.chains), p=chain_scores)
            selected_chain = self.chains[selected_idx]['name']
            
            # Check if we have enough balance
            price = chain_data[selected_chain]['price']
            if self.balance < price * 2:  # Need more capital for sandwich trades
                logger.debug(f"Insufficient balance for sandwich trade on {selected_chain}")
                return -0.1  # Higher penalty for failed sandwich attempt
            
            # Sandwich trades are complex - we simulate the outcome
            # Higher success probability with quantum advantage
            success_prob = 0.3 + (q_result.get("quantum_factor", 0) + 0.5) * 0.4
            
            # Determine if sandwich trade succeeds
            if random.random() < success_prob:
                # Successful sandwich provides higher rewards but uses more capital
                profit = self.balance * 0.03 * (1 + entanglement)  # 3% profit boosted by entanglement
                self.balance += profit
                logger.info(f"Successful sandwich trade on {selected_chain}, profit: ${profit:.2f}")
                return 0.2  # Higher base reward for successful sandwich
            else:
                # Failed sandwich costs gas and potential slippage
                loss = self.balance * 0.01  # 1% loss
                self.balance -= loss
                logger.info(f"Failed sandwich trade on {selected_chain}, loss: ${loss:.2f}")
                return -0.1
        
        return -0.05  # Penalty for unsuccessful sandwich attempt
    
    def execute_hold(self, chain_data, q_result):
        """Execute a hold action (do nothing)"""
        # Even though we're not trading, we still update portfolio values
        # This simulates price movements affecting our holdings
        
        # Calculate previous portfolio value
        prev_value = self.calculate_portfolio_value(chain_data)
        
        # Simulate price changes (slightly influenced by quantum factors)
        # Hold is often good in stable markets, so quantum factor should be small
        quantum_factor = q_result.get("quantum_factor", 0.0)
        
        # Small random market movements for each chain
        for chain in self.chains:
            chain_name = chain['name']
            if self.positions[chain_name] > 0:
                # Natural price movement plus quantum influence
                price_change = random.uniform(-0.02, 0.02) + quantum_factor * 0.01
                new_price = chain_data[chain_name]['price'] * (1 + price_change)
                chain_data[chain_name]['price'] = new_price
        
        # Calculate new portfolio value
        new_value = self.calculate_portfolio_value(chain_data)
        
        # Reward based on portfolio performance during hold
        value_change_pct = (new_value - prev_value) / prev_value if prev_value > 0 else 0
        
        # Hold is good if you're already gaining value, bad if losing
        reward = value_change_pct * 5  # Scale the change for reward
        
        # Add bonus if quantum system is highly confident in hold
        if q_result.get("hold_probability", 0) > 0.6:
            reward += 0.02
        
        logger.debug(f"Hold action, portfolio value change: {value_change_pct:.4%}")
        return reward

    def calculate_portfolio_value(self, chain_data):
        """Calculate the total value of the portfolio"""
        positions_value = sum(
            self.positions[chain['name']] * chain_data[chain['name']]['price']
            for chain in self.chains
        )
        return self.balance + positions_value
    
    def record_transaction(self, action, reward, chain_data, q_result):
        """Record transaction details for later analysis"""
        action_names = {0: "HOLD", 1: "BUY", 2: "SELL", 3: "SANDWICH"}
        
        # Convert numpy array to int if needed
        if hasattr(action, 'item'):
            action_key = action.item()
        else:
            action_key = int(action)
        
        transaction = {
            "timestamp": datetime.now().isoformat(),
            "action": action_names[action_key],
            "reward": reward,
            "balance": self.balance,
            "portfolio_value": self.calculate_portfolio_value(chain_data),
            "positions": self.positions.copy(),
            "quantum_factor": q_result.get("enhanced_quantum_factor", 0.0),
            "circuit_depth": q_result.get("circuit_params", {}).get("depth", 0),
            "entanglement": q_result.get("entanglement_factor", 0.0)
        }
        
        self.transaction_history.append(transaction)
        
        # Periodically save transaction history
        if len(self.transaction_history) % 100 == 0:
            self.save_transaction_history()
    
    def save_transaction_history(self):
        """Save transaction history to file for analysis"""
        if not self.transaction_history:
            return
            
        # Create directory if it doesn't exist
        os.makedirs("results/transactions", exist_ok=True)
        
        # Convert to DataFrame and save
        df = pd.DataFrame(self.transaction_history)
        df.to_csv(f"results/transactions/transactions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", index=False)


# Function to train RL agent with enhanced quantum features
def train_rl_agent(chains=CHAINS, total_timesteps=100000, algorithm="DQN", callback=None,
                learning_rate=0.0001, gamma=0.99, exploration_fraction=0.1, 
                checkpoint_freq=10000, tensorboard_log="./results/tensorboard/"):
    """Train RL agent with the quantum-enhanced environment
    
    Args:
        chains: List of chain dictionaries to trade on
        total_timesteps: Total number of timesteps to train for
        algorithm: RL algorithm to use ("DQN", "PPO", "A2C", etc.)
        callback: Optional callback function for monitoring progress
        learning_rate: Learning rate for the optimizer
        gamma: Discount factor
        exploration_fraction: Fraction of training to explore
        checkpoint_freq: Save checkpoints every n steps
        tensorboard_log: Directory to save tensorboard logs
        
    Returns:
        Trained model object
    """
    from stable_baselines3 import DQN, PPO, A2C, SAC
    from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
    import gym
    
    logger.info(f"Training RL agent with {algorithm} for {total_timesteps} timesteps")
    
    # Create training environment
    env = TradingEnv(chains)
    
    # Prepare directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("results/tensorboard", exist_ok=True)
    os.makedirs("results/checkpoints", exist_ok=True)
    
    # Unique run ID for this training session
    run_id = f"{algorithm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Set up model based on algorithm
    if algorithm == "DQN":
        model = DQN(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=learning_rate,
            buffer_size=100000,
            learning_starts=1000,
            batch_size=64,
            gamma=gamma,
            exploration_fraction=exploration_fraction,
            tensorboard_log=tensorboard_log
        )
    elif algorithm == "PPO":
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=learning_rate,
            gamma=gamma,
            tensorboard_log=tensorboard_log
        )
    elif algorithm == "A2C":
        model = A2C(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=learning_rate,
            gamma=gamma,
            tensorboard_log=tensorboard_log
        )
    elif algorithm == "SAC":
        # SAC is suitable for continuous action spaces, would need env modification
        logger.warning("SAC requires continuous action space, consider adaptation")
        model = SAC(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=learning_rate,
            gamma=gamma,
            tensorboard_log=tensorboard_log
        )
    else:
        # Default to DQN
        logger.warning(f"Algorithm {algorithm} not implemented, defaulting to DQN")
        model = DQN("MlpPolicy", env, verbose=1, tensorboard_log=tensorboard_log)
    
    # Set up callbacks for training monitoring
    callbacks = []
    
    # Checkpoint callback to save model during training
    if checkpoint_freq > 0:
        checkpoint_callback = CheckpointCallback(
            save_freq=checkpoint_freq,
            save_path="results/checkpoints/",
            name_prefix=run_id,
            save_replay_buffer=True,
            save_vecnormalize=True
        )
        callbacks.append(checkpoint_callback)
    
    # Add custom callback if provided
    if callback is not None:
        callbacks.append(callback)
    
    # Training performance metrics
    training_metrics = {
        "start_time": datetime.now().isoformat(),
        "algorithm": algorithm,
        "total_timesteps": total_timesteps,
        "chains": [chain["name"] for chain in chains],
        "parameters": {
            "learning_rate": learning_rate,
            "gamma": gamma,
            "exploration_fraction": exploration_fraction
        }
    }
    
    # Train the agent
    try:
        start_time = datetime.now()
        logger.info(f"Starting training at {start_time}")
        
        # Use callbacks if available
        if callbacks:
            model.learn(total_timesteps=total_timesteps, callback=callbacks)
        else:
            model.learn(total_timesteps=total_timesteps)
        
        training_time = datetime.now() - start_time
        logger.info(f"Training completed in {training_time}")
        
        # Update metrics
        training_metrics["end_time"] = datetime.now().isoformat()
        training_metrics["training_duration_seconds"] = training_time.total_seconds()
        
        # Save final model
        final_model_path = f"models/{run_id}_final"
        model.save(final_model_path)
        logger.info(f"Model saved to {final_model_path}")
        
        # Save training metrics
        with open(f"results/metrics/{run_id}_metrics.json", "w") as f:
            import json
            json.dump(training_metrics, f, indent=4)
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        # Save partial metrics if training fails
        training_metrics["error"] = str(e)
        training_metrics["error_time"] = datetime.now().isoformat()
        with open(f"results/metrics/{run_id}_error_metrics.json", "w") as f:
            import json
            json.dump(training_metrics, f, indent=4)
        raise
    
    logger.info(f"RL agent training completed successfully")
    return model
