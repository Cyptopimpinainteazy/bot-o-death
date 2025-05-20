#!/usr/bin/env python
"""
Test Quantum Integration
-----------------------
This script tests the integration of quantum computing with the trading strategy
by creating a quantum circuit and executing the quantum_trade_strategy function.
"""

import sys
import logging
import numpy as np
from quantum import create_quantum_circuit, quantum_trade_strategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("QuantumTest")

def test_quantum_strategy():
    """Test the quantum trading strategy with different parameters"""
    logger.info("Testing Quantum Trading Strategy Integration")
    
    test_scenarios = [
        {"name": "Buy Signal", "rsi": 0.8, "macd": 0.6, "imbalance": 0.3},
        {"name": "Sell Signal", "rsi": 0.2, "macd": -0.6, "imbalance": -0.3},
        {"name": "Hold Signal", "rsi": 0.5, "macd": 0.1, "imbalance": 0.0},
        {"name": "Sandwich Trade", "rsi": 0.5, "macd": 0.2, "imbalance": 0.8}
    ]
    
    for scenario in test_scenarios:
        logger.info(f"=== Testing {scenario['name']} ===")
        
        # Create quantum circuit with the given parameters
        circuit_config = create_quantum_circuit(
            depth=3,
            shots=1024,
            rsi=scenario["rsi"],
            macd=scenario["macd"],
            imbalance=scenario["imbalance"]
        )
        
        # Execute quantum trading strategy
        result = quantum_trade_strategy(circuit_config)
        
        # Log results
        logger.info(f"Action: {result['action']}")
        logger.info(f"Buy Probability: {result['buy_probability']:.4f}")
        logger.info(f"Sell Probability: {result['sell_probability']:.4f}")
        logger.info(f"Hold Probability: {result['hold_probability']:.4f}")
        logger.info(f"Quantum Factor: {result['quantum_factor']:.4f}")
        logger.info("")

if __name__ == "__main__":
    logger.info("=== Starting Quantum Integration Test ===")
    test_quantum_strategy()
    logger.info("=== Quantum Integration Test Completed ===")
