#!/usr/bin/env python3
"""
Enhanced Quantum Trade AI Runner Script
This script ensures all dependencies are available (mocked if necessary)
and runs the trading application
"""
import os
import sys
import importlib
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("EnhancedQuantumTradeAI")

# Ensure core directory is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

logger.info("Starting Enhanced Quantum Trade AI")
logger.info(f"Python version: {sys.version}")
logger.info(f"Working directory: {current_dir}")

# Create logs directory if it doesn't exist
if not os.path.exists(os.path.join(current_dir, 'logs')):
    os.makedirs(os.path.join(current_dir, 'logs'))
    logger.info("Created logs directory")

# Create empty contract file if needed for TradingLogic to load
contracts_dir = os.path.join(current_dir, 'contracts')
if not os.path.exists(contracts_dir):
    os.makedirs(contracts_dir)
    logger.info("Created contracts directory")

flashloan_contract_path = os.path.join(contracts_dir, 'FlashloanTrader.json')
if not os.path.exists(flashloan_contract_path):
    with open(flashloan_contract_path, 'w') as f:
        f.write('{"abi": []}')
    logger.info("Created empty FlashloanTrader.json")

# Run the app with command line arguments
try:
    import click
except ImportError:
    print("Using mock click implementation")
    import mock_click as click

# Import app in a way that bypasses click decorators
sys.path.append(current_dir)
from app import TradingSwarm

# Handle possible missing yaml dependency
try:
    import yaml
except ImportError:
    print("Using mock yaml implementation")
    import mock_yaml as yaml

# Add default symbols if none provided
symbols = ['ETH', 'USDC']
mode = 'paper'
chains = ['ethereum']

# Load config
config_path = os.path.join(current_dir, 'config.yaml')
if not os.path.exists(config_path):
    # Create minimal config file if it doesn't exist
    config = {
        'chains': ['ethereum', 'polygon', 'bsc', 'arbitrum_one'],
        'providers': {
            'ethereum': ['https://mainnet.infura.io/v3/your-api-key'],
            'polygon': ['https://polygon-rpc.com'],
            'bsc': ['https://bsc-dataseed.binance.org'],
            'arbitrum_one': ['https://arb1.arbitrum.io/rpc']
        },
        'contracts': {
            'flashloan': {
                'ethereum': '0x0000000000000000000000000000000000000000'
            }
        }
    }
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    logger.info("Created minimal config.yaml")
else:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

logger.info(f"Running Enhanced Quantum Trade AI with symbols={symbols}, mode={mode}, chains={chains}")
try:
    # Create trading swarm instance directly
    swarm = TradingSwarm(config)
    # Run stress test with specified parameters
    import asyncio
    asyncio.run(swarm.stress_test(chains, symbols, rps_target=10))
    logger.info("Trading application completed successfully")
except Exception as e:
    logger.error(f"Error running application: {str(e)}", exc_info=True)
    print(f"Error: {str(e)}")
    sys.exit(1)
