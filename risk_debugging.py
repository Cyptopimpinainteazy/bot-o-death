#!/usr/bin/env python
"""
Debug the risk parameters being used in the Exchange Risk Manager
"""

import json
from pathlib import Path
from exchange_risk_manager import ExchangeRiskManager

def debug_risk_parameters():
    """Print out all risk parameters currently being used"""
    print("\n=== RISK PARAMETER DEBUG ===")
    
    # Load config file directly
    config_path = Path("config") / "exchange_risk_config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print("From config file:")
    risk_params = config.get("risk_parameters", {})
    for key, value in risk_params.items():
        print(f"  {key}: {value}")
    
    # Load via risk manager
    risk_manager = ExchangeRiskManager()
    
    print("\nFrom risk manager object:")
    print(f"  min_profit_threshold: {risk_manager.min_profit_threshold}")
    print(f"  max_acceptable_slippage: {risk_manager.max_acceptable_slippage}")
    print(f"  max_liquidity_usage: {risk_manager.max_liquidity_usage}")
    print(f"  max_transfer_wait: {risk_manager.max_transfer_wait}")
    print(f"  min_confidence_score: {risk_manager.min_confidence_score}")
    
    # Feature state matrix for RL use
    print("\nExample feature state matrix for RL training:")
    state = {
        "spread": 0.02,  # 2% price difference
        "fees": 0.005,   # 0.5% fees
        "liquidity_score": 0.8,  # High liquidity
        "transfer_time": 30,  # 30 minutes
        "volatility": 0.003,  # Low volatility
        "market_trend": 1,  # Uptrend
    }
    
    for key, value in state.items():
        print(f"  {key}: {value}")
    
    return risk_manager

if __name__ == "__main__":
    debug_risk_parameters()
