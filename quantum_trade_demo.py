#!/usr/bin/env python3
import os
import sys
import time
import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from dotenv import load_dotenv
import logging
from web3 import Web3

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("BotX3-Demo")

# Load environment variables
load_dotenv()

# Demo configuration
DEMO_ASSETS = {
    "MATIC": {"price": 0.5, "liquidity": 1000000},
    "ETH": {"price": 3000, "liquidity": 500000},
    "BTC": {"price": 60000, "liquidity": 100000},
    "USDC": {"price": 1.0, "liquidity": 2000000}
}

# Simulate price movement
def simulate_price_movement(asset, quantum_factor):
    """Simulate price movement based on quantum factor"""
    # Negative quantum factor tends to drive prices down
    # Positive quantum factor tends to drive prices up
    price_change = asset["price"] * quantum_factor * np.random.normal(1, 0.5)
    new_price = asset["price"] + price_change
    return max(new_price, 0.01)  # Ensure price doesn't go below 0.01

# Simulate a trade execution
def simulate_trade(asset_name, amount, trade_type, quantum_factor):
    """Simulate a trade and its outcome"""
    asset = DEMO_ASSETS[asset_name]
    initial_value = amount * asset["price"]
    
    # Factor in slippage based on trade size relative to liquidity
    slippage = (amount / asset["liquidity"]) * 10
    
    # Apply quantum factor to influence trade success probability
    success_prob = 0.5 + (quantum_factor * 2 if trade_type == "buy" else -quantum_factor * 2)
    success_prob = max(0.1, min(0.95, success_prob))  # Keep within reasonable bounds
    
    # Transaction fees (higher for more complex strategies)
    tx_fee = 0.003  # 0.3% base fee
    if trade_type in ["sandwich", "flash", "mev"]:
        tx_fee = 0.01  # 1% for complex strategies
    
    # Calculate result
    price_after = simulate_price_movement(asset, quantum_factor)
    DEMO_ASSETS[asset_name]["price"] = price_after  # Update simulation price
    
    # Calculate final value
    if trade_type == "buy":
        final_value = amount * price_after * (1 - slippage - tx_fee)
    else:  # sell
        final_value = amount * price_after * (1 - slippage - tx_fee)
        
    profit_loss = final_value - initial_value
    profit_pct = (profit_loss / initial_value) * 100 if initial_value > 0 else 0
    
    # Determine if trade was successful based on probability
    successful = np.random.random() < success_prob
    
    return {
        "asset": asset_name,
        "trade_type": trade_type,
        "amount": amount,
        "initial_price": asset["price"],
        "final_price": price_after,
        "initial_value": initial_value,
        "final_value": final_value,
        "profit_loss": profit_loss,
        "profit_pct": profit_pct,
        "successful": successful,
        "quantum_factor": quantum_factor
    }

def run_quantum_trading_demo():
    """Run a full demonstration of quantum-influenced trading"""
    print("\n" + "="*70)
    print(" "*25 + "QUANTUM TRADING DEMO")
    print("="*70)
    
    # Wallet setup (demo only)
    private_key = os.getenv('PRIVATE_KEY', "")
    if private_key.startswith("0x"):
        private_key = private_key[2:]
    
    # Create demo wallet
    try:
        # Connect to Alchemy API (just for address derivation)
        alchemy_key = os.getenv('ALCHEMY_API_KEY', '')
        w3 = Web3(Web3.HTTPProvider(f"https://polygon-mumbai.g.alchemy.com/v2/{alchemy_key}"))
        account = w3.eth.account.from_key(private_key)
        wallet_address = account.address
        print(f"Connected to demo wallet: {wallet_address[:10]}...{wallet_address[-8:]}")
    except Exception as e:
        print(f"Warning: Using simulated wallet - {e}")
        wallet_address = "0xSimulatedWalletAddress"
    
    # Initial portfolio
    portfolio = {
        "MATIC": 1000,
        "ETH": 0.5,
        "BTC": 0.01,
        "USDC": 2000
    }
    
    print("\nInitial Portfolio:")
    total_value = 0
    for asset, amount in portfolio.items():
        value = amount * DEMO_ASSETS[asset]["price"]
        total_value += value
        print(f"  {asset}: {amount:.4f} (${value:.2f})")
    print(f"Total Portfolio Value: ${total_value:.2f}")
    
    # Run three different quantum circuits to demo different trading scenarios
    circuits = [
        {"name": "Conservative Circuit", "rz_params": [np.pi/8, np.pi/8, np.pi/8], 
         "extra_ops": False, "bias": 0.1},
        {"name": "Balanced Circuit", "rz_params": [np.pi/2, np.pi/2, np.pi/4], 
         "extra_ops": True, "bias": 0.3},
        {"name": "Aggressive Circuit", "rz_params": [np.pi, np.pi/2, 0], 
         "extra_ops": True, "bias": 0.6}
    ]
    
    for circuit_config in circuits:
        print(f"\n\n{'-'*70}")
        print(f"Running {circuit_config['name']} Trading Simulation")
        print(f"{'-'*70}")
        
        # Create and run quantum circuit
        qc = QuantumCircuit(3, 3)
        qc.h([0, 1, 2])  # Create superposition
        qc.cx(0, 1)      # Entangle qubits
        qc.cx(1, 2)
        
        # Apply rotations with circuit-specific parameters
        qc.rz(circuit_config['rz_params'][0], 0)
        qc.rz(circuit_config['rz_params'][1], 1)
        qc.rz(circuit_config['rz_params'][2], 2)
        
        # Add extra operations for more complex circuits
        if circuit_config.get('extra_ops', False):
            qc.h(0)  # Additional Hadamard on first qubit
            qc.z(1)  # Phase flip on second qubit
            qc.cx(2, 0)  # Additional entanglement
        
        # Measure
        qc.measure([0, 1, 2], [0, 1, 2])
        
        # Run simulation
        backend = AerSimulator()
        result = backend.run(qc, shots=1000).result().get_counts()
        
        # Calculate quantum factor with circuit-specific bias
        base_factor = (result.get('000', 0) - result.get('111', 0)) / 1000
        # Add circuit-specific bias to demonstrate different strategies
        quantum_factor = base_factor + circuit_config.get('bias', 0) * (-1 if circuit_config['name'] == 'Balanced Circuit' else 1)
        
        print(f"Quantum Circuit Results:")
        for state, count in sorted(result.items()):
            print(f"  State {state}: {count} counts ({count/10:.1f}%)")
        print(f"\nQuantum Factor: {quantum_factor:.4f}")
        
        # Determine trading strategy based on quantum factor
        if quantum_factor > 0.5:
            strategy = "sandwich"
            print("Strategy: SANDWICH TRADE (strongly positive signal)")
        elif quantum_factor > 0.3:
            strategy = "buy"
            print("Strategy: BUY (moderately positive signal)")
        elif quantum_factor < -0.5:
            strategy = "mev"
            print("Strategy: MEV OPPORTUNITY (strongly negative signal)")
        elif quantum_factor < -0.3:
            strategy = "sell"
            print("Strategy: SELL (moderately negative signal)")
        elif abs(quantum_factor) > 0.2:
            strategy = "flash"
            print("Strategy: FLASH LOAN ARBITRAGE (volatility signal)")
        else:
            strategy = "hold"
            print("Strategy: HOLD (neutral signal)")
        
        # Execute appropriate trades based on strategy
        if strategy != "hold":
            print("\nExecuting trades based on quantum analysis...")
            time.sleep(1)  # Simulate processing time
            
            # Choose assets to trade based on strategy
            if strategy == "buy":
                trades = [
                    {"asset": "ETH", "amount": portfolio["USDC"] * 0.3 / DEMO_ASSETS["ETH"]["price"], "type": "buy"},
                    {"asset": "MATIC", "amount": portfolio["USDC"] * 0.1 / DEMO_ASSETS["MATIC"]["price"], "type": "buy"}
                ]
            elif strategy == "sell":
                trades = [
                    {"asset": "ETH", "amount": portfolio["ETH"] * 0.5, "type": "sell"},
                    {"asset": "MATIC", "amount": portfolio["MATIC"] * 0.3, "type": "sell"}
                ]
            elif strategy == "sandwich":
                trades = [
                    {"asset": "ETH", "amount": portfolio["ETH"] * 0.7, "type": "sandwich"},
                    {"asset": "MATIC", "amount": portfolio["MATIC"] * 0.5, "type": "sandwich"}
                ]
            elif strategy == "flash":
                trades = [
                    {"asset": "ETH", "amount": portfolio["ETH"] * 0.3, "type": "flash"},
                    {"asset": "BTC", "amount": portfolio["BTC"] * 0.4, "type": "flash"}
                ]
            else:  # mev
                trades = [
                    {"asset": "ETH", "amount": portfolio["ETH"] * 0.25, "type": "mev"},
                    {"asset": "BTC", "amount": portfolio["BTC"] * 0.25, "type": "mev"}
                ]
            
            # Execute trades and track results
            for trade in trades:
                result = simulate_trade(
                    trade["asset"], 
                    trade["amount"], 
                    trade["type"], 
                    quantum_factor
                )
                
                # Update portfolio based on trade
                if result["successful"]:
                    if trade["type"] == "buy":
                        portfolio[trade["asset"]] += trade["amount"]
                        portfolio["USDC"] -= trade["amount"] * DEMO_ASSETS[trade["asset"]]["price"]
                    elif trade["type"] == "sell":
                        portfolio[trade["asset"]] -= trade["amount"]
                        portfolio["USDC"] += trade["amount"] * DEMO_ASSETS[trade["asset"]]["price"]
                    elif trade["type"] in ["sandwich", "flash", "mev"]:
                        # Complex strategies might generate profit directly
                        portfolio["USDC"] += result["profit_loss"]
                
                # Display trade result
                status = "✅ SUCCESS" if result["successful"] else "❌ FAILED"
                print(f"\nTrade: {result['trade_type'].upper()} {result['amount']:.4f} {result['asset']}")
                print(f"  Initial Price: ${result['initial_price']:.2f}")
                print(f"  Final Price: ${result['final_price']:.2f}")
                print(f"  P/L: ${result['profit_loss']:.2f} ({result['profit_pct']:+.2f}%)")
                print(f"  Outcome: {status}")
            
            # Display updated portfolio
            print("\nUpdated Portfolio:")
            new_total = 0
            for asset, amount in portfolio.items():
                value = amount * DEMO_ASSETS[asset]["price"]
                new_total += value
                print(f"  {asset}: {amount:.4f} (${value:.2f})")
            print(f"Total Portfolio Value: ${new_total:.2f}")
            print(f"Session P/L: ${new_total - total_value:.2f} ({(new_total/total_value - 1)*100:+.2f}%)")
            
            # Update total value for next circuit
            total_value = new_total
        else:
            print("\nHolding current positions - no trades executed.")
    
    print("\n" + "="*70)
    print(" "*25 + "DEMO COMPLETE")
    print("="*70)
    print("Note: This is a demonstration using simulated data.")
    print("Real quantum trading would incorporate additional market data,")
    print("technical analysis, and risk management strategies.")

if __name__ == "__main__":
    run_quantum_trading_demo()
