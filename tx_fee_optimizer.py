import os
import json
import time
import logging
import requests
from web3 import Web3
from dotenv import load_dotenv
from math import exp
import numpy as np
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TxFeeOptimizer")

# Load environment variables
load_dotenv()

class TxFeeOptimizer:
    """
    Transaction fee optimization module for blockchain operations.
    Analyzes gas prices, mempool congestion, and priority to optimize transaction fees.
    """
    
    def __init__(self):
        """Initialize the Transaction Fee Optimizer"""
        # API keys
        self.alchemy_api_key = os.getenv("ALCHEMY_API_KEY")
        
        # Web3 providers for different chains
        self.providers = {
            "ethereum": Web3(Web3.HTTPProvider(f"https://eth-mainnet.g.alchemy.com/v2/{self.alchemy_api_key}")),
            "polygon": Web3(Web3.HTTPProvider(f"https://polygon-mainnet.g.alchemy.com/v2/{self.alchemy_api_key}")),
            "arbitrum": Web3(Web3.HTTPProvider(f"https://arb-mainnet.g.alchemy.com/v2/{self.alchemy_api_key}")),
            "optimism": Web3(Web3.HTTPProvider(f"https://opt-mainnet.g.alchemy.com/v2/{self.alchemy_api_key}"))
        }
        
        # Gas price tracking
        self.gas_price_history = {chain: [] for chain in self.providers.keys()}
        self.max_history_length = 100  # Maximum entries to keep in history
        
        # Gas price confidence levels
        self.confidence_levels = {
            "low": 0.7,      # 70% chance to be included within target blocks
            "medium": 0.85,  # 85% chance to be included within target blocks
            "high": 0.95,    # 95% chance to be included within target blocks
            "urgent": 0.99   # 99% chance to be included within target blocks
        }
        
        # Default priority
        self.default_priority = "medium"
        
        # EIP-1559 parameters
        self.base_fee_multiplier = {
            "low": 1.1,      # 10% above base fee
            "medium": 1.3,   # 30% above base fee
            "high": 1.7,     # 70% above base fee
            "urgent": 2.5    # 150% above base fee
        }
        
        # Legacy gas price parameters (percentiles for different priority levels)
        self.legacy_percentiles = {
            "low": 25,       # 25th percentile
            "medium": 50,    # 50th percentile
            "high": 75,      # 75th percentile
            "urgent": 95     # 95th percentile
        }
        
        # Target confirmation blocks
        self.target_blocks = {
            "low": 6,        # Within ~1.5 minutes on ETH
            "medium": 3,     # Within ~45 seconds on ETH
            "high": 1,       # Within ~15 seconds on ETH
            "urgent": 1      # Highest priority for next block
        }
        
        logger.info("Transaction Fee Optimizer initialized")
    
    def update_gas_price(self, chain="ethereum"):
        """Update gas price data for a specific chain"""
        if chain not in self.providers:
            logger.error(f"Unsupported chain: {chain}")
            return None
        
        try:
            w3 = self.providers[chain]
            
            # Current time
            timestamp = int(time.time())
            
            # Get current gas prices (this differs by chain)
            gas_data = {}
            
            if chain == "ethereum":
                # For EIP-1559 compatible chains, get both base fee and priority fee
                latest_block = w3.eth.get_block('latest')
                
                # Base fee from the latest block
                base_fee = latest_block.baseFeePerGas
                
                # Get a few recent transactions to estimate priority fees
                priority_fees = []
                for i in range(min(10, len(latest_block.transactions))):
                    tx_hash = latest_block.transactions[i]
                    tx = w3.eth.get_transaction(tx_hash)
                    if hasattr(tx, 'maxPriorityFeePerGas'):
                        priority_fees.append(tx.maxPriorityFeePerGas)
                
                # Calculate average or median priority fee
                if priority_fees:
                    avg_priority_fee = sum(priority_fees) // len(priority_fees)
                else:
                    # Fallback to an estimate
                    avg_priority_fee = w3.eth.max_priority_fee
                
                gas_data = {
                    "type": "eip1559",
                    "baseFeePerGas": base_fee,
                    "maxPriorityFeePerGas": avg_priority_fee,
                    "timestamp": timestamp,
                    "block": latest_block.number,
                    "pending_tx_count": w3.eth.get_block_transaction_count('pending')
                }
                
                # Calculate max fee recommendations
                for priority, multiplier in self.base_fee_multiplier.items():
                    priority_fee = avg_priority_fee
                    if priority == "urgent":
                        priority_fee = max(priority_fees) if priority_fees else avg_priority_fee * 2
                    
                    max_fee_per_gas = int(base_fee * multiplier) + priority_fee
                    gas_data[f"{priority}_maxFeePerGas"] = max_fee_per_gas
                    gas_data[f"{priority}_maxPriorityFeePerGas"] = priority_fee
            
            elif chain in ["polygon", "arbitrum", "optimism"]:
                # Similar approach for other EVM chains with their own peculiarities
                latest_block = w3.eth.get_block('latest')
                gas_price = w3.eth.gas_price
                
                # Some chains don't fully support EIP-1559
                if hasattr(latest_block, 'baseFeePerGas'):
                    # EIP-1559 is supported
                    base_fee = latest_block.baseFeePerGas
                    
                    gas_data = {
                        "type": "eip1559",
                        "gasPrice": gas_price,
                        "baseFeePerGas": base_fee,
                        "timestamp": timestamp,
                        "block": latest_block.number,
                        "pending_tx_count": w3.eth.get_block_transaction_count('pending')
                    }
                    
                    # Calculate max fee recommendations (chain-specific adjustments)
                    multiplier_adjustments = 1.0
                    if chain == "polygon":
                        multiplier_adjustments = 1.2  # Polygon needs higher multipliers
                    elif chain == "arbitrum":
                        multiplier_adjustments = 1.5  # Arbitrum has custom gas logic
                    
                    for priority, multiplier in self.base_fee_multiplier.items():
                        adjusted_multiplier = multiplier * multiplier_adjustments
                        max_fee_per_gas = int(base_fee * adjusted_multiplier) + (gas_price // 10)
                        gas_data[f"{priority}_maxFeePerGas"] = max_fee_per_gas
                else:
                    # Legacy gas price only
                    gas_data = {
                        "type": "legacy",
                        "gasPrice": gas_price,
                        "timestamp": timestamp,
                        "block": latest_block.number,
                        "pending_tx_count": w3.eth.get_block_transaction_count('pending')
                    }
                    
                    # Legacy gas price recommendations
                    for priority, percentile in self.legacy_percentiles.items():
                        gas_data[f"{priority}_gasPrice"] = self._calculate_legacy_gas_price(
                            gas_price, percentile, chain
                        )
            
            # Store in history
            self.gas_price_history[chain].append(gas_data)
            
            # Trim history if too long
            if len(self.gas_price_history[chain]) > self.max_history_length:
                self.gas_price_history[chain].pop(0)
            
            logger.debug(f"Updated gas prices for {chain}")
            
            return gas_data
            
        except Exception as e:
            logger.error(f"Error updating gas prices for {chain}: {e}")
            return None
    
    def _calculate_legacy_gas_price(self, current_gas_price, percentile, chain):
        """Calculate legacy gas price based on percentile and chain-specific factors"""
        # Base multiplier from percentile (higher percentile = higher multiplier)
        base_multiplier = 0.8 + (percentile / 100 * 0.8)  # Range from 0.8 to 1.6
        
        # Chain-specific adjustments
        chain_multiplier = 1.0
        if chain == "polygon":
            chain_multiplier = 1.3  # Polygon typically needs higher gas prices
        elif chain == "arbitrum":
            chain_multiplier = 1.5  # Arbitrum has different gas dynamics
        elif chain == "optimism":
            chain_multiplier = 0.9  # Optimism tends to have lower gas costs
        
        # Apply multipliers
        return int(current_gas_price * base_multiplier * chain_multiplier)
    
    def optimize_gas_price(self, chain="ethereum", priority=None, max_cost=None, time_preference=None):
        """
        Optimize gas price based on chain, priority level, maximum cost, and time preference.
        
        Args:
            chain: Blockchain network (ethereum, polygon, etc.)
            priority: Priority level (low, medium, high, urgent)
            max_cost: Maximum acceptable gas cost in native currency
            time_preference: Balance between cost and time (0 to 1, where 0 is lowest cost, 1 is fastest)
            
        Returns:
            Optimized gas parameters
        """
        # Default to medium priority if not specified
        if priority is None:
            if time_preference is not None:
                # Map time preference to priority
                if time_preference < 0.25:
                    priority = "low"
                elif time_preference < 0.75:
                    priority = "medium"
                elif time_preference < 0.9:
                    priority = "high"
                else:
                    priority = "urgent"
            else:
                priority = self.default_priority
        
        # Validate priority
        if priority not in self.confidence_levels:
            logger.warning(f"Invalid priority: {priority}, using medium")
            priority = "medium"
        
        # Update gas prices if we don't have recent data
        if not self.gas_price_history.get(chain) or time.time() - self.gas_price_history[chain][-1]["timestamp"] > 60:
            self.update_gas_price(chain)
        
        # If we still don't have data, return None
        if not self.gas_price_history.get(chain):
            logger.error(f"No gas price data available for {chain}")
            return None
        
        # Get latest gas data
        gas_data = self.gas_price_history[chain][-1]
        
        # Analyze network congestion
        congestion_factor = self._analyze_congestion(chain)
        
        # Prepare result
        result = {
            "chain": chain,
            "priority": priority,
            "timestamp": int(time.time()),
            "congestion": congestion_factor,
            "target_blocks": self.target_blocks[priority],
            "confidence": self.confidence_levels[priority]
        }
        
        # Determine gas parameters based on chain and transaction type
        if gas_data["type"] == "eip1559":
            # EIP-1559 transaction
            max_fee_key = f"{priority}_maxFeePerGas"
            max_priority_fee_key = f"{priority}_maxPriorityFeePerGas"
            
            if max_fee_key in gas_data and max_priority_fee_key in gas_data:
                # Use pre-calculated values
                max_fee = gas_data[max_fee_key]
                max_priority_fee = gas_data[max_priority_fee_key]
            else:
                # Calculate based on base fee and congestion
                base_fee = gas_data["baseFeePerGas"]
                multiplier = self.base_fee_multiplier[priority] * (1 + congestion_factor * 0.5)
                max_fee = int(base_fee * multiplier)
                
                if "maxPriorityFeePerGas" in gas_data:
                    priority_fee = gas_data["maxPriorityFeePerGas"]
                    # Adjust priority fee based on congestion and priority
                    priority_multiplier = 1.0
                    if priority == "high":
                        priority_multiplier = 1.5
                    elif priority == "urgent":
                        priority_multiplier = 2.5
                    
                    max_priority_fee = int(priority_fee * priority_multiplier * (1 + congestion_factor))
                else:
                    # Fallback if we don't have priority fee data
                    max_priority_fee = max(1, max_fee // 10)
            
            # Apply max cost constraint if specified
            if max_cost is not None:
                # Convert max_cost to wei (simplified, should use actual conversion)
                max_cost_wei = max_cost * 1e9  # Convert Gwei to wei
                
                # Ensure max_fee doesn't exceed max_cost
                if max_fee > max_cost_wei:
                    max_fee = int(max_cost_wei)
                    max_priority_fee = min(max_priority_fee, max_fee // 2)
            
            # Add gas parameters to result
            result.update({
                "type": "eip1559",
                "maxFeePerGas": max_fee,
                "maxPriorityFeePerGas": max_priority_fee,
                "baseFeePerGas": gas_data["baseFeePerGas"],
                "estimatedCostWei": max_fee * 21000,  # Simple transfer gas limit
                "gasLimit": 21000  # Standard gas limit for transfers
            })
        
        else:
            # Legacy transaction
            gas_price_key = f"{priority}_gasPrice"
            
            if gas_price_key in gas_data:
                gas_price = gas_data[gas_price_key]
            else:
                # Calculate based on current gas price and congestion
                base_gas_price = gas_data["gasPrice"]
                percentile = self.legacy_percentiles[priority]
                gas_price = self._calculate_legacy_gas_price(base_gas_price, percentile, chain)
                
                # Adjust for congestion
                gas_price = int(gas_price * (1 + congestion_factor * 0.5))
            
            # Apply max cost constraint if specified
            if max_cost is not None:
                # Convert max_cost to wei
                max_cost_wei = max_cost * 1e9
                
                # Ensure gas_price doesn't exceed max_cost
                if gas_price > max_cost_wei:
                    gas_price = int(max_cost_wei)
            
            # Add gas parameters to result
            result.update({
                "type": "legacy",
                "gasPrice": gas_price,
                "estimatedCostWei": gas_price * 21000,  # Simple transfer gas limit
                "gasLimit": 21000  # Standard gas limit for transfers
            })
        
        # Convert to human-readable format
        result["readableMaxFee"] = f"{result.get('maxFeePerGas', result.get('gasPrice', 0)) / 1e9:.2f} Gwei"
        if "maxPriorityFeePerGas" in result:
            result["readableMaxPriorityFee"] = f"{result['maxPriorityFeePerGas'] / 1e9:.2f} Gwei"
        result["readableEstimatedCost"] = f"{result['estimatedCostWei'] / 1e18:.8f} {self._get_chain_currency(chain)}"
        
        return result
    
    def _analyze_congestion(self, chain):
        """Analyze network congestion based on recent gas price history and pending transactions"""
        if not self.gas_price_history.get(chain):
            return 0  # Default to no congestion adjustment
        
        try:
            # Get the last few entries from history
            history = self.gas_price_history[chain]
            if len(history) < 2:
                return 0
            
            # Calculate rate of change in base fee or gas price
            current = history[-1]
            previous = history[-2]
            
            if "baseFeePerGas" in current and "baseFeePerGas" in previous:
                # Use base fee for EIP-1559 chains
                current_price = current["baseFeePerGas"]
                previous_price = previous["baseFeePerGas"]
            else:
                # Use gas price for legacy chains
                current_price = current["gasPrice"]
                previous_price = previous["gasPrice"]
            
            # Calculate percentage change
            if previous_price > 0:
                price_change = (current_price - previous_price) / previous_price
            else:
                price_change = 0
            
            # Check pending transaction count if available
            pending_ratio = 0
            if "pending_tx_count" in current:
                # Compare to normal levels (this would be better with historical averages)
                pending_count = current["pending_tx_count"]
                
                # Simple heuristic: assume >500 pending is high congestion
                # Normalize to 0-1 range
                pending_ratio = min(1, pending_count / 500)
            
            # Combine signals (price change and pending ratio)
            congestion_factor = 0.7 * abs(price_change) + 0.3 * pending_ratio
            
            # Apply sigmoid function to get a value between 0 and 1
            congestion_factor = 1 / (1 + exp(-5 * congestion_factor))
            
            # Scale to reasonable range (0 to 0.5)
            return congestion_factor * 0.5
            
        except Exception as e:
            logger.error(f"Error analyzing congestion: {e}")
            return 0
    
    def _get_chain_currency(self, chain):
        """Get native currency for a chain"""
        currencies = {
            "ethereum": "ETH",
            "polygon": "MATIC",
            "arbitrum": "ETH",
            "optimism": "ETH"
        }
        return currencies.get(chain, "ETH")
    
    def estimate_gas_limit(self, chain, from_address, to_address, value=0, data="0x"):
        """Estimate gas limit for a transaction"""
        if chain not in self.providers:
            logger.error(f"Unsupported chain: {chain}")
            return 21000  # Default gas limit for transfers
        
        try:
            w3 = self.providers[chain]
            
            # Prepare transaction parameters
            tx_params = {
                "from": from_address,
                "to": to_address,
                "value": value
            }
            
            if data and data != "0x":
                tx_params["data"] = data
            
            # Estimate gas
            gas_limit = w3.eth.estimate_gas(tx_params)
            
            # Add safety margin
            gas_limit = int(gas_limit * 1.2)  # 20% safety margin
            
            return gas_limit
            
        except Exception as e:
            logger.error(f"Error estimating gas limit: {e}")
            
            # Default gas limits based on transaction type
            if data and data != "0x":
                return 150000  # Default for contract interaction
            else:
                return 21000  # Default for simple transfers
    
    def fetch_gas_oracle_data(self, chain="ethereum"):
        """Fetch gas price data from external gas oracles"""
        try:
            # Different APIs for different chains
            if chain == "ethereum":
                # Try Etherscan API if we have a key
                etherscan_key = os.getenv("ETHERSCAN_API_KEY")
                if etherscan_key:
                    url = f"https://api.etherscan.io/api?module=gastracker&action=gasoracle&apikey={etherscan_key}"
                    response = requests.get(url, timeout=5)
                    
                    if response.status_code == 200:
                        data = response.json()
                        if data["status"] == "1":
                            result = data["result"]
                            return {
                                "source": "etherscan",
                                "timestamp": int(time.time()),
                                "safeLow": int(float(result["SafeGasPrice"]) * 1e9),
                                "standard": int(float(result["ProposeGasPrice"]) * 1e9),
                                "fast": int(float(result["FastGasPrice"]) * 1e9),
                            }
                
                # Try ETH Gas Station Compatible API
                url = "https://ethgasstation.info/api/ethgasAPI.json"
                response = requests.get(url, timeout=5)
                
                if response.status_code == 200:
                    data = response.json()
                    # ETH Gas Station returns values in 10 Gwei
                    return {
                        "source": "ethgasstation",
                        "timestamp": int(time.time()),
                        "safeLow": int(data["safeLow"] * 1e8),
                        "standard": int(data["average"] * 1e8),
                        "fast": int(data["fast"] * 1e8),
                        "fastest": int(data["fastest"] * 1e8),
                    }
            
            elif chain == "polygon":
                # Try Polygon Gas Station
                url = "https://gasstation-mainnet.matic.network/v2"
                response = requests.get(url, timeout=5)
                
                if response.status_code == 200:
                    data = response.json()
                    return {
                        "source": "polygon_gasstation",
                        "timestamp": int(time.time()),
                        "safeLow": int(data["safeLow"]["maxFee"] * 1e9),
                        "standard": int(data["standard"]["maxFee"] * 1e9),
                        "fast": int(data["fast"]["maxFee"] * 1e9),
                        "maxPriorityFee": {
                            "safeLow": int(data["safeLow"]["maxPriorityFee"] * 1e9),
                            "standard": int(data["standard"]["maxPriorityFee"] * 1e9),
                            "fast": int(data["fast"]["maxPriorityFee"] * 1e9)
                        }
                    }
            
            # Fallback to our own calculations
            return self.update_gas_price(chain)
            
        except Exception as e:
            logger.error(f"Error fetching gas oracle data: {e}")
            return None
    
    def calculate_optimal_transaction_fee(self, chain, transaction_type="transfer", priority=None, time_constraint=None):
        """
        Calculate optimal transaction fee based on market conditions.
        
        Args:
            chain: Target blockchain
            transaction_type: Type of transaction (transfer, swap, contract_interaction)
            priority: Priority level (low, medium, high, urgent)
            time_constraint: Maximum acceptable confirmation time in seconds
            
        Returns:
            Optimized transaction fee parameters
        """
        # Default gas limits by transaction type
        gas_limits = {
            "transfer": 21000,
            "swap": 150000,
            "contract_interaction": 100000,
            "nft_mint": 200000,
            "complex_contract": 300000
        }
        gas_limit = gas_limits.get(transaction_type, 100000)
        
        # Convert time constraint to priority if provided
        if time_constraint is not None and priority is None:
            if time_constraint < 15:  # Need confirmation in under 15 seconds
                priority = "urgent"
            elif time_constraint < 60:  # Under 1 minute
                priority = "high"
            elif time_constraint < 300:  # Under 5 minutes
                priority = "medium"
            else:
                priority = "low"
        
        # Get optimized gas price
        gas_data = self.optimize_gas_price(chain, priority)
        if not gas_data:
            logger.error(f"Failed to optimize gas price for {chain}")
            return None
        
        # Calculate total fee in wei
        if gas_data["type"] == "eip1559":
            max_fee = gas_data["maxFeePerGas"]
            max_priority_fee = gas_data["maxPriorityFeePerGas"]
            estimated_fee_wei = max_fee * gas_limit
            
            result = {
                "type": "eip1559",
                "chain": chain,
                "gasLimit": gas_limit,
                "maxFeePerGas": max_fee,
                "maxPriorityFeePerGas": max_priority_fee,
                "totalFeeWei": estimated_fee_wei,
                "totalFee": estimated_fee_wei / 1e18,
                "currency": self._get_chain_currency(chain),
                "confirmation_time": self.target_blocks[gas_data["priority"]] * 12,  # Estimated seconds
                "priority": gas_data["priority"]
            }
        else:
            gas_price = gas_data["gasPrice"]
            estimated_fee_wei = gas_price * gas_limit
            
            result = {
                "type": "legacy",
                "chain": chain,
                "gasLimit": gas_limit,
                "gasPrice": gas_price,
                "totalFeeWei": estimated_fee_wei,
                "totalFee": estimated_fee_wei / 1e18,
                "currency": self._get_chain_currency(chain),
                "confirmation_time": self.target_blocks[gas_data["priority"]] * 12,  # Estimated seconds
                "priority": gas_data["priority"]
            }
        
        # Add human-readable values
        if "maxFeePerGas" in result:
            result["readableMaxFee"] = f"{result['maxFeePerGas'] / 1e9:.2f} Gwei"
        if "maxPriorityFeePerGas" in result:
            result["readableMaxPriorityFee"] = f"{result['maxPriorityFeePerGas'] / 1e9:.2f} Gwei"
        if "gasPrice" in result:
            result["readableGasPrice"] = f"{result['gasPrice'] / 1e9:.2f} Gwei"
        
        result["readableTotalFee"] = f"{result['totalFee']:.8f} {result['currency']}"
        
        return result


# Example usage
if __name__ == "__main__":
    # Initialize fee optimizer
    optimizer = TxFeeOptimizer()
    
    # Update gas prices
    eth_gas = optimizer.update_gas_price("ethereum")
    print("Ethereum Gas Data:", json.dumps(eth_gas, indent=2))
    
    # Optimize gas price for Ethereum transaction
    optimal_gas = optimizer.optimize_gas_price("ethereum", priority="medium")
    print("Optimal Gas Parameters:", json.dumps(optimal_gas, indent=2))
    
    # Calculate optimal fee for a swap transaction
    optimal_fee = optimizer.calculate_optimal_transaction_fee("ethereum", "swap", "high")
    print("Optimal Transaction Fee:", json.dumps(optimal_fee, indent=2))
